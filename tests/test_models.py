import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from faceloop.colour import Transfer
from faceloop.engine import StepContext, run_cycle
from faceloop.kernels import KernelResizer
from faceloop.models import ModelInfo, ModelManager, ModelRunner
from faceloop.recipe import Recipe, RecipeError, model_short
from faceloop.sizes import Size

WEIGHTS = Path(os.environ.get("FACELOOP_WEIGHTS", Path(__file__).resolve().parents[1] / "weights"))


class NearestX2(torch.nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, 2).repeat_interleave(2, 3)


class FakeManager:
    def __init__(self):
        info = ModelInfo("fake-x2", "dropin", "accuracy", 2, "test")
        self.runner = ModelRunner(info, NearestX2(), 2, torch.device("cpu"))

    def get(self, mid, device, precision="fp32"):
        return self.runner

    def known_ids(self):
        return {"fake-x2"}

    def gan_ids(self):
        return set()

    def describe_used(self, recipe):
        return {}


def test_catalog_is_complete_and_pinned():
    data = json.loads((Path(__file__).resolve().parents[1] / "faceloop/models/catalog.json").read_text())
    ids = [m["id"] for m in data["models"]]
    assert len(ids) == len(set(ids))
    for m in data["models"]:
        assert m["label"] in ("GAN", "accuracy")
        for f in m["files"]:
            if "manual" in f:  # weights only on a cloud drive: you download them yourself
                assert m["family"] == "irn" and f["manual"].startswith("https://")
                continue
            assert len(f["sha256"]) == 64
            assert f["url"].startswith("https://github.com/") or \
                f["url"].startswith("https://raw.githubusercontent.com/nihui/")
            if "raw.githubusercontent" in f["url"]:
                # pinned to a commit, never a branch
                assert len(f["url"].split("/")[5]) == 40


def test_model_step_lands_on_its_width():
    ctx = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                      resizer=KernelResizer(), models=FakeManager())
    x = torch.rand(2, 3, 48, 64)
    seen = []
    run_cycle(x, [{"op": "resize", "method": "area", "width": 16},
                  {"op": "model", "model": "fake-x2", "width": 40},
                  {"op": "model", "model": "fake-x2"}],
              ctx, capture=lambda i, t: seen.append(tuple(t.shape[-2:])))
    assert seen == [(12, 16), (30, 40), (60, 80)]  # the cycle ends where its last step ends


def test_runner_falls_back_to_smaller_batches_and_tiles():
    calls = {"n": 0}

    class Picky(torch.nn.Module):
        def forward(self, x):
            calls["n"] += 1
            if x.shape[0] > 1 or x.shape[-1] > 300:
                raise RuntimeError("MPS backend out of memory (fake)")
            return x.repeat_interleave(2, 2).repeat_interleave(2, 3)

    r = ModelRunner(ModelInfo("p", "dropin", "accuracy", 2, ""), Picky(), 2, torch.device("cpu"))
    x = torch.rand(3, 3, 200, 400)
    y = r(x)
    assert y.shape == (3, 3, 400, 800)
    assert torch.equal(y, NearestX2()(x))
    assert r.micro == 1 and r.tile is not None


def test_runner_pads_to_model_size_rules():
    class Mult8(torch.nn.Module):
        def forward(self, x):
            assert x.shape[-1] % 8 == 0 and x.shape[-2] % 8 == 0
            return NearestX2()(x)

    r = ModelRunner(ModelInfo("m", "dropin", "accuracy", 2, ""), Mult8(), 2, torch.device("cpu"), multiple_of=8)
    y = r(torch.rand(1, 3, 30, 45))
    assert y.shape == (1, 3, 60, 90)


def test_short_names_and_gan_flag():
    assert model_short("realesr-animevideov3") == "avv3"
    assert model_short("w2x-cunet-n1") == "w2xcunetn1"
    r = Recipe.from_dict({"lead_in": [{"op": "resize", "method": "area", "width": 160}],
                          "finish": [{"op": "resize", "method": "area", "width": "source"}],
                          "cycles": {"A": [{"op": "resize", "method": "area", "width": 40},
                                           {"op": "model", "model": "realesr-animevideov3", "width": 160}]},
                          "pattern": ["A"], "iterations": 3, "save_every": 1})
    assert r.slug({"realesr-animevideov3"}).endswith("_GAN")
    assert "avv3@160" in r.slug()


def test_dropin_labels(tmp_path):
    d = tmp_path / "dropin"
    d.mkdir()
    (d / "Mine_x4.pth").write_bytes(b"")
    (d / "Labelled.pth").write_bytes(b"")
    (d / "Labelled.json").write_text('{"label": "accuracy"}')
    mm = ModelManager(tmp_path)
    assert mm.catalog["dropin-Mine_x4"].label == "unlabelled"
    assert "dropin-Mine_x4" in mm.gan_ids()          # unlabelled counts as GAN
    assert "dropin-Labelled" not in mm.gan_ids()


# ---------------------------------------------------------------- with weights
needs_w2x = pytest.mark.skipif(not (WEIGHTS / "waifu2x/models-cunet/scale2.0x_model.bin").exists(),
                               reason="waifu2x weights not downloaded")


@needs_w2x
def test_waifu2x_port_matches_ncnn_reference():
    ncnn = pytest.importorskip("ncnn")
    from faceloop.models.ncnn_port import Waifu2x, model_files
    root = WEIGHTS / "waifu2x"
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((1, 3, 31, 45)).astype(np.float32))
    x = torch.round(F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1) * 255) / 255
    for variant, noise in [("cunet", 1), ("anime", 0), ("photo", -1)]:
        m = Waifu2x(root, variant, noise, 2)
        p, a = m.prepadding, m.align
        h, w = x.shape[-2:]
        xp = F.pad(x, (p, p + (-w) % a, p, p + (-h) % a), mode="replicate")
        net = ncnn.Net()
        for k in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage"):
            setattr(net.opt, k, False)
        pf, bf = model_files(variant, noise, 2)
        net.load_param(str(root / pf))
        net.load_model(str(root / bf))
        arr = np.ascontiguousarray(xp[0].numpy())
        mat = ncnn.Mat(arr)
        ex = net.create_extractor()
        ex.input("Input1", mat)
        _, out = ex.extract("Eltwise4")
        ref = torch.from_numpy(np.array(out))[None][..., :h * 2, :w * 2].clamp(0, 1)
        assert float((m(x) - ref).abs().max()) < 1e-4


@needs_w2x
def test_waifu2x_does_not_shift_the_picture():
    from faceloop.models.ncnn_port import Waifu2x
    ys, xs = torch.meshgrid(torch.arange(48) + 0.5, torch.arange(64) + 0.5, indexing="ij")
    blob = 0.2 + 0.6 * torch.exp(-(((xs - 33.3) / 7) ** 2 + ((ys - 22.7) / 6) ** 2))
    x = torch.round(blob.view(1, 1, 48, 64).repeat(1, 3, 1, 1) * 255) / 255
    y = Waifu2x(WEIGHTS / "waifu2x", "cunet", -1, 2)(x)

    def c(img, s):
        m = img[0].mean(0).double()
        m = (m - m.median()).clamp(min=0)
        yy, xx = torch.meshgrid(torch.arange(m.shape[0]) + 0.5, torch.arange(m.shape[1]) + 0.5, indexing="ij")
        return float((m * xx).sum() / m.sum()) / s, float((m * yy).sum() / m.sum()) / s

    (ax, ay), (bx, by) = c(x, 1), c(y, 2)
    assert abs(ax - bx) < 0.1 and abs(ay - by) < 0.1


def test_half_precision_runner_casts_and_returns_float32():
    seen = {}

    class Rec(torch.nn.Module):
        def forward(self, x):
            seen["dtype"] = x.dtype
            return NearestX2()(x)

    r = ModelRunner(ModelInfo("h", "dropin", "accuracy", 2, ""), Rec(), 2, torch.device("cpu"),
                    dtype=torch.bfloat16)
    y = r(torch.rand(2, 3, 8, 8))
    assert seen["dtype"] == torch.bfloat16 and y.dtype == torch.float32


def test_precision_in_recipe():
    base = {"lead_in": [{"op": "resize", "method": "area", "width": 16}], "finish": [],
            "pattern": ["A"], "iterations": 1, "save_every": 1}
    plain = {"A": [{"op": "resize", "method": "nearest", "width": 8}]}
    model = {"A": [{"op": "model", "model": "w2x-cunet", "width": "loop"}]}
    # without models precision changes nothing
    assert Recipe.from_dict({**base, "cycles": plain}).code() == \
        Recipe.from_dict({**base, "cycles": plain, "precision": "fp16"}).code()
    a = Recipe.from_dict({**base, "cycles": model})
    b = Recipe.from_dict({**base, "cycles": model, "precision": "fp16"})
    assert a.code() != b.code() and b.slug().endswith("_h16")


@needs_w2x
def test_waifu2x_half_precision_is_close():
    mm = ModelManager(WEIGHTS)
    dev = torch.device("cpu")
    x = torch.rand(1, 3, 24, 32)
    a = mm.get("w2x-cunet", dev, "fp32")(x)
    b = mm.get("w2x-cunet", dev, "fp16")(x)
    assert b.dtype == torch.float32
    assert float((a - b).abs().mean()) < 2 / 255


def test_half_precision_falls_back_when_a_model_mixes_types():
    """Like SwinIR: builds a float32 tensor inside forward, so converting the
    weights to half fails; autocast (or full precision) must take over."""
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            y = self.conv(x)
            mask = torch.ones(y.shape[-1], y.shape[-1])  # float32, whatever the weights are
            return NearestX2()(torch.matmul(y, mask) / y.shape[-1])

    r = ModelRunner(ModelInfo("m", "dropin", "accuracy", 2, ""), Mixed(), 2, torch.device("cpu"))
    ModelManager._make_half(ModelManager.__new__(ModelManager), r)
    assert "autocast" in r.precision_label or r.precision_label.startswith("float32")
    y = r(torch.rand(1, 3, 8, 8))
    assert y.shape == (1, 3, 16, 16) and y.dtype == torch.float32


def _fake_irn_weights(tmp_path, scale, name=None):
    from faceloop.models.irn import InvRescaleNet
    torch.manual_seed(scale)
    net = InvRescaleNet(scale)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if "haar" not in n:
                p.normal_(0, 0.01)
    d = tmp_path / "irn"
    d.mkdir(exist_ok=True)
    torch.save(net.state_dict(), d / (name or f"IRN_x{scale}.pth"))


def test_irn_steps_shrink_and_enlarge(tmp_path):
    _fake_irn_weights(tmp_path, 4)
    mm = ModelManager(tmp_path)
    assert mm.is_downloaded("irn-x4") and not mm.is_downloaded("irn-x2")
    ok, msg = mm.check("irn-x4", torch.device("cpu"))
    assert ok, msg
    ctx = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                      resizer=KernelResizer(), models=mm)
    seen = []
    run_cycle(torch.rand(2, 3, 48, 64), [{"op": "irn", "direction": "down", "scale": 4},
                                         {"op": "irn", "direction": "up", "scale": 4, "width": 40}],
              ctx, capture=lambda i, t: seen.append(tuple(t.shape[-2:])))
    assert seen == [(12, 16), (30, 40)]
    with pytest.raises(FileNotFoundError):
        mm.get_irn("irn-x2", torch.device("cpu"))


def test_irn_variants(tmp_path):
    _fake_irn_weights(tmp_path, 8)
    _fake_irn_weights(tmp_path, 4, "IRN+_x4.pth")
    _fake_irn_weights(tmp_path, 2, "IRN_x2_finetune.pth")
    mm = ModelManager(tmp_path)
    for mid in ("irn-x8", "irnplus-x4", "irn-x2-finetune"):
        ok, msg = mm.check(mid, torch.device("cpu"))
        assert ok, (mid, msg)
    assert "irnplus-x4" in mm.gan_ids() and "irn-x8" not in mm.gan_ids()
    m = mm.get_irn("irn-x8", torch.device("cpu"))
    assert m.down(torch.rand(1, 3, 64, 48)).shape == (1, 3, 8, 6)
    base = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "finish": [],
            "pattern": ["A"], "iterations": 1, "save_every": 1}
    r = Recipe.from_dict({**base, "cycles": {"A": [{"op": "irn", "direction": "down", "scale": 4, "variant": "plus"}]}})
    assert r.model_ids() == {"irnplus-x4"} and "irnPD4" in r.slug() and r.slug(mm.gan_ids()).endswith("_GAN")
    with pytest.raises(RecipeError):
        Recipe.from_dict({**base, "cycles": {"A": [{"op": "irn", "direction": "down", "scale": 2, "variant": "plus"}]}})


def test_irn_in_recipes():
    base = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "finish": [],
            "pattern": ["A"], "iterations": 1, "save_every": 1}
    r = Recipe.from_dict({**base, "cycles": {"A": [{"op": "irn", "direction": "down", "scale": 2},
                                                   {"op": "irn", "direction": "up", "scale": 2, "width": "loop"}]}})
    assert r.model_ids() == {"irn-x2"} and "irnD2-irnU2@L" in r.slug()
    for bad in ({"op": "irn", "direction": "down", "scale": 3}, {"op": "irn", "direction": "sideways", "scale": 2}):
        with pytest.raises(RecipeError):
            Recipe.from_dict({**base, "cycles": {"A": [bad]}})
