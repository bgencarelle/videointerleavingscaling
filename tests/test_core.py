import json
import math

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from faceloop.colour import (GammaCurve, ParametricCurve, SRGBCurve, TableCurve, Transfer,
                             curves_from_icc, quantize)
from faceloop.engine import run_pass, run_cycle, StepContext, Controls, Progress
from faceloop.kernels import METHODS, KernelResizer, weights_1d
from faceloop.recipe import Recipe, RecipeError, auto_finish_steps
from faceloop.sizes import Size, aspect_error, height_for, size_for


# ------------------------------------------------------------------ sizes
def test_heights_follow_aspect():
    src = Size(960, 720)
    assert height_for(40, src) == 30
    assert height_for(20, src) == 15
    assert size_for(960, src) == src
    odd = Size(1437, 1080)
    s = size_for(40, odd)
    assert s.height == 30 and aspect_error(s, odd) < 0.5
    wide = Size(1920, 1080)
    assert height_for(64, wide) == 36


# ------------------------------------------------------------------ colour
def _levels():
    return torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16) / 255.0


@pytest.mark.parametrize("curve", [
    SRGBCurve(), GammaCurve(2.2),
    ParametricCurve(3, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045]),
    ParametricCurve(0, [1.8]),
    TableCurve([round(65535 * (i / 1023) ** 2.2) for i in range(1024)]),
])
def test_every_8bit_level_survives_decode_encode(curve):
    x = _levels()
    back = curve.encode(curve.decode(x))
    assert torch.equal(torch.round(back * 255), torch.round(x * 255))


def test_para3_srgb_matches_analytic():
    p = ParametricCurve(3, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
    x = _levels()
    assert torch.allclose(p.decode(x), SRGBCurve().decode(x), atol=2e-5)


def test_icc_curves_from_real_profile():
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    curves, warn = curves_from_icc(icc)
    assert warn is None
    x = _levels()
    for c in curves:
        assert torch.allclose(c.decode(x), SRGBCurve().decode(x), atol=2e-3)
    curves, warn = curves_from_icc(None)
    assert warn and isinstance(curves[0], SRGBCurve)


def test_quantize_modes():
    t = Transfer()
    x = torch.rand(1, 3, 8, 8)
    q = quantize(x, "srgb", t)
    assert torch.equal(q, torch.round(q * 255) / 255)
    ql = quantize(x, "linear", t)
    lin = t.decode(ql)
    assert torch.allclose(lin * 255, torch.round(lin * 255), atol=2e-3)
    # linear rounding is coarser in the shadows
    dark = torch.linspace(0, 0.1, 200).view(1, 1, 1, 200).expand(1, 3, 1, 200)
    assert quantize(dark, "linear", t).unique().numel() < quantize(dark, "srgb", t).unique().numel()


# ------------------------------------------------------------------ kernels
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_in,n_out", [(960, 40), (40, 960), (240, 160), (30, 45), (7, 3)])
def test_rows_sum_to_one(method, n_in, n_out):
    W = weights_1d(n_in, n_out, method)
    assert torch.allclose(W.sum(1), torch.ones(n_out, dtype=W.dtype))


def test_area_is_block_mean_and_exact_roundtrip():
    x = torch.rand(1, 3, 48, 64)
    r = KernelResizer()
    down = r.resize(x, 16, 12, "area")
    ref = x.view(1, 3, 12, 4, 16, 4).mean(dim=(3, 5))
    assert torch.allclose(down, ref, atol=1e-6)
    up = r.resize(down, 64, 48, "area")
    assert torch.allclose(r.resize(up, 16, 12, "area"), down, atol=1e-6)
    assert torch.allclose(up, r.resize(down, 64, 48, "nearest"), atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_flat_image_stays_flat(method):
    x = torch.full((1, 3, 30, 40), 0.37)
    r = KernelResizer()
    for w, h in [(13, 10), (100, 75), (40, 30)]:
        x = r.resize(x, w, h, method)
        assert torch.allclose(x, torch.full_like(x, 0.37), atol=1e-5)


def _centroid(img):
    m = img[0, 0].double()
    h, w = m.shape
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    return float((m * xs).sum() / m.sum()), float((m * ys).sum() / m.sum())


@pytest.mark.parametrize("down", METHODS)
@pytest.mark.parametrize("up", METHODS)
def test_no_drift_over_many_cycles(down, up):
    """The face must not slide: 40 round trips 240 -> 40 -> 240 keep the centroid."""
    h, w = 180, 240
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    blob = torch.exp(-(((xs - 131.0) / 20) ** 2 + ((ys - 87.0) / 16) ** 2))
    x = blob.view(1, 1, h, w).repeat(1, 3, 1, 1)
    c0 = _centroid(x)
    r = KernelResizer()
    for _ in range(40):
        x = r.resize(x, 40, 30, down).clamp(0, 1)
        x = r.resize(x, w, h, up).clamp(0, 1)
    c1 = _centroid(x)
    assert abs(c1[0] - c0[0]) < 1.5 and abs(c1[1] - c0[1]) < 1.5, (c0, c1)


def test_no_antialias_aliases_more():
    # fine stripes: a proper filter averages them out, the no-AA one does not
    x = torch.zeros(1, 3, 64, 256)
    x[..., 0::2] = 1.0
    r = KernelResizer()
    aa = r.resize(x, 26, 64, "bilinear")
    na = r.resize(x, 26, 64, "bilinear", antialias=False)
    assert aa.std() < 0.05 and na.std() > 0.2


def test_fixed_point_differs_but_stays_close():
    x = torch.rand(1, 3, 30, 40)
    r = KernelResizer()
    a = r.resize(x, 97, 73, "bicubic")
    b = r.resize(x, 97, 73, "bicubic", fixed_point=True)
    d = (a - b).abs()
    assert d.max() > 1 / 255 and d.mean() < 0.05


# ------------------------------------------------------------------ recipe
def _recipe(**kw):
    d = {
        "lead_in": [{"op": "resize", "method": "area", "width": 32}],
        "finish": [{"op": "resize", "method": "lanczos", "width": "source"}],
        "cycles": {
            "A": [{"op": "resize", "method": "area", "width": 16},
                  {"op": "resize", "method": "lanczos", "width": "loop"}],
            "B": [{"op": "resize", "method": "nearest", "width": 8}],
        },
        "pattern": ["A", "B"], "iterations": 5, "save_every": 2,
    }
    d.update(kw)
    return Recipe.from_dict(d)


def test_recipe_code_ignores_iterations_and_label_only_in_slug():
    a, b = _recipe(), _recipe(iterations=50, save_every=10)
    assert a.code() == b.code()
    assert a.code() != _recipe(rounding="linear").code()
    assert "rlin" in _recipe(rounding="linear").slug()
    assert a.save_points() == [2, 4, 5]
    assert [a.cycle_at(i) for i in range(1, 6)] == ["A", "B", "A", "B", "A"]
    assert _recipe(name="my test").slug().startswith("my-test_")


@pytest.mark.parametrize("bad", [
    {"pattern": ["C"]},
    {"iterations": 0},
    {"cycles": {"A": [{"op": "resize", "method": "sinc", "width": 4}]}, "pattern": ["A"]},
    {"cycles": {"A": [{"op": "resize", "method": "area", "width": -3}]}, "pattern": ["A"]},
    {"rounding": "dither"},
])
def test_recipe_rejects_bad(bad):
    with pytest.raises((RecipeError, KeyError)):
        _recipe(**bad)


# ------------------------------------------------------------------ engine
def _make_source(tmp_path, n=8, w=64, h=48):
    src = tmp_path / "faces_src"
    src.mkdir()
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rng = np.random.default_rng(1)
    for i in range(n):
        img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(src / f"frame{i:04d}.png", icc_profile=icc)
    return src, icc


def test_run_pass_end_to_end_and_resume(tmp_path):
    src, icc = _make_source(tmp_path)
    out = tmp_path / "out"
    r = _recipe()
    res = run_pass(r, src, out, device=torch.device("cpu"), batch_size=3)
    folder = res.folder
    for c in ("c0002", "c0004", "c0005"):
        files = sorted((folder / c).glob("*.webp"))
        assert len(files) == 8
        im = Image.open(files[0])
        assert im.size == (64, 48)
        assert im.info.get("icc_profile") == icc
    assert not (folder / "c0003").exists()
    rec = json.loads((folder / "recipe.json").read_text())
    assert rec["code"] == r.code()
    import csv
    rows = list(csv.DictReader(open(out / "catalog.csv")))
    assert [row["code"] for row in rows] == [r.code()]
    again = run_pass(r, src, out, device=torch.device("cpu"), batch_size=3)
    assert again.skipped_frames == 8


def test_cycle_lands_on_source_size():
    ctx = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                      resizer=KernelResizer(), models=None)
    x = torch.rand(2, 3, 48, 64)
    y = run_cycle(x, [{"op": "resize", "method": "bicubic", "width": 20}], ctx)
    assert y.shape == x.shape
    assert torch.equal(y, torch.round(y * 255) / 255)


def test_stop_and_resume(tmp_path):
    src, _ = _make_source(tmp_path)
    out = tmp_path / "out"
    r = _recipe(iterations=4)
    ctl = Controls()
    calls = {"n": 0}

    def log(_):
        calls["n"] += 1
        if "batch 1/" in _:
            ctl.stop()

    res = run_pass(r, src, out, device=torch.device("cpu"), batch_size=4, controls=ctl, log=log)
    assert res.stopped
    res2 = run_pass(r, src, out, device=torch.device("cpu"), batch_size=4)
    assert res2.skipped_frames == 4 and not res2.stopped
    assert len(list((res2.folder / "c0004").glob("*.webp"))) == 8


def test_linear_light_differs_from_encoded(tmp_path):
    ctx_lin = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                          resizer=KernelResizer(), models=None)
    ctx_enc = StepContext(src=Size(64, 48), transfer=Transfer(), light="encoded", rounding="srgb",
                          resizer=KernelResizer(), models=None)
    x = torch.zeros(1, 3, 48, 64)
    x[..., ::2, ::2] = 1.0  # fine bright detail
    steps = [{"op": "resize", "method": "area", "width": 16}]
    a = run_cycle(x, steps, ctx_lin)
    b = run_cycle(x, steps, ctx_enc)
    assert a.mean() > b.mean()  # linear light keeps bright detail brighter


# ------------------------------------------------------------------ new pass shape
def test_auto_finish_chain():
    def plan(cur, threshold=2.0):
        return [(s.get("model", s.get("method")), s["width"]) for s in auto_finish_steps(cur, 810, threshold)]
    assert plan(162) == [("swinir-classical-x4", 648), ("area", "source")]
    assert plan(40) == [("swinir-classical-x4", 160), ("swinir-classical-x4", 640), ("area", "source")]
    assert plan(300) == [("swinir-classical-x2", 600), ("area", "source")]
    assert plan(450) == [("area", "source")]                      # under 2x left: no AI
    assert plan(450, 1.5) == [("swinir-classical-x2", 900), ("area", "source")]
    assert plan(810) == [("area", "source")]


def test_cycles_run_at_loop_size_and_saves_are_full_size(tmp_path):
    src, _ = _make_source(tmp_path)
    sizes = []

    def on_contact(cycle, step_i, names, imgs):
        sizes.append((cycle, step_i, imgs.shape[1:3]))

    r = _recipe(iterations=2, save_every=1)
    res = run_pass(r, src, tmp_path / "out", device=torch.device("cpu"), batch_size=8, on_contact=on_contact)
    # cycle A: area 16 -> 12x16, lanczos back to the loop size 32x24
    assert ("A", 0, (12, 16)) in sizes and ("A", 1, (24, 32)) in sizes
    # cycle B: nearest 8, then the automatic landing back on the loop size
    assert ("B", 0, (6, 8)) in sizes and ("B", 1, (24, 32)) in sizes
    for c in ("c0001", "c0002"):
        im = Image.open(next((res.folder / c).glob("*.webp")))
        assert im.size == (64, 48)


def test_lead_in_cannot_use_loop_width():
    with pytest.raises(RecipeError):
        _recipe(lead_in=[{"op": "resize", "method": "area", "width": "loop"}])
    with pytest.raises(RecipeError):
        _recipe(lead_in=[])


def test_finish_changes_code_but_not_loop():
    a = _recipe()
    b = _recipe(finish="auto")
    assert a.code() != b.code()
    assert b.slug().endswith("_swin")
    assert "swinir-classical-x4" in b.model_ids()
