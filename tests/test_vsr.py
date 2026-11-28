"""BasicVSR step: frames mix only when asked, inside a batch, in frame order."""
import numpy as np
import pytest
import torch

from faceloop.models import ModelManager, default_weights_dir
from faceloop.models.basicvsr import BasicVSRNet, plan_windows, run_vsr
from faceloop.recipe import Recipe, RecipeError


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    n = BasicVSRNet(mid=8, blocks=2)          # tiny random network: same code paths
    for p in n.parameters():
        p.data.mul_(0.3)
    return n.eval()


@pytest.fixture(scope="module")
def frames():
    torch.manual_seed(1)
    base = torch.rand(1, 3, 20, 24)
    return torch.cat([torch.roll(base, (k, 2 * k), (2, 3)) for k in range(6)], 0)


NAMES = [f"f{k}.png" for k in range(6)]


def test_window_plan():
    assert list(plan_windows(0, 6, 1)) == [(k, k, k + 1) for k in range(6)]
    assert list(plan_windows(0, 6, 4)) == [(0, 0, 4), (1, 4, 6)]
    # counted from the sequence's frame 0, cut at the batch end
    assert list(plan_windows(5, 6, 4)) == [(1, 0, 3), (2, 3, 6)]


def test_full_window_is_the_network_as_trained(net, frames):
    whole = net(frames.unsqueeze(0))[0]
    got = run_vsr(net, frames, NAMES, 0, memory=6)
    assert torch.allclose(whole, got, atol=1e-6)


def test_default_is_every_frame_alone(net, frames):
    got = run_vsr(net, frames, NAMES, 0)
    assert got.shape == (6, 3, 80, 96)
    for k in range(6):
        alone = net(frames[k:k + 1].unsqueeze(0))[0]
        assert torch.allclose(got[k:k + 1], alone, atol=1e-6)
    other = frames.clone()
    other[3] = 0.5
    assert torch.equal(run_vsr(net, other, NAMES, 0)[[0, 1, 2, 4, 5]], got[[0, 1, 2, 4, 5]])


def test_no_names_no_mixing(net, frames):
    assert torch.allclose(run_vsr(net, frames, None, 0, memory=6), run_vsr(net, frames, NAMES, 0), atol=1e-6)


def test_mixing_stays_inside_its_window(net, frames):
    a = run_vsr(net, frames, NAMES, 0, memory=3)
    other = frames.clone()
    other[4] = 0.5                                       # window 2 (frames 3-5)
    b = run_vsr(net, other, NAMES, 0, memory=3)
    assert torch.equal(a[:3], b[:3])
    assert not torch.equal(a[3], b[3])                    # its window-mates do see it


def test_frame_number_order_from_names(net, frames):
    names = ["f10.png", "f2.png", "f9.png", "f1.png"]   # natural order: f1, f2, f9, f10
    x = frames[:4]
    got = run_vsr(net, x, names, 0, memory=4)
    order = [3, 1, 2, 0]
    ref = net(x[order].unsqueeze(0))[0]
    for pos, k in enumerate(order):
        assert torch.allclose(got[k], ref[pos], atol=1e-6)


@pytest.mark.parametrize("order", ["reverse", "shuffle"])
def test_reordered_frames_go_back_to_their_own_place(net, frames, order):
    got = run_vsr(net, frames, NAMES, 0, memory=6, order=order, seed=3)
    assert torch.equal(got, run_vsr(net, frames, NAMES, 0, memory=6, order=order, seed=3))
    up = torch.nn.functional.interpolate(frames, scale_factor=4, mode="bilinear", align_corners=False)
    for k in range(6):   # each output is still its own frame, not a neighbour
        d = [(got[k] - up[j]).abs().mean() for j in range(6)]
        assert int(np.argmin(d)) == k


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_one_direction(net, frames, direction):
    got = run_vsr(net, frames, NAMES, 0, memory=6, direction=direction)
    alone = net(frames[0:1].unsqueeze(0))[0]
    edge = 0 if direction == "forward" else 5          # the first frame it meets has no memory yet
    assert torch.allclose(got[edge], net(frames[edge:edge + 1].unsqueeze(0))[0], atol=1e-6)
    assert alone.shape == got[0:1].shape


def test_flow_zero(net, frames):
    a = run_vsr(net, frames, NAMES, 0, memory=6, flow=0.0)
    assert a.shape == (6, 3, 80, 96) and torch.isfinite(a).all()


BASE = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A"], "iterations": 2,
        "save_every": 2, "finish": []}


def test_recipe_checks_and_slugs():
    r = Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "resize", "method": "area", "width": 8},
                                                   {"op": "vsr", "width": 32}]}})
    assert "bvsrreds4" in r.slug() and "basicvsr-reds4" in r.model_ids()
    r = Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "vsr", "memory": 7, "direction": "forward",
                                                    "order": "shuffle", "flow": 0.5, "width": 32}]}})
    assert "bvsrreds4m7fshuffl50" in r.slug()
    for bad in ({"memory": 0}, {"memory": 2.5}, {"direction": "sideways"}, {"order": "random"},
                {"flow": 4}, {"model": "swinir-classical-x2"}):
        with pytest.raises(RecipeError):
            Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "vsr", **bad}]}})


class _FakeModels:
    """ModelManager stand-in: a tiny network, and a record of what it was fed."""

    def __init__(self, net):
        self.net, self.catalog, self.seen = net, {}, []

    def get_vsr(self, mid, device, precision="fp32"):
        return self

    def __getattr__(self, name):
        return getattr(self.net, name)

    def run_window(self, frames, *a, **k):
        self.seen.append(len(frames))
        return self.net.run_window(frames, *a, **k)

    def describe_used(self, recipe):
        return {}

    def gan_ids(self):
        return set()

    def known_ids(self):
        return {"basicvsr-reds4"}

    def precision_of(self, mid, device, precision):
        return "float32"

    def take_notes(self, seen, mids=None):
        return []

    def is_downloaded(self, mid):
        return True


@pytest.mark.parametrize("memory", [1, 3])
def test_in_a_pass(tmp_path, net, memory):
    from PIL import Image
    from faceloop.engine import run_pass
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(5):
        Image.fromarray(rng.integers(0, 256, (32, 40, 3)).astype(np.uint8)).save(src / f"frame{i + 1}.png")
    rec = Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "resize", "method": "area", "width": 8},
                                                   {"op": "vsr", "memory": memory, "width": 32}]}})
    fm = _FakeModels(net)
    res = run_pass(rec, src, tmp_path / "out", models=fm, device=torch.device("cpu"), contact=False,
                   batch_size=4)
    out = sorted(p.name for p in (res.folder / "c0002").glob("*.webp"))
    assert out == [f"frame{i + 1}.webp" for i in range(5)]
    # batches of 4 then 1; windows of 3 cut at the batch edge: 3, 1 | 1
    want = [1] * 5 if memory == 1 else [3, 1, 1]
    assert fm.seen == want * 2


def _weights():
    mm = ModelManager(default_weights_dir())
    if not mm.is_downloaded("basicvsr-reds4"):
        pytest.skip("basicvsr-reds4 weights not downloaded")
    return mm


def test_real_weights_beat_bilinear_and_memory_helps():
    mm = _weights()
    net = mm.get_vsr("basicvsr-reds4", torch.device("cpu"))
    torch.manual_seed(0)
    # a textured scene panning a few pixels per frame, shrunk x4 with bicubic
    h, w = 96, 128
    yy, xx = torch.meshgrid(torch.arange(h + 40).float(), torch.arange(w + 40).float(), indexing="ij")
    scene = torch.stack([0.5 + 0.4 * torch.sin(xx / 3.1 + torch.sin(yy / 7.0)),
                         0.5 + 0.4 * torch.sin(yy / 2.7 + xx / 11.0),
                         0.5 + 0.4 * torch.cos((xx + yy) / 4.3)], 0).unsqueeze(0)
    hr = torch.cat([scene[..., 2 * k:2 * k + h, 3 * k:3 * k + w] for k in range(7)], 0)
    lr = torch.nn.functional.interpolate(hr, scale_factor=0.25, mode="bicubic", antialias=True,
                                         align_corners=False).clamp(0, 1)
    names = [f"f{k}.png" for k in range(7)]

    def psnr(a):
        return float(-10 * torch.log10(((a - hr) ** 2).mean()))
    bil = psnr(torch.nn.functional.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False))
    alone = psnr(run_vsr(net, lr, names, 0, memory=1))
    mixed = psnr(run_vsr(net, lr, names, 0, memory=7))
    assert alone > bil + 0.5
    assert mixed > alone
