"""Old video hardware: fields and squash & stretch."""
import numpy as np
import pytest
import torch

from faceloop.kernels import KernelResizer
from faceloop.recipe import Recipe, RecipeError
from faceloop.video import field_kept, fields, squash


def _col(h):
    return (torch.arange(h, dtype=torch.float32) / 10).view(1, 1, h, 1)


def test_fields_keep_one_parity():
    x = _col(8)
    assert fields(x, "top", "double").flatten().tolist() == pytest.approx([0, 0, .2, .2, .4, .4, .6, .6])
    assert fields(x, "bottom", "double").flatten().tolist() == pytest.approx([.1, .1, .3, .3, .5, .5, .7, .7])
    assert fields(x, "top", "linear").flatten().tolist() == pytest.approx([0, .1, .2, .3, .4, .5, .6, .6])
    assert fields(x, "bottom", "linear").flatten().tolist() == pytest.approx([.1, .1, .2, .3, .4, .5, .6, .7])
    # the kept lines are untouched in every mode
    for f, keep in (("top", 0), ("bottom", 1)):
        for r in ("double", "linear", "cubic"):
            assert torch.equal(fields(x, f, r)[..., keep::2, :], x[..., keep::2, :])


def test_alternate_follows_the_cycle():
    assert [field_kept("alternate", c) for c in (1, 2, 3, 4)] == [0, 1, 0, 1]


def _centroid(y):
    ys = torch.arange(y.shape[-2]) + 0.5
    return float((y.flatten() * ys).sum() / y.sum())


@pytest.mark.parametrize("rebuild", ["linear", "cubic"])
def test_bob_does_not_move_the_picture(rebuild):
    ys = torch.arange(60) + 0.5
    y = torch.exp(-((ys - 27.3) / 5) ** 2).view(1, 1, 60, 1)
    for c in range(1, 41):
        y = fields(y, "alternate", rebuild, c)
    assert abs(_centroid(y) - 27.3) < 0.01


def test_line_doubling_moves_half_a_line_once():
    ys = torch.arange(60) + 0.5
    b = torch.exp(-((ys - 27.3) / 5) ** 2).view(1, 1, 60, 1)
    for f, want in (("top", 27.8), ("bottom", 26.8), ("alternate", 27.8)):
        y = b
        for c in range(1, 101):
            y = fields(y, f, "double", c)
        assert abs(_centroid(y) - want) < 0.01


def test_odd_heights():
    x = torch.rand(1, 3, 7, 5)
    for f in ("top", "bottom"):
        for r in ("double", "linear", "cubic"):
            assert fields(x, f, r).shape == x.shape


def test_squash_keeps_size_and_place():
    r = KernelResizer()
    rs = lambda t, w, h, m: r.resize(t, w, h, m)  # noqa: E731
    h, w = 40, 64
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    x = torch.exp(-(((xs - 29.3) / 7) ** 2 + ((ys - 17.6) / 5) ** 2)).view(1, 1, h, w).repeat(1, 3, 1, 1)
    for axis in ("x", "y"):
        y = x
        for _ in range(20):
            y = squash(y, rs, axis, 0.4, "area", "bicubic").clamp(0, 1)
        assert y.shape == x.shape
        m = y[0, 0]
        cx, cy = float((m * xs).sum() / m.sum()), float((m * ys).sum() / m.sum())
        assert abs(cx - 29.3) < 0.05 and abs(cy - 17.6) < 0.05
    # only the squeezed axis loses detail
    stripes = torch.zeros(1, 3, 16, 16)
    stripes[..., ::2] = 1
    assert squash(stripes, rs, "y", 0.5).sub(stripes).abs().max() < 1e-5
    assert squash(stripes, rs, "x", 0.5).std() < 0.05


BASE = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A", "B"], "iterations": 4,
        "save_every": 2, "finish": []}


def test_recipe_checks_and_slugs():
    r = Recipe.from_dict({**BASE, "cycles": {
        "A": [{"op": "video", "mode": "fields", "field": "alternate", "rebuild": "double"}],
        "B": [{"op": "video", "mode": "squash", "axis": "x", "factor": 0.75, "down": "area", "up": "lanczos"}]}})
    assert "flda" in r.slug() and "sqx75" in r.slug()
    for bad in ({"mode": "fields", "field": "left"}, {"mode": "fields", "rebuild": "nnedi3"},
                {"mode": "squash", "factor": 1.5}, {"mode": "squash", "factor": 0.5, "up": "dpid"},
                {"mode": "squash", "factor": 0.5, "width": 16}, {"mode": "noise"}):
        with pytest.raises(RecipeError):
            Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "video", **bad}], "B": []}})


def test_in_a_pass(tmp_path):
    from PIL import Image
    from faceloop.engine import run_pass
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, (24, 32, 3)).astype(np.uint8)).save(src / f"f{i}.png")
    rec = Recipe.from_dict({**BASE, "cycles": {
        "A": [{"op": "video", "mode": "fields", "field": "alternate", "rebuild": "cubic"}],
        "B": [{"op": "video", "mode": "squash", "axis": "y", "factor": 0.5, "down": "dpid", "up": "catrom"}]}})
    res = run_pass(rec, src, tmp_path / "out", device=torch.device("cpu"), contact=False)
    assert len(list((res.folder / "c0004").glob("*.webp"))) == 2
