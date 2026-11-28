"""Emulator pixel-art scalers (faceloop/retro.py)."""
import hashlib
import shutil

import numpy as np
import pytest
import torch

from faceloop import retro
from faceloop.edi import upscale
from faceloop.recipe import Recipe, RecipeError


def _t(img):
    return torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0


def _u8(y):
    return torch.round(y[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)


def _images():
    rng = np.random.default_rng(2024)
    out = []
    for t in range(12):
        pal = rng.integers(0, 256, (int(rng.integers(2, 6)), 3)).astype(np.uint8)
        img = pal[rng.integers(0, len(pal), (int(rng.integers(6, 14)), int(rng.integers(6, 14))))]
        if t % 4 == 3:
            img = rng.integers(0, 256, img.shape).astype(np.uint8)
        out.append(img)
    return out


# checksums of ScummVM's own sai.cpp templates (compiled, 8888 masks, edges
# repeated two pixels out) on _images(); the port matched them bit for bit
SCUMMVM = {"2xsai": ("sai2x", "b051954d67898b50"), "super2xsai": ("super2xsai", "fffba21babe06708"),
           "supereagle": ("supereagle", "82b8d4ce6bd7f6f0")}


@pytest.mark.parametrize("name", sorted(SCUMMVM))
def test_sai_family_matches_scummvm(name):
    fn, want = SCUMMVM[name]
    h = hashlib.sha256()
    for img in _images():
        h.update(_u8(getattr(retro, fn)(_t(img))).tobytes())
    assert h.hexdigest()[:16] == want


# checksum of libretro's own scalefx.slangp passes on _images(), rendered with
# Mesa (llvmpipe, OpenGL 4.5), float buffers for passes 0-1 and 8-bit after,
# clamp to edge; the port matched it on every pixel of 40 test images
SCALEFX_GL = "3365f4313068d75f"


def test_scalefx_matches_the_shaders():
    h = hashlib.sha256()
    for img in _images():
        h.update(_u8(retro.scalefx(_t(img))).tobytes())
    assert h.hexdigest()[:16] == SCALEFX_GL


def test_scalefx_uses_only_the_frames_colours_and_smooths_stairs():
    x = torch.rand(1, 3, 9, 11)
    y = retro.scalefx(x)
    assert set(torch.unique(y).tolist()) <= set(torch.unique(x).tolist())
    img = torch.zeros(1, 3, 12, 12)
    for r in range(12):
        img[..., r, :r] = 1
    y = retro.scalefx(img)[0, 0] > 0.5
    # a one-pixel staircase becomes a line one new pixel per row
    widths = y.sum(1)[4:30]
    assert (widths.diff() == 1).all()


def _scale2x_ref(img):
    h, w, _ = img.shape
    P = lambda y, x: tuple(img[min(max(y, 0), h - 1), min(max(x, 0), w - 1)])  # noqa: E731
    out = np.zeros((2 * h, 2 * w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            B, D, E, F, H = P(y - 1, x), P(y, x - 1), P(y, x), P(y, x + 1), P(y + 1, x)
            if B != H and D != F:
                e = [D if D == B else E, F if B == F else E, D if D == H else E, F if H == F else E]
            else:
                e = [E] * 4
            out[2 * y, 2 * x], out[2 * y, 2 * x + 1], out[2 * y + 1, 2 * x], out[2 * y + 1, 2 * x + 1] = e
    return out


def _scale3x_ref(img):
    h, w, _ = img.shape
    P = lambda y, x: tuple(img[min(max(y, 0), h - 1), min(max(x, 0), w - 1)])  # noqa: E731
    out = np.zeros((3 * h, 3 * w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            A, B, C = P(y - 1, x - 1), P(y - 1, x), P(y - 1, x + 1)
            D, E, F = P(y, x - 1), P(y, x), P(y, x + 1)
            G, H, I = P(y + 1, x - 1), P(y + 1, x), P(y + 1, x + 1)
            if B != H and D != F:
                e = [D if D == B else E,
                     B if (D == B and E != C) or (B == F and E != A) else E,
                     F if B == F else E,
                     D if (D == B and E != G) or (D == H and E != A) else E,
                     E,
                     F if (B == F and E != I) or (H == F and E != C) else E,
                     D if D == H else E,
                     H if (D == H and E != I) or (H == F and E != G) else E,
                     F if H == F else E]
            else:
                e = [E] * 9
            for k, v in enumerate(e):
                out[3 * y + k // 3, 3 * x + k % 3] = v
    return out


def test_scale2x_and_scale3x_follow_the_published_rules():
    for img in _images()[:8]:
        assert np.array_equal(_u8(retro.scale2x(_t(img))), _scale2x_ref(img))
        assert np.array_equal(_u8(retro.scale3x(_t(img))), _scale3x_ref(img))


def test_superxbr_matches_the_shaders():
    # tie-free inputs: on flat two-colour art the edge tests can tie exactly,
    # and which way a tie goes depends on the order of float additions
    # (on a GPU too)
    from tests.superxbr_ref import ref
    rng = np.random.default_rng(3)
    for t in range(4):
        img = rng.random((int(rng.integers(5, 9)), int(rng.integers(5, 9)), 3))
        if t % 2:
            yy, xx = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
            img = 0.5 + 0.4 * np.tanh((xx - 0.7 * yy - 1.3)[..., None] * np.array([1.0, 0.6, -0.8])) + 0.01 * img
        got = retro.superxbr(torch.from_numpy(img).permute(2, 0, 1)[None].double())[0].permute(1, 2, 0).numpy()
        assert np.abs(ref(img) - got).max() < 1e-12


def _hqx_ok():
    if not shutil.which("c++") or not shutil.which("cc"):
        pytest.skip("no C compiler")


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_hqx_runs_the_library(scale):
    _hqx_ok()
    flat = torch.full((2, 3, 6, 7), 100 / 255)
    y = retro.hqx(flat, scale)
    assert y.shape == (2, 3, 6 * scale, 7 * scale)
    assert torch.allclose(y, flat[:, :, :1, :1].expand_as(y))
    img = _images()[0]
    out = _u8(retro.hqx(_t(img), scale))
    assert out.shape == (img.shape[0] * scale, img.shape[1] * scale, 3)
    # where a pixel and all eight neighbours share a colour, hqx copies it
    h, w, _ = img.shape
    for y0 in range(1, h - 1):
        for x0 in range(1, w - 1):
            if (img[y0 - 1:y0 + 2, x0 - 1:x0 + 2] == img[y0, x0]).all():
                assert (out[y0 * scale:(y0 + 1) * scale, x0 * scale:(x0 + 1) * scale] == img[y0, x0]).all()


@pytest.mark.parametrize("method", retro.METHODS)
def test_picture_stays_put(method):
    if method == "hqx":
        _hqx_ok()
    h, w = 40, 48
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    x = torch.exp(-(((xs - 21.3) / 6) ** 2 + ((ys - 17.8) / 5) ** 2)).view(1, 1, h, w).repeat(1, 3, 1, 1)
    for s in retro.SCALES[method][:2]:
        y = upscale(x, method, s)[0, 0].double()
        H, W = y.shape
        yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
        cx, cy = float((y * xx).sum() / y.sum()) / s, float((y * yy).sum() / y.sum()) / s
        assert abs(cx - 21.3) < 0.03 and abs(cy - 17.8) < 0.03, (s, cx, cy)


def test_raw_keeps_the_drift():
    x = torch.rand(1, 3, 10, 12)
    assert not torch.equal(upscale(x, "supereagle", 2), upscale(x, "supereagle", 2, center=False))
    assert torch.equal(upscale(x, "supereagle", 2, center=False), retro.supereagle(x))


BASE = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A"], "iterations": 1,
        "save_every": 1, "finish": []}


def test_recipe_scales_and_slugs():
    r = Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "edi", "method": "scale3x", "scale": 9}]}})
    assert "sc3x9" in r.slug()
    r = Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "edi", "method": "supereagle", "scale": 2,
                                                     "center": False}]}})
    assert "seagle2raw" in r.slug()
    for bad in ({"method": "scale3x", "scale": 2}, {"method": "hqx", "scale": 8}, {"method": "2xsai", "scale": 3}):
        with pytest.raises(RecipeError):
            Recipe.from_dict({**BASE, "cycles": {"A": [{"op": "edi", **bad}]}})


@pytest.mark.parametrize("method", retro.METHODS)
def test_in_a_pass(tmp_path, method):
    from PIL import Image
    from faceloop.engine import run_pass
    if method == "hqx":
        _hqx_ok()
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, (24, 32, 3)).astype(np.uint8)).save(src / f"f{i}.png")
    s = retro.SCALES[method][0]
    rec = Recipe.from_dict({**BASE, "iterations": 2, "save_every": 2,
                            "cycles": {"A": [{"op": "resize", "method": "area", "width": 16},
                                             {"op": "edi", "method": method, "scale": s, "width": 32}]}})
    res = run_pass(rec, src, tmp_path / "out", device=torch.device("cpu"), contact=False)
    assert len(list((res.folder / "c0002").glob("*.webp"))) == 2
