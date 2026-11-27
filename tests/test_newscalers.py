import numpy as np
import pytest
import torch

from faceloop import contentaware as ca
from faceloop import edi
from faceloop.eedi import eedi3_rows
from faceloop.engine import plan_sizes, run_pass
from faceloop.kernels import KernelResizer
from faceloop.recipe import Recipe, RecipeError
from faceloop.sizes import Size

torch.manual_seed(0)


def _img(h=24, w=32, c=3):
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    base = 0.5 + 0.3 * torch.sin(xx / 3) * torch.cos(yy / 4)
    x = base.expand(1, c, h, w).clone()
    if c > 1:
        x[:, 1] = 1 - x[:, 1]
    x[..., h // 3:h // 2, :] = 0.9  # a hard edge
    return torch.round(x.clamp(0, 1) * 255) / 255


# ------------------------------------------------------------------ perceptual
def _perceptual_reference(H, s, np_=2, eps=1e-6):
    """Algorithm 1 of the paper, literally, for a whole-number factor s."""
    def conv_valid(X, k):
        h, w = X.shape
        return np.array([[X[i:i + k, j:j + k].mean() for j in range(w - k + 1)] for i in range(h - k + 1)])

    def conv_full(X, k):
        return conv_valid(np.pad(X, k - 1), k)

    def sub(X, k):
        return X[::k, ::k]
    L = sub(conv_valid(H, s), s)
    L2 = sub(conv_valid(H * H, s), s)
    M = conv_valid(L, np_)
    Sl = conv_valid(L * L, np_) - M * M
    Sh = conv_valid(L2, np_) - M * M
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.sqrt(Sh / Sl)
    R[Sl < eps] = 0
    N = conv_full(np.ones_like(M), np_)
    T = conv_full(R * M, np_)
    M = conv_full(M, np_)
    R = conv_full(R, np_)
    return (M + R * L - T) / N


def test_perceptual_matches_the_paper():
    x = _img(24, 32, 1)
    kr = KernelResizer()
    area = lambda t, w, h: kr.resize(t, w, h, "area")
    got = ca.perceptual(x, 8, 6, area)[0, 0].numpy()
    want = _perceptual_reference(x[0, 0].double().numpy(), 4)
    assert np.allclose(got, want, atol=1e-5)


# ------------------------------------------------------------------ Kopf
def test_kopf_keeps_flat_colour_and_shapes():
    c = torch.full((2, 3, 16, 20), 0.4)
    y = ca.kopf(c, 5, 4)
    assert y.shape == (2, 3, 4, 5) and torch.allclose(y, c[..., :4, :5], atol=1e-4)
    x = _img() ** 2.2
    y = ca.kopf(x, 8, 6)
    assert y.shape == (1, 3, 6, 8) and torch.isfinite(y).all()
    box = torch.nn.functional.adaptive_avg_pool2d(x, (6, 8))
    assert (y - box).abs().mean() < 0.15  # crisper than a box average, but the same picture


# ------------------------------------------------------------------ 2x upscalers
def test_mmpx_copies_input_pixels_and_keeps_flat_areas():
    x = torch.zeros(1, 3, 8, 8)
    x[..., 2:6, 2:6] = 1.0
    x[..., 3, 3] = 0.5
    y = edi.mmpx(x)
    assert y.shape == (1, 3, 16, 16)
    assert set(torch.unique(y).tolist()) <= set(torch.unique(x).tolist())
    flat = torch.full((1, 3, 5, 5), 0.25)
    assert torch.equal(edi.mmpx(flat), torch.full((1, 3, 10, 10), 0.25))


@pytest.mark.parametrize("method", ["dcci", "nedi", "icbi", "eedi3"])
def test_lattice_methods_keep_originals_and_ramps(method):
    x = _img()
    y = edi.upscale2(x, method, center=False)
    assert y.shape == (1, 3, 48, 64)
    assert torch.allclose(y[..., 0::2, 0::2], x, atol=1e-5)
    yy, xx = np.mgrid[0:16, 0:16]
    ramp = torch.from_numpy(((xx * 3 + yy * 5) / 255.0).astype(np.float32))[None, None].repeat(1, 3, 1, 1)
    o = edi.upscale2(ramp, method, center=False)[0, 0].numpy() * 255
    want = xx[2:-3, 2:-3] * 3 + 1.5 + yy[2:-3, 2:-3] * 5
    assert np.abs(o[4:-6:2, 5:-6:2] - want).max() < 0.05


def test_centring_removes_the_half_pixel_creep():
    x = torch.zeros(1, 3, 24, 32)
    x[..., 8:14, 10:18] = 1.0  # one bright block: its centre is easy to follow
    kr = KernelResizer()
    for method in ("dcci",):
        a, b = x.clone(), x.clone()
        for _ in range(6):
            a = kr.resize(edi.upscale2(a, method, center=True), 32, 24, "area")
            b = kr.resize(edi.upscale2(b, method, center=False), 32, 24, "area")

        def cx(t):
            l = t[0].mean(0)
            w = torch.arange(l.shape[1], dtype=torch.float32)
            return float((l * w).sum() / l.sum())
        assert abs(cx(a) - cx(x)) < 0.1 < abs(cx(b) - cx(x))


def test_dcci_is_mirror_symmetric():
    x = _img()
    a = edi.upscale2(x, "dcci")
    b = edi.upscale2(x.flip(3), "dcci").flip(3)
    assert torch.allclose(a[..., 6:-6, 6:-6], b[..., 6:-6, 6:-6], atol=1e-5)


def test_eedi3_rows_shape_and_flat():
    S = torch.full((2, 6, 9), 100.0)
    out = eedi3_rows(S)
    assert out.shape == (2, 12, 9) and torch.allclose(out, torch.full_like(out, 100.0))


# ------------------------------------------------------------------ in a pass
def test_seam_methods_are_gone():
    rec = {"lead_in": [{"op": "resize", "method": "area", "width": 32}],
           "cycles": {"A": [{"op": "resize", "method": "seam", "width": 16}]},
           "pattern": ["A"], "iterations": 1, "save_every": 1, "finish": []}
    with pytest.raises(RecipeError):
        Recipe.from_dict(rec)


@pytest.mark.parametrize("cycle", [
    [{"op": "resize", "method": "perceptual", "width": 16}, {"op": "edi", "method": "dcci", "scale": 2}],
    [{"op": "resize", "method": "kopf", "width": 8, "iterations": 20}, {"op": "edi", "method": "nedi", "scale": 4}],
    [{"op": "edi", "method": "mmpx", "scale": 2, "width": 32}],
    [{"op": "resize", "method": "area", "width": 16}, {"op": "edi", "method": "icbi", "scale": 2, "center": False}],
    [{"op": "resize", "method": "area", "width": 16}, {"op": "edi", "method": "eedi3", "scale": 2}],
])
def test_new_methods_in_a_pass(tmp_path, cycle):
    from PIL import Image
    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        Image.fromarray((_img(24, 32)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(src / f"f{i}.png")
    rec = Recipe.from_dict({"lead_in": [{"op": "resize", "method": "area", "width": 32}], "cycles": {"A": cycle},
                            "pattern": ["A"], "iterations": 2, "save_every": 2, "finish": []})
    res = run_pass(rec, src, tmp_path / "out", device=torch.device("cpu"), contact=False)
    assert len(list((res.folder / "c0002").glob("*.webp"))) == 2
    assert rec.slug()
