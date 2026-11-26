import math

import numpy as np
import pytest
import torch

from faceloop.dpid import dpid


def reference(img, ow, oh, lam):
    """Loop-for-loop transcription of the authors' CUDA kernel (0..1 floats)."""
    ih, iw, ch = img.shape
    pw, ph = iw / ow, ih / oh

    def cov(ix, iy, sx, ex, sy, ey):
        f = 1.0
        if ix < sx: f *= 1.0 - (sx - ix)
        if ix + 1 > ex: f *= 1.0 - ((ix + 1) - ex)
        if iy < sy: f *= 1.0 - (sy - iy)
        if iy + 1 > ey: f *= 1.0 - ((iy + 1) - ey)
        return f

    def box(px, py):
        return max(px * pw, 0), min((px + 1) * pw, iw), max(py * ph, 0), min((py + 1) * ph, ih)

    avg = np.zeros((oh, ow, ch))
    for py in range(oh):
        for px in range(ow):
            sx, ex, sy, ey = box(px, py)
            acc, fs = np.zeros(ch), 0.0
            for iy in range(math.floor(sy), math.ceil(ey)):
                for ix in range(math.floor(sx), math.ceil(ex)):
                    f = cov(ix, iy, sx, ex, sy, ey)
                    acc += img[iy, ix] * f
                    fs += f
            avg[py, px] = acc / fs
    out = np.zeros((oh, ow, ch))
    for py in range(oh):
        for px in range(ow):
            g, gw = np.zeros(ch), 0.0
            for dy, dx, wt in [(-1, -1, 1), (-1, 0, 2), (-1, 1, 1), (0, -1, 2), (0, 0, 4), (0, 1, 2),
                               (1, -1, 1), (1, 0, 2), (1, 1, 1)]:
                y, x = py + dy, px + dx
                if 0 <= y < oh and 0 <= x < ow:
                    g += avg[y, x] * wt
                    gw += wt
            g /= gw
            sx, ex, sy, ey = box(px, py)
            acc, fs = np.zeros(ch), 0.0
            for iy in range(math.floor(sy), math.ceil(ey)):
                for ix in range(math.floor(sx), math.ceil(ex)):
                    if lam == 0:
                        f = 1.0
                    else:
                        f = (np.linalg.norm(g - img[iy, ix]) / math.sqrt(3.0)) ** lam
                    f *= cov(ix, iy, sx, ex, sy, ey)
                    acc += img[iy, ix] * f
                    fs += f
            out[py, px] = g if fs == 0 else acc / fs
    return out


@pytest.mark.parametrize("ih,iw,oh,ow", [(24, 18, 8, 6), (30, 40, 7, 9), (17, 23, 5, 4), (12, 12, 12, 12), (6, 8, 9, 12)])
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
def test_dpid_matches_the_reference(ih, iw, oh, ow, lam):
    rng = np.random.default_rng(ih * 100 + ow)
    img = rng.random((ih, iw, 3))
    img[ih // 3, iw // 2] = [1.0, 1.0, 1.0]  # a small bright detail
    ref = reference(img, ow, oh, lam)
    got = dpid(torch.from_numpy(img).permute(2, 0, 1)[None].float(), ow, oh, lam)[0].permute(1, 2, 0).numpy()
    assert np.abs(got - ref).max() < 1e-5


def test_dpid_keeps_a_thin_highlight_that_area_washes_out():
    x = torch.full((1, 3, 60, 60), 0.2)
    x[..., 30, 10:50] = 1.0  # one-pixel bright line
    from faceloop.kernels import KernelResizer
    area = KernelResizer().resize(x, 6, 6, "area")
    d = dpid(x, 6, 6, 1.0)
    assert float(d.max()) > float(area.max()) + 0.2


def test_dpid_flat_image_stays_flat():
    x = torch.full((2, 3, 30, 40), 0.37)
    assert torch.allclose(dpid(x, 9, 7, 1.0), torch.full((2, 3, 7, 9), 0.37), atol=1e-6)
