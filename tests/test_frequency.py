"""DCT / FFT / wavelet resizing."""
import numpy as np
import pytest
import torch

from faceloop.kernels import FREQUENCY, KernelResizer, _trig, _wavelet_down, _WAVELETS, weights_1d
from faceloop.recipe import Recipe, RecipeError

scipy_fft = pytest.importorskip("scipy.fft")


@pytest.mark.parametrize("n_in,n_out", [(240, 160), (40, 96), (31, 17), (64, 64 * 3)])
def test_dct_is_cut_or_padded_dct(n_in, n_out):
    x = np.random.default_rng(0).random(n_in)
    X = scipy_fft.dct(x, norm="ortho")
    Y = np.zeros(n_out)
    m = min(n_in, n_out)
    Y[:m] = X[:m]
    ref = scipy_fft.idct(Y, norm="ortho") * np.sqrt(n_out / n_in)
    got = (weights_1d(n_in, n_out, "dct") @ torch.tensor(x)).numpy()
    assert np.abs(ref - got).max() < 1e-10


@pytest.mark.parametrize("n_in,n_out", [(40, 96), (31, 17), (30, 45)])
def test_fft_matches_scipy_resample_on_its_grid(n_in, n_out):
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(1).random(n_in)
    q = torch.arange(n_out, dtype=torch.float64) * n_in / n_out
    got = (_trig(n_in, q, min(n_in, n_out), "fft") @ torch.tensor(x)).numpy()
    assert np.abs(signal.resample(x, n_out) - got).max() < 1e-10


def test_fft_wraps_around_dct_does_not():
    x = torch.zeros(1, 1, 8, 64)
    x[..., :8] = 1.0                     # bright strip on the left edge only
    r = KernelResizer()
    f = r.resize(x, 128, 8, "fft")
    d = r.resize(x, 128, 8, "dct")
    assert f[..., -2:].abs().max() > 0.1     # it shows up on the right
    assert d[..., -2:].abs().max() < 0.02


def test_wavelet_halving_is_the_low_band():
    h = _WAVELETS["wavelet97"][0]
    W = _wavelet_down(16, h)
    x = torch.rand(16, dtype=torch.float64)
    k = 3                                 # an interior sample: the plain filter
    hn = torch.tensor(h, dtype=torch.float64) / sum(h)
    assert torch.isclose((W @ x)[k], (hn * x[2 * k - 4:2 * k + 5]).sum())


@pytest.mark.parametrize("method", FREQUENCY)
def test_bump_stays_put(method):
    n = 240
    x = torch.exp(-((torch.arange(n) - 100.5) ** 2) / (2 * 8 ** 2)).double()
    for o in (120, 60, 80, 47, 480):
        y = weights_1d(n, o, method) @ x
        pos = (torch.arange(o) + 0.5) * (n / o) - 0.5
        assert abs(float((y * pos).sum() / y.sum()) - 100.5) < 1e-4


@pytest.mark.parametrize("method", ("dct", "fft"))
def test_no_antialias_keeps_more(method):
    x = torch.zeros(1, 3, 8, 256)
    x[..., 0::2] = 1.0
    r = KernelResizer()
    aa = r.resize(x, 100, 8, method)
    na = r.resize(x, 100, 8, method, antialias=False)
    assert na.std() > aa.std() * 2


BASE = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A"], "iterations": 1,
        "save_every": 1, "finish": []}


def test_recipe_rejects_fixed_point_and_slugs():
    base = BASE
    r = Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": "wavelet97", "width": 16}]}})
    assert "wav97" in r.slug()
    with pytest.raises(RecipeError):
        Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": "dct", "width": 16,
                                                      "fixed_point": True}]}})
