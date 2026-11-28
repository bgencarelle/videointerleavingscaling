import math

import pytest
import torch

from faceloop.kernels import EWA, METHODS, SEPARABLE, KernelResizer, _FILTERS, ewa_plan, weights_1d
from faceloop.recipe import Recipe, RecipeError


@pytest.mark.parametrize("name", sorted(_FILTERS))
def test_filters_are_continuous_and_vanish_at_their_edge(name):
    f, sup = _FILTERS[name]
    for t in [k for k in (1.0, 2.0, 3.0, 4.0) if k < sup]:
        assert abs(f(t - 1e-7) - f(t + 1e-7)) < 1e-5
    assert abs(f(sup - 1e-9)) < 1e-6 and f(sup + 1e-6) == 0.0


@pytest.mark.parametrize("name", SEPARABLE)
def test_rows_sum_to_one(name):
    for n_in, n_out in ((40, 13), (13, 40), (64, 64)):
        W = weights_1d(n_in, n_out, name)
        assert torch.allclose(W.sum(1), torch.ones(n_out, dtype=torch.float64))


@pytest.mark.parametrize("name", ["catrom", "mitchell", "bspline", "spline16", "spline36", "spline64", "lanczos4"])
def test_linear_ramps_survive_enlarging(name):
    x = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32).expand(1, 3, 4, 32) / 64
    y = KernelResizer().resize(x, 96, 4, name)
    want = ((torch.arange(96) + 0.5) / 3 - 0.5) / 64
    assert (y[0, 0, 0, 12:-12] - want[12:-12]).abs().max() < (2e-3 if name == "lanczos4" else 1e-5)


@pytest.mark.parametrize("name", EWA)
def test_ewa_keeps_flat_colour_and_is_round(name):
    kr = KernelResizer()
    c = torch.full((2, 3, 30, 40), 0.42)
    for w, h in ((20, 15), (80, 60), (13, 9)):
        assert torch.allclose(kr.resize(c, w, h, name), torch.full((2, 3, h, w), 0.42), atol=1e-5)
    # a single dot comes out round: same spread along x and y
    dot = torch.zeros(1, 1, 21, 21)
    dot[..., 10, 10] = 1
    y = kr.resize(dot, 63, 63, name)[0, 0]
    assert torch.allclose(y, y.t(), atol=1e-6)
    iy, ix, wgt = ewa_plan(30, 40, 15, 20, name)
    assert torch.allclose(wgt.sum((-1, -2)), torch.ones(15, 20, dtype=wgt.dtype))


def test_recipe_knows_the_new_kernels():
    base = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A"], "iterations": 1,
            "save_every": 1, "finish": []}
    for m in METHODS:
        r = Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": m, "width": 16}]}})
        assert r.slug()
    r = Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": "gaussian", "width": 16, "sigma": 1.2}]}})
    assert "gauss1.2" in r.slug()
    with pytest.raises(RecipeError):
        Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": "bicubic", "width": 16, "sigma": 1}]}})
    with pytest.raises(RecipeError):
        Recipe.from_dict({**base, "cycles": {"A": [{"op": "resize", "method": "ewa_lanczos", "width": 16,
                                                      "fixed_point": True}]}})
