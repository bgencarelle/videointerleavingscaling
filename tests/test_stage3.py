import pytest
import torch

from faceloop.chroma import subsample
from faceloop.colour import Transfer
from faceloop.engine import StepContext, run_steps
from faceloop.kernels import KernelResizer
from faceloop.pixelart import xbr
from faceloop.recipe import Recipe, RecipeError
from faceloop.sizes import Size


def _sprite():
    g = torch.Generator().manual_seed(0)
    pal = torch.tensor([[0, 0, 0], [1, 1, 1], [0.9, 0.2, 0.1], [0.1, 0.3, 0.9]])
    idx = torch.randint(0, 4, (1, 12, 16), generator=g)
    idx = torch.nn.functional.interpolate(idx[None].float(), scale_factor=2, mode="nearest")[0].long()
    return pal[idx].permute(0, 3, 1, 2)


@pytest.mark.parametrize("s", [2, 3, 4])
def test_xbr_is_symmetric(s):
    """Rotating/mirroring the input must rotate/mirror the output: catches any
    mixed-up direction in the port."""
    x = _sprite()
    y = xbr(x, s)
    assert y.shape == (1, 3, 24 * s, 32 * s)
    for k in (1, 2, 3):
        assert torch.allclose(xbr(torch.rot90(x, k, (2, 3)), s), torch.rot90(y, k, (2, 3)), atol=1e-6)
    assert torch.allclose(xbr(torch.flip(x, (3,)), s), torch.flip(y, (3,)), atol=1e-6)
    assert torch.allclose(xbr(x.transpose(2, 3), s), y.transpose(2, 3), atol=1e-6)


def test_xbr_keeps_flat_areas_and_smooths_diagonals():
    assert torch.allclose(xbr(torch.full((1, 3, 8, 8), 0.4), 3), torch.full((1, 3, 24, 24), 0.4))
    x = torch.zeros(1, 3, 16, 16)
    for i in range(16):
        x[..., i, : i + 1] = 1.0  # staircase
    y = xbr(x, 4)
    blocky = x.repeat_interleave(4, 2).repeat_interleave(4, 3)
    assert not torch.equal(y, blocky)  # edges were reshaped
    assert ((y > 0.01) & (y < 0.99)).any()  # with anti-aliased blending


@pytest.mark.parametrize("mode,block", [("422", (1, 2)), ("420", (2, 2)), ("411", (1, 4)), ("410", (4, 4))])
def test_chroma_blocks(mode, block):
    # moderate colours, so averaged colour stays inside RGB (no clipping)
    x = 0.35 + 0.3 * torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(3))
    y = subsample(x, mode)
    assert y.shape == x.shape
    kr, kb = 0.299, 0.114
    cb = (y[:, 2] - (kr * y[:, 0] + (1 - kr - kb) * y[:, 1] + kb * y[:, 2]))
    bh, bw = block
    blocks = cb.view(1, 16 // bh, bh, 16 // bw, bw)
    spread = blocks.amax(dim=(2, 4)) - blocks.amin(dim=(2, 4))
    assert float(spread.max()) < 0.03  # colour is (nearly) constant per block
    gray = torch.rand(1, 1, 9, 11).repeat(1, 3, 1, 1)
    assert float((subsample(gray, mode) - gray).abs().max()) <= 1.5 / 255


def test_steps_run_in_a_cycle():
    ctx = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                      resizer=KernelResizer(), models=None, loop=Size(32, 24))
    seen = []
    x = torch.rand(2, 3, 24, 32)
    run_steps(x, [{"op": "resize", "method": "nearest", "width": 16},
                  {"op": "chroma", "mode": "411"},
                  {"op": "xbr", "scale": 3, "width": 40},
                  {"op": "xbr", "scale": 2}],
              ctx, Size(32, 24), capture=lambda i, t: seen.append(tuple(t.shape[-2:])))
    assert seen == [(12, 16), (12, 16), (30, 40), (60, 80), (24, 32)]


def test_recipe_checks_new_steps():
    base = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "finish": [],
            "pattern": ["A"], "iterations": 2, "save_every": 1}
    ok = Recipe.from_dict({**base, "cycles": {"A": [{"op": "chroma", "mode": "410"},
                                                    {"op": "xbr", "scale": 4, "width": "loop"}]}})
    assert "c410" in ok.slug() and "xbr4@L" in ok.slug()
    for bad in ([{"op": "xbr", "scale": 5}], [{"op": "chroma", "mode": "444"}],
                [{"op": "chroma", "mode": "420", "width": 16}]):
        with pytest.raises(RecipeError):
            Recipe.from_dict({**base, "cycles": {"A": bad}})
