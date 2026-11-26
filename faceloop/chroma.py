"""Colour-resolution (chroma subsampling) step, as video codecs do it.

RGB (stored values) -> Y'CbCr (BT.601 or BT.709, full range, 8-bit), colour
averaged over blocks, then brought back with nearest (each block repeats its
colour), -> RGB. Brightness (Y') keeps full resolution. The frame size does
not change.

Edges that don't fill a whole block are padded by repeating the last
row/column, as encoders do.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# mode -> (block width, block height)
MODES = {
    "422": (2, 1),
    "420": (2, 2),
    "411": (4, 1),   # DV NTSC
    "410": (4, 4),   # YUV9 (Indeo, Cinepak era)
}
MATRICES = {"601": (0.299, 0.114), "709": (0.2126, 0.0722)}  # (Kr, Kb)


def _q8(t):
    return torch.round(t * 255.0) / 255.0


@torch.no_grad()
def subsample(x: torch.Tensor, mode: str = "420", matrix: str = "601") -> torch.Tensor:
    if mode not in MODES:
        raise ValueError(f"chroma mode must be one of {sorted(MODES)}")
    if matrix not in MATRICES:
        raise ValueError(f"matrix must be one of {sorted(MATRICES)}")
    bw, bh = MODES[mode]
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb)) + 0.5
    cr = (r - y) / (2 * (1 - kr)) + 0.5
    y, cb, cr = _q8(y), _q8(cb), _q8(cr)

    n, _, h, w = x.shape
    ph, pw = (-h) % bh, (-w) % bw
    chroma = torch.cat([cb, cr], 1)
    if ph or pw:
        chroma = F.pad(chroma, (0, pw, 0, ph), mode="replicate")
    small = _q8(F.avg_pool2d(chroma, (bh, bw)))
    back = small.repeat_interleave(bh, 2).repeat_interleave(bw, 3)[..., :h, :w]
    cb, cr = back[:, 0:1] - 0.5, back[:, 1:2] - 0.5

    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return torch.cat([r, g, b], 1).clamp(0, 1)
