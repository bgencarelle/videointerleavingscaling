"""Old video hardware, as resizing: both keep the frame's size.

fields
    What an interlaced camera or deck does to a still frame: keep one field
    (every other line), then rebuild the missing lines.
      field    top (lines 0, 2, 4...), bottom (1, 3, 5...), or alternate:
               top on odd cycles, bottom on even ones
      rebuild  double: repeat each line (line doubling, the cheap
               scan converter look). It moves the picture half a line (down
               for top, up for bottom) once: after that the line pairs match,
               and doubling again, with either field, changes nothing, so it
               doesn't creep.
               linear / cubic: interpolate the missing lines between their
               neighbours (bob deinterlacing); the kept lines stay where they
               were, so nothing moves.
squash
    Anamorphic squeeze and stretch: shrink one axis by `factor` with one
    kernel, stretch it straight back with another (a camera or deck with
    less horizontal resolution than lines, a 4:3 picture squeezed into 16:9
    and back). Every kernel here is centred, so nothing moves.
"""
from __future__ import annotations

import torch

FIELDS = ("top", "bottom", "alternate")
REBUILD = ("double", "linear", "cubic")
AXES = ("x", "y")


def field_kept(field: str, cycle: int) -> int:
    """Which line parity is kept: 0 = top, 1 = bottom."""
    if field == "alternate":
        return 0 if cycle % 2 else 1
    return 0 if field == "top" else 1


def fields(x: torch.Tensor, field: str = "alternate", rebuild: str = "linear", cycle: int = 1) -> torch.Tensor:
    h = x.shape[-2]
    if h < 2:
        return x
    keep = field_kept(field, cycle)
    f = x[..., keep::2, :]
    m = f.shape[-2]
    rows = torch.arange(h, device=x.device)
    # position of every output line in field-line units (field line k sits at 2k + keep)
    pos = (rows - keep).to(x.dtype) / 2
    if rebuild == "double":
        idx = torch.div(rows - keep, 2, rounding_mode="floor").clamp(0, m - 1)
        if keep == 1:      # bottom: line 2k takes the field line just below it
            idx = torch.div(rows, 2, rounding_mode="floor").clamp(0, m - 1)
        return f.index_select(-2, idx)
    lo = torch.floor(pos)
    t = (pos - lo).view(-1, 1)
    lo = lo.long()

    def at(k):
        return f.index_select(-2, k.clamp(0, m - 1))
    if rebuild == "linear":
        return at(lo) * (1 - t) + at(lo + 1) * t
    # cubic: Catmull-Rom between field lines (t is 0 or 1/2 here)
    t2, t3 = t * t, t * t * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1
    w2 = -1.5 * t3 + 2 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return (at(lo - 1) * w0 + at(lo) * w1 + at(lo + 1) * w2 + at(lo + 2) * w3).clamp(0, 1)


def squash(x: torch.Tensor, resize, axis: str = "x", factor: float = 0.5, down: str = "area",
           up: str = "bicubic") -> torch.Tensor:
    """resize(x, width, height, method) -> tensor."""
    h, w = x.shape[-2:]
    if axis == "x":
        small = resize(x, max(1, round(w * factor)), h, down)
    else:
        small = resize(x, w, max(1, round(h * factor)), down)
    return resize(small, w, h, up)
