"""Plain resize kernels, built as one weight matrix per axis.

Convention (the same everywhere, so nothing drifts): output pixel i covers
the input interval [i*s, (i+1)*s) with s = n_in / n_out, and its centre sits
at (i + 0.5) * s in input pixel units. Input pixel j has its centre at j+0.5.

A resize is   out = Wh @ img @ Ww^T   (horizontal first, then vertical).

Methods
-------
nearest    the input pixel under the output centre
area       exact area-weighted average of the covered input (pixel mixing);
           works both ways and invents nothing
bilinear   triangle filter
bicubic    Keys cubic, a = -0.5 (same as Pillow)
lanczos    Lanczos-3

Variants (params)
-----------------
antialias=False   shrink without prefiltering: the filter is not widened
                  when downscaling (old games / browsers: shimmer, moire)
fixed_point=True  low-precision maths like old hardware scalers: filter
                  phase quantised to `phase_bits`, weights to `coef_bits`,
                  and the intermediate rounded to 8 bits between passes
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

METHODS = ("nearest", "area", "bilinear", "bicubic", "lanczos")


def _triangle(x):
    x = abs(x)
    return 1.0 - x if x < 1.0 else 0.0


def _keys(x, a=-0.5):
    x = abs(x)
    if x < 1.0:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    if x < 2.0:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return 0.0


def _sinc(x):
    if x == 0.0:
        return 1.0
    x *= math.pi
    return math.sin(x) / x


def _lanczos3(x):
    if -3.0 < x < 3.0:
        return _sinc(x) * _sinc(x / 3.0)
    return 0.0


_FILTERS = {
    "bilinear": (_triangle, 1.0),
    "bicubic": (_keys, 2.0),
    "lanczos": (_lanczos3, 3.0),
}


def _quant(v: float, bits: int) -> float:
    q = 1 << bits
    return round(v * q) / q


@lru_cache(maxsize=512)
def weights_1d(n_in: int, n_out: int, method: str, antialias: bool = True,
               fixed_point: bool = False, phase_bits: int = 4, coef_bits: int = 6) -> torch.Tensor:
    """Dense (n_out, n_in) float64 matrix whose rows sum to 1."""
    if method not in METHODS:
        raise ValueError(f"unknown kernel {method!r}; choose from {METHODS}")
    s = n_in / n_out
    W = torch.zeros(n_out, n_in, dtype=torch.float64)

    if method == "nearest":
        for i in range(n_out):
            pos = (i + 0.5) * s
            r = round(pos)
            if abs(pos - r) < 1e-9 and 0 < r < n_in:
                # centre falls exactly between two input pixels (even
                # factors). Any fixed choice shifts the picture half a pixel
                # per resize; alternating left/right cancels that out.
                j = r - 1 if i % 2 == 0 else r
            else:
                j = int(math.floor(pos))
            W[i, min(n_in - 1, max(0, j))] = 1.0
        return W

    if method == "area":
        for i in range(n_out):
            a, b = i * s, (i + 1) * s
            j0, j1 = int(math.floor(a)), min(n_in, int(math.ceil(b)))
            for j in range(j0, j1):
                ov = min(b, j + 1) - max(a, j)
                if ov > 0:
                    W[i, j] = ov / s
        return W

    f, support = _FILTERS[method]
    scale = max(s, 1.0) if antialias else 1.0  # widen filter when shrinking
    radius = support * scale
    for i in range(n_out):
        centre = (i + 0.5) * s  # in input units; input j centre = j + 0.5
        if fixed_point:
            # sub-pixel phase quantised, as an old scaler's phase accumulator
            centre = math.floor(centre - 0.5) + 0.5 + _quant((centre - 0.5) % 1.0, phase_bits)
        lo = int(math.floor(centre - radius))
        hi = int(math.ceil(centre + radius))
        row = {}
        for j in range(lo, hi + 1):
            w = f((j + 0.5 - centre) / scale)
            if w == 0.0:
                continue
            jj = min(max(j, 0), n_in - 1)  # replicate edges
            row[jj] = row.get(jj, 0.0) + w
        total = sum(row.values())
        for jj, w in row.items():
            W[i, jj] = w / total
        if fixed_point:
            # integer weights that still sum to exactly 1
            q = 1 << coef_bits
            ints = {jj: round(float(W[i, jj]) * q) for jj in row}
            err = q - sum(ints.values())
            if err:
                big = max(ints, key=lambda k: abs(ints[k]))
                ints[big] += err
            for jj, v in ints.items():
                W[i, jj] = v / q
    return W


class KernelResizer:
    """Applies 1-D weight matrices on a device, caching float32 copies."""

    def __init__(self):
        self._cache: dict = {}

    def _mat(self, n_in, n_out, method, device, **kw):
        key = (n_in, n_out, method, str(device), tuple(sorted(kw.items())))
        m = self._cache.get(key)
        if m is None:
            m = weights_1d(n_in, n_out, method, **kw).float().to(device)
            self._cache[key] = m
        return m

    def resize(self, x: torch.Tensor, width: int, height: int, method: str,
               antialias: bool = True, fixed_point: bool = False,
               phase_bits: int = 4, coef_bits: int = 6) -> torch.Tensor:
        """x: (N, C, H, W) float. Returns (N, C, height, width)."""
        n, c, h, w = x.shape
        kw = {}
        if method not in ("nearest", "area"):
            kw = dict(antialias=antialias, fixed_point=fixed_point,
                      phase_bits=phase_bits, coef_bits=coef_bits)
        if w != width:
            Ww = self._mat(w, width, method, x.device, **kw)
            x = torch.matmul(x, Ww.t())
            if fixed_point:
                x = torch.round(x.clamp(0, 1) * 255.0) / 255.0
        if h != height:
            Wh = self._mat(h, height, method, x.device, **kw)
            x = torch.matmul(Wh, x)
        return x
