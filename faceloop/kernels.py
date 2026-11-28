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
lanczos    Lanczos-3; lanczos2 / lanczos4 / lanczos8 with that many lobes
mitchell   Mitchell-Netravali BC-spline (B = C = 1/3)
catrom     Catmull-Rom (B = 0, C = 1/2)
bspline    cubic B-spline (B = 1, C = 0): very soft
hermite    Hermite (B = C = 0)
spline16/36/64   the Panorama Tools / AviSynth splines
gaussian   exp(-x^2 / 2 sigma^2), sigma in input pixels (default 0.5)
magic      Magic Kernel Sharp 2021 (Costella): the magic kernel convolved
           with the Sharp 2021 taps [-1, 6, -35, 204, -35, 6, -1] / 144
ewa_lanczos, ewa_lanczossharp, ewa_robidoux, ewa_robidouxsharp
           elliptical weighted averages (round, 2-D filters, as mpv and
           ImageMagick's distort resize use): jinc-windowed jinc (3 lobes;
           "sharp" with mpv's blur 0.9812505644269356), and ImageMagick's
           Robidoux / RobidouxSharp cylindrical cubics
dct        resize in the frequency domain: the frame's DCT-II, cut or
           padded with zeros to the new size, transformed back (what
           "spectral" resizing in image codecs does). Shrinking keeps every
           frequency the new size can hold and drops the rest at once: the
           sharpest possible shrink, with ringing at hard edges. Mirror-
           symmetric at the borders, so no wrap-around.
fft        the same with the plain Fourier transform (zero-padding the FFT):
           the frame is treated as periodic, so the left edge bleeds into
           the right and the top into the bottom
wavelet53, wavelet97
           JPEG 2000's wavelets (LeGall 5/3, CDF 9/7): each halving keeps
           the low band of one wavelet level, each doubling runs the inverse
           with the detail bands empty. What's left of the factor after the
           whole halvings/doublings (e.g. x1.5) is done the dct way, which
           also moves the result back by the wavelets' quarter-pixel offset,
           so the picture doesn't drift. Each halving's low band already
           removes the finest detail, so antialias=False changes little

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

SEPARABLE = ("bilinear", "bicubic", "lanczos", "lanczos2", "lanczos4", "lanczos8", "mitchell", "catrom", "bspline",
             "hermite", "spline16", "spline36", "spline64", "gaussian", "magic")
EWA = ("ewa_lanczos", "ewa_lanczossharp", "ewa_robidoux", "ewa_robidouxsharp")
FREQUENCY = ("dct", "fft", "wavelet53", "wavelet97")
METHODS = ("nearest", "area") + SEPARABLE + EWA + FREQUENCY


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


def _lanczos_n(n):
    def f(x):
        return _sinc(x) * _sinc(x / n) if -n < x < n else 0.0
    return f


def _bc(B, C):
    def f(x):
        x = abs(x)
        if x < 1.0:
            return ((12 - 9 * B - 6 * C) * x ** 3 + (-18 + 12 * B + 6 * C) * x ** 2 + (6 - 2 * B)) / 6.0
        if x < 2.0:
            return ((-B - 6 * C) * x ** 3 + (6 * B + 30 * C) * x ** 2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6.0
        return 0.0
    return f


def _spline16(x):
    x = abs(x)
    if x < 1.0:
        return ((x - 9 / 5) * x - 1 / 5) * x + 1
    if x < 2.0:
        x -= 1
        return ((-1 / 3 * x + 4 / 5) * x - 7 / 15) * x
    return 0.0


def _spline36(x):
    x = abs(x)
    if x < 1.0:
        return ((13 / 11 * x - 453 / 209) * x - 3 / 209) * x + 1
    if x < 2.0:
        x -= 1
        return ((-6 / 11 * x + 270 / 209) * x - 156 / 209) * x
    if x < 3.0:
        x -= 2
        return ((1 / 11 * x - 45 / 209) * x + 26 / 209) * x
    return 0.0


def _spline64(x):
    x = abs(x)
    if x < 1.0:
        return ((49 / 41 * x - 6387 / 2911) * x - 3 / 2911) * x + 1
    if x < 2.0:
        x -= 1
        return ((-24 / 41 * x + 4032 / 2911) * x - 2328 / 2911) * x
    if x < 3.0:
        x -= 2
        return ((6 / 41 * x - 1008 / 2911) * x + 582 / 2911) * x
    if x < 4.0:
        x -= 3
        return ((-1 / 41 * x + 168 / 2911) * x - 97 / 2911) * x
    return 0.0


def _magic(x):
    x = abs(x)
    if x <= 0.5:
        return 0.75 - x * x
    if x <= 1.5:
        return 0.5 * (x - 1.5) ** 2
    return 0.0


_SHARP2021 = (-1 / 144, 6 / 144, -35 / 144, 204 / 144, -35 / 144, 6 / 144, -1 / 144)


def _magic_sharp(x):
    return sum(t * _magic(x - k) for k, t in zip(range(-3, 4), _SHARP2021))


_FILTERS = {
    "bilinear": (_triangle, 1.0),
    "bicubic": (_keys, 2.0),
    "lanczos": (_lanczos3, 3.0),
    "lanczos2": (_lanczos_n(2), 2.0),
    "lanczos4": (_lanczos_n(4), 4.0),
    "lanczos8": (_lanczos_n(8), 8.0),
    "mitchell": (_bc(1 / 3, 1 / 3), 2.0),
    "catrom": (_bc(0.0, 0.5), 2.0),
    "bspline": (_bc(1.0, 0.0), 2.0),
    "hermite": (_bc(0.0, 0.0), 1.0),
    "spline16": (_spline16, 2.0),
    "spline36": (_spline36, 3.0),
    "spline64": (_spline64, 4.0),
    "magic": (_magic_sharp, 4.5),
}


def _quant(v: float, bits: int) -> float:
    q = 1 << bits
    return round(v * q) / q


@lru_cache(maxsize=512)
def weights_1d(n_in: int, n_out: int, method: str, antialias: bool = True,
               fixed_point: bool = False, phase_bits: int = 4, coef_bits: int = 6,
               sigma: float = 0.5) -> torch.Tensor:
    """Dense (n_out, n_in) float64 matrix whose rows sum to 1."""
    if method not in METHODS or method in EWA:
        raise ValueError(f"unknown kernel {method!r}; choose from {METHODS}")
    if method in FREQUENCY:
        if fixed_point:
            raise ValueError("fixed-point isn't available for the frequency methods")
        return frequency_weights(n_in, n_out, method, antialias)
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

    if method == "gaussian":
        f, support = (lambda t: math.exp(-t * t / (2 * sigma * sigma))), max(1.0, 3.0 * sigma)
    else:
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


# ===================================================================== frequency
def _trig(n_in: int, q: torch.Tensor, keep: int, kind: str) -> torch.Tensor:
    """(len(q), n_in) matrix evaluating the band-limited interpolant of a
    length-n_in signal at fractional sample positions q (sample j at j),
    keeping the lowest `keep` frequencies. dct: the DCT-II basis (mirror
    edges); fft: the DFT basis (periodic), with the Nyquist bin split in
    half when `keep` is even so the result stays real."""
    j = torch.arange(n_in, dtype=torch.float64)
    q = q.to(torch.float64)
    if kind == "dct":
        k = torch.arange(keep, dtype=torch.float64)
        a = torch.full((keep,), 2.0 / n_in, dtype=torch.float64)
        a[0] = 1.0 / n_in
        out_b = torch.cos(math.pi * torch.outer(q + 0.5, k) / n_in)        # (n_q, keep)
        in_b = torch.cos(math.pi * torch.outer(k, j + 0.5) / n_in)         # (keep, n_in)
        return (out_b * a) @ in_b
    half = keep // 2
    k = torch.arange(0, half + 1, dtype=torch.float64)
    c = torch.full((half + 1,), 2.0, dtype=torch.float64)
    c[0] = 1.0
    if keep % 2 == 0 and half > 0:
        c[-1] = 1.0                               # +-Nyquist, half each
    aq = 2 * math.pi * torch.outer(q, k) / n_in    # cos(a(q-j)) = cos aq cos aj + sin aq sin aj
    aj = 2 * math.pi * torch.outer(k, j) / n_in
    return ((torch.cos(aq) * c) @ torch.cos(aj) + (torch.sin(aq) * c) @ torch.sin(aj)) / n_in


_WAVELETS = {
    # (analysis low-pass, synthesis low-pass), both odd and symmetric
    "wavelet53": ([-1, 2, 6, 2, -1], [1, 2, 1]),
    "wavelet97": ([0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443, 0.602949018236,
                   0.266864118443, -0.078223266529, -0.016864118443, 0.026748757411],
                  [-0.091271763114, -0.057543526229, 0.591271763114, 1.115087052457, 0.591271763114,
                   -0.057543526229, -0.091271763114]),
}


def _mirror(i: int, n: int) -> int:
    """Whole-sample symmetric extension (JPEG 2000's)."""
    if n == 1:
        return 0
    p = 2 * (n - 1)
    i %= p
    return p - i if i >= n else i


def _wavelet_down(n: int, h) -> torch.Tensor:
    h = [v / sum(h) for v in h]
    c = len(h) // 2
    m = (n + 1) // 2
    W = torch.zeros(m, n, dtype=torch.float64)
    for k in range(m):
        for t, v in enumerate(h):
            W[k, _mirror(2 * k + t - c, n)] += v
    return W


def _wavelet_up(n: int, g) -> torch.Tensor:
    g = [2 * v / sum(g) for v in g]
    c = len(g) // 2
    L = 2 * n - 1                                  # the zero-stuffed signal, originals at even places
    W = torch.zeros(2 * n, n, dtype=torch.float64)
    for p in range(2 * n):
        for t, v in enumerate(g):
            u = _mirror(p + t - c, L)
            if u % 2 == 0:
                W[p, u // 2] += v
    return W


def frequency_weights(n_in: int, n_out: int, method: str, antialias: bool = True) -> torch.Tensor:
    """(n_out, n_in) matrix for dct / fft / wavelet53 / wavelet97. Output
    pixel i is centred at input position t_i = (i + 0.5) n_in / n_out - 0.5
    (sample j at j), the same convention as the other kernels."""
    t = (torch.arange(n_out, dtype=torch.float64) + 0.5) * (n_in / n_out) - 0.5
    if method in ("dct", "fft"):
        keep = min(n_in, n_out) if antialias else n_in
        return _trig(n_in, t, keep, method)
    h, g = _WAVELETS[method]
    W = torch.eye(n_in, dtype=torch.float64)
    n, step = n_in, 1.0                            # sample k of the current signal sits at input k * step
    while n >= 2 * n_out and n >= 2:
        W = _wavelet_down(n, h) @ W
        n, step = (n + 1) // 2, step * 2
    while 2 * n <= n_out:
        W = _wavelet_up(n, g) @ W
        n, step = 2 * n, step / 2
    keep = min(n, n_out) if antialias else n
    return _trig(n, t / step, keep, "dct") @ W


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
               phase_bits: int = 4, coef_bits: int = 6, sigma: float = 0.5) -> torch.Tensor:
        """x: (N, C, H, W) float. Returns (N, C, height, width)."""
        n, c, h, w = x.shape
        if method in EWA:
            return self._ewa(x, width, height, method, antialias)
        kw = {}
        if method in FREQUENCY:
            kw = dict(antialias=antialias)
        elif method not in ("nearest", "area"):
            kw = dict(antialias=antialias, fixed_point=fixed_point,
                      phase_bits=phase_bits, coef_bits=coef_bits)
            if method == "gaussian":
                kw["sigma"] = float(sigma)
        if w != width:
            Ww = self._mat(w, width, method, x.device, **kw)
            x = torch.matmul(x, Ww.t())
            if fixed_point:
                x = torch.round(x.clamp(0, 1) * 255.0) / 255.0
        if h != height:
            Wh = self._mat(h, height, method, x.device, **kw)
            x = torch.matmul(Wh, x)
        return x


# ===================================================================== EWA
JINC_ZERO1 = 1.2196698912665045   # first zero of jinc(x) = 2 J1(pi x) / (pi x)
JINC_ZERO3 = 3.2383154841662362   # third zero: mpv's ewa_lanczos radius
_EWA_FILTERS = {  # name -> (radius, blur)
    "ewa_lanczos": (JINC_ZERO3, 1.0),
    "ewa_lanczossharp": (JINC_ZERO3, 0.9812505644269356),
    "ewa_robidoux": (2.0, 1.0),
    "ewa_robidouxsharp": (2.0, 1.0),
}
_ROBIDOUX = (0.37821575509399867, 0.31089212245300067)
_ROBIDOUX_SHARP = (0.2620145123990142, 0.3689927438004929)


def _jinc(r: torch.Tensor) -> torch.Tensor:
    pr = math.pi * r
    return torch.where(r.abs() < 1e-12, torch.ones_like(r), 2 * torch.special.bessel_j1(pr) / pr.clamp(min=1e-30))


def _ewa_weight(method: str, r: torch.Tensor) -> torch.Tensor:
    radius, blur = _EWA_FILTERS[method]
    r = r / blur
    if method.startswith("ewa_lanczos"):
        w = _jinc(r) * _jinc(r * JINC_ZERO1 / radius)
    else:
        B, C = _ROBIDOUX if method == "ewa_robidoux" else _ROBIDOUX_SHARP
        x = r.abs()
        w = torch.where(x < 1, ((12 - 9 * B - 6 * C) * x ** 3 + (-18 + 12 * B + 6 * C) * x ** 2 + (6 - 2 * B)) / 6,
                        ((-B - 6 * C) * x ** 3 + (6 * B + 30 * C) * x ** 2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6)
    return torch.where(r < radius, w, torch.zeros_like(w))


@lru_cache(maxsize=64)
def ewa_plan(h_in: int, w_in: int, h_out: int, w_out: int, method: str, antialias: bool = True):
    """(row index (h_out, ky), column index (w_out, kx), normalised weights
    (h_out, w_out, ky, kx)) for an elliptical weighted average: every output
    pixel is a radial filter over the input pixels around its centre
    (stretched by the shrink factor on each axis when antialiasing). Edges
    replicate, like the separable kernels."""
    radius, blur = _EWA_FILTERS[method]
    sx, sy = w_in / w_out, h_in / h_out
    fx = max(sx, 1.0) if antialias else 1.0
    fy = max(sy, 1.0) if antialias else 1.0
    reach_x, reach_y = radius * blur * fx, radius * blur * fy

    def axis(n_in, n_out, s, reach):
        c = (torch.arange(n_out, dtype=torch.float64) + 0.5) * s
        lo = torch.floor(c - reach).long()
        k = int(math.ceil(2 * reach)) + 2
        idx = lo.view(-1, 1) + torch.arange(k).view(1, -1)
        d = idx.double() + 0.5 - c.view(-1, 1)
        return idx.clamp(0, n_in - 1), d

    iy, dy = axis(h_in, h_out, sy, reach_y)
    ix, dx = axis(w_in, w_out, sx, reach_x)
    r = torch.sqrt((dy.view(h_out, 1, -1, 1) / fy) ** 2 + (dx.view(1, w_out, 1, -1) / fx) ** 2)
    wgt = _ewa_weight(method, r)
    wgt = wgt / wgt.sum((-1, -2), keepdim=True)
    return iy, ix, wgt


def _ewa(self, x: torch.Tensor, width: int, height: int, method: str, antialias: bool = True) -> torch.Tensor:
    n, c, h, w = x.shape
    key = ("ewa", h, w, height, width, method, antialias, str(x.device), x.dtype)
    plan = self._cache.get(key)
    if plan is None:
        iy, ix, wgt = ewa_plan(h, w, height, width, method, antialias)
        plan = (iy.to(x.device), ix.to(x.device), wgt.to(x.device, x.dtype))
        self._cache[key] = plan
    iy, ix, wgt = plan
    out = torch.zeros((n, c, height, width), dtype=x.dtype, device=x.device)
    for a in range(iy.shape[1]):
        rows = x[:, :, iy[:, a]]                             # (n, c, height, w)
        for b in range(ix.shape[1]):
            wab = wgt[:, :, a, b]
            if not bool((wab != 0).any()):
                continue
            out += rows[..., ix[:, b]] * wab
    return out


KernelResizer._ewa = _ewa
