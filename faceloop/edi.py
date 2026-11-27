"""2x upscalers from the pixel-art and edge-directed families, batched on
(N,3,H,W) encoded frames in 0..1, any device. Each doubles width and height;
4x and 8x repeat it.

mmpx   McGuire & Gagiu, "MMPX Style-Preserving Pixel-Art Magnification",
       JCGT 2021. Ported from the authors' MIT-licensed C reference
       (rules compare exact 8-bit colours; every output pixel is a copy of
       an input pixel).
dcci   Zhou, Shen & Dong, "Image Zooming Using Directional Cubic Convolution
       Interpolation", 2012: diagonal gaps first, then the rest; cubic
       (-1, 9, 9, -1)/16 along an edge, weighted mix in smooth areas
       (threshold 1.15, exponent 5). Edge strengths from luma, all channels
       interpolated with them.
nedi   Li & Orchard, "New Edge-Directed Interpolation", IEEE TIP 2001:
       each new pixel is a linear prediction from its 4 neighbours with
       weights fitted on the low-resolution neighbourhood (geometric
       duality), diagonal gaps first, then the rest. Weights from luma,
       applied to all channels. Flat areas (local variance below a
       threshold) use the plain average, as in the paper.
icbi   Giachetti & Asuni, "Real-Time Artifact-Free Image Upscaling", IEEE
       TIP 2011 (iterative curvature-based interpolation): FCBI start, then
       greedy +-delta steps on curvature-continuity / enhancement energies,
       per colour channel. Follows the published code's defaults and step
       schedule; pixels are updated all at once per iteration (the
       original sweeps in raster order), so it isn't bit-identical.

The lattice methods (dcci, nedi, icbi, and eedi3 in eedi.py) keep the
original pixels at even positions, which moves the picture by half a new
pixel up and left. Repeated in a loop that walks the image off-frame, so
by default the result is shifted back by half a pixel with a Lanczos-3
resampler, as VapourSynth/AviSynth "rpow2" scripts do ("center": false
keeps the raw lattice and its drift).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LATTICE = ("dcci", "nedi", "icbi", "eedi3", "eedi2")
METHODS = ("mmpx",) + LATTICE


def _luma255(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255.0


# ===================================================================== centring
def _lanczos_taps(shift: float, a: int = 3):
    """Taps (offsets, weights) to sample at x + shift."""
    offs = list(range(-a + 1, a + 1)) if shift > 0 else list(range(-a, a))
    w = []
    for o in offs:
        t = o - shift
        w.append(1.0 if t == 0 else (math.sin(math.pi * t) * math.sin(math.pi * t / a)) / (math.pi * math.pi * t * t / a))
    s = sum(w)
    return offs, [v / s for v in w]


def half_pixel_shift(x: torch.Tensor) -> torch.Tensor:
    """Resample at -0.5 pixel in x and y (Lanczos-3). A lattice result holds
    at pixel o what belongs at o + 0.5, so this puts it back."""
    offs, w = _lanczos_taps(-0.5)
    k = torch.tensor(w, dtype=x.dtype, device=x.device)
    pad_l, pad_r = -offs[0], offs[-1]
    c = x.shape[1]
    y = F.pad(x, (pad_l, pad_r, 0, 0), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.pad(y, (0, 0, pad_l, pad_r), mode="replicate")
    return F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def upscale2(x: torch.Tensor, method: str, center: bool = True, **kw) -> torch.Tensor:
    if method == "mmpx":
        return mmpx(x)
    if method == "dcci":
        y = dcci(x, **kw)
    elif method == "nedi":
        y = nedi(x, **kw)
    elif method == "icbi":
        y = icbi(x, **kw)
    elif method == "eedi3":
        from .eedi import eedi3_2x
        y = eedi3_2x(x, **kw)
    elif method == "eedi2":
        from .eedi import eedi2_2x
        y = eedi2_2x(x, **kw)
    else:
        raise ValueError(f"unknown 2x method {method!r}")
    return half_pixel_shift(y) if center else y


def upscale(x: torch.Tensor, method: str, scale: int, center: bool = True, check=None, **kw) -> torch.Tensor:
    while scale > 1:
        if check:
            check()
        x = upscale2(x, method, center, **kw).clamp(0, 1)
        scale //= 2
    return x


def _lattice(x: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) -> (N,C,2H,2W) with originals at even positions, NaN elsewhere."""
    n, c, h, w = x.shape
    out = torch.full((n, c, 2 * h, 2 * w), float("nan"), dtype=x.dtype, device=x.device)
    out[:, :, ::2, ::2] = x
    return out


class _View:
    """Shifted reads of a padded image: v(dy, dx) is the image moved so that
    position (y, x) holds the original (y+dy, x+dx)."""

    def __init__(self, t: torch.Tensor, pad: int, h: int, w: int):
        self.t, self.p, self.h, self.w = t, pad, h, w

    def __call__(self, dy: int, dx: int) -> torch.Tensor:
        p = self.p
        return self.t[..., p + dy:p + dy + self.h, p + dx:p + dx + self.w]


# ===================================================================== MMPX
def mmpx(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    codes = torch.round(x.clamp(0, 1) * 255.0).to(torch.int32)
    packed = (codes[:, 0] << 16) | (codes[:, 1] << 8) | codes[:, 2]                 # (N,H,W)
    luma = codes[:, 0] + codes[:, 1] + codes[:, 2] + 1                              # opaque: (r+g+b+1)*1
    P = 3
    pv = _View(F.pad(packed.unsqueeze(1).float(), (P,) * 4, mode="replicate").squeeze(1).to(torch.int32), P, h, w)
    fv = _View(F.pad(x, (P,) * 4, mode="replicate"), P, h, w)
    lv = _View(F.pad(luma.unsqueeze(1).float(), (P,) * 4, mode="replicate").squeeze(1), P, h, w)

    def px(dx, dy):  # src(x+dx, y+dy): (packed, float)
        return pv(dy, dx), fv(dy, dx)

    A, B, C = px(-1, -1), px(0, -1), px(1, -1)
    D, E, F_ = px(-1, 0), px(0, 0), px(1, 0)
    G, H, I = px(-1, 1), px(0, 1), px(1, 1)
    Q, R = px(-2, 0), px(2, 0)
    P_, S = px(0, -2), px(0, 2)
    a, b, cc, d, e, f, g, hh, i = (t[0] for t in (A, B, C, D, E, F_, G, H, I))
    q, r, p, s = Q[0], R[0], P_[0], S[0]
    Bl, Dl, El, Fl, Hl = lv(-1, 0), lv(0, -1), lv(0, 0), lv(0, 1), lv(1, 0)

    out = {k: [e.clone(), E[1].clone()] for k in "JKLM"}

    def setv(mask, dst, src):
        """dst = src where mask; src is a (packed, float) pair or an output name."""
        sp, sf = (out[src][0], out[src][1]) if isinstance(src, str) else src
        out[dst][0] = torch.where(mask, sp, out[dst][0])
        out[dst][1] = torch.where(mask.unsqueeze(1), sf, out[dst][1])

    def any3(v, a0, a1, a2):
        return (v == a0) | (v == a1) | (v == a2)

    def all2(v, a0, a1):
        return (v == a0) & (v == a1)

    def all3(v, a0, a1, a2):
        return (v == a0) & (v == a1) & (v == a2)

    def all4(v, a0, a1, a2, a3):
        return (v == a0) & (v == a1) & (v == a2) & (v == a3)

    def none2(v, a0, a1):
        return (v != a0) & (v != a1)

    def none4(v, a0, a1, a2, a3):
        return (v != a0) & (v != a1) & (v != a2) & (v != a3)

    busy = ~((a == e) & (b == e) & (cc == e) & (d == e) & (f == e) & (g == e) & (hh == e) & (i == e))
    # 1:1 slope rules
    setv(busy & ((d == b) & (d != hh) & (d != f)) & ((El >= Dl) | (e == a)) & any3(e, a, cc, g)
         & ((El < Dl) | (a != d) | (e != p) | (e != q)), "J", D)
    setv(busy & ((b == f) & (b != d) & (b != hh)) & ((El >= Bl) | (e == cc)) & any3(e, a, cc, i)
         & ((El < Bl) | (cc != b) | (e != p) | (e != r)), "K", B)
    setv(busy & ((hh == d) & (hh != f) & (hh != b)) & ((El >= Hl) | (e == g)) & any3(e, a, g, i)
         & ((El < Hl) | (g != hh) | (e != s) | (e != q)), "L", H)
    setv(busy & ((f == hh) & (f != b) & (f != d)) & ((El >= Fl) | (e == i)) & any3(e, cc, g, i)
         & ((El < Fl) | (i != hh) | (e != r) | (e != s)), "M", F_)
    # intersection rules
    m = busy & (e != f) & all4(e, cc, i, d, q) & all2(f, b, hh) & (f != px(3, 0)[0])
    setv(m, "K", F_); setv(m, "M", F_)
    m = busy & (e != d) & all4(e, a, g, f, r) & all2(d, b, hh) & (d != px(-3, 0)[0])
    setv(m, "J", D); setv(m, "L", D)
    m = busy & (e != hh) & all4(e, g, i, b, p) & all2(hh, d, f) & (hh != px(0, 3)[0])
    setv(m, "L", H); setv(m, "M", H)
    m = busy & (e != b) & all4(e, a, cc, hh, s) & all2(b, d, f) & (b != px(0, -3)[0])
    setv(m, "J", B); setv(m, "K", B)
    m = busy & (Bl < El) & all4(e, g, hh, i, s) & none4(e, a, d, cc, f)
    setv(m, "J", B); setv(m, "K", B)
    m = busy & (Hl < El) & all4(e, a, b, cc, p) & none4(e, d, g, i, f)
    setv(m, "L", H); setv(m, "M", H)
    m = busy & (Fl < El) & all4(e, a, d, g, q) & none4(e, b, cc, i, hh)
    setv(m, "K", F_); setv(m, "M", F_)
    m = busy & (Dl < El) & all4(e, cc, f, i, r) & none4(e, b, a, g, hh)
    setv(m, "J", D); setv(m, "L", D)
    # 2:1 slope rules
    hb = busy & (hh != b)
    m1 = hb & (hh != a) & (hh != e) & (hh != cc)
    setv(m1 & all3(hh, g, f, r) & none2(hh, d, px(2, -1)[0]), "L", "M")
    setv(m1 & all3(hh, i, d, q) & none2(hh, f, px(-2, -1)[0]), "M", "L")
    m2 = hb & (b != i) & (b != g) & (b != e)
    setv(m2 & all3(b, a, f, r) & none2(b, d, px(2, 1)[0]), "J", "K")
    setv(m2 & all3(b, cc, d, q) & none2(b, f, px(-2, 1)[0]), "K", "J")
    fd = busy & (f != d)
    m3 = fd & (d != i) & (d != e) & (d != cc)
    setv(m3 & all3(d, a, hh, s) & none2(d, b, px(1, 2)[0]), "J", "L")
    setv(m3 & all3(d, g, b, p) & none2(d, hh, px(1, -2)[0]), "L", "J")
    m4 = fd & (f != e) & (f != a) & (f != g)
    setv(m4 & all3(f, cc, hh, s) & none2(f, b, px(-1, 2)[0]), "K", "M")
    setv(m4 & all3(f, i, b, p) & none2(f, hh, px(-1, -2)[0]), "M", "K")

    y = torch.empty((n, c, 2 * h, 2 * w), dtype=x.dtype, device=x.device)
    y[:, :, 0::2, 0::2] = out["J"][1]
    y[:, :, 0::2, 1::2] = out["K"][1]
    y[:, :, 1::2, 0::2] = out["L"][1]
    y[:, :, 1::2, 1::2] = out["M"][1]
    return y


# ===================================================================== DCCI
def _cubic(p0, p1, p2, p3):
    return (-p0 + 9 * p1 + 9 * p2 - p3) / 16.0


def dcci(x: torch.Tensor, threshold: float = 1.15, exponent: float = 5.0) -> torch.Tensor:
    n, c, h, w = x.shape
    PAD = 4
    # step 1: diagonal gaps, from the original pixels (indices in the low-res grid)
    xp = F.pad(x, (PAD,) * 4, mode="replicate")
    lp = _luma255(xp)
    v = _View(xp, PAD, h, w)      # v(dy, dx) = P(x+dx, y+dy) for the gap at (y+1/2, x+1/2)
    lv = _View(lp, PAD, h, w)
    # 4x4 block P(X,Y), X,Y in 0..3 = low-res offsets (X-1, Y-1) from the gap's top-left pixel
    Pl = lambda X, Y: lv(Y - 1, X - 1)
    Pc = lambda X, Y: v(Y - 1, X - 1)
    d1 = sum((Pl(X, Y) - Pl(X - 1, Y + 1)).abs() for X in range(1, 4) for Y in range(0, 3))
    d2 = sum((Pl(X, Y) - Pl(X + 1, Y + 1)).abs() for X in range(0, 3) for Y in range(0, 3))
    down_right = _cubic(Pc(0, 0), Pc(1, 1), Pc(2, 2), Pc(3, 3))
    up_right = _cubic(Pc(3, 0), Pc(2, 1), Pc(1, 2), Pc(0, 3))
    w1 = 1.0 / (1.0 + d1 ** exponent)
    w2 = 1.0 / (1.0 + d2 ** exponent)
    smooth = (down_right * w1 + up_right * w2) / (w1 + w2)
    diag = torch.where((1 + d1) > threshold * (1 + d2), down_right,
                       torch.where((1 + d2) > threshold * (1 + d1), up_right, smooth))
    lat = x.repeat_interleave(2, 2).repeat_interleave(2, 3)  # gaps start as copies (only edges read them)
    lat[:, :, 1::2, 1::2] = diag
    # step 2: the rest, from originals and diagonal pixels on the lattice
    lpad = F.pad(lat, (2 * PAD,) * 4, mode="replicate")
    L = _luma255(lpad)
    H2, W2 = 2 * h, 2 * w
    lvw = _View(lpad, 2 * PAD, H2, W2)
    llw = _View(L, 2 * PAD, H2, W2)
    Q = lambda dx, dy: llw(dy, dx)
    Qc = lambda dx, dy: lvw(dy, dx)
    hd = ((Q(1, -2) - Q(-1, -2)).abs() + (Q(2, -1) - Q(0, -1)).abs() + (Q(0, -1) - Q(-2, -1)).abs()
          + (Q(3, 0) - Q(1, 0)).abs() + (Q(1, 0) - Q(-1, 0)).abs() + (Q(-1, 0) - Q(-3, 0)).abs()
          + (Q(2, 1) - Q(0, 1)).abs() + (Q(0, 1) - Q(-2, 1)).abs() + (Q(1, 2) - Q(-1, 2)).abs())
    vd = ((Q(-2, 1) - Q(-2, -1)).abs() + (Q(-1, 2) - Q(-1, 0)).abs() + (Q(-1, 0) - Q(-1, -2)).abs()
          + (Q(0, 3) - Q(0, 1)).abs() + (Q(0, 1) - Q(0, -1)).abs() + (Q(0, -1) - Q(0, -3)).abs()
          + (Q(1, 2) - Q(1, 0)).abs() + (Q(1, 0) - Q(1, -2)).abs() + (Q(2, 1) - Q(2, -1)).abs())
    vert = _cubic(Qc(0, -3), Qc(0, -1), Qc(0, 1), Qc(0, 3))
    horz = _cubic(Qc(-3, 0), Qc(-1, 0), Qc(1, 0), Qc(3, 0))
    w1 = 1.0 / (1.0 + hd ** exponent)
    w2 = 1.0 / (1.0 + vd ** exponent)
    smooth = (vert * w1 + horz * w2) / (w1 + w2)
    rest = torch.where((1 + hd) > threshold * (1 + vd), vert,
                       torch.where((1 + vd) > threshold * (1 + hd), horz, smooth))
    gaps = torch.zeros((1, 1, H2, W2), dtype=torch.bool, device=x.device)
    gaps[..., 0::2, 1::2] = True
    gaps[..., 1::2, 0::2] = True
    return torch.where(gaps, rest, lat)


# ===================================================================== NEDI
def _solve4(A: torch.Tensor, b: torch.Tensor):
    """Batched 4x4 SPD solve by Cholesky; returns (x, ok)."""
    L = torch.zeros_like(A)
    ok = torch.ones(A.shape[:-2], dtype=torch.bool, device=A.device)
    for i in range(4):
        for j in range(i + 1):
            s = A[..., i, j] - (L[..., i, :j] * L[..., j, :j]).sum(-1)
            if i == j:
                ok = ok & (s > 1e-12)
                L[..., i, i] = torch.sqrt(s.clamp(min=1e-12))
            else:
                L[..., i, j] = s / L[..., j, j]
    y = torch.zeros_like(b)
    for i in range(4):
        y[..., i] = (b[..., i] - (L[..., i, :i] * y[..., :i]).sum(-1)) / L[..., i, i]
    xs = torch.zeros_like(b)
    for i in reversed(range(4)):
        xs[..., i] = (y[..., i] - (L[..., i + 1:, i] * xs[..., i + 1:]).sum(-1)) / L[..., i, i]
    return xs, ok


def _nedi_fill(img, lum, targets, nb, train_nb, window_kernel, var_kernel, var_thresh):
    """Predict pixels at `targets` (bool map) of the lattice image `img`
    from their 4 neighbours `nb` (offsets), with weights fitted on the
    training samples selected by `window_kernel` (their neighbours at
    offsets `train_nb`). img/lum: (N,C,H,W)/(N,1,H,W) with known pixels."""
    n, c, h, w = img.shape
    pad = 8
    lp = F.pad(lum, (pad,) * 4, mode="replicate")
    ip = F.pad(img, (pad,) * 4, mode="replicate")
    LV = _View(lp, pad, h, w)
    IV = _View(ip, pad, h, w)
    tn = [LV(dy, dx) for dy, dx in train_nb]      # neighbours of a sample at each position
    y0 = LV(0, 0)
    kw = window_kernel.shape[-1]
    conv = lambda t: F.conv2d(F.pad(t, (kw // 2,) * 4, mode="replicate"), window_kernel)
    A = torch.empty((n, 1, h, w, 4, 4), dtype=img.dtype, device=img.device)
    for i in range(4):
        for j in range(i, 4):
            s = conv(tn[i] * tn[j])
            A[..., i, j] = s
            A[..., j, i] = s
    bvec = torch.stack([conv(tn[i] * y0) for i in range(4)], -1)
    cnt = float(window_kernel.sum())
    A = A / cnt
    bvec = bvec / cnt
    A = A + torch.eye(4, dtype=img.dtype, device=img.device) * 1e-2   # (0..255)^2 scale: tiny
    a, ok = _solve4(A, bvec)
    # local variance decides edge (NEDI) vs flat (average)
    mean = F.conv2d(F.pad(lum, (var_kernel.shape[-1] // 2,) * 4, mode="replicate"), var_kernel) / float(var_kernel.sum())
    mean2 = F.conv2d(F.pad(lum * lum, (var_kernel.shape[-1] // 2,) * 4, mode="replicate"), var_kernel) / float(var_kernel.sum())
    edge = ((mean2 - mean * mean) > var_thresh) & ok
    s = a.sum(-1)
    edge = edge & (s > 0.5) & (s < 1.5) & torch.isfinite(s)
    nbs = [IV(dy, dx) for dy, dx in nb]                   # (N,C,H,W) each
    pred = sum(a[..., k] * nbs[k] for k in range(4))      # a (N,1,H,W) broadcasts over C
    avg = sum(nbs) / 4.0
    val = torch.where(edge, pred, avg)
    return torch.where(targets, val, img)


def nedi(x: torch.Tensor, window: int = 4, var_threshold: float = 64.0) -> torch.Tensor:
    """window: training window side in low-res pixels (4 -> 16 samples).
    var_threshold: local variance (0..255 scale) below which the plain
    average of the 4 neighbours is used."""
    n, c, h, w = x.shape
    H2, W2 = 2 * h, 2 * w
    # one replicated row/column past the edge so the last gaps have neighbours;
    # gaps start as copies of their top-left original (only read at the edges)
    xe = F.pad(x, (0, 1, 0, 1), mode="replicate")
    full = xe.repeat_interleave(2, 2).repeat_interleave(2, 3)
    lum = _luma255(full)
    # window kernel over the even lattice: window x window originals around the gap
    k = 2 * window
    wk = torch.zeros((1, 1, k + 1, k + 1), dtype=x.dtype, device=x.device)
    for yy in range(window):
        for xx in range(window):
            wk[0, 0, 1 + 2 * yy, 1 + 2 * xx] = 1.0
    # centre the window on the gap: gap at offset (k/2, k/2) -> originals at odd offsets from the gap
    var_k = torch.zeros((1, 1, 3, 3), dtype=x.dtype, device=x.device)
    var_k[0, 0, 0::2, 0::2] = 1.0  # the 4 diagonal neighbours of a gap
    t1 = torch.zeros((1, 1, full.shape[2], full.shape[3]), dtype=torch.bool, device=x.device)
    t1[..., 1::2, 1::2] = True
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    full = _nedi_fill(full * 255.0, lum, t1, diag, [(-2, -2), (-2, 2), (2, -2), (2, 2)], wk, var_k,
                      var_threshold) / 255.0
    # step 2: the rest from axial neighbours; training on the quincunx lattice, neighbours at (+-2,0),(0,+-2)
    lum = _luma255(full)
    k2 = 2 * window - 1
    wk2 = torch.zeros((1, 1, k2, k2), dtype=x.dtype, device=x.device)
    cc = k2 // 2
    for yy in range(k2):
        for xx in range(k2):
            if (yy - cc + xx - cc) % 2 != 0 and abs(yy - cc) + abs(xx - cc) <= window:
                wk2[0, 0, yy, xx] = 1.0
    var_k2 = torch.zeros((1, 1, 3, 3), dtype=x.dtype, device=x.device)
    var_k2[0, 0, 0, 1] = var_k2[0, 0, 2, 1] = var_k2[0, 0, 1, 0] = var_k2[0, 0, 1, 2] = 1.0
    t2 = torch.zeros_like(t1)
    t2[..., 0::2, 1::2] = True
    t2[..., 1::2, 0::2] = True
    axial = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    full = _nedi_fill(full * 255.0, lum, t2, axial, [(-2, 0), (2, 0), (0, -2), (0, 2)], wk2, var_k2,
                      var_threshold) / 255.0
    return full[:, :, :H2, :W2]


# ===================================================================== ICBI
def icbi(x: torch.Tensor, iterations: int = 20, tm: float = 100.0, tc: float = 50.0, ts: float = 100.0,
         alpha: float = 1.0, beta: float = -1.0, gamma: float = 5.0, potential: int = 1) -> torch.Tensor:
    """Per channel, on 0..255 values. Defaults are the published code's
    (ST=20, TM=100, TC=50, TS=100, AL=1, BT=-1, GM=5, PF=1)."""
    n, c, h, w = x.shape
    # one extra low-res row/column (replicated) so the 2h-1 grid covers 2h
    xe = F.pad(x, (0, 1, 0, 1), mode="replicate") * 255.0
    hh, ww = h + 1, w + 1
    mm, nn = 2 * hh - 1, 2 * ww - 1
    I = torch.zeros((n, c, mm, nn), dtype=x.dtype, device=x.device)
    I[:, :, 0::2, 0::2] = xe
    # borders: average of the 2 neighbours along the border
    I[:, :, 1:mm - 1:2, 0] = (I[:, :, 0:mm - 2:2, 0] + I[:, :, 2::2, 0]) / 2
    I[:, :, 1:mm - 1:2, nn - 1] = (I[:, :, 0:mm - 2:2, nn - 1] + I[:, :, 2::2, nn - 1]) / 2
    I[:, :, 0, 1:nn - 1:2] = (I[:, :, 0, 0:nn - 2:2] + I[:, :, 0, 2::2]) / 2
    I[:, :, mm - 1, 1:nn - 1:2] = (I[:, :, mm - 1, 0:nn - 2:2] + I[:, :, mm - 1, 2::2]) / 2
    PADI = 6
    ii = torch.arange(mm, device=x.device).view(mm, 1)
    jj = torch.arange(nn, device=x.device).view(1, nn)

    def view(t):
        return _View(F.pad(t, (PADI,) * 4, mode="replicate"), PADI, mm, nn)

    for s in (0, 1):
        # ---- FCBI for this class
        if s == 0:
            cls = (ii % 2 == 1) & (jj % 2 == 1)
            rng = (ii >= 1) & (ii < mm - 1) & (jj >= 1) & (jj < nn - 1)
        else:
            cls = ((ii + jj) % 2 == 1)
            rng = (ii >= 1) & (ii < mm - 1) & (jj >= 1) & (jj < nn - 1)
        tgt = cls & rng
        V = view(I)
        g = lambda di, dj: V(di, dj)
        v1 = (g(-1, -1 + s) - g(1, 1 - s)).abs()
        v2 = (g(1 - s, -1) - g(-1 + s, 1)).abs()
        p1 = (g(-1, -1 + s) + g(1, 1 - s)) / 2
        p2 = (g(1 - s, -1) + g(-1 + s, 1)) / 2
        inner = (ii > 2 - s) & (ii < mm - 4 - s) & (jj > 2 - s) & (jj < nn - 4 - s)
        use2 = (v1 < tm) & (v2 < tm) & inner & ((p1 - p2).abs() < tm)
        lhs = (g(-1 - s, -3 + 2 * s) + g(-3 + s, -1 + 2 * s) + g(1 + s, 3 - 2 * s) + g(3 - s, 1 - 2 * s)
               + 2 * p2 - 6 * p1).abs()
        rhs = (g(-3 + 2 * s, 1 + s) + g(-1 + 2 * s, 3 - s) + g(3 - 2 * s, -1 - s) + g(1 - 2 * s, -3 + s)
               + 2 * p1 - 6 * p2).abs()
        val = torch.where(use2, torch.where(lhs > rhs, p1, p2), torch.where(v1 < v2, p1, p2))
        I = torch.where(tgt, val, I)
        # ---- iterative refinement (greedy +-step on the energy)
        if s == 0:
            upd = cls & (ii >= 5) & (ii < mm - 5) & (jj >= 5) & (jj < nn - 5)
        else:
            upd = cls & (ii >= 2) & (ii < mm - 2) & (jj >= 2) & (jj < nn - 2)
        step = 4.0 / (1 + s)
        for it in range(iterations):
            if it < iterations / 4 - 1:
                step = 1.0
            elif it < iterations / 2 - 1:
                step = 2.0
            elif it < 3 * iterations / 4 - 1:
                step = 2.0
            V = view(I)
            g = lambda di, dj: V(di, dj)
            C1 = (g(-1 + s, -1) - g(1 - s, 1)) / 2
            C2 = (g(1 - 2 * s, -1 + s) - g(-1 + 2 * s, 1 - s)) / 2
            D1 = g(-1 + s, -1) + g(1 - s, 1) - 2 * I
            D2 = g(1, -1 + s) + g(-1, 1 - s) - 2 * I
            D3 = (g(-s, -2 + s) - g(-2 + s, s) + g(s, 2 - s) - g(2 - s, -s)) / 2
            vd1, vd2 = view(D1), view(D2)
            c1 = ((g(1 - s, 1) - I).abs() <= tc).to(I.dtype)
            c2 = ((g(-1 + s, -1) - I).abs() <= tc).to(I.dtype)
            c3 = ((g(1, -1 + s) - I).abs() <= tc).to(I.dtype)
            c4 = ((g(-1, 1 - s) - I).abs() <= tc).to(I.dtype)

            def terms(off):
                a1 = c1 * (D1 - vd1(1 - s, 1) - off).abs() + c2 * (D1 - vd1(-1 + s, -1) - off).abs()
                a2 = c3 * (D1 - vd1(1, -1 + s) - off).abs() + c4 * (D1 - vd1(-1, 1 - s) - off).abs()
                a3 = c1 * (D2 - vd2(1 - s, 1) - off).abs() + c2 * (D2 - vd2(-1 + s, -1) - off).abs()
                a4 = c3 * (D2 - vd2(1, -1 + s) - off).abs() + c4 * (D2 - vd2(-1, 1 - s) - off).abs()
                return a1 + a2 + a3 + a4

            def ends(off):
                e5 = (g(-2 + 2 * s, -2) + g(2 - 2 * s, 2) - 2 * I - off).abs()
                e6 = (g(2, -2 + 2 * s) + g(-2, 2 - 2 * s) - 2 * I - off).abs()
                return e5 + e6

            den = C1 * C1 + C2 * C2
            eiso = torch.where(den > 0, (C1 * C1 * D2 - 2 * C1 * C2 * D3 + C2 * C2 * D1) / den.clamp(min=1e-12),
                               torch.zeros_like(den))
            eiso = torch.where(eiso.abs() < 0.2, torch.zeros_like(eiso), eiso)
            sg = torch.sign(eiso)
            EN = alpha * terms(0.0)
            EA = alpha * terms(3 * step)
            ES = alpha * terms(-3 * step)
            if potential == 1:
                EN, EA, ES = EN + beta * ends(0.0), EA + beta * ends(2 * step), ES + beta * ends(-2 * step)
            elif potential == 2:
                EA, ES = EA - gamma * sg, ES - gamma * sg
            else:
                EN, EA, ES = (EN + beta * ends(0.0), EA + beta * ends(2 * step) - gamma * sg,
                              ES + beta * ends(-2 * step) + gamma * sg)
            up = upd & (EN > EA) & (ES > EA)
            down = upd & (EN > ES) & (EA > ES) & ~up
            I = I + step * up.to(I.dtype) - step * down.to(I.dtype)
            changed = float((up | down).sum()) * step / max(1, n * c)
            if changed < ts:
                break
    return (I[:, :, :2 * h, :2 * w] / 255.0)
