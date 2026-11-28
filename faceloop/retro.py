"""Old pixel-art scalers from emulators, batched on (N,3,H,W) encoded frames in
0..1, any device. Reads past the edge repeat the border pixel.

scale2x, scale3x
    Andrea Mazzoleni's Scale2x / Scale3x (AdvanceMAME; Scale2x is EPX),
    the rules as published on scale2x.it. Every output pixel is a copy of
    an input pixel; colours are compared as exact 8-bit values.
2xsai, super2xsai, supereagle
    Derek Liauw Kie Fa's 2xSaI, Super 2xSaI and Super Eagle, ported from
    ScummVM's graphics/scaler/sai.cpp (GPL-3.0-or-later, so this file is
    too), with ScummVM's 8-bit integer blends, including the one that reads
    the second colour's low bits twice in its 6:1:1 mix. Each builds its 2x2
    block from the pixel and the ones right of and below it, which moves the
    picture up and left (2xSaI by half a new pixel each way, Super 2xSaI half
    across and one down, Super Eagle one each way). By default that is put
    back after every x2: whole pixels by moving the frame (edge repeated),
    halves with the same Lanczos-3 half-pixel step as the lattice methods
    ("center": false keeps the raw drift).
hqx
    Maxim Stepin's hq2x / hq3x / hq4x: the hqx library itself (Cameron
    Zemek's C version, LGPL 2.1) from faceloop/native/hqx, compiled the first
    time, on the CPU.
superxbr
    Hyllian's Super-xBR, passes 0-2 of libretro's super-xbr.slangp
    (slang-shaders commit 02bf134555c4595bd6b69be89feaff923da83754, MIT,
    Copyright (c) 2015 Hyllian), with the preset's settings: mode 1
    ("details"), edge strength 3 in every pass, filter weight 0 in pass 0
    and 1 in the others. Pass 2 already puts the result on the right pixel
    centres; the preset's display passes after it (jinc2 to the screen size,
    deblur) aren't part of the 2x step and are left out.
scalefx
    Sp00kyFox's ScaleFX (MIT, Copyright (c) 2016 Sp00kyFox), the five
    passes of libretro's scalefx.slangp at the same commit, x3 (x9 by
    running it twice), with its defaults (threshold 0.5, filter AA on,
    filter corners on). Every output pixel is a copy of the input pixel
    itself or of one up to two away, so only the frame's own colours appear.
    The passes' 8-bit framebuffers only ever hold the packed flags and
    indices, which are carried here as they are.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SCALES = {"scale2x": (2, 4, 8), "scale3x": (3, 9), "2xsai": (2, 4, 8), "super2xsai": (2, 4, 8),
          "supereagle": (2, 4, 8), "hqx": (2, 3, 4), "superxbr": (2, 4, 8), "scalefx": (3, 9)}
METHODS = tuple(SCALES)


def _codes(x):
    return torch.round(x.clamp(0, 1) * 255.0).to(torch.int32)


def _pack(c):
    return (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]


class _Nb:
    """Neighbours of every pixel: nb(dx, dy) is the image moved so each
    position holds the pixel dx right and dy down of it (border repeated)."""

    def __init__(self, t, pad):
        self.p, self.h, self.w = pad, t.shape[-2], t.shape[-1]
        if t.is_floating_point():
            self.t = F.pad(t, (pad,) * 4, mode="replicate")
        else:   # replicate padding wants floats; codes below 2^24 are exact in float32 (no float64 on MPS)
            self.t = F.pad(t.float(), (pad,) * 4, mode="replicate").to(t.dtype)

    def __call__(self, dx, dy):
        p = self.p
        return self.t[..., p + dy:p + dy + self.h, p + dx:p + dx + self.w]


def _interleave(blocks, s):
    """blocks[v][u]: (N,C,H,W) -> (N,C,H*s,W*s) with block (v,u) at sub-pixel (v,u)."""
    n, c, h, w = blocks[0][0].shape
    rows = torch.stack([torch.stack(r, dim=-1) for r in blocks], dim=-3)   # (N,C,H,s,W,s)
    return rows.reshape(n, c, h * s, w * s)


# ===================================================================== Scale2x / Scale3x
def scale2x(x):
    c = _codes(x)
    pk = _Nb(_pack(c).unsqueeze(1), 1)
    cn = _Nb(c, 1)
    B, D, E, Fq, H = pk(0, -1), pk(-1, 0), pk(0, 0), pk(1, 0), pk(0, 1)
    cB, cD, cE, cF, cH = cn(0, -1), cn(-1, 0), cn(0, 0), cn(1, 0), cn(0, 1)
    go = (B != H) & (D != Fq)

    def pick(cond, col):
        return torch.where(go & cond, col, cE)

    e0, e1 = pick(D == B, cD), pick(B == Fq, cF)
    e2, e3 = pick(D == H, cD), pick(H == Fq, cF)
    return _interleave([[e0, e1], [e2, e3]], 2).to(x.dtype) / 255.0


def scale3x(x):
    c = _codes(x)
    pk = _Nb(_pack(c).unsqueeze(1), 1)
    cn = _Nb(c, 1)
    A, B, C = pk(-1, -1), pk(0, -1), pk(1, -1)
    D, E, Fq = pk(-1, 0), pk(0, 0), pk(1, 0)
    G, H, I = pk(-1, 1), pk(0, 1), pk(1, 1)
    cB, cD, cE, cF, cH = cn(0, -1), cn(-1, 0), cn(0, 0), cn(1, 0), cn(0, 1)
    go = (B != H) & (D != Fq)

    def pick(cond, col):
        return torch.where(go & cond, col, cE)

    e0 = pick(D == B, cD)
    e1 = pick(((D == B) & (E != C)) | ((B == Fq) & (E != A)), cB)
    e2 = pick(B == Fq, cF)
    e3 = pick(((D == B) & (E != G)) | ((D == H) & (E != A)), cD)
    e5 = pick(((B == Fq) & (E != I)) | ((H == Fq) & (E != C)), cF)
    e6 = pick(D == H, cD)
    e7 = pick(((D == H) & (E != I)) | ((H == Fq) & (E != G)), cH)
    e8 = pick(H == Fq, cF)
    return _interleave([[e0, e1, e2], [e3, cE, e5], [e6, e7, e8]], 3).to(x.dtype) / 255.0


# ===================================================================== 2xSaI family
# ScummVM's interpolate32_* on 8-bit channels (exact integer results)
def _i11(a, b):
    return (a >> 1) + (b >> 1) + (a & b & 1)


def _i31(a, b):
    return (a >> 2) * 3 + (b >> 2) + ((((a & 3) * 3 + (b & 3)) >> 2) & 3)


def _i611(a, b, c):
    return (a >> 3) * 6 + (b >> 3) + (c >> 3) + ((((a & 7) * 6 + (b & 7) + (b & 7)) >> 3) & 7)


def _i1111(a, b, c, d):
    return (a >> 2) + (b >> 2) + (c >> 2) + (d >> 2) + ((((a & 3) + (b & 3) + (c & 3) + (d & 3)) >> 2) & 3)


def _get_result(A, B, C, D):
    ac, bc, ad, bd = (A == C), (B == C), (A == D), (B == D)
    x = ac.int() + ad.int()
    y = (bc & ~ac).int() + (bd & ~ad).int()
    return (y >> 1) - (x >> 1)


class _Grid:
    """Packed colours (for ==) and channels (for blends) at offsets."""

    def __init__(self, x):
        c = _codes(x)
        self.pk = _Nb(_pack(c), 2)
        self.ch = _Nb(c, 2)

    def __call__(self, dx, dy):
        return self.pk(dx, dy), self.ch(dx, dy)


def _sel(*pairs, default):
    """First matching (cond, value) wins; conditions are (N,H,W), values (N,3,H,W)."""
    out = default
    for cond, val in reversed(pairs):
        out = torch.where(cond.unsqueeze(1), val, out)
    return out


def sai2x(x):
    g = _Grid(x)
    (I, cI), (E, cE), (Fq, cF), (J, cJ) = g(-1, -1), g(0, -1), g(1, -1), g(2, -1)
    (G, cG), (A, cA), (B, cB), (K, cK) = g(-1, 0), g(0, 0), g(1, 0), g(2, 0)
    (H, cH), (C, cC), (D, cD), (L, cL) = g(-1, 1), g(0, 1), g(1, 1), g(2, 1)
    (M, cM), (N, cN), (O, cO) = g(-1, 2), g(0, 2), g(1, 2)
    ab, ac, a4 = _i11(cA, cB), _i11(cA, cC), _i1111(cA, cB, cC, cD)

    c1 = (A == D) & (B != C)
    c2 = (B == C) & (A != D)
    c3 = (A == D) & (B == C)
    same = A == B
    r = (_get_result(A, B, G, E) - _get_result(B, A, K, Fq) - _get_result(B, A, H, N) + _get_result(A, B, L, O))

    p_c1 = _sel((((A == E) & (B == L)) | ((A == C) & (A == Fq) & (B != E) & (B == J)), cA), default=ab)
    q_c1 = _sel((((A == G) & (C == O)) | ((A == B) & (A == H) & (G != C) & (C == M)), cA), default=ac)
    p_c2 = _sel((((B == Fq) & (A == H)) | ((B == E) & (B == D) & (A != Fq) & (A == I)), cB), default=ab)
    q_c2 = _sel((((C == H) & (A == Fq)) | ((C == G) & (C == D) & (A != H) & (A == I)), cC), default=ac)
    p_c4 = _sel(((A == C) & (A == Fq) & (B != E) & (B == J), cA), ((B == E) & (B == D) & (A != Fq) & (A == I), cB),
                default=ab)
    q_c4 = _sel(((A == B) & (A == H) & (G != C) & (C == M), cA), ((C == G) & (C == D) & (A != H) & (A == I), cC),
                default=ac)

    product = _sel((c1, p_c1), (c2, p_c2), (c3 & same, cA), (c3, ab), default=p_c4)
    product1 = _sel((c1, q_c1), (c2, q_c2), (c3 & same, cA), (c3, ac), default=q_c4)
    product2 = _sel((c1, cA), (c2, cB), (c3 & same, cA), (c3 & (r > 0), cA), (c3 & (r < 0), cB), default=a4)
    return _interleave([[cA, product], [product1, product2]], 2).to(x.dtype) / 255.0


def super2xsai(x):
    g = _Grid(x)
    (B0, _), (B1, _), (B2, _), (B3, _) = g(-1, -1), g(0, -1), g(1, -1), g(2, -1)
    (c4, k4), (c5, k5), (c6, k6), (S2, _) = g(-1, 0), g(0, 0), g(1, 0), g(2, 0)
    (c1, _), (c2, k2), (c3, k3), (S1, _) = g(-1, 1), g(0, 1), g(1, 1), g(2, 1)
    (A0, _), (A1, _), (A2, _), (A3, _) = g(-1, 2), g(0, 2), g(1, 2), g(2, 2)

    r = (_get_result(c6, c5, c1, A1) + _get_result(c6, c5, c4, B1)
         + _get_result(c6, c5, A2, S1) + _get_result(c6, c5, B2, S2))
    m56 = _i11(k5, k6)
    t1 = (c2 == c6) & (c5 != c3)
    t2 = (c5 == c3) & (c2 != c6)
    t3 = (c5 == c3) & (c2 == c6)
    pb_t3 = _sel((r > 0, k6), (r < 0, k5), default=m56)
    p2b_else = _sel(((c6 == c3) & (c3 == A1) & (c2 != A2) & (c3 != A0), _i31(k3, k2)),
                    ((c5 == c2) & (c2 == A2) & (A1 != c3) & (c2 != A3), _i31(k2, k3)), default=_i11(k2, k3))
    p1b_else = _sel(((c6 == c3) & (c6 == B1) & (c5 != B2) & (c6 != B0), _i31(k6, k5)),
                    ((c5 == c2) & (c5 == B2) & (B1 != c6) & (c5 != B3), _i31(k5, k6)), default=m56)
    product2b = _sel((t1, k2), (t2, k5), (t3, pb_t3), default=p2b_else)
    product1b = _sel((t1, k2), (t2, k5), (t3, pb_t3), default=p1b_else)
    m25 = _i11(k2, k5)
    product2a = _sel(((c5 == c3) & (c2 != c6) & (c4 == c5) & (c5 != A2), m25),
                     ((c5 == c1) & (c6 == c5) & (c4 != c2) & (c5 != A0), m25), default=k2)
    product1a = _sel(((c2 == c6) & (c5 != c3) & (c1 == c2) & (c2 != B2), m25),
                     ((c4 == c2) & (c3 == c2) & (c1 != c5) & (c2 != B0), m25), default=k5)
    return _interleave([[product1a, product1b], [product2a, product2b]], 2).to(x.dtype) / 255.0


def supereagle(x):
    g = _Grid(x)
    (B1, _), (B2, _) = g(0, -1), g(1, -1)
    (c4, k4), (c5, k5), (c6, k6), (S2, _) = g(-1, 0), g(0, 0), g(1, 0), g(2, 0)
    (c1, _), (c2, k2), (c3, k3), (S1, _) = g(-1, 1), g(0, 1), g(1, 1), g(2, 1)
    (A1, _), (A2, _) = g(0, 2), g(1, 2)

    m56, m23 = _i11(k5, k6), _i11(k2, k3)
    ne53, eq26 = c5 != c3, c2 == c6
    r = (_get_result(c6, c5, c1, A1) + _get_result(c6, c5, c4, B1)
         + _get_result(c6, c5, A2, S1) + _get_result(c6, c5, B2, S2))
    # branch 1: c5 != c3 and c2 == c6
    a1 = ne53 & eq26
    # branch 2: c5 != c3 and c2 != c6
    a2 = ne53 & ~eq26
    # branch 3: c5 == c3 and c2 != c6
    a3 = ~ne53 & ~eq26
    # branch 4: c5 == c3 and c2 == c6
    rp, rn = r > 0, r < 0

    product1a = _sel((a1 & ((c1 == c2) | (c6 == B2)), _i31(k2, k5)), (a1, m56),
                     (a2, _i611(k5, k2, k6)), (a3, k5), (rp, m56), default=k5)
    product2b = _sel((a1 & ((c6 == S2) | (c2 == A1)), _i31(k2, k3)), (a1, m23),
                     (a2, _i611(k3, k2, k6)), (a3, k5), (rp, m56), default=k5)
    product1b = _sel((a1, k2), (a2, _i611(k6, k5, k3)),
                     (a3 & ((B1 == c5) | (c3 == S1)), _i31(k5, k6)), (a3, m56),
                     (rp, k2), (rn, m56), default=k2)
    product2a = _sel((a1, k2), (a2, _i611(k2, k5, k3)),
                     (a3 & ((c3 == A2) | (c4 == c5)), _i31(k5, k2)), (a3, m23),
                     (rp, k2), (rn, m56), default=k2)
    return _interleave([[product1a, product1b], [product2a, product2b]], 2).to(x.dtype) / 255.0


# ===================================================================== hqx (native)
_HQX = None
_HQX_LOCK = __import__("threading").Lock()


def _hqx_lib():
    global _HQX
    with _HQX_LOCK:
        if _HQX is None:
            import ctypes
            from .native.build import library
            path = library("hqx", "hqx_shim.cpp", ["hqx/hqx.h", "hqx/common.h"],
                           c_files=["hqx/hq2x.c", "hqx/hq3x.c", "hqx/hq4x.c", "hqx/init.c"])
            lib = ctypes.CDLL(str(path))
            lib.hqx_scale.restype = ctypes.c_int
            lib.hqx_scale.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _HQX = lib
        return _HQX


def hqx(x, scale):
    import os
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    if scale not in (2, 3, 4):
        raise ValueError("hqx comes in x2, x3 and x4")
    lib = _hqx_lib()
    n, _, h, w = x.shape
    packed = _pack(_codes(x)).cpu().numpy().astype(np.uint32)            # (N,H,W)
    out = np.empty((n, h * scale, w * scale), dtype=np.uint32)

    def one(i):
        src = np.ascontiguousarray(packed[i])
        if lib.hqx_scale(src.ctypes.data, out[i].ctypes.data, w, h, scale):
            raise RuntimeError("hqx failed")
    with ThreadPoolExecutor(max_workers=max(1, min(n, os.cpu_count() or 4))) as ex:
        list(ex.map(one, range(n)))
    rgb = np.stack([(out >> 16) & 255, (out >> 8) & 255, out & 255], axis=1).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).to(x.device, x.dtype)


# ===================================================================== Super-xBR
_SX_Y = (0.2126, 0.7152, 0.0722)
SUPERXBR = {"edge_strength": 3.0, "weight0": 0.0, "weight": 1.0}


def _df(a, b):
    return (a - b).abs()


def _d_wd(wp, b0, b1, c0, c1, c2, d0, d1, d2, d3, e1, e2, e3, f2, f3):
    w1, w2, w3, w4, w5, w6 = wp
    return (w1 * (_df(c1, c2) + _df(c1, c0) + _df(e2, e1) + _df(e2, e3)) + w2 * (_df(d2, d3) + _df(d0, d1))
            + w3 * (_df(d1, d3) + _df(d0, d2)) + w4 * _df(d1, d2) + w5 * (_df(c0, c2) + _df(e1, e3))
            + w6 * (_df(b0, b1) + _df(f2, f3)))


def _hv_wd(wp, i1, i2, i3, i4, e1, e2, e3, e4):
    w1, w2, w3, w4, w5, w6 = wp
    return (w4 * (_df(i1, i2) + _df(i3, i4)) + w1 * (_df(i1, e1) + _df(i2, e2) + _df(i3, e3) + _df(i4, e4))
            + w3 * (_df(i1, e2) + _df(i3, e4) + _df(e1, i2) + _df(e3, i4)))


def _sx_core(s, wp, weight, strength):
    """The shared body of the three passes. s: dict of name -> (rgb, luma)."""
    w1 = (-weight * 1.29633 / 10.0, weight * 1.29633 / 10.0 + 0.5)
    w2 = (-weight * 1.75068 / 10.0 / 2.0, weight * 1.75068 / 10.0 / 2.0 + 0.25)
    L = {k: v[1] for k, v in s.items()}
    C = {k: v[0] for k, v in s.items()}
    d_edge = (_d_wd(wp, L["D"], L["B"], L["G"], L["E"], L["C"], L["P2"], L["H"], L["F"], L["P1"], L["H5"], L["I"],
                    L["F4"], L["I5"], L["I4"])
              - _d_wd(wp, L["C"], L["F4"], L["B"], L["F"], L["I4"], L["P0"], L["E"], L["I"], L["P3"], L["D"], L["H"],
                      L["I5"], L["G"], L["H5"]))
    hv_edge = (_hv_wd(wp, L["F"], L["I"], L["E"], L["H"], L["C"], L["I5"], L["B"], L["H5"])
               - _hv_wd(wp, L["E"], L["F"], L["H"], L["I"], L["D"], L["F4"], L["G"], L["I4"]))
    t = (d_edge.abs() / (strength + 0.000001)).clamp(0, 1)
    edge_strength = t * t * (3 - 2 * t)

    def taps(w, a, b, c, d):
        return a * w[0] + b * w[1] + c * w[1] + d * w[0]

    c3 = taps(w2, C["D"] + C["G"], C["E"] + C["H"], C["F"] + C["I"], C["F4"] + C["I4"])
    c4 = taps(w2, C["C"] + C["B"], C["F"] + C["E"], C["I"] + C["H"], C["I5"] + C["H5"])
    c1 = taps(w1, C["P2"], C["H"], C["F"], C["P1"])
    c2 = taps(w1, C["P0"], C["E"], C["I"], C["P3"])
    a = torch.where(d_edge >= 0, c2, c1)
    b = torch.where(hv_edge >= 0, c4, c3)
    color = a + (b - a) * (1.0 - edge_strength)
    lo = torch.minimum(torch.minimum(C["E"], C["F"]), torch.minimum(C["H"], C["I"]))
    hi = torch.maximum(torch.maximum(C["E"], C["F"]), torch.maximum(C["H"], C["I"]))
    return torch.minimum(torch.maximum(color, lo), hi)


def _luma(rgb):
    return _SX_Y[0] * rgb[:, 0:1] + _SX_Y[1] * rgb[:, 1:2] + _SX_Y[2] * rgb[:, 2:3]


# 4x4 neighbourhood of passes 0 and 2, as (dx, dy) from the top-left of the centre 2x2 (E)
_SX_TAPS = {"P0": (-1, -1), "B": (0, -1), "C": (1, -1), "P1": (2, -1),
            "D": (-1, 0), "E": (0, 0), "F": (1, 0), "F4": (2, 0),
            "G": (-1, 1), "H": (0, 1), "I": (1, 1), "I4": (2, 1),
            "P2": (-1, 2), "H5": (0, 2), "I5": (1, 2), "P3": (2, 2)}
# pass 1: (source, k along g1, m along g2)
_SX_P1 = {"P0": ("o", -3, 0), "P1": ("d", 0, -3), "P2": ("d", 0, 3), "P3": ("o", 3, 0),
          "B": ("d", -2, -1), "C": ("o", -1, -2), "D": ("d", -2, 1), "E": ("o", -1, 0),
          "F": ("d", 0, -1), "G": ("o", -1, 2), "H": ("d", 0, 1), "I": ("o", 1, 0),
          "F4": ("o", 1, -2), "I4": ("d", 2, -1), "H5": ("o", 1, 2), "I5": ("d", 2, 1)}


def _floor2(v):
    return v // 2   # Python floor division: also right for negatives


@torch.no_grad()
def superxbr(x, edge_strength=None):
    p = SUPERXBR
    es = p["edge_strength"] if edge_strength is None else float(edge_strength)
    n, c, h, w = x.shape
    orig = x
    orig_l = _luma(orig)
    rgbl = torch.cat([orig, orig_l], 1)

    # pass 0: the centre of every 2x2 block, at source resolution
    nb = _Nb(rgbl, 3)
    s = {k: (nb(dx, dy)[:, :3], nb(dx, dy)[:, 3:]) for k, (dx, dy) in _SX_TAPS.items()}
    diag = _sx_core(s, (2.0, 1.0, -1.0, 4.0, -1.0, 1.0), p["weight0"], es)
    diag_l = _luma(diag)

    # pass 1: the points between, on the 2x grid (originals at even/even, diagonals at odd/odd)
    on = _Nb(rgbl, 3)
    dn = _Nb(torch.cat([diag, diag_l], 1), 3)

    def mixed(case):
        out = {}
        for k, (src, kk, mm) in _SX_P1.items():
            if case == "A":            # X odd, Y even: g1 = +x, g2 = +y
                dx, dy = _floor2(1 + kk), _floor2(mm)
            else:                      # X even, Y odd: g1 = +y, g2 = +x
                dx, dy = _floor2(mm), _floor2(1 + kk)
            t = (on if src == "o" else dn)(dx, dy)
            out[k] = (t[:, :3], t[:, 3:])
        return _sx_core(out, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), p["weight"], es)

    a = mixed("A")
    b = mixed("B")
    big = _interleave([[orig, a], [b, diag]], 2)

    # pass 2: the 4x4 rule again on the 2x grid, sampled half a pixel back (re-centres it)
    nb2 = _Nb(torch.cat([big, _luma(big)], 1), 3)
    s2 = {k: (nb2(dx - 1, dy - 1)[:, :3], nb2(dx - 1, dy - 1)[:, 3:]) for k, (dx, dy) in _SX_TAPS.items()}
    return _sx_core(s2, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0), p["weight"], es)


# ===================================================================== ScaleFX
class _V:
    """A vec4 per pixel: (N,4,H,W), with GLSL-style component access."""

    def __init__(self, t):
        self.t = t

    def __getattr__(self, name):
        if name in ("x", "y", "z", "w"):
            return self.t[:, "xyzw".index(name)]
        if set(name) <= set("xyzw"):
            return torch.stack([self.t[:, "xyzw".index(c)] for c in name], 1)
        raise AttributeError(name)


def _v4(*comps):
    return torch.stack(comps, 1)


def _GE(a, b):   # 1 - step(b, a)... GE(x, y) = x > y
    return (a > b).to(torch.float32)


def _LE(a, b):
    return (a < b).to(torch.float32)


def _LEQ(a, b):
    return (a <= b).to(torch.float32)


def _NOT(a):
    return 1 - a


def _sfx_dist(A, B):
    r = 0.5 * (A[:, 0] + B[:, 0])
    d = A - B
    return torch.sqrt((2 + r) * d[:, 0] ** 2 + 4 * d[:, 1] ** 2 + (3 - r) * d[:, 2] ** 2) / 3


SCALEFX = {"threshold": 0.5}


def _comp(t, i):
    return t[:, i]


@torch.no_grad()
def scalefx(x, threshold=None):
    clr = SCALEFX["threshold"] if threshold is None else float(threshold)
    ref = x
    q = _codes(x).to(torch.float32) / 255.0                 # the passes read 8-bit textures
    nb = _Nb(q, 1)
    E = nb(0, 0)
    # ---- pass 0: metric x,y,z,w = dist(E, A), dist(E, B), dist(E, C), dist(E, F)
    M = _v4(_sfx_dist(E, nb(-1, -1)), _sfx_dist(E, nb(0, -1)), _sfx_dist(E, nb(1, -1)), _sfx_dist(E, nb(1, 0)))

    # ---- pass 1: corner strength
    m = _Nb(M, 1)
    A, B = _V(m(-1, -1)), _V(m(0, -1))
    D, E1, F_ = _V(m(-1, 0)), _V(m(0, 0)), _V(m(1, 0))
    G, H, I = _V(m(-1, 1)), _V(m(0, 1)), _V(m(1, 1))

    def strength(d, ax, ay, bx, by):
        diff = ax - ay
        w1 = (clr - d).clamp(min=0) / clr
        w2 = ((1 - d) + torch.where(torch.minimum(ax, bx) + ax > torch.minimum(ay, by) + ay, diff, -diff)).clamp(0, 1)
        return (w1 * w2) * (ax * ay)                      # filter AA on: always this branch
    S = _v4(strength(D.z, D.w, E1.y, A.w, D.y), strength(F_.x, E1.w, E1.y, B.w, F_.y),
            strength(H.z, E1.w, H.y, H.w, I.y), strength(H.x, D.w, H.y, G.w, G.y))

    # ---- pass 2: junctions, edges, orientation
    sn = _Nb(S, 1)
    As, Bs, Cs = _V(sn(-1, -1)), _V(sn(0, -1)), _V(sn(1, -1))
    Ds, Es, Fs = _V(sn(-1, 0)), _V(sn(0, 0)), _V(sn(1, 0))
    Gs, Hs, Is = _V(sn(-1, 1)), _V(sn(0, 1)), _V(sn(1, 1))

    def dom(x3, y3, z3, w3):
        return 2 * _v4(x3[:, 1], y3[:, 1], z3[:, 1], w3[:, 1]) - (_v4(x3[:, 0], y3[:, 0], z3[:, 0], w3[:, 0])
                                                                  + _v4(x3[:, 2], y3[:, 2], z3[:, 2], w3[:, 2]))
    jSx, jDx = _v4(As.z, Bs.w, Es.x, Ds.y), dom(As.yzw, Bs.zwx, Es.wxy, Ds.xyz)
    jSy, jDy = _v4(Bs.z, Cs.w, Fs.x, Es.y), dom(Bs.yzw, Cs.zwx, Fs.wxy, Es.xyz)
    jSz, jDz = _v4(Es.z, Fs.w, Is.x, Hs.y), dom(Es.yzw, Fs.zwx, Is.wxy, Hs.xyz)
    jSw, jDw = _v4(Ds.z, Es.w, Hs.x, Gs.y), dom(Ds.yzw, Es.zwx, Hs.wxy, Gs.xyz)

    def vote(jD):
        v = _V(jD)
        return torch.clamp(_GE(jD, 0) * (_LEQ(v.yzwx, 0) * _LEQ(v.wxyz, 0) + _GE(jD + v.zwxy, v.yzwx + v.wxyz)),
                           max=1)
    jx, jy, jz, jw = _V(vote(jDx)), _V(vote(jDy)), _V(vote(jDz)), _V(vote(jDw))
    sx, sy, sz, sw = _V(jSx), _V(jSy), _V(jSz), _V(jSw)
    res = _v4(
        torch.clamp(jx.z + _NOT(jx.y) * _NOT(jx.w) * _GE(sx.z, 0) * (jx.x + _GE(sx.x + sx.z, sx.y + sx.w)), max=1),
        torch.clamp(jy.w + _NOT(jy.z) * _NOT(jy.x) * _GE(sy.w, 0) * (jy.y + _GE(sy.y + sy.w, sy.x + sy.z)), max=1),
        torch.clamp(jz.x + _NOT(jz.w) * _NOT(jz.y) * _GE(sz.x, 0) * (jz.z + _GE(sz.x + sz.z, sz.y + sz.w)), max=1),
        torch.clamp(jw.y + _NOT(jw.x) * _NOT(jw.z) * _GE(sw.y, 0) * (jw.w + _GE(sw.y + sw.w, sw.x + sw.z)), max=1))
    rv = _V(res)
    res = torch.clamp(res * (_v4(jx.z, jy.w, jz.x, jw.y) + _NOT(rv.wxyz * rv.yzwx)), max=1)

    def clear(cx, cy, ax, ay, bx, by):
        return ((cx >= torch.maximum(torch.minimum(ax, ay), torch.minimum(bx, by)))
                & (cy >= torch.maximum(torch.minimum(ax, by), torch.minimum(bx, ay)))).to(torch.float32)
    E = E1
    clr4 = _v4(clear(D.z, E.x, D.w, E.y, A.w, D.y), clear(F_.x, E.z, E.w, E.y, B.w, F_.y),
               clear(H.z, I.x, E.w, H.y, H.w, I.y), clear(H.x, G.z, D.w, H.y, G.w, G.y))
    h = _v4(torch.minimum(D.w, A.w), torch.minimum(E.w, B.w), torch.minimum(E.w, H.w), torch.minimum(D.w, G.w))
    v = _v4(torch.minimum(E.y, D.y), torch.minimum(E.y, F_.y), torch.minimum(H.y, I.y), torch.minimum(H.y, G.y))
    orien = _GE(h + _v4(D.w, E.w, E.w, D.w), v + _v4(E.y, E.y, H.y, H.y))
    hori = _LE(h, v) * clr4
    vert = _GE(h, v) * clr4
    flags = torch.cat([res, hori, vert, orien], 1) > 0.5           # (N,16,H,W): c, h, v, o

    # ---- pass 3: pick the subpixel sources
    fl = _Nb(flags, 3)

    class _P:
        def __init__(self, dx, dy):
            t = fl(dx, dy)
            self.c, self.h, self.v, self.o = _V(t[:, 0:4]), _V(t[:, 4:8]), _V(t[:, 8:12]), _V(t[:, 12:16])
    Ep = _P(0, 0)
    Dp, D0, D1 = _P(-1, 0), _P(-2, 0), _P(-3, 0)
    Fp, F0, F1 = _P(1, 0), _P(2, 0), _P(3, 0)
    Bp, B0, B1 = _P(0, -1), _P(0, -2), _P(0, -3)
    Hp, H0, H1 = _P(0, 1), _P(0, 2), _P(0, 3)
    Ec, Eh, Ev, Eo = Ep.c, Ep.h, Ep.v, Ep.o
    Dc, Dh, Do = Dp.c, Dp.h, Dp.o
    D0c, D0h, D1h = D0.c, D0.h, D1.h
    Fc, Fh, Fo = Fp.c, Fp.h, Fp.o
    F0c, F0h, F1h = F0.c, F0.h, F1.h
    Bc, Bv, Bo = Bp.c, Bp.v, Bp.o
    B0c, B0v, B1v = B0.c, B0.v, B1.v
    Hc, Hv, Ho = Hp.c, Hp.v, Hp.o
    H0c, H0v, H1v = H0.c, H0.v, H1.v

    lvl1x, lvl1y, lvl1z, lvl1w = Ec.x, Ec.y, Ec.z, Ec.w            # filter corners on
    lvl2x = ((Ec.x & Eh.y) & Dc.z, (Ec.y & Eh.x) & Fc.w)
    lvl2y = ((Ec.y & Ev.z) & Bc.w, (Ec.z & Ev.y) & Hc.x)
    lvl2z = ((Ec.w & Eh.z) & Dc.y, (Ec.z & Eh.w) & Fc.x)
    lvl2w = ((Ec.x & Ev.w) & Bc.z, (Ec.w & Ev.x) & Hc.y)
    lvl3x = (lvl2x[1] & (Dh.y & Dh.x) & Fh.z, lvl2w[1] & (Bv.w & Bv.x) & Hv.z)
    lvl3y = (lvl2x[0] & (Fh.x & Fh.y) & Dh.w, lvl2y[1] & (Bv.z & Bv.y) & Hv.w)
    lvl3z = (lvl2z[0] & (Fh.w & Fh.z) & Dh.x, lvl2y[0] & (Hv.y & Hv.z) & Bv.x)
    lvl3w = (lvl2z[1] & (Dh.z & Dh.w) & Fh.y, lvl2w[0] & (Hv.x & Hv.w) & Bv.y)
    lvl4x = ((Dc.x & Dh.y & Eh.x & Eh.y & Fh.x & Fh.y) & (D0c.z & D0h.w),
             (Bc.x & Bv.w & Ev.x & Ev.w & Hv.x & Hv.w) & (B0c.z & B0v.y))
    lvl4y = ((Fc.y & Fh.x & Eh.y & Eh.x & Dh.y & Dh.x) & (F0c.w & F0h.z),
             (Bc.y & Bv.z & Ev.y & Ev.z & Hv.y & Hv.z) & (B0c.w & B0v.x))
    lvl4z = ((Fc.z & Fh.w & Eh.z & Eh.w & Dh.z & Dh.w) & (F0c.x & F0h.y),
             (Hc.z & Hv.y & Ev.z & Ev.y & Bv.z & Bv.y) & (H0c.x & H0v.w))
    lvl4w = ((Dc.w & Dh.z & Eh.w & Eh.z & Fh.w & Fh.z) & (D0c.y & D0h.x),
             (Hc.w & Hv.x & Ev.w & Ev.x & Bv.w & Bv.x) & (H0c.y & H0v.z))
    lvl5x = (lvl4x[0] & (F0h.x & F0h.y) & (D1h.z & D1h.w), lvl4y[0] & (D0h.y & D0h.x) & (F1h.w & F1h.z))
    lvl5y = (lvl4y[1] & (H0v.y & H0v.z) & (B1v.w & B1v.x), lvl4z[1] & (B0v.z & B0v.y) & (H1v.x & H1v.w))
    lvl5z = (lvl4w[0] & (F0h.w & F0h.z) & (D1h.y & D1h.x), lvl4z[0] & (D0h.z & D0h.w) & (F1h.x & F1h.y))
    lvl5w = (lvl4x[1] & (H0v.x & H0v.w) & (B1v.z & B1v.y), lvl4w[1] & (B0v.w & B0v.x) & (H1v.y & H1v.z))
    lvl6x = (lvl5x[1] & (D1h.y & D1h.x), lvl5w[1] & (B1v.w & B1v.x))
    lvl6y = (lvl5x[0] & (F1h.x & F1h.y), lvl5y[1] & (B1v.z & B1v.y))
    lvl6z = (lvl5z[0] & (F1h.w & F1h.z), lvl5y[0] & (H1v.y & H1v.z))
    lvl6w = (lvl5z[1] & (D1h.z & D1h.w), lvl5w[0] & (H1v.x & H1v.w))

    def chain(pairs, default=0):
        out = torch.full_like(Ec.x, default, dtype=torch.int64)
        for cond, val in reversed(pairs):
            out = torch.where(cond, torch.as_tensor(val) if not torch.is_tensor(val) else val, out)
        return out

    def ifelse(c, a, b):
        return torch.where(c, a, b)
    n_ = torch.zeros_like(Ec.x, dtype=torch.int64)

    def k(v):
        return n_ + v

    crn_x = chain([((lvl1x & Eo.x) | (lvl3x[0] & Eo.y) | (lvl4x[0] & Do.x) | (lvl6x[0] & Fo.y), 5),
                   (lvl1x | (lvl3x[1] & ~Eo.w) | (lvl4x[1] & ~Bo.x) | (lvl6x[1] & ~Ho.w), 1),
                   (lvl3x[0], 3), (lvl3x[1], 7), (lvl4x[0], 2), (lvl4x[1], 6), (lvl6x[0], 4), (lvl6x[1], 8)])
    crn_y = chain([((lvl1y & Eo.y) | (lvl3y[0] & Eo.x) | (lvl4y[0] & Fo.y) | (lvl6y[0] & Do.x), 5),
                   (lvl1y | (lvl3y[1] & ~Eo.z) | (lvl4y[1] & ~Bo.y) | (lvl6y[1] & ~Ho.z), 3),
                   (lvl3y[0], 1), (lvl3y[1], 7), (lvl4y[0], 4), (lvl4y[1], 6), (lvl6y[0], 2), (lvl6y[1], 8)])
    crn_z = chain([((lvl1z & Eo.z) | (lvl3z[0] & Eo.w) | (lvl4z[0] & Fo.z) | (lvl6z[0] & Do.w), 7),
                   (lvl1z | (lvl3z[1] & ~Eo.y) | (lvl4z[1] & ~Ho.z) | (lvl6z[1] & ~Bo.y), 3),
                   (lvl3z[0], 1), (lvl3z[1], 5), (lvl4z[0], 4), (lvl4z[1], 8), (lvl6z[0], 2), (lvl6z[1], 6)])
    crn_w = chain([((lvl1w & Eo.w) | (lvl3w[0] & Eo.z) | (lvl4w[0] & Do.w) | (lvl6w[0] & Fo.z), 7),
                   (lvl1w | (lvl3w[1] & ~Eo.x) | (lvl4w[1] & ~Ho.w) | (lvl6w[1] & ~Bo.x), 1),
                   (lvl3w[0], 3), (lvl3w[1], 5), (lvl4w[0], 2), (lvl4w[1], 8), (lvl6w[0], 4), (lvl6w[1], 6)])
    mid_x = chain([((lvl2x[0] & Eo.x) | (lvl2x[1] & Eo.y) | (lvl5x[0] & Do.x) | (lvl5x[1] & Fo.y), 5),
                   (lvl2x[0], 1), (lvl2x[1], 3), (lvl5x[0], 2), (lvl5x[1], 4),
                   (Ec.x & Dc.z & Ec.y & Fc.w, ifelse(Eo.x, ifelse(Eo.y, k(5), k(3)), k(1)))])
    mid_y = chain([((lvl2y[0] & ~Eo.y) | (lvl2y[1] & ~Eo.z) | (lvl5y[0] & ~Bo.y) | (lvl5y[1] & ~Ho.z), 3),
                   (lvl2y[0], 5), (lvl2y[1], 7), (lvl5y[0], 6), (lvl5y[1], 8),
                   (Ec.y & Bc.w & Ec.z & Hc.x, ifelse(~Eo.y, ifelse(~Eo.z, k(3), k(7)), k(5)))])
    mid_z = chain([((lvl2z[0] & Eo.w) | (lvl2z[1] & Eo.z) | (lvl5z[0] & Do.w) | (lvl5z[1] & Fo.z), 7),
                   (lvl2z[0], 1), (lvl2z[1], 3), (lvl5z[0], 2), (lvl5z[1], 4),
                   (Ec.z & Fc.x & Ec.w & Dc.y, ifelse(Eo.z, ifelse(Eo.w, k(7), k(1)), k(3)))])
    mid_w = chain([((lvl2w[0] & ~Eo.x) | (lvl2w[1] & ~Eo.w) | (lvl5w[0] & ~Bo.x) | (lvl5w[1] & ~Ho.w), 1),
                   (lvl2w[0], 5), (lvl2w[1], 7), (lvl5w[0], 6), (lvl5w[1], 8),
                   (Ec.w & Hc.y & Ec.x & Bc.z, ifelse(~Eo.w, ifelse(~Eo.x, k(1), k(5)), k(7)))])

    # ---- pass 4: 3x3 subpixels, each a copy of a nearby original pixel
    layout = [[crn_x, mid_x, crn_y], [mid_w, n_, mid_y], [crn_w, mid_z, crn_z]]
    offs = [(0, 0), (-1, 0), (-2, 0), (1, 0), (2, 0), (0, -1), (0, -2), (0, 1), (0, 2)]
    rn = _Nb(ref, 2)
    cands = torch.stack([rn(dx, dy) for dx, dy in offs], 0)        # (9,N,C,H,W)
    blocks = []
    for row in layout:
        out_row = []
        for sp in row:
            idx = sp.unsqueeze(0).unsqueeze(2).expand(1, -1, ref.shape[1], -1, -1)
            out_row.append(torch.gather(cands, 0, idx)[0])
        blocks.append(out_row)
    return _interleave(blocks, 3)


# ===================================================================== dispatch
# how far (in new pixels, x then y) each x2 moves the picture up/left
OFFSET = {"2xsai": (0.5, 0.5), "super2xsai": (0.5, 1.0), "supereagle": (1.0, 1.0)}


def _shift_back(x, ox, oy):
    """Content at pixel o belongs at o + offset: move it there."""
    from .edi import _lanczos_taps
    for axis, off in ((3, ox), (2, oy)):
        whole = int(off)
        if whole:
            n = x.shape[axis]
            idx = (torch.arange(n, device=x.device) - whole).clamp(0, n - 1)
            x = x.index_select(axis, idx)
        if off - whole:
            offs, w = _lanczos_taps(-(off - whole))
            k = torch.tensor(w, dtype=x.dtype, device=x.device)
            c = x.shape[1]
            if axis == 3:
                y = F.pad(x, (-offs[0], offs[-1], 0, 0), mode="replicate")
                x = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
            else:
                y = F.pad(x, (0, 0, -offs[0], offs[-1]), mode="replicate")
                x = F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


_TWO = {"scale2x": scale2x, "2xsai": sai2x, "super2xsai": super2xsai, "supereagle": supereagle,
        "superxbr": superxbr}


def upscale(x, method, scale, center=True, check=None):
    if scale not in SCALES[method]:
        raise ValueError(f"{method} comes in x{', x'.join(map(str, SCALES[method]))}")
    if method == "hqx":
        return hqx(x, scale)
    f, step = ((scale3x, 3) if method == "scale3x" else (scalefx, 3) if method == "scalefx"
               else (_TWO[method], 2))
    while scale > 1:
        if check:
            check()
        x = f(x)
        if center and method in OFFSET:
            x = _shift_back(x, *OFFSET[method])
        x = x.clamp(0, 1)
        scale //= step
    return x
