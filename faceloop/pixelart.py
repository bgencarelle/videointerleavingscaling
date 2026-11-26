"""xBR pixel-art upscaling: Hyllian's xBR-lv2 shader, ported to PyTorch.

Source: libretro/slang-shaders, edge-smoothing/xbr/shaders/xbr-lv2-standalone.slang
(commit 02bf134555c4595bd6b69be89feaff923da83754), with its default
parameters: colour distinction threshold 0.32, smoothness 0.30, AA blending
on, corner detection C.

The shader runs once per output pixel and works out its position inside the
source pixel (`fp`). For an exact x2/x3/x4 scale that position takes only
s*s values, so here the edge rules are computed once per source pixel for the
whole frame, and the blending once per sub-pixel position.

Like the shader, it works on the stored (encoded) values, and texture reads
past the edge clamp to the border.

Original licence (MIT):

   Hyllian's xBR-lv2 Shader
   Copyright (C) 2011-2022 Hyllian - sergiogdb@gmail.com

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to permit
   persons to whom the Software is furnished to do so, subject to the
   following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.

   Incorporates some of the ideas from SABR shader. Thanks to Joshua Street.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EQ_THRESHOLD = 0.32
LV2_COEFFICIENT = 0.30
LV2_CF = LV2_COEFFICIENT + 2.0

_Y = (0.2627, 0.6780, 0.0593)
_Ao = (1.0, -1.0, -1.0, 1.0)
_Bo = (1.0, 1.0, -1.0, -1.0)
_Co = (1.5, 0.5, -0.5, 0.5)
_Ax = (1.0, -1.0, -1.0, 1.0)
_Bx = (0.5, 2.0, -0.5, -2.0)
_Cx = (1.0, 1.0, -0.5, 0.0)
_Ay = (1.0, -1.0, -1.0, 1.0)
_By = (2.0, 0.5, -2.0, -0.5)
_Cy = (2.0, 0.0, -1.0, 0.5)
_Ci = (0.25, 0.25, 0.25, 0.25)

SCALES = (2, 3, 4)


def _vec(v, ref):
    return torch.tensor(v, dtype=ref.dtype, device=ref.device).view(1, 4, 1, 1)


def _swz(t, order):
    """GLSL swizzle on the direction axis: 'yzwx' -> indices [1,2,3,0]."""
    idx = ["xyzw".index(ch) for ch in order]
    return t[:, idx]


@torch.no_grad()
def xbr(x: torch.Tensor, scale: int) -> torch.Tensor:
    """x: (N, 3, H, W) in 0..1 -> (N, 3, H*scale, W*scale)."""
    if scale not in SCALES:
        raise ValueError(f"xBR scale must be one of {SCALES}")
    n, c, h, w = x.shape
    p = F.pad(x, (2, 2, 2, 2), mode="replicate")
    Yw = torch.tensor(_Y, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)

    def at(dy, dx):
        return p[..., 2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    A1, B1, C1 = at(-2, -1), at(-2, 0), at(-2, 1)
    A, B, C = at(-1, -1), at(-1, 0), at(-1, 1)
    D, E, Fp = at(0, -1), at(0, 0), at(0, 1)
    G, H, I = at(1, -1), at(1, 0), at(1, 1)
    G5, H5, I5 = at(2, -1), at(2, 0), at(2, 1)
    A0, D0, G0 = at(-1, -2), at(0, -2), at(1, -2)
    C4, F4, I4 = at(-1, 2), at(0, 2), at(1, 2)

    def m(*cols):  # mat4x3 -> (N, 4, 3, H, W)
        return torch.stack(cols, dim=1)

    b, c_, d = m(B, D, H, Fp), m(C, A, G, I), m(D, H, Fp, B)
    e = m(E, E, E, E)
    f, g, hh, i = m(Fp, B, D, H), m(G, I, C, A), m(H, Fp, B, D), m(I, C, A, G)
    i4, i5, h5, f4 = m(I4, C1, A0, G5), m(I5, C4, A1, G0), m(H5, F4, B1, D0), m(F4, B1, D0, H5)

    def code(t):  # exact colour identity (the shader's v2f encoding), on 8-bit values
        q = torch.round(t.clamp(0, 1) * 255).to(torch.int32)
        return q[:, :, 0] * 65536 + q[:, :, 1] * 256 + q[:, :, 2]

    b_, cc_, e_ = code(b), code(c_), code(e)
    d_, f_, h_ = _swz(b_, "yzwx"), _swz(b_, "wxyz"), _swz(b_, "zwxy")
    g_ = _swz(cc_, "zwxy")

    def dist4(a, bb):
        return ((a - bb).abs() * Yw.unsqueeze(1)).sum(2)

    def eq(a, bb):
        return (dist4(a, bb) <= EQ_THRESHOLD).to(x.dtype)

    def neq(a, bb):
        return 1.0 - eq(a, bb)

    def diff(a, bb):
        return (a != bb).to(x.dtype)

    def wdist(a, bb, cc, dd, ee, ff, gg, hh_):
        return dist4(a, bb) + dist4(a, cc) + dist4(dd, ee) + dist4(dd, ff) + 4.0 * dist4(gg, hh_)

    irlv0 = diff(e_, f_) * diff(e_, h_)
    irlv1 = (irlv0 * (neq(f, b) * neq(f, c_) + neq(hh, d) * neq(hh, g)
                      + eq(e, i) * (neq(f, f4) * neq(f, i4) + neq(hh, h5) * neq(hh, i5))
                      + eq(e, g) + eq(e, c_))).clamp(0, 1)
    irlv2l = diff(e_, g_) * diff(d_, g_)
    irlv2u = diff(e_, cc_) * diff(b_, cc_)

    wd1 = wdist(e, c_, g, i, h5, f4, hh, f)
    wd2 = wdist(hh, d, i5, f, i4, b, e, i)
    d_fg = dist4(f, g)
    d_hc = dist4(hh, c_)

    edri = (wd2 >= wd1).to(x.dtype) * irlv0
    edr = (wd1 < wd2).to(x.dtype) * irlv1 * (1.0 - _swz(edri, "yzwx") * _swz(edri, "wxyz"))
    edr_l = (d_hc >= LV2_CF * d_fg).to(x.dtype) * irlv2l * edr * ((1.0 - _swz(edri, "yzwx")) * eq(e, c_))
    edr_u = (d_fg >= LV2_CF * d_hc).to(x.dtype) * irlv2u * edr * ((1.0 - _swz(edri, "wxyz")) * eq(e, g))

    px = (dist4(e, hh) >= dist4(e, f)).to(x.dtype)  # (N,4,H,W)

    aa = 2.0 / scale
    delta = aa
    deltaL = _vec((0.5, 1.0, 0.5, 1.0), x) * aa
    deltaU = _vec((1.0, 0.5, 1.0, 0.5), x) * aa
    Ao, Bo, Co, Ci = (_vec(v, x) for v in (_Ao, _Bo, _Co, _Ci))
    Ax, Bx, Cx = (_vec(v, x) for v in (_Ax, _Bx, _Cx))
    Ay, By, Cy = (_vec(v, x) for v in (_Ay, _By, _Cy))

    def dist(a, bb):  # (N,3,H,W) -> (N,1,H,W)
        return ((a - bb).abs() * Yw).sum(1, keepdim=True)

    def mix(a, bb, t):
        return a + (bb - a) * t

    out = torch.empty(n, c, h * scale, w * scale, dtype=x.dtype, device=x.device)
    for v in range(scale):
        fpy = (v + 0.5) / scale
        for u in range(scale):
            fpx = (u + 0.5) / scale
            fx = Ao * fpy + Bo * fpx
            fx_l = Ax * fpy + Bx * fpx
            fx_u = Ay * fpy + By * fpx
            fx45i = (0.5 + (fx - Co - Ci) / delta).clamp(0, 1) * edri
            fx45 = (0.5 + (fx - Co) / delta).clamp(0, 1) * edr
            fx30 = (0.5 + (fx_l - Cx) / deltaL).clamp(0, 1) * edr_l
            fx60 = (0.5 + (fx_u - Cy) / deltaU).clamp(0, 1) * edr_u
            mx = torch.maximum(torch.maximum(fx30, fx60), torch.maximum(fx45, fx45i))

            def pick(k):
                return mx[:, k:k + 1], px[:, k:k + 1]

            m0, p0 = pick(0)
            m1, p1 = pick(1)
            m2, p2 = pick(2)
            m3, p3 = pick(3)
            r1 = mix(E, mix(H, Fp, p0), m0)
            r2 = mix(E, mix(B, D, p2), m2)
            res1a = mix(r1, r2, (dist(E, r2) >= dist(E, r1)).to(x.dtype))
            r1 = mix(E, mix(Fp, B, p1), m1)
            r2 = mix(E, mix(D, H, p3), m3)
            res1b = mix(r1, r2, (dist(E, r2) >= dist(E, r1)).to(x.dtype))
            res = mix(res1a, res1b, (dist(E, res1b) >= dist(E, res1a)).to(x.dtype))
            out[..., v::scale, u::scale] = res
    return out
