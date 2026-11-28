"""EEDI3, EEDI2 and NNEDI3 as 2x upscalers.

Port of the VapourSynth EEDI3 filter (Kevin Stone 2010, Laurent de Soras
2013, VapourSynth port by HolyWu), which is licensed under the GNU GPL
version 2 or later, and so is this file:

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

What it does: each new line between two original lines is found as the
cheapest non-crossing warping between the lines above and below (dynamic
programming over a connection cost), then interpolated along that path;
"vcheck" blends back towards plain cubic where the path looks unreliable.
Doubling a frame runs it on the rows (dh=True, field=1: originals on the
even rows) and then on the columns, as eedi3_rpow2 scripts do. Every plane
(here R, G and B) on its own, in 0..255 floats, like the filter's float
path. Defaults are the filter's: alpha 0.2, beta 0.25, gamma 20, nrad 2,
mdis 20, cost3, ucubic, vcheck 2 (vthresh 32 / 64 / 4), no sclip or mclip.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BIG = 3.0e38


def _reflect_index(i: torch.Tensor, n: int) -> torch.Tensor:
    """Mirror without repeating the edge sample (as the filter's margins)."""
    if n == 1:
        return torch.zeros_like(i)
    period = 2 * (n - 1)
    i = i.remainder(period)
    return torch.where(i >= n, period - i, i)


def _rows(S: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Rows of S (B,h,w) at indices k, with the filter's vertical padding:
    above the top mirrored without the edge (-1 -> 1), below the bottom
    mirrored with it (h -> h-1, h+1 -> h-2)."""
    h = S.shape[1]
    k = torch.where(k < 0, (-k).clamp(max=h - 1), k)
    k = torch.where(k >= h, (2 * h - 1 - k).clamp(min=0), k)
    return S[:, k]


def eedi3_rows(S: torch.Tensor, alpha=0.2, beta=0.25, gamma=20.0, nrad=2, mdis=20, cost3=True, ucubic=True,
               vcheck=2, vthresh0=32.0, vthresh1=64.0, vthresh2=4.0) -> torch.Tensor:
    """S: (B,h,w) in 0..255. Returns (B,2h,w): original rows at 0,2,..,
    new rows at 1,3,.."""
    B, h, w = S.shape
    dev, dt = S.device, S.dtype
    rem = 1.0 - alpha - beta
    a = alpha / 3.0 if cost3 else alpha
    U = 2 * mdis + 1
    kk = torch.arange(h, device=dev)
    p3, p1, n1, n3 = _rows(S, kk - 1), _rows(S, kk), _rows(S, kk + 1), _rows(S, kk + 2)  # (B,h,w)
    M = mdis * 2 + nrad + 2
    cols = torch.arange(-M, w + M, device=dev)
    ridx = _reflect_index(cols, w)

    def padc(t):  # (B,h,w) -> (B,h,w+2M), column c+M holds column c (mirrored outside)
        return t[..., ridx]

    P3, P1, N1, N3 = padc(p3), padc(p1), padc(n1), padc(n3)
    x = torch.arange(w, device=dev)
    xe = torch.arange(-nrad, w + nrad, device=dev)  # columns the window sums reach
    we = xe.numel()

    def at(T, off, xs=x):  # T at column xs+off, off a (U,) tensor or int -> (B,h,len(xs),U)
        if isinstance(off, int):
            return T[..., M + off + xs].unsqueeze(-1).expand(B, h, xs.numel(), U)
        return T[..., M + xs.view(-1, 1) + off.view(1, U)]

    u = torch.arange(-mdis, mdis + 1, device=dev)

    # sum over k = -nrad..nrad of |a[x+k+off_a] - b[x+k+off_b]| for the three row pairs
    # (the terms at columns past the edge read the mirrored margins, as the filter does)
    def pair_cost(off_a, off_b):
        t = ((at(P3, off_a, xe) - at(P1, off_b, xe)).abs() + (at(P1, off_a, xe) - at(N1, off_b, xe)).abs()
             + (at(N1, off_a, xe) - at(N3, off_b, xe)).abs())
        return _boxsum(t, nrad)

    s0 = pair_cost(u, -u)
    if cost3:
        s1 = pair_cost(0, -2 * u)
        s2 = pair_cost(2 * u, 0)
        xx = x.view(1, 1, w, 1)
        uu = u.view(1, 1, 1, U)
        ok1 = ((uu >= 0) & (xx >= 2 * uu)) | ((uu <= 0) & (xx < w + 2 * uu))
        ok2 = ((uu <= 0) & (xx >= -2 * uu)) | ((uu >= 0) & (xx < w + 2 * uu))
        s1v = torch.where(ok1, s1, torch.where(ok2, s2, s0))
        s2v = torch.where(ok2, s2, s1v)
        s = s0 + s1v + s2v
    else:
        s = s0
    ip = (at(P1, u) + at(N1, -u)) / 2.0
    v = (at(P1, 0) - ip).abs() + (at(N1, 0) - ip).abs()
    cost = a * s + beta * u.abs().to(dt).view(1, 1, 1, U) + rem * v
    umax = torch.minimum(torch.minimum(x, w - 1 - x), torch.tensor(mdis, device=dev))  # (w,)
    valid = u.abs().view(1, U) <= umax.view(w, 1)                                        # (w,U)
    cost = torch.where(valid.view(1, 1, w, U), cost, torch.full((), BIG, device=dev, dtype=dt))

    # dynamic programming along each new row: pick v in {u-1, u, u+1}
    R = B * h
    cost = cost.reshape(R, w, U)
    back = torch.zeros((R, w, U), dtype=torch.int8, device=dev)
    pc = torch.full((R, U), BIG, device=dev, dtype=dt)
    pc[:, mdis] = cost[:, 0, mdis]
    inf = torch.full((R, 1), BIG, device=dev, dtype=dt)
    for xi in range(1, w):
        left = torch.cat([inf, pc[:, :-1]], 1) + gamma    # v = u-1
        right = torch.cat([pc[:, 1:], inf], 1) + gamma    # v = u+1
        stack = torch.stack([left, pc, right], -1)
        best, arg = stack.min(-1)                          # first of equal minima = lowest v
        best = torch.where(valid[xi].view(1, U), best + cost[:, xi], torch.full_like(best, BIG))
        back[:, xi] = (arg - 1).to(torch.int8)
        pc = best.clamp(max=BIG)
    path = torch.zeros((R, w), dtype=torch.long, device=dev)
    for xi in range(w - 2, -1, -1):  # fpath[w-1] = 0
        nxt = path[:, xi + 1]
        path[:, xi] = nxt + back[:, xi + 1].gather(1, (nxt + mdis).unsqueeze(1)).squeeze(1).long()
    path = path.view(B, h, w)

    # interpolate along the path
    gx = lambda T, off: T.gather(2, (M + x.view(1, 1, w) + off).clamp(0, T.shape[2] - 1))
    d1 = path
    d3 = path * 3
    lin = (gx(P1, d1) + gx(N1, -d1)) / 2.0
    if ucubic:
        cub = 0.5625 * (gx(P1, d1) + gx(N1, -d1)) - 0.0625 * (gx(P3, d3) + gx(N3, -d3))
        okc = (x.view(1, 1, w) >= d3.abs()) & (x.view(1, 1, w) <= w - 1 - d3.abs())
        new = torch.where(okc, cub, lin)
    else:
        new = lin

    if vcheck > 0 and h >= 3:
        new = _vcheck(new, path, p3, p1, n1, n3, vcheck, vthresh0, vthresh1, vthresh2)
    out = torch.empty((B, 2 * h, w), device=dev, dtype=dt)
    out[:, 0::2] = S
    out[:, 1::2] = new
    return out


def _boxsum(t: torch.Tensor, r: int) -> torch.Tensor:
    """(B,h,w+2r,U) -> (B,h,w,U): sums of 2r+1 neighbouring columns."""
    w = t.shape[2] - 2 * r
    out = t[:, :, 0:w]
    for k in range(1, 2 * r + 1):
        out = out + t[:, :, k:k + w]
    return out


def _vcheck(new, dmap, p3, p1, n1, n3, mode, vt0, vt1, vt2):
    """The filter's vertical check, row by row (each row uses the already
    checked row above it)."""
    B, h, w = new.shape
    x = torch.arange(w, device=new.device).view(1, w)
    rows = [new[:, k] for k in range(h)]
    for k in range(1, h - 1):
        cur, above, below = rows[k], rows[k - 1], rows[k + 1]
        dst1p, dst1n, dst3p, dst3n = p1[:, k], n1[:, k], p3[:, k], n3[:, k]
        dirc = dmap[:, k]
        dirt, dirb = dmap[:, k - 1], dmap[:, k + 1]
        cint = 0.5625 * (dst1p + dst1n) - 0.0625 * (dst3p + dst3n)
        g = lambda t, off: t.gather(1, (x + off).clamp(0, w - 1))
        it = (g(above, dirc) + g(cur, -dirc)) / 2.0
        ib = (g(cur, dirc) + g(below, -dirc)) / 2.0
        vt = (g(above, dirc) - g(dst1p, dirc)).abs() + (g(cur, dirc) - g(dst1p, dirc)).abs()
        vb = (g(below, -dirc) - g(dst1n, -dirc)).abs() + (g(cur, -dirc) - g(dst1n, -dirc)).abs()
        vc = (cur - dst1p).abs() + (cur - dst1n).abs()
        d0, d1 = (it - dst1p).abs(), (ib - dst1n).abs()
        d2, d3 = (vt - vc).abs(), (vb - vc).abs()
        if mode == 1:
            m0, m1 = torch.minimum(d0, d1), torch.minimum(d2, d3)
        elif mode == 2:
            m0, m1 = (d0 + d1) / 2, (d2 + d3) / 2
        else:
            m0, m1 = torch.maximum(d0, d1), torch.maximum(d2, d3)
        a = torch.maximum(torch.maximum(m0 / vt0, m1 / vt1), ((vt2 - dirc.abs()) / vt2).clamp(min=0)).clamp(max=1)
        blended = (1 - a) * cur + a * cint
        plain = (dirc == 0) | (torch.maximum(dirc * dirt, dirc * dirb) < 0) | ((dirt == dirb) & (dirt == 0))
        rows[k] = torch.where(plain, cint, blended)
    return torch.stack(rows, 1)


def eedi3_2x(x: torch.Tensor, **kw) -> torch.Tensor:
    """(N,C,H,W) encoded 0..1 -> (N,C,2H,2W): rows, then columns; originals
    stay at even positions (the caller re-centres)."""
    n, c, h, w = x.shape
    S = (x * 255.0).reshape(n * c, h, w)
    t = eedi3_rows(S, **kw)                                   # (nc, 2h, w)
    t = eedi3_rows(t.transpose(1, 2).contiguous(), **kw)      # (nc, 2w, 2h)
    return (t.transpose(1, 2) / 255.0).reshape(n, c, 2 * h, 2 * w)


# ===================================================================== EEDI2
# The filter itself, compiled from faceloop/native (GPL-2+, unchanged) and
# called per plane: rows (field=1: originals on the even rows), then columns.
_LIB = None
_LIB_LOCK = __import__("threading").Lock()


def _eedi2_lib():
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        import ctypes
        from .native.build import library
        L = ctypes.CDLL(str(library("eedi2", "eedi2_shim.cpp", ["EEDI2.cpp", "VapourSynth.h", "VSHelper.h"])))
        u8p = ctypes.POINTER(ctypes.c_uint8)
        L.eedi2_plane.argtypes = [u8p, ctypes.c_int, ctypes.c_int, u8p] + [ctypes.c_int] * 9
        L.eedi2_plane.restype = ctypes.c_int
        _LIB = L
        return L


def _eedi2_rows(plane, p):
    """uint8 (h, w) -> (2h, w)."""
    import ctypes
    import numpy as np
    h, w = plane.shape
    src = np.ascontiguousarray(plane)
    out = np.empty((2 * h, w), np.uint8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    rc = _eedi2_lib().eedi2_plane(src.ctypes.data_as(u8p), w, h, out.ctypes.data_as(u8p), 1, p["mthresh"],
                                  p["lthresh"], p["vthresh"], p["estr"], p["dstr"], p["maxd"], p["nt"], p["pp"])
    if rc:
        raise ValueError(f"EEDI2 can't work on {w}x{h} (it needs at least 8x7)")
    return out


EEDI2_DEFAULTS = {"mthresh": 10, "lthresh": 20, "vthresh": 20, "estr": 2, "dstr": 4, "maxd": 24, "nt": 50, "pp": 1}


def eedi2_2x(x: torch.Tensor, **kw) -> torch.Tensor:
    """(N,C,H,W) encoded 0..1 -> (N,C,2H,2W) with the filter's defaults
    (mthresh 10, lthresh 20, vthresh 20, estr 2, dstr 4, maxd 24, nt 50,
    pp 1). EEDI2 is 8-bit: frames go through it as 8-bit codes."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    p = {**EEDI2_DEFAULTS, **kw}
    n, c, h, w = x.shape
    if w < 8 or h < 7 or 2 * h < 8 or w < 7:
        raise ValueError(f"EEDI2 needs frames at least 8x7, these are {w}x{h}")
    codes = torch.round(x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().reshape(n * c, h, w)

    def one(plane):
        t = _eedi2_rows(plane, p)                       # (2h, w)
        return _eedi2_rows(np.ascontiguousarray(t.T), p).T  # (2w, 2h) -> (2h, 2w)
    import os
    with ThreadPoolExecutor(max_workers=max(1, min(len(codes), os.cpu_count() or 4))) as ex:
        planes = list(ex.map(one, codes))
    out = torch.from_numpy(np.stack(planes).reshape(n, c, 2 * h, 2 * w).astype(np.float32) / 255.0)
    return out.to(x.device, x.dtype)


# ===================================================================== NNEDI3
NNEDI3_WEIGHTS = {
    "url": "https://raw.githubusercontent.com/dubhater/vapoursynth-nnedi3/"
           "82993ff21cf569776fc1a7e5bb60235c00bbeea3/src/nnedi3_weights.bin",
    "sha256": "27f382430435bb7613deb1c52f3c79c300c9869812cfe29079432a9c82251d42",
}
# "fast" is the filter's own default; "best" what nnedi3_rpow2 scripts use for upscaling
NNEDI3_QUALITY = {"fast": dict(nsize=6, nns=1, qual=1), "best": dict(nsize=0, nns=3, qual=2)}
_NN = None
_NN_LOCK = __import__("threading").Lock()


def nnedi3_weights() -> "Path":
    """weights/nnedi3/nnedi3_weights.bin, downloaded (13.5 MB, checked) the first time."""
    import hashlib
    import os
    import urllib.request
    from .models import default_weights_dir, sha256_file
    dest = default_weights_dir() / "nnedi3" / "nnedi3_weights.bin"
    if dest.exists() and sha256_file(dest) == NNEDI3_WEIGHTS["sha256"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(NNEDI3_WEIGHTS["url"], headers={"User-Agent": "faceloop"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
            h.update(b)
    if h.hexdigest() != NNEDI3_WEIGHTS["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("nnedi3_weights.bin: checksum mismatch")
    os.replace(tmp, dest)
    return dest


def _nnedi3_lib():
    global _NN
    with _NN_LOCK:
        if _NN is not None:
            return _NN
        import ctypes
        from .native.build import library
        L = ctypes.CDLL(str(library("nnedi3", "nnedi3_shim.cpp", ["nnedi3.cpp", "cpufeatures.h", "VapourSynth.h",
                                                                   "VSHelper.h"], arm_extra=["simd_neon.c"])))
        L.nnedi3_setup.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int] + [ctypes.c_int] * 5 + \
            [ctypes.c_char_p, ctypes.c_int]
        L.nnedi3_setup.restype = ctypes.c_int
        u8p = ctypes.POINTER(ctypes.c_uint8)
        L.nnedi3_plane.argtypes = [u8p, ctypes.c_int, ctypes.c_int, u8p]
        L.nnedi3_plane.restype = ctypes.c_int
        _NN = L
        return L


def _nnedi3_rows(planes, quality: str):
    """uint8 (P, h, w) -> (P, 2h, w), originals on the even rows (field 1, dh)."""
    import ctypes
    import os
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    L = _nnedi3_lib()
    P, h, w = planes.shape
    q = NNEDI3_QUALITY[quality]
    err = ctypes.create_string_buffer(512)
    with _NN_LOCK:  # one plane size at a time: the filter is set up per size
        if L.nnedi3_setup(str(nnedi3_weights().parent).encode(), w, h, q["nsize"], q["nns"], q["qual"], 0, 2,
                          err, 512):
            raise ValueError(err.value.decode(errors="replace"))
        out = np.empty((P, 2 * h, w), np.uint8)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        src = np.ascontiguousarray(planes)

        def one(i):
            if L.nnedi3_plane(src[i].ctypes.data_as(u8p), w, h, out[i].ctypes.data_as(u8p)):
                raise RuntimeError("nnedi3 failed")
        with ThreadPoolExecutor(max_workers=max(1, min(P, os.cpu_count() or 4))) as ex:
            list(ex.map(one, range(P)))
    return out


def nnedi3_2x(x: torch.Tensor, quality: str = "fast", **_) -> torch.Tensor:
    """(N,C,H,W) encoded 0..1 -> (N,C,2H,2W): the nnedi3 filter on 8-bit
    codes, rows then columns, prescreener 2 (the filter's default for 8-bit),
    etype 0. quality "fast" = nsize 6, nns 1, qual 1 (the filter's
    defaults); "best" = nsize 0, nns 3, qual 2 (nnedi3_rpow2's)."""
    import numpy as np
    n, c, h, w = x.shape
    codes = torch.round(x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy().reshape(n * c, h, w)
    t = _nnedi3_rows(codes, quality)                                   # (P, 2h, w)
    t = _nnedi3_rows(np.ascontiguousarray(t.transpose(0, 2, 1)), quality)  # (P, 2w, 2h)
    t = t.transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(t).reshape(n, c, 2 * h, 2 * w).astype(np.float32) / 255.0).to(
        x.device, x.dtype)
