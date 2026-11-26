"""DPID: Rapid, Detail-Preserving Image Downscaling (Weber et al., SIGGRAPH Asia 2016).

Follows the authors' CUDA reference (github.com/mergian/dpid, commit
dba82881, BSD 3-Clause, copyright (c) 2016 Nicolas Weber and Sandra C.
Amend / GCC / TU-Darmstadt), written as whole-frame tensor operations:

1. avg   = area downscale (each input pixel weighted by how much of it the
           output pixel covers)
2. guide = avg smoothed with the 3x3 kernel [1 2 1; 2 4 2; 1 2 1], only over
           neighbours that exist (renormalised at the edges)
3. out   = for each output pixel, the input pixels it covers, weighted by
           coverage x (distance to guide)^lambda, where distance is the RGB
           Euclidean distance divided by sqrt(3) (0..1). If all weights are 0
           (a perfectly flat patch), out = guide.

lambda = 0 is a plain box filter; larger values favour pixels that stand out
from their surroundings (the reference default is 1.0).

The MATLAB version in the same repository has a typo in the bottom-edge
coverage test (compares with the start instead of the end); the CUDA
version, followed here, has it right.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _footprints(n_in: int, n_out: int):
    """For each output index: the covered input indices and coverage (0..1),
    padded to the same length K (padding has coverage 0)."""
    s = n_in / n_out
    rows = []
    for o in range(n_out):
        a, b = o * s, min((o + 1) * s, n_in)
        lo, hi = int(math.floor(a)), int(math.ceil(b))
        row = []
        for i in range(lo, hi):
            f = 1.0
            if i < a:
                f *= 1.0 - (a - i)
            if i + 1 > b:
                f *= 1.0 - ((i + 1) - b)
            row.append((i, f))
        rows.append(row)
    k = max(len(r) for r in rows)
    idx = torch.zeros(n_out, k, dtype=torch.long)
    cov = torch.zeros(n_out, k)
    for o, r in enumerate(rows):
        for j, (i, f) in enumerate(r):
            idx[o, j] = i
            cov[o, j] = f
    return idx, cov


_CACHE: dict = {}


def _fp(n_in, n_out, device):
    key = (n_in, n_out, str(device))
    if key not in _CACHE:
        idx, cov = _footprints(n_in, n_out)
        _CACHE[key] = (idx.to(device), cov.to(device))
    return _CACHE[key]


@torch.no_grad()
def dpid(x: torch.Tensor, width: int, height: int, lam: float = 1.0) -> torch.Tensor:
    """x: (N, C, H, W) in 0..1 -> (N, C, height, width)."""
    n, c, h, w = x.shape
    iy, cy = _fp(h, height, x.device)   # (oh, ky)
    ix, cx = _fp(w, width, x.device)    # (ow, kx)
    # gather every output pixel's footprint: (N, C, oh, ky, ow, kx)
    patch = x[:, :, iy][:, :, :, :, ix]
    cov = cy[:, :, None, None] * cx[None, None, :, :]  # (oh, ky, ow, kx)
    covs = cov.sum(dim=(1, 3))                         # (oh, ow)
    avg = (patch * cov).sum(dim=(3, 5)) / covs         # (N, C, oh, ow)

    # guide: 3x3 [1 2 1; 2 4 2; 1 2 1], renormalised where neighbours are missing
    k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=x.device, dtype=x.dtype)
    kc = k.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    num = F.conv2d(F.pad(avg, (1, 1, 1, 1)), kc, groups=c)
    den = F.conv2d(F.pad(torch.ones(1, 1, height, width, device=x.device, dtype=x.dtype), (1, 1, 1, 1)),
                   k.view(1, 1, 3, 3))
    guide = num / den

    if lam == 0:
        return avg
    dist = ((patch - guide[:, :, :, None, :, None]) ** 2).sum(1, keepdim=True).sqrt() / math.sqrt(3.0)
    wgt = dist ** lam if lam != 1.0 else dist
    wgt = wgt * cov
    wsum = wgt.sum(dim=(3, 5))                          # (N, 1, oh, ow)
    out = (patch * wgt).sum(dim=(3, 5)) / wsum.clamp_min(1e-12)
    return torch.where(wsum > 0, out, guide)
