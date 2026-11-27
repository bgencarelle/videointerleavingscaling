"""Content-aware downscaling: perceptual downscaling and content-adaptive
downscaling. All work on (N,3,H,W) float frames on any
device, frames independent.

perceptual   Öztireli & Gross, "Perceptually Based Downscaling of Images",
             SIGGRAPH 2015, Algorithm 1 (appendix B). For whole-number
             factors it is exactly their algorithm; other factors use the
             same box (area) average for its first two lines.
kopf         Kopf, Shamir & Peers, "Content-Adaptive Image Downscaling",
             SIGGRAPH Asia 2013, following their supplementary pseudocode:
             one bilateral kernel per output pixel (spatial Gaussian x
             CIELAB colour Gaussian), refined by EM with their correction
             step. See kopf() for how the paper's units are read.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ===================================================================== perceptual
def perceptual(x: torch.Tensor, width: int, height: int, area, patch: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """x: (N,C,H,W) in the light the caller chose. area(t, w, h) is a box
    (area) downscale. patch is sqrt(np) in the paper (2 = their np = 4)."""
    L = area(x, width, height)
    L2 = area(x * x, width, height)
    if height < patch or width < patch:
        return L
    valid = lambda t: F.avg_pool2d(t, patch, stride=1)                       # convValid
    full = lambda t: F.avg_pool2d(F.pad(t, (patch - 1,) * 4), patch, stride=1)  # convFull, zero padded
    M = valid(L)
    Sl = valid(L * L) - M * M
    Sh = valid(L2) - M * M
    R = torch.sqrt(Sh.clamp(min=0) / Sl.clamp(min=eps))
    R = torch.where(Sl < eps, torch.zeros_like(R), R)
    N = full(torch.ones_like(M))
    T = full(R * M)
    Mf = full(M)
    Rf = full(R)
    return (Mf + Rf * L - T) / N


# ===================================================================== Kopf 2013
_M_RGB2XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
# CIELAB scaled to about 0..1 (the paper starts colours at (1/2, 1/2, 1/2))
_LAB_OFF = torch.tensor([0.0, 128.0, 128.0], dtype=torch.float64)
_LAB_SCALE = torch.tensor([100.0, 255.0, 255.0], dtype=torch.float64)


def _lab(lin: torch.Tensor) -> torch.Tensor:
    """linear sRGB (N,3,...) -> scaled CIELAB (N,3,...)."""
    dt, dev = lin.dtype, lin.device
    m = _M_RGB2XYZ.to(dev, dt)
    xyz = torch.einsum("ij,nj...->ni...", m, lin) / _WHITE.to(dev, dt).view(1, 3, *([1] * (lin.dim() - 2)))
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-12) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    lab = torch.stack([L, a, b], 1)
    sh = (1, 3) + (1,) * (lin.dim() - 2)
    return (lab + _LAB_OFF.to(dev, dt).view(sh)) / _LAB_SCALE.to(dev, dt).view(sh)


def _unlab(lab: torch.Tensor) -> torch.Tensor:
    """scaled CIELAB (N,3,...) -> linear sRGB."""
    dt, dev = lab.dtype, lab.device
    sh = (1, 3) + (1,) * (lab.dim() - 2)
    lab = lab * _LAB_SCALE.to(dev, dt).view(sh) - _LAB_OFF.to(dev, dt).view(sh)
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    d = 6 / 29
    inv = lambda f: torch.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29))
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], 1) * _WHITE.to(dev, dt).view(sh)
    m = torch.linalg.inv(_M_RGB2XYZ).to(dev, dt)
    return torch.einsum("ij,nj...->ni...", m, xyz)


def kopf(lin: torch.Tensor, width: int, height: int, max_iter: int = 200, tol: float = 0.002,
         check=None) -> torch.Tensor:
    """Content-adaptive downscaling of linear-light RGB (N,3,H,W) to
    (N,3,height,width), in CIELAB internally.

    Reading of the paper's units: kernel centres, pixel positions and the
    support Rk (|x - xk| < 2 rx) are in input pixels, as the paper says
    ("the center of output pixel k scaled to input image dimensions").
    The constants that only make sense relative to an output pixel - the
    singular-value clamp [0.05, 0.1], the directional-variance threshold
    0.2 and the edge-strength threshold 0.08 - are applied in output-pixel
    units (input offsets divided by rx, ry; kernel weights normalised), so
    the method behaves the same at any downscaling ratio.
    Stops when no kernel colour moves by more than `tol` or after max_iter.
    """
    dev = lin.device
    dt = torch.float32
    n, _, hi, wi = lin.shape
    wo, ho = int(width), int(height)
    rx, ry = wi / wo, hi / ho
    lab = _lab(lin.to(torch.float64)).to(dt).reshape(n, 3, hi * wi)

    # kernel grid and each kernel's window of input pixels (Rk)
    kx = (torch.arange(wo, device=dev, dtype=dt) + 0.5) * rx
    ky = (torch.arange(ho, device=dev, dtype=dt) + 0.5) * ry
    wx, wy = int(math.floor(4 * rx)) + 2, int(math.floor(4 * ry)) + 2
    x0 = torch.floor(kx - 2 * rx).long()
    y0 = torch.floor(ky - 2 * ry).long()
    ox = torch.arange(wx, device=dev)
    oy = torch.arange(wy, device=dev)
    px = (x0.view(wo, 1) + ox.view(1, wx))                     # (wo, wx) input columns
    py = (y0.view(ho, 1) + oy.view(1, wy))                     # (ho, wy) input rows
    okx = (px >= 0) & (px < wi) & ((px.to(dt) + 0.5 - kx.view(wo, 1)).abs() < 2 * rx)
    oky = (py >= 0) & (py < hi) & ((py.to(dt) + 0.5 - ky.view(ho, 1)).abs() < 2 * ry)
    K, P = wo * ho, wx * wy
    # (K, P) index, validity and position (input pixel centres, then output units)
    gy = py.view(ho, 1, wy, 1).expand(ho, wo, wy, wx)
    gx = px.view(1, wo, 1, wx).expand(ho, wo, wy, wx)
    valid = (oky.view(ho, 1, wy, 1) & okx.view(1, wo, 1, wx)).reshape(K, P)
    idx = (gy.clamp(0, hi - 1) * wi + gx.clamp(0, wi - 1)).reshape(K, P)
    pos = torch.stack([(gx.to(dt) + 0.5) / rx, (gy.to(dt) + 0.5) / ry], -1).reshape(K, P, 2)  # output units
    centre = torch.stack(torch.meshgrid(torch.arange(ho, device=dev, dtype=dt) + 0.5,
                                        torch.arange(wo, device=dev, dtype=dt) + 0.5, indexing="ij")[::-1],
                         -1).reshape(K, 2)                     # (x, y) in output units
    col = lab.gather(2, idx.view(1, 1, K * P).expand(n, 3, K * P)).view(n, 3, K, P).permute(0, 2, 3, 1)  # (n,K,P,3)
    flat_idx = idx.view(1, K * P).expand(n, K * P)

    # neighbours on the kernel grid
    def shift(t, dy, dx, fill):
        """out[:, y, x] = t[:, y+dy, x+dx], `fill` where that's outside."""
        out = torch.full_like(t, fill)
        ys, yd = slice(max(0, -dy), ho - max(0, dy)), slice(max(0, dy), ho - max(0, -dy))
        xs, xd = slice(max(0, -dx), wo - max(0, dx)), slice(max(0, dx), wo - max(0, -dx))
        out[:, ys, xs] = t[:, yd, xd]
        return out

    # --- initialise (paper eq. 6; Sigma diag(rx/3, ry/3) in input pixels -> output units)
    mu = centre.view(1, K, 2).repeat(n, 1, 1)
    sxx = torch.full((n, K), (rx / 3) / (rx * rx), device=dev, dtype=dt)
    syy = torch.full((n, K), (ry / 3) / (ry * ry), device=dev, dtype=dt)
    sxy = torch.zeros((n, K), device=dev, dtype=dt)
    nu = torch.full((n, K, 3), 0.5, device=dev, dtype=dt)
    sig = torch.full((n, K), 1e-4, device=dev, dtype=dt)
    neg = torch.tensor(-1e30, device=dev, dtype=dt)
    pix_n = n
    for it in range(max_iter):
        if check:
            check()
        # --- E-step: log w_k(i), normalised per kernel, then per pixel (in logs: no underflow)
        d = pos.view(1, K, P, 2) - mu.view(n, K, 1, 2)
        det = (sxx * syy - sxy * sxy).clamp(min=1e-12)
        ixx, iyy, ixy = syy / det, sxx / det, -sxy / det
        maha = (ixx.view(n, K, 1) * d[..., 0] ** 2 + 2 * ixy.view(n, K, 1) * d[..., 0] * d[..., 1]
                + iyy.view(n, K, 1) * d[..., 1] ** 2)
        cdist = ((col - nu.view(n, K, 1, 3)) ** 2).sum(-1)
        logw = -0.5 * maha - cdist / (2 * sig.view(n, K, 1) ** 2)
        logw = torch.where(valid.view(1, K, P), logw, neg)
        logw = logw - torch.logsumexp(logw, -1, keepdim=True)           # w_k / wsum
        m_pix = torch.full((pix_n, hi * wi), -1e30, device=dev, dtype=dt)
        m_pix = m_pix.scatter_reduce(1, flat_idx, torch.where(valid.view(1, K, P), logw, neg).reshape(n, K * P),
                                     "amax", include_self=True)
        mx = m_pix.gather(1, flat_idx).view(n, K, P)
        e = torch.where(valid.view(1, K, P), torch.exp(logw - mx), torch.zeros((), device=dev, dtype=dt))
        s_pix = torch.zeros((pix_n, hi * wi), device=dev, dtype=dt).scatter_add(1, flat_idx, e.reshape(n, K * P))
        gam = e / s_pix.gather(1, flat_idx).view(n, K, P).clamp(min=1e-30)   # gamma_k(i)
        # --- M-step (Sigma with the old mean, then the mean; as in the pseudocode)
        ws = gam.sum(-1).clamp(min=1e-12)
        dxx = (gam * d[..., 0] * d[..., 0]).sum(-1) / ws
        dyy = (gam * d[..., 1] * d[..., 1]).sum(-1) / ws
        dxy = (gam * d[..., 0] * d[..., 1]).sum(-1) / ws
        mu = (gam.unsqueeze(-1) * pos.view(1, K, P, 2)).sum(2) / ws.unsqueeze(-1)
        nu_new = (gam.unsqueeze(-1) * col).sum(2) / ws.unsqueeze(-1)
        moved = (nu_new - nu).abs().max().item() if it else 1.0
        nu = nu_new
        # --- C-step: spatial constraints (eq. 11)
        m2 = mu.view(n, ho, wo, 2)
        acc = torch.zeros_like(m2)
        cnt = torch.zeros((n, ho, wo, 1), device=dev, dtype=dt)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            has = shift(torch.ones((n, ho, wo, 1), device=dev, dtype=dt), dy, dx, 0.0)
            acc = acc + shift(m2, dy, dx, 0.0)
            cnt = cnt + has
        mbar = (acc / cnt.clamp(min=1)).view(n, K, 2)
        mu = 0.5 * mbar + 0.5 * mu
        c = centre.view(1, K, 2)
        mu = torch.max(torch.min(mu, c + 0.25), c - 0.25)
        # --- C-step: clamp the singular values of Sigma to [0.05, 0.1] (output units)
        tr = dxx + dyy
        disc = torch.sqrt(((dxx - dyy) * 0.5) ** 2 + dxy ** 2)
        l1, l2 = tr * 0.5 + disc, tr * 0.5 - disc
        ang = 0.5 * torch.atan2(2 * dxy, dxx - dyy)
        cs, sn = torch.cos(ang), torch.sin(ang)
        l1, l2 = l1.clamp(0.05, 0.1), l2.clamp(0.05, 0.1)
        sxx = l1 * cs * cs + l2 * sn * sn
        syy = l1 * sn * sn + l2 * cs * cs
        sxy = (l1 - l2) * cs * sn
        # --- C-step: shape constraints (dominant kernels, staircasing)
        grow = torch.zeros((n, ho, wo), dtype=torch.bool, device=dev)
        grow_nb = {}
        dk = pos.view(1, K, P, 2) - mu.view(n, K, 1, 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                proj = (dk[..., 0] * dx + dk[..., 1] * dy).clamp(min=0)
                s = (gam * proj * proj).sum(-1) / ws                     # directional variance
                trig = (s > 0.2).view(n, ho, wo)
                if dy == 0 or dx == 0:
                    # edge strength between k and its 4-neighbour n: sum gamma_k gamma_n
                    gn = _neighbour_gamma(gam, x0, y0, wx, wy, wo, ho, dy, dx, rx, ry, valid)
                    f = (gam * gn).sum(-1) / ws
                    ratio = gam / (gam + gn).clamp(min=1e-12)
                    rr = ratio.view(n, K, wy, wx)
                    both = ((gam + gn) > 1e-6).view(n, K, wy, wx)
                    gx_ = torch.zeros_like(rr)
                    gy_ = torch.zeros_like(rr)
                    gx_[..., 1:-1] = (rr[..., 2:] - rr[..., :-2]) * 0.5
                    gy_[..., 1:-1, :] = (rr[..., 2:, :] - rr[..., :-2, :]) * 0.5
                    gx_ = torch.where(both, gx_, torch.zeros_like(gx_)).sum((-1, -2))
                    gy_ = torch.where(both, gy_, torch.zeros_like(gy_)).sum((-1, -2))
                    # angle between the kernels' edge normal and the pixel edge normal (as lines)
                    dot = (gx_ * rx * dx + gy_ * ry * dy).abs()
                    norm = torch.sqrt((gx_ * rx) ** 2 + (gy_ * ry) ** 2).clamp(min=1e-12)
                    ang = torch.rad2deg(torch.acos((dot / norm).clamp(max=1.0)))
                    trig = trig | ((f < 0.08) & (ang > 25)).view(n, ho, wo)
                grow = grow | trig
                grow_nb[(dy, dx)] = trig
        for (dy, dx), trig in grow_nb.items():  # the neighbour of a triggering pair grows too
            grow = grow | shift(trig, -dy, -dx, False)
        sig = torch.where(grow.view(n, K), sig * 1.1, sig)
        kopf.last_iterations = it + 1
        if it and moved < tol:  # colours settled (some kernels may keep softening: harmless)
            break
    out = _unlab(nu.permute(0, 2, 1).reshape(n, 3, ho, wo).to(torch.float64)).to(lin.dtype)
    return out.clamp(0, 1)


def _neighbour_gamma(gam, x0, y0, wx, wy, wo, ho, dy, dx, rx, ry, valid):
    """gamma of the 4-neighbour (dy, dx) of each kernel, at the pixels of
    the kernel's own window (0 where the neighbour's window doesn't reach)."""
    n, K, P = gam.shape
    g = gam.view(n, ho, wo, wy, wx)
    out = torch.zeros_like(g)
    # neighbour kernel indices and the offset of its window relative to ours
    ky = torch.arange(ho, device=gam.device)
    kx = torch.arange(wo, device=gam.device)
    ny = (ky + dy).clamp(0, ho - 1)
    nx = (kx + dx).clamp(0, wo - 1)
    sy = (y0[ny] - y0)  # (ho,) rows our window is ahead of the neighbour's
    sx = (x0[nx] - x0)  # (wo,)
    has_y = (ky + dy >= 0) & (ky + dy < ho)
    has_x = (kx + dx >= 0) & (kx + dx < wo)
    gn = g[:, ny][:, :, nx]  # (n, ho, wo, wy, wx): neighbour's window
    iy = torch.arange(wy, device=gam.device).view(1, wy) - sy.view(ho, 1)      # our row -> neighbour row
    ix = torch.arange(wx, device=gam.device).view(1, wx) - sx.view(wo, 1)
    oky = (iy >= 0) & (iy < wy) & has_y.view(ho, 1)
    okx = (ix >= 0) & (ix < wx) & has_x.view(wo, 1)
    gi = gn.gather(3, iy.clamp(0, wy - 1).view(1, ho, 1, wy, 1).expand(n, ho, wo, wy, wx))
    gi = gi.gather(4, ix.clamp(0, wx - 1).view(1, 1, wo, 1, wx).expand(n, ho, wo, wy, wx))
    m = oky.view(1, ho, 1, wy, 1) & okx.view(1, 1, wo, 1, wx)
    out = torch.where(m, gi, out)
    return out.view(n, K, P) * valid.view(1, K, P)
