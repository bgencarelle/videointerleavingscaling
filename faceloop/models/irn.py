"""IRN: Invertible Rescaling Network (Xiao et al., ECCV 2020 / IJCV 2022).

The architecture is ported from the authors' code (github.com/pkuxmq/
Invertible-Image-Rescaling, commit 0ebe8f9d, codes/models/modules/
Inv_arch.py and Subnet_constructor.py; Apache License 2.0), keeping the
module names so their pretrained .pth files load unchanged. Training-only
parts (Jacobians, initialisation) are left out.

IRN is one invertible network used both ways:
- down: run forward; the first 3 channels are the small image (the rest is
  "lost detail" the network learned to push into a latent z, thrown away).
- up:   run backwards on the small image plus a latent z. The paper samples
  z at random; here z = 0 (its mean), so results are repeatable.
Frames are replicate-padded to a multiple of the scale and cropped after.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1))


class InvBlockExp(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=1.0):
        super().__init__()
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp
        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)
        if not rev:
            y1 = x1 + self.F(x2)
            s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(s)) + self.G(y1)
        else:
            s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(s))
            y1 = x1 - self.F(y2)
        return torch.cat((y1, y2), 1)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.channel_in = channel_in
        w = torch.ones(4, 1, 2, 2)
        w[1, 0, 0, 1] = -1
        w[1, 0, 1, 1] = -1
        w[2, 0, 1, 0] = -1
        w[2, 0, 1, 1] = -1
        w[3, 0, 1, 0] = -1
        w[3, 0, 0, 1] = -1
        self.haar_weights = nn.Parameter(torch.cat([w] * channel_in, 0), requires_grad=False)

    def forward(self, x, rev=False):
        c = self.channel_in
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=c) / 4.0
            out = out.reshape([x.shape[0], c, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            return out.reshape([x.shape[0], c * 4, x.shape[2] // 2, x.shape[3] // 2])
        out = x.reshape([x.shape[0], 4, c, x.shape[2], x.shape[3]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], c * 4, x.shape[2], x.shape[3]])
        return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=c)


class InvRescaleNet(nn.Module):
    """IRN_x2 (and its JPEG finetune): block_num [8]; IRN_x4 and IRN+_x4: [8, 8];
    IRN_x8: [8, 8, 8] (the authors' test configs)."""

    def __init__(self, scale: int, channel_in=3, channel_out=3):
        super().__init__()
        down_num = int(math.log(scale, 2))
        block_num = [8] * down_num
        ops = []
        cur = channel_in
        for i in range(down_num):
            ops.append(HaarDownsampling(cur))
            cur *= 4
            for _ in range(block_num[i]):
                ops.append(InvBlockExp(cur, channel_out))
        self.operations = nn.ModuleList(ops)

    def forward(self, x, rev=False):
        out = x
        for op in (reversed(self.operations) if rev else self.operations):
            out = op(out, rev)
        return out


class IRN(nn.Module):
    def __init__(self, path: Path, scale: int):
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("IRN scale must be 2, 4 or 8")
        self.scale = scale
        self.net = InvRescaleNet(scale)
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd, strict=True)
        self.net.eval().requires_grad_(False)

    # ------------------------------------------------------------ running
    # IRN holds ~140 channels per pixel inside its blocks, so a whole batch
    # of frames rarely fits. Frames go through a few at a time; on running
    # out of memory the group shrinks, and as a last resort frames are cut
    # into overlapping tiles.
    micro = 4
    tile: int | None = None

    @torch.no_grad()
    def _down_raw(self, x):
        s = self.scale
        h, w = x.shape[-2:]
        ph, pw = (-h) % s, (-w) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return self.net(x)[:, :3][..., : -(-h // s), : -(-w // s)]

    @torch.no_grad()
    def _up_raw(self, x):
        s = self.scale
        n, c, h, w = x.shape
        z = torch.zeros(n, c * (s * s - 1), h, w, device=x.device, dtype=x.dtype)
        return self.net(torch.cat((x, z), 1), rev=True)

    def _tiled(self, fn, x, down: bool):
        s = self.scale
        n, c, h, w = x.shape
        # tile and overlap in input pixels; multiples of s when shrinking so tiles line up
        t, o = (self.tile // s * s, 32 // s * s or s) if down else (self.tile, 8)
        oh, ow = (-(-h // s), -(-w // s)) if down else (h * s, w * s)
        out = torch.empty(n, c, oh, ow, device=x.device, dtype=x.dtype)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(h, y0 + t), min(w, x0 + t)
                ya, xa = max(0, y0 - o), max(0, x0 - o)
                yb, xb = min(h, y1 + o), min(w, x1 + o)
                r = fn(x[..., ya:yb, xa:xb])
                if down:
                    oy0, oy1, ox0, ox1 = y0 // s, -(-y1 // s), x0 // s, -(-x1 // s)
                    ry, rx = (y0 - ya) // s, (x0 - xa) // s
                else:
                    oy0, oy1, ox0, ox1 = y0 * s, y1 * s, x0 * s, x1 * s
                    ry, rx = (y0 - ya) * s, (x0 - xa) * s
                out[..., oy0:oy1, ox0:ox1] = r[..., ry:ry + (oy1 - oy0), rx:rx + (ox1 - ox0)]
        return out

    def _safe(self, fn, x, down: bool):
        outs, i = [], 0
        while i < x.shape[0]:
            chunk = x[i:i + self.micro]
            try:
                y = self._tiled(fn, chunk, down) if self.tile else fn(chunk)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                _free(x.device)
                if self.micro > 1:
                    self.micro //= 2
                elif self.tile is None:
                    self.tile = 256
                elif self.tile > 32:
                    self.tile //= 2
                else:
                    raise
                continue
            outs.append(y)
            i += chunk.shape[0]
        return torch.cat(outs, 0)

    def down(self, x: torch.Tensor) -> torch.Tensor:
        return self._safe(self._down_raw, x, True)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        return self._safe(self._up_raw, x, False)


def _free(device):
    try:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
