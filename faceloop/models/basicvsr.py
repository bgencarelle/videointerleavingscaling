"""BasicVSR (Chan, Wang, Yu, Dong, Loy, CVPR 2021): video x4 with a memory.

Two recurrent chains of 30 residual blocks carry features along the
sequence, one backward and one forward in time; before each step the
carried features are moved onto the current frame by optical flow from
SPyNet (Ranjan & Black). The two are fused and upsampled x4 with pixel
shuffle, on top of a bilinear x4 of the frame. Not generative: trained
with a plain (Charbonnier) loss against the real frames.

The layers are written out here with the names of OpenMMLab's
BasicVSRNet (mmagic / mmediting, Apache-2.0), so its released checkpoints
load as they are; the maths follows BasicSR's basicvsr_arch.py and
spynet_arch.py (Apache-2.0) line for line, and the result matches it
(tests).

What faceloop adds is how the frames are fed (see run_vsr):
    memory      frames per window (default 1: every frame alone, no mixing).
                Windows are counted from the sequence's first frame and never
                reach past the batch in memory; nothing carries between
                batches, so a resumed run gives the same frames
    flow        multiply every flow (0 = carry features without aligning them)
    direction   both (as trained), forward or backward: the other chain then
                sees only the current frame
    order       normal (frame-number order), reverse or shuffle in a window
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x, flow, padding_mode="zeros"):
    """Sample x at (pixel + flow); flow (N,2,H,W) in pixels, x then y."""
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    fx = gx + flow[:, 0].to(x.dtype)
    fy = gy + flow[:, 1].to(x.dtype)
    grid = torch.stack((2.0 * fx / max(w - 1, 1) - 1.0, 2.0 * fy / max(h - 1, 1) - 1.0), dim=3)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


class _Conv(nn.Module):           # mmcv ConvModule without norm: .conv then an optional ReLU
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.act else x


class _SPyNetBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(_Conv(8, 32), _Conv(32, 64), _Conv(64, 32), _Conv(32, 16),
                                          _Conv(16, 2, act=False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([_SPyNetBasic() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            up = flow if level == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear",
                                                       align_corners=True) * 2.0
            flow = up + self.basic_module[level](
                torch.cat([ref[level], flow_warp(supp[level], up, padding_mode="border"), up], 1))
        return flow

    def forward(self, ref, supp):
        """Flow from ref to supp: supp sampled at (pixel + flow) lines up with ref."""
        h, w = ref.shape[2:]
        wu = w if w % 32 == 0 else 32 * (w // 32 + 1)
        hu = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(hu, wu), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(hu, wu), mode="bilinear", align_corners=False)
        flow = F.interpolate(self._flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        return torch.stack((flow[:, 0] * (w / wu), flow[:, 1] * (h / hu)), 1)


class _ResBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class _Chain(nn.Module):          # ResidualBlocksWithInputConv
    def __init__(self, cin, c=64, blocks=30):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, c, 3, 1, 1), nn.LeakyReLU(0.1),
                                  nn.Sequential(*[_ResBlock(c) for _ in range(blocks)]))

    def forward(self, x):
        return self.main(x)


class _Shuffle(nn.Module):        # PixelShufflePack
    def __init__(self, cin, cout):
        super().__init__()
        self.upsample_conv = nn.Conv2d(cin, cout * 4, 3, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), 2)


class BasicVSRNet(nn.Module):
    def __init__(self, mid=64, blocks=30):
        super().__init__()
        self.mid = mid
        self.spynet = SPyNet()
        self.backward_resblocks = _Chain(mid + 3, mid, blocks)
        self.forward_resblocks = _Chain(mid + 3, mid, blocks)
        self.fusion = nn.Conv2d(mid * 2, mid, 1, 1, 0)
        self.upsample1 = _Shuffle(mid, mid)
        self.upsample2 = _Shuffle(mid, 64)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    @classmethod
    def from_checkpoint(cls, path) -> "BasicVSRNet":
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
        net = cls()
        net.load_state_dict(sd, strict=True)
        return net.eval()

    def flow(self, ref, supp):
        dt = self.spynet.mean.dtype
        return self.spynet(ref.to(dt), supp.to(dt))

    def step(self, chain, lr, feat):
        return chain(torch.cat([lr, feat], 1))

    def reconstruct(self, lr, fb, ff):
        out = F.leaky_relu(self.fusion(torch.cat([fb, ff], 1)), 0.1)
        out = F.leaky_relu(self.upsample1(out), 0.1)
        out = F.leaky_relu(self.upsample2(out), 0.1)
        out = F.leaky_relu(self.conv_hr(out), 0.1)
        out = self.conv_last(out)
        return out + F.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def forward(self, lrs):
        """lrs (N,T,3,H,W) in 0..1 -> (N,T,3,4H,4W): the network exactly as trained."""
        outs, _ = self.run_window([f.unsqueeze(0) for f in lrs[0].unbind(0)], "both", 1.0)
        return torch.cat(outs, 0).unsqueeze(0)

    # --------------------------------------------------------------- faceloop
    @torch.no_grad()
    def run_window(self, frames, direction="both", flow=1.0, carry=None, check=None):
        """frames: list of (1,3,H,W). carry: (previous frame, forward features)
        to continue the forward chain from. Returns (outputs, carry for the
        next piece of the same window)."""
        dt = self.fusion.weight.dtype
        frames = [f.to(dt) for f in frames]
        t = len(frames)
        h, w = frames[0].shape[-2:]
        zeros = frames[0].new_zeros(1, self.mid, h, w)

        def fl(a, b):
            if flow == 0:
                return a.new_zeros(1, 2, h, w)
            return self.flow(a, b) * flow

        # backward chain: frame i takes the features of i+1
        back = [None] * t
        if direction in ("both", "backward"):
            feat = zeros
            for i in range(t - 1, -1, -1):
                if check:
                    check()
                if i < t - 1:
                    feat = flow_warp(feat, fl(frames[i], frames[i + 1]))
                feat = self.step(self.backward_resblocks, frames[i], feat)
                back[i] = feat
        else:
            for i in range(t):
                back[i] = self.step(self.backward_resblocks, frames[i], zeros)

        outs = []
        prev, feat = carry if carry is not None else (None, zeros)
        for i in range(t):
            if check:
                check()
            if direction in ("both", "forward"):
                if prev is not None:
                    feat = flow_warp(feat, fl(frames[i], prev.to(dt)))
                feat = self.step(self.forward_resblocks, frames[i], feat)
            else:
                feat = self.step(self.forward_resblocks, frames[i], zeros)
            outs.append(self.reconstruct(frames[i], back[i], feat).clamp(0, 1))
            prev = frames[i]
        return outs, (prev, feat)


def plan_windows(first: int, count: int, memory: int):
    """Split frames first..first+count-1 (positions in the whole sequence)
    into windows of `memory` frames counted from frame 0, cut at the ends
    of this batch. Yields (window number, start, stop), local to the batch.
    memory 1: every frame is its own window."""
    i = 0
    while i < count:
        wid = (first + i) // memory
        end = min(count, (wid + 1) * memory - first)
        yield wid, i, end
        i = end


def run_vsr(net: BasicVSRNet, x: torch.Tensor, names=None, first: int = 0, direction="both", memory=1,
            flow=1.0, order="normal", seed=0, check=None) -> torch.Tensor:
    """x (B,3,H,W): one batch, frames in sequence order; names: their file
    names. Frames are only mixed within a window inside this batch, never
    across batches, and only when `names` is given (else every frame alone).
    Inside a window the frames are put in frame-number order from the
    names before anything else (then reversed or shuffled if asked), and
    every output goes back to the frame it came from: one out per one in."""
    from ..imageio import _natural_key
    from pathlib import Path
    b = x.shape[0]
    if names is None or len(names) != b:
        memory = 1
    memory = max(1, int(memory))
    outs = [None] * b
    for wid, a, e in plan_windows(first, b, memory):
        idx = list(range(a, e))
        if memory > 1:
            idx.sort(key=lambda k: _natural_key(Path(str(names[k]))))
            if order == "reverse":
                idx = idx[::-1]
            elif order == "shuffle":
                g = torch.Generator().manual_seed(hash((int(seed), wid)) & 0x7FFFFFFF)
                idx = [idx[k] for k in torch.randperm(len(idx), generator=g).tolist()]
        got, _ = net.run_window([x[k:k + 1] for k in idx], direction if memory > 1 else "both",
                                flow, None, check)
        for k, o in zip(idx, got):
            outs[k] = o
    return torch.cat(outs, 0).to(x.dtype)
