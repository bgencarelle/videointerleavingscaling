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
Half precision (the pass's precision switch) is used only if it passes a
test run against full precision when the network loads.
Frames are replicate-padded to a multiple of the scale and cropped after.

Engines: the first time a pass shrinks or enlarges frames of a given size,
IRN tries the ways it can run on this Mac on a few of those frames, checks
each against PyTorch's own result, times them, and keeps the fastest:
PyTorch on the GPU (as stored, or channels-last), and on macOS with
coremltools, the network converted to Core ML for the Neural Engine
(half precision only; the Neural Engine has no float32) or for Core ML's
GPU path. Converted networks are kept in weights/irn/coreml/.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

    # The authors do this as a grouped stride-2 convolution plus reshapes that
    # depend on the frame size. Same sums here as pixel (un)shuffle plus a
    # fixed 1x1 mixing convolution: no size arithmetic, so the network can be
    # traced at a fixed size for Core ML, and it's one plain conv on the GPU.
    def _mix(self, rev: bool, like: torch.Tensor) -> torch.Tensor:
        key = (rev, like.dtype, like.device)
        cache = self.__dict__.setdefault("_mix_cache", {})
        if key not in cache:
            c = self.channel_in
            # built on the CPU in float64 (MPS has no float64), then moved
            m = self.haar_weights.detach()[:4, 0].reshape(4, 4).cpu().double()  # [filter, 2x2 position]
            eye = torch.eye(c, dtype=torch.float64, device=m.device)
            # forward: out[f*c + ch] = sum_k m[f, k] * in[ch*4 + k] / 4
            fwd = torch.einsum("fk,gc->fgck", m, eye).reshape(4 * c, 4 * c) / 4.0
            w = fwd.t() * 4.0 if rev else fwd  # m m^T = 4 I, so the inverse is 4 * fwd^T
            cache[key] = w.reshape(4 * c, 4 * c, 1, 1).to(device=like.device, dtype=like.dtype)
        return cache[key]

    def forward(self, x, rev=False):
        if not rev:
            return F.conv2d(F.pixel_unshuffle(x, 2), self._mix(False, x))
        return F.pixel_shuffle(F.conv2d(x, self._mix(True, x)), 2)

    def _apply(self, fn, *args, **kwargs):  # .to() / .half(): rebuild the mixing matrices
        self.__dict__.pop("_mix_cache", None)
        return super()._apply(fn, *args, **kwargs)


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


class _Down(nn.Module):
    """Shrink at a fixed (already padded) size, for Core ML."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[:, :3]


class _Up(nn.Module):
    """Enlarge at a fixed size with z = 0, for Core ML."""

    def __init__(self, net, scale, h, w):
        super().__init__()
        self.net = net
        self.register_buffer("z", torch.zeros(1, 3 * (scale * scale - 1), h, w))

    def forward(self, x):
        return self.net(torch.cat((x, self.z), 1), rev=True)


@dataclass
class Engine:
    name: str                       # "torch", "torch-cl", "coreml-ne", ...
    label: str                      # for the log: "the Neural Engine (float16)"
    fn: Callable[[torch.Tensor], torch.Tensor]
    ms: float | None = None         # per frame, when it was timed


ACCURACY = {"fp16": 2 / 255, "fp32": 0.25 / 255}  # mean difference allowed vs PyTorch
SETUP = {"seconds": 0.0}  # total time spent choosing engines (converting, test runs), this process


class IRN(nn.Module):
    def __init__(self, path: Path, scale: int):
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("IRN scale must be 2, 4 or 8")
        self.path = Path(path)
        self.engines: dict[tuple, Engine] = {}
        self.notes: list[str] = []      # what was chosen, for the pass log
        self.precision = "fp32"         # requested: "fp16" allows the Neural Engine
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
    micro = 8
    tile: int | None = None
    dtype = torch.float32          # float16 when half precision is on and passes its test
    precision_label = "float32"

    @torch.no_grad()
    def _down_raw(self, x, net=None):
        s = self.scale
        h, w = x.shape[-2:]
        ph, pw = (-h) % s, (-w) % s
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return (net or self.net)(x)[:, :3][..., : -(-h // s), : -(-w // s)]

    @torch.no_grad()
    def _up_raw(self, x, net=None):
        s = self.scale
        n, c, h, w = x.shape
        z = torch.zeros(n, c * (s * s - 1), h, w, device=x.device, dtype=x.dtype)
        return (net or self.net)(torch.cat((x, z), 1), rev=True)

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

    def make_half(self, device) -> None:
        """Half precision if it works here: the network in float16, checked on a
        test frame against full precision (shrink then enlarge); else stays fp32."""
        x = torch.rand(1, 3, 64, 64, device=device)
        ref = self._up_raw(self._down_raw(x))
        self.net.half()
        self.dtype = torch.float16
        try:
            got = self._up_raw(self._down_raw(x.half())).float()
            if device.type == "mps":
                torch.mps.synchronize()
            ok = bool(torch.isfinite(got).all()) and float((got - ref).abs().mean()) < 2 / 255
        except (RuntimeError, TypeError, NotImplementedError):
            ok = False
        self.precision = "fp16"  # asked for: lets Core ML try half precision even if PyTorch's failed
        if ok:
            self.precision_label = "float16"
        else:
            self.net.float()
            self.dtype = torch.float32
            self.precision_label = "float32 (half precision didn't hold up for IRN here)"

    def _safe(self, fn, x, down: bool):
        x = x.to(self.dtype)
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
            outs.append(y.float())
            i += chunk.shape[0]
        return torch.cat(outs, 0)

    def down(self, x: torch.Tensor) -> torch.Tensor:
        return self._run("down", x)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        return self._run("up", x)

    # ------------------------------------------------------------ engines
    def _run(self, d: str, x: torch.Tensor) -> torch.Tensor:
        key = (d, x.shape[-2], x.shape[-1])
        eng = self.engines.get(key)
        if eng is None:
            eng = self.engines[key] = self._choose(d, x)
        return eng.fn(x)

    def _torch_engine(self, d: str) -> Engine:
        raw, down = (self._down_raw, True) if d == "down" else (self._up_raw, False)
        return Engine("torch", f"PyTorch on {self._device()} ({self.precision_label.split(' ')[0]})",
                      lambda x: self._safe(raw, x, down))

    def _device(self):
        return next(self.net.parameters()).device

    def _candidates(self, d: str, h: int, w: int):
        """(name, label, builder) for every other way to run this here."""
        dev = self._device()
        if dev.type == "mps":
            yield "torch-cl", "PyTorch on mps, channels-last", lambda: self._channels_last_engine(d)
        from .coreml import UNIT_LABEL, unavailable_reason
        if dev.type == "cpu" or unavailable_reason() is not None:
            return
        if self.precision == "fp16":
            for units in ("ne", "all"):
                yield (f"coreml-{units}", f"{UNIT_LABEL[units]} (float16)",
                       lambda u=units: self._coreml_engine(d, h, w, "fp16", u))
        else:
            yield "coreml-gpu", f"{UNIT_LABEL['gpu']} (float32)", lambda: self._coreml_engine(d, h, w, "fp32", "gpu")

    def _channels_last_engine(self, d: str) -> Engine:
        net = self.__dict__.get("_cl_net")  # not a submodule: stays out of state_dict
        if net is None:
            net = self.__dict__["_cl_net"] = copy.deepcopy(self.net).to(memory_format=torch.channels_last)
        cl = torch.channels_last
        if d == "down":
            return Engine("torch-cl", "", lambda x: self._safe(
                lambda t: self._down_raw(t.contiguous(memory_format=cl), net), x, True))
        return Engine("torch-cl", "", lambda x: self._safe(
            lambda t: self._up_raw(t.contiguous(memory_format=cl), net), x, False))

    def _coreml_engine(self, d: str, h: int, w: int, precision: str, units: str) -> Engine:
        from .coreml import FORMAT_VERSION, CoreMLRunner, convert
        s = self.scale
        st = self.path.stat()
        tag = hashlib.sha256(f"{self.path.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]
        if d == "down":
            ph, pw = h + (-h) % s, w + (-w) % s
            module, shape = _Down(copy.deepcopy(self.net)), (1, 3, ph, pw)
        else:
            module, shape = _Up(copy.deepcopy(self.net), s, h, w), (1, 3, h, w)
        name = f"{self.path.stem}-{tag}-{d}-{shape[3]}x{shape[2]}-{precision}-v{FORMAT_VERSION}.mlpackage"
        pkg = convert(module, shape, self.path.parent / "coreml" / name, precision)
        run = CoreMLRunner(pkg, units, precision)
        if d == "down":
            def fn(x):
                if ph != h or pw != w:
                    x = F.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
                return run(x)  # (ph/s, pw/s) is already ceil(h/s), ceil(w/s)
        else:
            fn = run
        return Engine(f"coreml-{units}", "", fn)

    def _choose(self, d: str, x: torch.Tensor) -> Engine:
        """Try every engine on a few of these frames; keep the fastest that
        matches PyTorch's result."""
        h, w = x.shape[-2:]
        best = self._torch_engine(d)
        cands = list(self._candidates(d, h, w))
        if not cands:
            return best
        t_setup = time.time()
        try:
            return self._choose_from(d, x, best, cands)
        finally:
            SETUP["seconds"] += time.time() - t_setup

    def _choose_from(self, d, x, best, cands) -> Engine:
        h, w = x.shape[-2:]
        sample = x[: min(4, x.shape[0])].float()
        ref, best.ms = _timed(best.fn, sample)
        torch_ms = best.ms
        tried = []
        tol = ACCURACY[self.precision]
        for name, label, build in cands:
            try:
                t0 = time.time()
                eng = build()
                eng.label = label
                setup = time.time() - t0
                out, eng.ms = _timed(eng.fn, sample)
                diff = float((out.float() - ref).abs().mean()) if out.shape == ref.shape else float("inf")
                if not torch.isfinite(out).all() or diff > tol:
                    tried.append(f"{label}: off by {diff * 255:.2f}/255 on average, not used")
                    continue
                tried.append(f"{label}: {eng.ms:.1f} ms, off by {diff * 255:.2f}/255 on average"
                             + (f", set up in {setup:.0f} s" if setup >= 2 else ""))
                if eng.ms < best.ms * 0.95:
                    best = eng
            except Exception as e:  # noqa: BLE001
                tried.append(f"{label}: couldn't run ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})")
        self.notes.append(f"IRN x{self.scale} {'shrink' if d == 'down' else 'enlarge'} of {w}x{h} frames runs on "
                          f"{best.label}: {best.ms:.1f} ms per frame (PyTorch {torch_ms:.1f} ms; tried "
                          + "; ".join(tried) + ")")
        return best


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _timed(fn, x: torch.Tensor, repeats: int = 2) -> tuple[torch.Tensor, float]:
    """(output, ms per frame): one warm-up call on a frame, then the best of
    `repeats` calls on all of x."""
    fn(x[:1])
    _sync(x.device)
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(x)
        _sync(x.device)
        best = min(best, time.perf_counter() - t0)
    return out, best / x.shape[0] * 1000


def _free(device):
    try:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
