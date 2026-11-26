"""waifu2x and Real-CUGAN as plain PyTorch, from their original ncnn files.

The ncnn model files (param + bin, from nihui/waifu2x-ncnn-vulkan and
nihui/realcugan-ncnn-vulkan) are read directly and run as a small graph of
torch ops, so results match the ncnn programs, but on the Mac GPU.

Pre/post-processing follows each program: values / 255, pad by the model's
`prepadding` (plus alignment on the bottom and right), run, then keep the
top-left (w*scale, h*scale).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FP16_TAG = 0x01306B47


@dataclass
class _Layer:
    type: str
    name: str
    inputs: list[str]
    outputs: list[str]
    params: dict
    weights: dict


def _parse_params(tokens: list[str]) -> dict:
    p = {}
    for t in tokens:
        k, v = t.split("=", 1)
        k = int(k)
        if k <= -23300:  # array: -233xx=count,v1,v2...
            vals = v.split(",")
            p[-k - 23300] = [float(x) for x in vals[1:]]
        else:
            p[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    return p


class _Bin:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def raw_f32(self, n: int) -> np.ndarray:
        a = np.frombuffer(self.d, dtype="<f4", count=n, offset=self.o).copy()
        self.o += 4 * n
        return a

    def tagged(self, n: int) -> np.ndarray:
        (flag,) = struct.unpack_from("<I", self.d, self.o)
        self.o += 4
        if flag == FP16_TAG:
            a = np.frombuffer(self.d, dtype="<f2", count=n, offset=self.o).astype(np.float32)
            self.o += (2 * n + 3) // 4 * 4
            return a
        if flag == 0:
            return self.raw_f32(n)
        raise ValueError(f"unsupported ncnn weight storage flag {flag:#x}")


def load_ncnn(param_path: Path, bin_path: Path) -> list[_Layer]:
    lines = Path(param_path).read_text().split("\n")
    if lines[0].strip() != "7767517":
        raise ValueError(f"{param_path} is not an ncnn param file")
    nl, _ = map(int, lines[1].split())
    b = _Bin(Path(bin_path).read_bytes())
    layers = []
    for line in lines[2:2 + nl]:
        tok = line.split()
        typ, name, ni, no = tok[0], tok[1], int(tok[2]), int(tok[3])
        ins = tok[4:4 + ni]
        outs = tok[4 + ni:4 + ni + no]
        params = _parse_params(tok[4 + ni + no:])
        w = {}
        if typ in ("Convolution", "Deconvolution"):
            w["weight"] = b.tagged(params[6])
            if params.get(5, 0):
                w["bias"] = b.raw_f32(params[0])
        elif typ == "InnerProduct":
            w["weight"] = b.tagged(params[2])
            if params.get(1, 0):
                w["bias"] = b.raw_f32(params[0])
        elif typ == "Scale":
            if params.get(0, 0) != -233:
                raise ValueError("Scale with stored data is not used by waifu2x")
        elif typ not in ("Input", "Split", "Pooling", "Crop", "Eltwise", "BinaryOp", "PixelShuffle"):
            raise ValueError(f"ncnn layer type {typ} not supported")
        layers.append(_Layer(typ, name, ins, outs, params, w))
    if b.o != len(b.d):
        raise ValueError(f"{bin_path}: {len(b.d) - b.o} bytes left over; weights misread")
    return layers


def _act(x, p):
    a = p.get(9, 0)
    if a == 0:
        return x
    if a == 1:
        return F.relu(x)
    if a == 2:
        slope = p.get(10, [0.0])[0]
        return F.leaky_relu(x, slope)
    if a == 4:
        return torch.sigmoid(x)
    raise ValueError(f"activation {a} not supported")


class NcnnGraph(torch.nn.Module):
    def __init__(self, layers: list[_Layer]):
        super().__init__()
        self.layers = layers
        self._params = torch.nn.ParameterDict()
        for i, L in enumerate(layers):
            if L.type in ("Convolution", "Deconvolution"):
                k = L.params[1]
                kh = L.params.get(11, k)
                cout = L.params[0]
                n = L.params[6]
                cin = n // (cout * k * kh)
                wt = torch.from_numpy(L.weights["weight"])
                if L.type == "Convolution":
                    wt = wt.view(cout, cin, kh, k)
                else:
                    # ncnn keeps deconvolution weights as (out, in, kh, kw);
                    # torch's conv_transpose2d wants (in, out, kh, kw)
                    wt = wt.view(cout, cin, kh, k).transpose(0, 1).contiguous()
                self._params[f"w{i}"] = torch.nn.Parameter(wt, requires_grad=False)
                if "bias" in L.weights:
                    self._params[f"b{i}"] = torch.nn.Parameter(torch.from_numpy(L.weights["bias"]), requires_grad=False)
            elif L.type == "InnerProduct":
                cout = L.params[0]
                wt = torch.from_numpy(L.weights["weight"]).view(cout, -1)
                self._params[f"w{i}"] = torch.nn.Parameter(wt, requires_grad=False)
                if "bias" in L.weights:
                    self._params[f"b{i}"] = torch.nn.Parameter(torch.from_numpy(L.weights["bias"]), requires_grad=False)
        outs = [o for L in layers for o in L.outputs]
        self.output_name = "out0" if "out0" in outs else layers[-1].outputs[0]

    def forward(self, x):
        blobs = {}
        for i, L in enumerate(self.layers):
            p = L.params
            ins = [blobs[n] for n in L.inputs]
            t = L.type
            if t == "Input":
                y = x
            elif t == "Split":
                for o in L.outputs:
                    blobs[o] = ins[0]
                continue
            elif t == "Convolution":
                y = F.conv2d(ins[0], self._params[f"w{i}"], self._params[f"b{i}"] if f"b{i}" in self._params else None,
                             stride=p.get(3, 1), padding=p.get(4, 0), dilation=p.get(2, 1))
                y = _act(y, p)
            elif t == "Deconvolution":
                y = F.conv_transpose2d(ins[0], self._params[f"w{i}"],
                                       self._params[f"b{i}"] if f"b{i}" in self._params else None,
                                       stride=p.get(3, 1), padding=p.get(4, 0), dilation=p.get(2, 1))
                y = _act(y, p)
            elif t == "Pooling":
                if not p.get(4, 0):
                    raise ValueError("only global pooling is used by waifu2x")
                y = ins[0].mean(dim=(2, 3)) if p.get(0, 0) == 1 else ins[0].amax(dim=(2, 3))
            elif t == "InnerProduct":
                y = F.linear(ins[0], self._params[f"w{i}"], self._params[f"b{i}"] if f"b{i}" in self._params else None)
                y = _act(y, p)
            elif t == "Scale":
                y = ins[0] * ins[1].view(ins[1].shape[0], -1, 1, 1)
            elif t == "Crop":
                if len(ins) == 2:  # crop to the size of the second input
                    ref = ins[1]
                    wo, ho = p.get(0, 0), p.get(1, 0)
                    y = ins[0][..., ho:ho + ref.shape[-2], wo:wo + ref.shape[-1]]
                else:  # starts / ends / axes arrays (ncnn axes: -2 = h, -1 = w)
                    y = ins[0]
                    starts = [int(v) for v in p.get(9, [])]
                    ends = [int(v) for v in p.get(10, [])]
                    axes = [int(v) for v in p.get(11, [])]
                    for st, en, ax in zip(starts, ends, axes):
                        dim = {-2: 2, -1: 3, 1: 2, 2: 3}[ax]
                        n = y.shape[dim]
                        en = n + en if en <= 0 else min(en, n)
                        y = y.narrow(dim, st, en - st)
            elif t == "BinaryOp":
                a, b = ins[0], ins[1]
                if a.dim() == 2 and b.dim() == 4:
                    a = a.view(a.shape[0], -1, 1, 1)
                if b.dim() == 2 and a.dim() == 4:
                    b = b.view(b.shape[0], -1, 1, 1)
                op = p.get(0, 0)
                if op == 0:
                    y = a + b
                elif op == 1:
                    y = a - b
                elif op == 2:
                    y = a * b
                else:
                    raise ValueError(f"BinaryOp {op} not supported")
            elif t == "PixelShuffle":
                if p.get(1, 0):
                    raise ValueError("PixelShuffle mode 1 not supported")
                y = F.pixel_shuffle(ins[0], int(p[0]))
            elif t == "Eltwise":
                if p.get(0, 1) != 1 or 1 in p:
                    raise ValueError("only plain Eltwise sum is used by waifu2x")
                y = ins[0] + ins[1]
            else:
                raise ValueError(t)
            blobs[L.outputs[0]] = y
        return blobs[self.output_name]


VARIANTS = {
    "cunet": "models-cunet",
    "anime": "models-upconv_7_anime_style_art_rgb",
    "photo": "models-upconv_7_photo",
}


def model_files(variant: str, noise: int, scale: int) -> tuple[str, str]:
    """Relative (param, bin) paths inside the waifu2x model folder."""
    d = VARIANTS[variant]
    if scale == 1:
        if variant != "cunet" or noise < 0:
            raise ValueError("denoise-only (x1) exists for cunet with noise 0-3")
        stem = f"noise{noise}_model"
    else:
        stem = "scale2.0x_model" if noise < 0 else f"noise{noise}_scale2.0x_model"
    return f"{d}/{stem}.param", f"{d}/{stem}.bin"


class Waifu2x(torch.nn.Module):
    """Callable on (N, 3, H, W) encoded floats in 0..1 -> (N, 3, H*s, W*s)."""

    def __init__(self, root: Path, variant: str, noise: int, scale: int = 2):
        super().__init__()
        param, binf = model_files(variant, noise, scale)
        self.graph = NcnnGraph(load_ncnn(Path(root) / param, Path(root) / binf))
        self.scale = scale
        self.variant = variant
        if variant == "cunet":
            self.prepadding = 28 if scale == 1 else 18
            # The ncnn binary aligns x1 to 2, which loses the last row or
            # column for some sizes; 4 always gives the full frame.
            self.align = 4
        else:
            self.prepadding = 7
            self.align = 1

    @torch.no_grad()
    def forward(self, x):
        n, c, h, w = x.shape
        p, a = self.prepadding, self.align
        pb = p + ((h + a - 1) // a * a - h)
        pr = p + ((w + a - 1) // a * a - w)
        xp = F.pad(x, (p, pr, p, pb), mode="replicate")
        y = self.graph(xp)
        y = y[..., :h * self.scale, :w * self.scale]
        # the ncnn binary writes v*255+0.5 truncated to uint8, i.e. rounded
        return y.clamp(0, 1)


REALCUGAN_SETS = {"se": "models-se", "pro": "models-pro", "nose": "models-nose"}


def realcugan_files(kind: str, scale: int, mode: str) -> tuple[str, str]:
    """mode: 'conservative' | 'no-denoise' | 'denoise1x' | 'denoise2x' | 'denoise3x'."""
    d = REALCUGAN_SETS[kind]
    stem = f"up{scale}x-{mode}"
    return f"{d}/{stem}.param", f"{d}/{stem}.bin"


class RealCUGAN(torch.nn.Module):
    """Real-CUGAN from the official ncnn model files (nihui/realcugan-ncnn-vulkan),
    with that program's pre/post-processing (whole frame, no tiling)."""

    def __init__(self, root: Path, kind: str, scale: int, mode: str):
        super().__init__()
        param, binf = realcugan_files(kind, scale, mode)
        self.graph = NcnnGraph(load_ncnn(Path(root) / param, Path(root) / binf))
        self.scale = scale
        self.prepadding = {2: 18, 3: 14, 4: 19}[scale]
        self.align = 4 if scale == 3 else 2

    @torch.no_grad()
    def forward(self, x):
        n, c, h, w = x.shape
        p, a = self.prepadding, self.align
        pb = p + ((h + a - 1) // a * a - h)
        pr = p + ((w + a - 1) // a * a - w)
        xp = _pad_reflect(x, p, pr, p, pb)
        y = self.graph(xp)[..., :h * self.scale, :w * self.scale]
        if self.scale == 4:  # the 4x net predicts a residual over nearest
            y = y + x.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
        return y.clamp(0, 1)


def _pad_reflect(x, l, r, t, b):
    """Reflect padding (ncnn BORDER_REFLECT) that also works when the pad is
    larger than the image, by reflecting repeatedly."""
    while l or r or t or b:
        h, w = x.shape[-2:]
        sl, sr = min(l, w - 1), min(r, w - 1)
        st, sb = min(t, h - 1), min(b, h - 1)
        if not (sl or sr or st or sb):  # 1-pixel image: replicate
            return F.pad(x, (l, r, t, b), mode="replicate")
        x = F.pad(x, (sl, sr, st, sb), mode="reflect")
        l, r, t, b = l - sl, r - sr, t - st, b - sb
    return x
