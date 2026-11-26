"""Tone curves, linear light and 8-bit rounding.

Frames live as float32 tensors holding *encoded* values (what an image file
stores) in 0..1. Kernel steps convert to linear light, resize, and convert
back. Model steps see encoded values, because that is what they were
trained on.

The tone curve comes from the source ICC profile (rTRC/gTRC/bTRC tags:
'curv' or 'para'). Profiles without readable curves fall back to sRGB, and
the fallback is reported so the run log can say so.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------
# Curves. Each maps encoded -> linear (decode) and back (encode), on tensors.
# --------------------------------------------------------------------------

class Curve:
    name = "curve"

    def decode(self, x: torch.Tensor) -> torch.Tensor:  # encoded -> linear
        raise NotImplementedError

    def encode(self, y: torch.Tensor) -> torch.Tensor:  # linear -> encoded
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class SRGBCurve(Curve):
    name = "sRGB"

    def decode(self, x):
        x = x.clamp(0.0, 1.0)
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def encode(self, y):
        y = y.clamp(0.0, 1.0)
        return torch.where(y <= 0.0031308, y * 12.92, 1.055 * y.pow(1 / 2.4) - 0.055)


class GammaCurve(Curve):
    def __init__(self, gamma: float):
        self.gamma = float(gamma)
        self.name = f"gamma {self.gamma:.4g}"

    def decode(self, x):
        return x.clamp(0.0, 1.0) ** self.gamma

    def encode(self, y):
        return y.clamp(0.0, 1.0) ** (1.0 / self.gamma)


class _LUTCurve(Curve):
    """Curve defined by an exact float64 decode function, evaluated through
    float32 lookup tables on the frame's device (the Mac GPU has no float64).

    decode: table uniform in the encoded value.
    encode: table uniform in sqrt(linear), so the steep start of gamma-like
            inverses is sampled densely.
    """
    K = 65536

    def _decode64(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _tables(self, device):
        cache = self.__dict__.setdefault("_lut_cache", {})
        key = str(device)
        if key not in cache:
            k = self.K
            xs = torch.linspace(0, 1, k, dtype=torch.float64)
            dec = self._decode64(xs).clamp(0, 1)
            dec = torch.cummax(dec, dim=0).values  # force monotonic
            # inverse: for linear targets y = s^2, find x with dec(x) = y
            s = torch.linspace(0, 1, k, dtype=torch.float64)
            y = (s * s).clamp(float(dec[0]), float(dec[-1]))
            idx = torch.searchsorted(dec.contiguous(), y.contiguous()).clamp(1, k - 1)
            y0, y1 = dec[idx - 1], dec[idx]
            span = y1 - y0
            frac = torch.where(span > 0, (y - y0) / torch.where(span > 0, span, torch.ones_like(span)),
                               torch.zeros_like(y))
            enc = (xs[idx - 1] + frac * (xs[idx] - xs[idx - 1])).clamp(0, 1)
            cache[key] = (dec.float().to(device), enc.float().to(device))
        return cache[key]

    @staticmethod
    def _lookup(table: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        n = table.numel() - 1
        p = pos.clamp(0, 1) * n
        i0 = p.floor().clamp(0, n - 1)
        frac = p - i0
        i0 = i0.long()
        return table[i0] * (1 - frac) + table[i0 + 1] * frac

    def decode(self, x):
        dec, _ = self._tables(x.device)
        return self._lookup(dec, x)

    def encode(self, y):
        _, enc = self._tables(y.device)
        return self._lookup(enc, y.clamp(0, 1).sqrt())


class ParametricCurve(_LUTCurve):
    """ICC 'para' curve, function types 0-4."""

    def __init__(self, ftype: int, p: list[float]):
        self.ftype = ftype
        self.p = p
        self.name = f"para{ftype}(" + ",".join(f"{v:.4g}" for v in p) + ")"

    def _decode64(self, x):
        t, p = self.ftype, self.p
        g = p[0]
        if t == 0:
            return x ** g
        if t == 1:
            a, b = p[1], p[2]
            return torch.where(x >= -b / a, (a * x + b).clamp(min=0) ** g, torch.zeros_like(x))
        if t == 2:
            a, b, c = p[1], p[2], p[3]
            return torch.where(x >= -b / a, (a * x + b).clamp(min=0) ** g + c, torch.full_like(x, c))
        if t == 3:
            a, b, c, d = p[1], p[2], p[3], p[4]
            return torch.where(x >= d, (a * x + b).clamp(min=0) ** g, c * x)
        if t == 4:
            a, b, c, d, e, f = p[1], p[2], p[3], p[4], p[5], p[6]
            return torch.where(x >= d, (a * x + b).clamp(min=0) ** g + e, c * x + f)
        raise ValueError(f"unknown para type {t}")


class TableCurve(_LUTCurve):
    """ICC 'curv' with a lookup table."""

    def __init__(self, table: list[int]):
        self._t = torch.tensor(table, dtype=torch.float64) / 65535.0
        self.name = f"table{len(table)}"

    def _decode64(self, x):
        t = self._t
        n = len(t) - 1
        p = x * n
        i0 = p.floor().clamp(0, n - 1).long()
        frac = p - i0.double()
        return t[i0] * (1 - frac) + t[i0 + 1] * frac


# --------------------------------------------------------------------------
# ICC parsing (only what we need: the three tone curves)
# --------------------------------------------------------------------------

def _s15f16(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536.0


def _parse_curve(data: bytes, off: int) -> Curve | None:
    sig = data[off:off + 4]
    if sig == b"curv":
        (count,) = struct.unpack(">I", data[off + 8:off + 12])
        if count == 0:
            return GammaCurve(1.0)
        if count == 1:
            (g,) = struct.unpack(">H", data[off + 12:off + 14])
            return GammaCurve(g / 256.0)
        table = list(struct.unpack(f">{count}H", data[off + 12:off + 12 + 2 * count]))
        return TableCurve(table)
    if sig == b"para":
        (ftype,) = struct.unpack(">H", data[off + 8:off + 10])
        nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(ftype)
        if nparams is None:
            return None
        p = [_s15f16(data[off + 12 + 4 * i:off + 16 + 4 * i]) for i in range(nparams)]
        # sRGB written as para3 is common: keep the analytic version
        if ftype == 3 and abs(p[0] - 2.4) < 1e-3 and abs(p[1] - 1 / 1.055) < 1e-3 \
                and abs(p[3] - 1 / 12.92) < 1e-3 and abs(p[4] - 0.04045) < 2e-3:
            return SRGBCurve()
        return ParametricCurve(ftype, p)
    return None


def curves_from_icc(icc: bytes | None) -> tuple[list[Curve], str | None]:
    """Return ([r, g, b] curves, warning or None)."""
    if not icc:
        return [SRGBCurve()] * 3, "no colour profile: assuming sRGB"
    try:
        (tag_count,) = struct.unpack(">I", icc[128:132])
        tags = {}
        for i in range(tag_count):
            sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i:144 + 12 * i])
            tags[sig] = off
        keys = [b"rTRC", b"gTRC", b"bTRC"]
        if all(k in tags for k in keys):
            curves = [_parse_curve(icc, tags[k]) for k in keys]
        elif b"kTRC" in tags:
            c = _parse_curve(icc, tags[b"kTRC"])
            curves = [c, c, c]
        else:
            return [SRGBCurve()] * 3, "colour profile has no tone curves: assuming sRGB"
        if any(c is None for c in curves):
            return [SRGBCurve()] * 3, "colour profile tone curves unreadable: assuming sRGB"
        return curves, None
    except Exception as e:  # malformed profile
        return [SRGBCurve()] * 3, f"colour profile unreadable ({e}): assuming sRGB"


# --------------------------------------------------------------------------
# Transfer: applies per-channel curves to (N, 3, H, W) tensors
# --------------------------------------------------------------------------

@dataclass
class Transfer:
    curves: list[Curve] = field(default_factory=lambda: [SRGBCurve()] * 3)

    @property
    def same(self) -> bool:
        return all(c is self.curves[0] for c in self.curves)

    def describe(self) -> str:
        names = [c.describe() for c in self.curves]
        return names[0] if len(set(names)) == 1 else "/".join(names)

    def _apply(self, x: torch.Tensor, fn: str) -> torch.Tensor:
        if self.same:
            return getattr(self.curves[0], fn)(x)
        return torch.cat([getattr(c, fn)(x[:, i:i + 1]) for i, c in enumerate(self.curves)], dim=1)

    def decode(self, x):
        return self._apply(x, "decode")

    def encode(self, y):
        return self._apply(y, "encode")


def quantize(x: torch.Tensor, mode: str, transfer: Transfer) -> torch.Tensor:
    """Round encoded frames to 8 bits, as if saved after a step.

    mode "srgb":   round the encoded values (like saving a normal 8-bit file).
    mode "linear": round in linear light (harsh shadow banding); the result is
                   kept as encoded floats so the next step sees exactly those
                   linear values.
    """
    if mode == "srgb":
        return torch.round(x.clamp(0, 1) * 255.0) / 255.0
    if mode == "linear":
        lin = transfer.decode(x)
        lin = torch.round(lin.clamp(0, 1) * 255.0) / 255.0
        return transfer.encode(lin)
    raise ValueError(f"unknown rounding mode {mode!r}")
