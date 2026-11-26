"""Pass recipes: what a pass does, as plain JSON-able data.

A pass is a pattern of cycles (A, B, C ...) that alternate in order.
`iterations` counts single cycles: A, B, A, B ... until `iterations` cycles
have run. Every `save_every` cycles (and after the last one) the frames are
saved.

A cycle is a list of steps. Each step picks a width (height follows the
source aspect) and a method. Cycle 1 starts from the source size; after a
cycle's last step the frame is landed exactly on the source size with an
area resize if it is not there already.

Step kinds
----------
{"op": "resize", "method": "lanczos", "width": 320}
    plain kernel; optional "antialias": false, "fixed_point": true,
    "phase_bits": 4, "coef_bits": 6
{"op": "model", "model": "<model id>", "width": 160}
    run the model at its fixed factor, then area-resize to `width`
    ("width" may be omitted: keep the model's own output size)

`width` may be the string "source" for the source width.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Bump when pixel output for the same recipe would change.
ENGINE_PIXELS_VERSION = 1

KERNEL_SHORT = {"nearest": "near", "area": "area", "bilinear": "bil",
                "bicubic": "bic", "lanczos": "lanc"}


class RecipeError(ValueError):
    pass


@dataclass
class Recipe:
    cycles: dict[str, list[dict[str, Any]]]
    pattern: list[str]
    iterations: int
    save_every: int
    rounding: str = "srgb"          # "srgb" | "linear"
    light: str = "linear"           # kernel steps: "linear" | "encoded"
    output: dict[str, Any] = field(default_factory=lambda: {
        "format": "webp", "quality": 95, "sharp_yuv": True})
    name: str = ""                  # optional label, shown in the slug

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        known = {"cycles", "pattern", "iterations", "save_every", "rounding",
                 "light", "output", "name"}
        extra = set(d) - known - {"version", "code", "slug"}
        if extra:
            raise RecipeError(f"unknown recipe keys: {sorted(extra)}")
        r = cls(
            cycles={k: [dict(s) for s in v] for k, v in d["cycles"].items()},
            pattern=list(d["pattern"]),
            iterations=int(d["iterations"]),
            save_every=int(d["save_every"]),
            rounding=d.get("rounding", "srgb"),
            light=d.get("light", "linear"),
            output=dict(d.get("output") or {"format": "webp", "quality": 95, "sharp_yuv": True}),
            name=d.get("name", ""),
        )
        r.validate()
        return r

    @classmethod
    def load(cls, path) -> "Recipe":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "version": ENGINE_PIXELS_VERSION,
            "name": self.name,
            "cycles": copy.deepcopy(self.cycles),
            "pattern": list(self.pattern),
            "iterations": self.iterations,
            "save_every": self.save_every,
            "rounding": self.rounding,
            "light": self.light,
            "output": dict(self.output),
            "code": self.code(),
            "slug": self.slug(),
        }

    # ------------------------------------------------------------ checks
    def validate(self, known_models: set[str] | None = None) -> None:
        if not self.pattern:
            raise RecipeError("pattern is empty")
        for c in self.pattern:
            if c not in self.cycles:
                raise RecipeError(f"pattern uses cycle {c!r}, which is not defined")
        if self.iterations < 1:
            raise RecipeError("iterations must be >= 1")
        if self.save_every < 1:
            raise RecipeError("save_every must be >= 1")
        if self.rounding not in ("srgb", "linear"):
            raise RecipeError("rounding must be 'srgb' or 'linear'")
        if self.light not in ("linear", "encoded"):
            raise RecipeError("light must be 'linear' or 'encoded'")
        fmt = self.output.get("format", "webp")
        if fmt not in ("webp", "png"):
            raise RecipeError("output format must be 'webp' or 'png'")
        for name, steps in self.cycles.items():
            if not steps:
                raise RecipeError(f"cycle {name!r} has no steps")
            for i, s in enumerate(steps):
                where = f"cycle {name} step {i + 1}"
                op = s.get("op")
                w = s.get("width")
                if w is not None and w != "source" and (not isinstance(w, int) or w < 1):
                    raise RecipeError(f"{where}: width must be a positive integer or 'source'")
                if op == "resize":
                    from .kernels import METHODS
                    if s.get("method") not in METHODS:
                        raise RecipeError(f"{where}: unknown kernel {s.get('method')!r}")
                    if w is None:
                        raise RecipeError(f"{where}: resize needs a width")
                elif op == "model":
                    m = s.get("model")
                    if not m:
                        raise RecipeError(f"{where}: model step needs a model id")
                    if known_models is not None and m not in known_models:
                        raise RecipeError(f"{where}: unknown model {m!r}")
                else:
                    raise RecipeError(f"{where}: unknown op {op!r}")

    # ------------------------------------------------------- identity
    def pixel_identity(self) -> dict:
        """Everything that decides the saved pixels (not how many/when)."""
        used = {k: self.cycles[k] for k in sorted(set(self.pattern))}
        return {
            "v": ENGINE_PIXELS_VERSION,
            "cycles": used,
            "pattern": self.pattern,
            "rounding": self.rounding,
            "light": self.light,
            "output": self.output,
        }

    def code(self) -> str:
        blob = json.dumps(self.pixel_identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:4]

    def uses_gan(self, gan_models: set[str]) -> bool:
        return any(s.get("op") == "model" and s.get("model") in gan_models
                   for k in set(self.pattern) for s in self.cycles[k])

    def slug(self, gan_models: set[str] | None = None, max_len: int = 72) -> str:
        parts = []
        for k in _unique(self.pattern):
            parts.append("-".join(_step_slug(s) for s in self.cycles[k]))
        body = "+".join(parts)
        flags = []
        if gan_models and self.uses_gan(gan_models):
            flags.append("GAN")
        if self.rounding == "linear":
            flags.append("rlin")
        if self.light == "encoded":
            flags.append("enc")
        prefix = _clean(self.name) + "_" if self.name else ""
        tail = ("_" + "_".join(flags)) if flags else ""
        room = max(8, max_len - len(prefix) - len(tail))
        if len(body) > room:
            body = body[:room - 1] + "~"
        return prefix + body + tail

    def folder_name(self, gan_models: set[str] | None = None) -> str:
        return f"{self.slug(gan_models)}_{self.code()}"

    def cycle_at(self, index: int) -> str:
        """Cycle name for 1-based cycle index."""
        return self.pattern[(index - 1) % len(self.pattern)]

    def save_points(self) -> list[int]:
        pts = list(range(self.save_every, self.iterations + 1, self.save_every))
        if not pts or pts[-1] != self.iterations:
            pts.append(self.iterations)
        return pts


def _unique(seq):
    seen = []
    for x in seq:
        if x not in seen:
            seen.append(x)
    return seen


def _clean(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in s.strip())
    return out.strip("-")[:30]


def _step_slug(s: dict) -> str:
    w = s.get("width")
    ws = "S" if w == "source" else ("" if w is None else str(w))
    if s.get("op") == "resize":
        m = KERNEL_SHORT.get(s["method"], s["method"])
        if s.get("antialias") is False:
            m += "NA"
        if s.get("fixed_point"):
            m += "FP"
        return f"{m}{ws}"
    if s.get("op") == "model":
        return f"{model_short(s['model'])}@{ws}" if ws else model_short(s['model'])
    return f"{s.get('op')}{ws}"


_SHORT = [("realesrgan-x4plus-anime", "esrAnime4"), ("realesrgan-x4plus", "esrgan4"),
          ("realesrgan-x2plus", "esrgan2"), ("realesrnet-x4", "esrnet4"),
          ("realesr-animevideov3", "avv3"), ("realesr-general-wdn-x4v3", "genwdn4"),
          ("realesr-general-x4v3", "gen4"), ("swinir-classical-", "swin"),
          ("w2x-cunet-denoise-", "w2xdn"), ("w2x-", "w2x"), ("rcugan-", "cug"), ("dropin-", "d")]


def model_short(mid: str) -> str:
    """Short readable model code for folder names, e.g. w2xcunetn1, avv3, cugx2dn3."""
    for pre, short in _SHORT:
        if mid.startswith(pre):
            rest = mid[len(pre):]
            return (short + "".join(ch for ch in rest if ch.isalnum()))[:14]
    return "".join(ch for ch in mid if ch.isalnum())[:14]
