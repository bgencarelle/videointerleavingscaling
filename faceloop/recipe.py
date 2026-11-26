"""Pass recipes: what a pass does, as plain JSON-able data.

Shape of a pass
---------------
lead_in   steps run once, taking the source frames down to the *loop size*
          (the size after the lead-in).
cycles    A, B, C ... alternate in `pattern` order. Every cycle starts and
          ends at the loop size: if a cycle's last step doesn't end there,
          an area resize lands it. `iterations` counts single cycles.
finish    takes a *copy* of the frames back to the source size at every save
          point (every `save_every` cycles and after the last). The loop
          itself carries on from the small frames, untouched.
          "auto" (default): RealESRNet x4 while `finish_threshold` (default
          2x) or more is still needed (overshooting is fine), then area for
          the rest. Or a list of your own steps.
          Either way an area resize lands exactly on the source size.

Step kinds
----------
{"op": "resize", "method": "lanczos", "width": 320}
    plain kernel; optional "antialias": false, "fixed_point": true,
    "phase_bits": 4, "coef_bits": 6
{"op": "model", "model": "<model id>", "width": 160}
    run the model at its fixed factor, then area-resize to `width`
    ("width" may be omitted: keep the model's own output size)

`width` is pixels, or "source" (source width), or "loop" (the loop size;
not allowed in the lead-in, which defines it).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Bump when pixel output for the same recipe would change.
ENGINE_PIXELS_VERSION = 3

KERNEL_SHORT = {"nearest": "near", "area": "area", "bilinear": "bil",
                "bicubic": "bic", "lanczos": "lanc"}

AUTO_FINISH_MODEL = "realesrnet-x4"  # accuracy-trained, upscaling only, x4


class RecipeError(ValueError):
    pass


@dataclass
class Recipe:
    lead_in: list[dict[str, Any]]
    cycles: dict[str, list[dict[str, Any]]]
    pattern: list[str]
    iterations: int
    save_every: int
    finish: Any = "auto"            # "auto" | list of steps
    finish_threshold: float = 2.0   # auto finish: AI x2 only while this much is left
    rounding: str = "srgb"          # "srgb" | "linear"
    light: str = "linear"           # kernel steps: "linear" | "encoded"
    output: dict[str, Any] = field(default_factory=lambda: {
        "format": "webp", "quality": 95, "sharp_yuv": True})
    name: str = ""                  # optional label, shown in the slug

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        known = {"lead_in", "cycles", "pattern", "iterations", "save_every", "finish",
                 "finish_threshold", "rounding", "light", "output", "name"}
        extra = set(d) - known - {"version", "code", "slug", "engine", "source"}
        if extra:
            raise RecipeError(f"unknown recipe keys: {sorted(extra)}")
        if "lead_in" not in d:
            raise RecipeError("recipe needs a lead_in (steps from the source down to the loop size)")
        fin = d.get("finish", "auto")
        r = cls(
            lead_in=[dict(s) for s in d["lead_in"]],
            cycles={k: [dict(s) for s in v] for k, v in d["cycles"].items()},
            pattern=list(d["pattern"]),
            iterations=int(d["iterations"]),
            save_every=int(d["save_every"]),
            finish=fin if fin == "auto" else [dict(s) for s in fin],
            finish_threshold=float(d.get("finish_threshold", 2.0)),
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
            "lead_in": copy.deepcopy(self.lead_in),
            "cycles": copy.deepcopy(self.cycles),
            "pattern": list(self.pattern),
            "iterations": self.iterations,
            "save_every": self.save_every,
            "finish": copy.deepcopy(self.finish),
            "finish_threshold": self.finish_threshold,
            "rounding": self.rounding,
            "light": self.light,
            "output": dict(self.output),
            "code": self.code(),
            "slug": self.slug(),
        }

    # ------------------------------------------------------------ checks
    def all_steps(self):
        """(where, step) for every step the recipe can run (auto finish excluded)."""
        for i, s in enumerate(self.lead_in):
            yield f"lead-in step {i + 1}", s
        for name in dict.fromkeys(self.pattern):
            for i, s in enumerate(self.cycles.get(name, [])):
                yield f"cycle {name} step {i + 1}", s
        if isinstance(self.finish, list):
            for i, s in enumerate(self.finish):
                yield f"finish step {i + 1}", s

    def model_ids(self) -> set[str]:
        ids = {s["model"] for _, s in self.all_steps() if s.get("op") == "model"}
        if self.finish == "auto":
            ids.add(AUTO_FINISH_MODEL)
        return ids

    def validate(self, known_models: set[str] | None = None) -> None:
        if not self.lead_in:
            raise RecipeError("lead_in is empty: it needs at least one step down to the loop size")
        if not self.pattern:
            raise RecipeError("pattern is empty")
        for c in self.pattern:
            if c not in self.cycles:
                raise RecipeError(f"pattern uses cycle {c!r}, which is not defined")
        for name in self.pattern:
            if not self.cycles[name]:
                raise RecipeError(f"cycle {name!r} has no steps")
        if self.iterations < 1:
            raise RecipeError("iterations must be >= 1")
        if self.save_every < 1:
            raise RecipeError("save_every must be >= 1")
        if self.rounding not in ("srgb", "linear"):
            raise RecipeError("rounding must be 'srgb' or 'linear'")
        if self.light not in ("linear", "encoded"):
            raise RecipeError("light must be 'linear' or 'encoded'")
        if not (self.finish == "auto" or isinstance(self.finish, list)):
            raise RecipeError("finish must be 'auto' or a list of steps")
        if self.finish_threshold <= 1.0:
            raise RecipeError("finish_threshold must be more than 1")
        fmt = self.output.get("format", "webp")
        if fmt not in ("webp", "png"):
            raise RecipeError("output format must be 'webp' or 'png'")
        from .kernels import METHODS
        for where, s in self.all_steps():
            op = s.get("op")
            w = s.get("width")
            if w is not None and w not in ("source", "loop") and (not isinstance(w, int) or w < 1):
                raise RecipeError(f"{where}: width must be a positive integer, 'source' or 'loop'")
            if w == "loop" and where.startswith("lead-in"):
                raise RecipeError(f"{where}: the lead-in defines the loop size, so it can't use 'loop'")
            if op == "resize":
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
        if known_models is not None and self.finish == "auto" and AUTO_FINISH_MODEL not in known_models:
            raise RecipeError(f"auto finish needs model {AUTO_FINISH_MODEL}")

    # ------------------------------------------------------- identity
    def pixel_identity(self) -> dict:
        """Everything that decides the saved pixels (not how many/when)."""
        used = {k: self.cycles[k] for k in sorted(set(self.pattern))}
        return {
            "v": ENGINE_PIXELS_VERSION,
            "lead_in": self.lead_in,
            "cycles": used,
            "pattern": self.pattern,
            "finish": self.finish,
            "finish_threshold": self.finish_threshold if self.finish == "auto" else None,
            "rounding": self.rounding,
            "light": self.light,
            "output": self.output,
        }

    def code(self) -> str:
        blob = json.dumps(self.pixel_identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:4]

    def uses_gan(self, gan_models: set[str]) -> bool:
        return any(m in gan_models for m in self.model_ids())

    def slug(self, gan_models: set[str] | None = None, max_len: int = 80) -> str:
        lead = "-".join(_step_slug(s) for s in self.lead_in)
        cyc = "+".join("-".join(_step_slug(s) for s in self.cycles[k]) for k in _unique(self.pattern))
        if self.finish == "auto":
            fin = "esrnet" if self.finish_threshold == 2.0 else f"esrnet{self.finish_threshold:g}"
        else:
            fin = "-".join(_step_slug(s) for s in self.finish) or "area"
        flags = []
        if gan_models and self.uses_gan(gan_models):
            flags.append("GAN")
        if self.rounding == "linear":
            flags.append("rlin")
        if self.light == "encoded":
            flags.append("enc")
        prefix = _clean(self.name) + "_" if self.name else ""
        tail = ("_" + "_".join(flags)) if flags else ""
        fixed = len(prefix) + len(lead) + len(fin) + len(tail) + 2
        room = max(8, max_len - fixed)
        if len(cyc) > room:
            cyc = cyc[:room - 1] + "~"
        return f"{prefix}{lead}_{cyc}_{fin}{tail}"

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


def auto_finish_steps(cur_width: int, src_width: int, threshold: float = 2.0) -> list[dict]:
    """RealESRNet x4 while `threshold` or more is left, then area lands."""
    steps = []
    w = cur_width
    while src_width / w >= threshold:
        w *= 4
        steps.append({"op": "model", "model": AUTO_FINISH_MODEL, "width": w})
    steps.append({"op": "resize", "method": "area", "width": "source"})
    return steps


def _unique(seq):
    seen = []
    for x in seq:
        if x not in seen:
            seen.append(x)
    return seen


def _clean(s: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in s.strip())
    return out.strip("-")[:30]


def _width_slug(w) -> str:
    return {"source": "S", "loop": "L", None: ""}.get(w, str(w))


def _step_slug(s: dict) -> str:
    ws = _width_slug(s.get("width"))
    if s.get("op") == "resize":
        m = KERNEL_SHORT.get(s["method"], s["method"])
        if s.get("antialias") is False:
            m += "NA"
        if s.get("fixed_point"):
            m += "FP"
        return f"{m}{ws}"
    if s.get("op") == "model":
        return f"{model_short(s['model'])}@{ws}" if ws else model_short(s["model"])
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
