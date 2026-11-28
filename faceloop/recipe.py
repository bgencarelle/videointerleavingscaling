"""Pass recipes: what a pass does, as plain JSON-able data.

Shape of a pass
---------------
lead_in   steps run once, taking the source frames down to the *loop size*
          (the size after the lead-in).
cycles    A, B, C ... alternate in `pattern` order. The first cycle starts
          at the loop size; each cycle ends wherever its last step ends and
          the next one takes it from there (no automatic resize in between).
          The sizes must settle into a repeating pattern (checked before a
          run). `iterations` counts single cycles.
finish    runs on a *copy* of the frames at every save point (every
          `save_every` cycles and after the last); that copy is what gets
          saved. The loop itself carries on from its frames, untouched.
          You choose the finish steps like any others (in the GUI).
          land_on_source (default true): if the finish doesn't end exactly
          at the source size, an area resize lands it there (an empty finish
          means just that resize). With it off, frames are saved at whatever
          size the finish ends at (an empty finish saves the loop frames).

Step kinds
----------
{"op": "resize", "method": "lanczos", "width": 320}
    plain kernel; optional "antialias": false, "fixed_point": true,
    "phase_bits": 4, "coef_bits": 6
{"op": "resize", "method": "dpid", "width": 54, "lambda": 1.0}
    detail-preserving downscaling (Weber et al. 2016); lambda 0 = box filter
{"op": "model", "model": "<model id>", "width": 160}
    run the model at its fixed factor, then area-resize to `width`
    ("width" may be omitted: keep the model's own output size)
{"op": "xbr", "scale": 2, "width": 160}
    xBR pixel-art upscale (x2/x3/x4), then area-resize to `width` (optional)
{"op": "irn", "direction": "down", "scale": 4, "variant": "standard", "width": 54}
    IRN (Invertible Rescaling Network) learned shrink ("down") or enlarge
    ("up"), then area-resize to `width` (optional). variant "standard"
    (x2/x4/x8), "plus" (IRN+, GAN-trained, x4) or "finetune" (JPEG-robust x2)
{"op": "chroma", "mode": "420", "matrix": "601"}
    colour resolution reduced like a video codec (422, 420, 411, 410=YUV9),
    colour back with nearest; size unchanged, so no width

`width` is pixels, or "source" (source width), or "loop" (the loop size;
not allowed in the lead-in, which defines it). The GUI offers only widths
that divide the step's incoming width exactly (or whole multiples of it,
for plain resizes), plus the loop and source sizes; the engine takes any.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Bump when pixel output for the same recipe would change.
ENGINE_PIXELS_VERSION = 5

KERNEL_SHORT = {"nearest": "near", "area": "area", "bilinear": "bil",
                "bicubic": "bic", "lanczos": "lanc", "dpid": "dpid", "perceptual": "perc", "kopf": "kopf",
                "lanczos2": "lanc2", "lanczos4": "lanc4", "lanczos8": "lanc8", "mitchell": "mitch",
                "catrom": "crom", "bspline": "bspl", "hermite": "herm", "spline16": "spl16", "spline36": "spl36",
                "spline64": "spl64", "gaussian": "gauss", "magic": "mks", "ewa_lanczos": "ewaL",
                "ewa_lanczossharp": "ewaLS", "ewa_robidoux": "ewaR", "ewa_robidouxsharp": "ewaRS",
                "dct": "dct", "fft": "fft", "wavelet53": "wav53", "wavelet97": "wav97"}
CONTENT_AWARE = ("perceptual", "kopf")



class RecipeError(ValueError):
    pass


@dataclass
class Recipe:
    lead_in: list[dict[str, Any]]
    cycles: dict[str, list[dict[str, Any]]]
    pattern: list[str]
    iterations: int
    save_every: int
    finish: list[dict[str, Any]] = field(default_factory=list)
    rounding: str = "srgb"          # "srgb" | "linear"
    light: str = "linear"           # kernel steps: "linear" | "encoded"
    precision: str = "fp32"         # AI models: "fp32" | "fp16" (half; bf16 where fp16 isn't supported)
    output: dict[str, Any] = field(default_factory=lambda: {
        "format": "webp", "quality": 95, "sharp_yuv": True})
    name: str = ""                  # optional label, shown in the slug
    land_on_source: bool = True     # finish ends with an area resize to the source size

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        known = {"lead_in", "cycles", "pattern", "iterations", "save_every", "finish",
                 "rounding", "light", "precision", "output", "name", "land_on_source"}
        extra = set(d) - known - {"version", "code", "slug", "engine", "source"}
        if extra:
            raise RecipeError(f"unknown recipe keys: {sorted(extra)}")
        if "lead_in" not in d:
            raise RecipeError("recipe needs a lead_in (steps from the source down to the loop size)")
        if "finish" not in d:
            raise RecipeError("recipe needs a finish (steps run on a copy at each save; [] = none)")
        r = cls(
            lead_in=[dict(s) for s in d["lead_in"]],
            cycles={k: [dict(s) for s in v] for k, v in d["cycles"].items()},
            pattern=list(d["pattern"]),
            iterations=int(d["iterations"]),
            save_every=int(d["save_every"]),
            finish=[dict(s) for s in d["finish"]],
            rounding=d.get("rounding", "srgb"),
            light=d.get("light", "linear"),
            precision=d.get("precision", "fp32"),
            output=dict(d.get("output") or {"format": "webp", "quality": 95, "sharp_yuv": True}),
            name=d.get("name", ""),
            land_on_source=bool(d.get("land_on_source", True)),
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
            "land_on_source": self.land_on_source,
            "rounding": self.rounding,
            "light": self.light,
            "precision": self.precision,
            "output": dict(self.output),
            "code": self.code(),
            "slug": self.slug(),
        }

    # ------------------------------------------------------------ checks
    def all_steps(self):
        """(where, step) for every step the recipe can run."""
        for i, s in enumerate(self.lead_in):
            yield f"lead-in step {i + 1}", s
        for name in dict.fromkeys(self.pattern):
            for i, s in enumerate(self.cycles.get(name, [])):
                yield f"cycle {name} step {i + 1}", s
        for i, s in enumerate(self.finish):
            yield f"finish step {i + 1}", s

    def model_ids(self) -> set[str]:
        ids = {s["model"] for _, s in self.all_steps() if s.get("op") == "model"}
        ids |= {s.get("model", VSR_DEFAULT) for _, s in self.all_steps() if s.get("op") == "vsr"}
        return ids | {irn_id(s) for _, s in self.all_steps() if s.get("op") == "irn"}

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
        if self.precision not in ("fp32", "fp16"):
            raise RecipeError("precision must be 'fp32' or 'fp16'")
        if not isinstance(self.finish, list):
            raise RecipeError("finish must be a list of steps")
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
                if s.get("method") not in METHODS + ("dpid",) + CONTENT_AWARE:
                    raise RecipeError(f"{where}: unknown kernel {s.get('method')!r}")
                if s.get("method") == "gaussian":
                    sg = s.get("sigma", 0.5)
                    if not isinstance(sg, (int, float)) or not (0.2 <= sg <= 3.0):
                        raise RecipeError(f"{where}: gaussian sigma must be 0.2 to 3")
                elif "sigma" in s:
                    raise RecipeError(f"{where}: sigma is only for the gaussian kernel")
                if s.get("method", "").startswith("ewa_") and s.get("fixed_point"):
                    raise RecipeError(f"{where}: fixed-point isn't available for EWA filters")
                if s.get("method") in ("dct", "fft", "wavelet53", "wavelet97") and s.get("fixed_point"):
                    raise RecipeError(f"{where}: fixed-point isn't available for the frequency methods")
                if s.get("method") == "perceptual" and s.get("patch", 2) not in (2, 3, 4):
                    raise RecipeError(f"{where}: perceptual patch must be 2, 3 or 4")
                if s.get("method") == "kopf":
                    it = s.get("iterations", 200)
                    if not isinstance(it, int) or not (10 <= it <= 1000):
                        raise RecipeError(f"{where}: Kopf iterations must be 10 to 1000")
                if s.get("method") == "dpid":
                    lam = s.get("lambda", 1.0)
                    if not isinstance(lam, (int, float)) or not (0 <= lam <= 10):
                        raise RecipeError(f"{where}: DPID lambda must be a number from 0 to 10")
                if w is None:
                    raise RecipeError(f"{where}: resize needs a width")
            elif op == "model":
                m = s.get("model")
                if not m:
                    raise RecipeError(f"{where}: model step needs a model id")
                if known_models is not None and m not in known_models:
                    raise RecipeError(f"{where}: unknown model {m!r}")
            elif op == "vsr":
                m = s.get("model", VSR_DEFAULT)
                if known_models is not None and m not in known_models:
                    raise RecipeError(f"{where}: unknown video model {m!r}")
                if not m.startswith("basicvsr"):
                    raise RecipeError(f"{where}: {m!r} isn't a video model")
                mem = s.get("memory", 1)
                if not isinstance(mem, int) or isinstance(mem, bool) or not 1 <= mem <= 100:
                    raise RecipeError(f"{where}: memory must be a whole number of frames, 1 to 100")
                if s.get("direction", "both") not in ("both", "forward", "backward"):
                    raise RecipeError(f"{where}: direction must be both, forward or backward")
                if s.get("order", "normal") not in ("normal", "reverse", "shuffle"):
                    raise RecipeError(f"{where}: order must be normal, reverse or shuffle")
                fl = s.get("flow", 1.0)
                if not isinstance(fl, (int, float)) or isinstance(fl, bool) or not 0 <= fl <= 3:
                    raise RecipeError(f"{where}: flow strength must be between 0 and 3")
            elif op == "xbr":
                if s.get("scale") not in (2, 3, 4):
                    raise RecipeError(f"{where}: xBR scale must be 2, 3 or 4")
            elif op == "edi":
                from .edi import METHODS as EDI_METHODS, SCALES as EDI_SCALES
                if s.get("method") not in EDI_METHODS:
                    raise RecipeError(f"{where}: unknown 2x method {s.get('method')!r}")
                ok = EDI_SCALES[s["method"]]
                if s.get("scale") not in ok:
                    raise RecipeError(f"{where}: {s['method']} scale must be {', '.join(map(str, ok[:-1]))} or {ok[-1]}")
                if s.get("method") == "nnedi3" and s.get("quality", "fast") not in ("fast", "best"):
                    raise RecipeError(f"{where}: nnedi3 quality must be 'fast' or 'best'")
            elif op == "irn":
                v = s.get("variant", "standard")
                ok = {"standard": (2, 4, 8), "plus": (4,), "finetune": (2,)}
                if v not in ok:
                    raise RecipeError(f"{where}: IRN variant must be standard, plus or finetune")
                if s.get("scale") not in ok[v]:
                    raise RecipeError(f"{where}: IRN {v} comes in x{', x'.join(map(str, ok[v]))}")
                if s.get("direction") not in ("down", "up"):
                    raise RecipeError(f"{where}: IRN direction must be 'down' or 'up'")
                if known_models is not None and irn_id(s) not in known_models:
                    raise RecipeError(f"{where}: {irn_id(s)} isn't available")
            elif op == "video":
                from .video import AXES, FIELDS, REBUILD
                from .kernels import METHODS as KMETHODS
                mode = s.get("mode")
                if w is not None:
                    raise RecipeError(f"{where}: a video step keeps the size, so it takes no width")
                if mode == "fields":
                    if s.get("field", "alternate") not in FIELDS:
                        raise RecipeError(f"{where}: field must be top, bottom or alternate")
                    if s.get("rebuild", "linear") not in REBUILD:
                        raise RecipeError(f"{where}: rebuild must be double, linear or cubic")
                elif mode == "squash":
                    if s.get("axis", "x") not in AXES:
                        raise RecipeError(f"{where}: squash axis must be x or y")
                    f = s.get("factor")
                    if not isinstance(f, (int, float)) or not 0.05 <= f <= 0.99:
                        raise RecipeError(f"{where}: squash factor must be between 0.05 and 0.99")
                    if s.get("down", "area") not in KMETHODS + ("dpid",):
                        raise RecipeError(f"{where}: unknown squash method {s.get('down')!r}")
                    if s.get("up", "bicubic") not in KMETHODS:
                        raise RecipeError(f"{where}: unknown stretch method {s.get('up')!r}")
                else:
                    raise RecipeError(f"{where}: video mode must be fields or squash")
            elif op == "chroma":
                from .chroma import MATRICES, MODES
                if s.get("mode", "420") not in MODES:
                    raise RecipeError(f"{where}: chroma mode must be one of {sorted(MODES)}")
                if s.get("matrix", "601") not in MATRICES:
                    raise RecipeError(f"{where}: chroma matrix must be one of {sorted(MATRICES)}")
                if w is not None:
                    raise RecipeError(f"{where}: a chroma step keeps the size, so it takes no width")
            else:
                raise RecipeError(f"{where}: unknown op {op!r}")

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
            "rounding": self.rounding,
            "light": self.light,
            "output": self.output,
            # absent when on (the default) so old codes stay
            **({} if self.land_on_source else {"land_on_source": False}),
            # only matters when a model is used; absent for fp32 so old codes stay
            **({"precision": self.precision} if self.precision != "fp32" and self.model_ids() else {}),
        }

    def code(self) -> str:
        blob = json.dumps(self.pixel_identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:4]

    def uses_gan(self, gan_models: set[str]) -> bool:
        return any(m in gan_models for m in self.model_ids())

    def slug(self, gan_models: set[str] | None = None, max_len: int = 80) -> str:
        lead = "-".join(_step_slug(s) for s in self.lead_in)
        cyc = "+".join("-".join(_step_slug(s) for s in self.cycles[k]) for k in _unique(self.pattern))
        fin = "-".join(_step_slug(s) for s in self.finish)
        if self.land_on_source:
            fin = fin or "area"
        else:
            fin = (fin + "-free") if fin else "free"
        flags = []
        if gan_models and self.uses_gan(gan_models):
            flags.append("GAN")
        if self.rounding == "linear":
            flags.append("rlin")
        if self.light == "encoded":
            flags.append("enc")
        if self.precision == "fp16" and self.model_ids():
            flags.append("h16")
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
        if s["method"] == "dpid" and s.get("lambda", 1.0) != 1.0:
            m += f"{s['lambda']:g}"
        if s["method"] == "gaussian" and s.get("sigma", 0.5) != 0.5:
            m += f"{s['sigma']:g}"
        if s["method"] == "perceptual" and s.get("patch", 2) != 2:
            m += f"p{s['patch']}"
        if s["method"] == "kopf" and s.get("iterations", 200) != 200:
            m += f"i{s['iterations']}"
        if s.get("antialias") is False:
            m += "NA"
        if s.get("fixed_point"):
            m += "FP"
        return f"{m}{ws}"
    if s.get("op") == "model":
        return f"{model_short(s['model'])}@{ws}" if ws else model_short(s["model"])
    if s.get("op") == "xbr":
        return f"xbr{s['scale']}@{ws}" if ws else f"xbr{s['scale']}"
    if s.get("op") == "edi":
        from .edi import CENTRABLE
        short = {"2xsai": "sai", "super2xsai": "ssai", "supereagle": "seagle", "superxbr": "sxbr",
                 "scale2x": "sc2x", "scale3x": "sc3x", "hqx": "hq", "scalefx": "sfx"}.get(s["method"], s["method"])
        tag = (f"{short}{s['scale']}" + ("q" if s.get("quality") == "best" else "")
               + ("raw" if s.get("center") is False and s["method"] in CENTRABLE else ""))
        return f"{tag}@{ws}" if ws else tag
    if s.get("op") == "irn":
        v = {"plus": "P", "finetune": "F"}.get(s.get("variant", "standard"), "")
        tag = f"irn{v}{'D' if s['direction'] == 'down' else 'U'}{s['scale']}"
        return f"{tag}@{ws}" if ws else tag
    if s.get("op") == "vsr":
        m = s.get("model", VSR_DEFAULT).replace("basicvsr-", "bvsr")
        mem = s.get("memory", 1)
        if mem == 1:
            return m
        extra = {"forward": "f", "backward": "b"}.get(s.get("direction", "both"), "")
        extra += {"reverse": "rev", "shuffle": "shuf"}.get(s.get("order", "normal"), "")
        fl = s.get("flow", 1.0)
        return f"{m}m{mem}{extra}" + ("" if fl == 1 else f"fl{round(fl * 100)}")
    if s.get("op") == "video":
        if s["mode"] == "fields":
            return f"fld{s.get('field', 'alternate')[0]}{s.get('rebuild', 'linear')[0]}"
        d, u = s.get("down", "area"), s.get("up", "bicubic")
        return (f"sq{s.get('axis', 'x')}{round(s['factor'] * 100)}"
                f"{KERNEL_SHORT.get(d, d)}{KERNEL_SHORT.get(u, u)}")
    if s.get("op") == "chroma":
        mx = "" if s.get("matrix", "601") == "601" else s["matrix"]
        return f"c{s.get('mode', '420')}{mx}"
    return f"{s.get('op')}{ws}"


_SHORT = [("realesrgan-x4plus-anime", "esrAnime4"), ("realesrgan-x4plus", "esrgan4"),
          ("realesrgan-x2plus", "esrgan2"), ("realesrnet-x4", "esrnet4"),
          ("realesr-animevideov3", "avv3"), ("realesr-general-wdn-x4v3", "genwdn4"),
          ("realesr-general-x4v3", "gen4"), ("swinir-classical-", "swin"),
          ("w2x-cunet-denoise-", "w2xdn"), ("w2x-", "w2x"), ("rcugan-", "cug"), ("dropin-", "d")]


VSR_DEFAULT = "basicvsr-reds4"


def irn_id(step: dict) -> str:
    """Model id for an IRN step: irn-x4, irnplus-x4, irn-x2-finetune ..."""
    v = step.get("variant", "standard")
    if v == "plus":
        return f"irnplus-x{step['scale']}"
    if v == "finetune":
        return f"irn-x{step['scale']}-finetune"
    return f"irn-x{step['scale']}"


def model_short(mid: str) -> str:
    """Short readable model code for folder names, e.g. w2xcunetn1, avv3, cugx2dn3."""
    for pre, short in _SHORT:
        if mid.startswith(pre):
            rest = mid[len(pre):]
            return (short + "".join(ch for ch in rest if ch.isalnum()))[:14]
    return "".join(ch for ch in mid if ch.isalnum())[:14]
