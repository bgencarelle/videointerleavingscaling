"""Runs a pass over one source folder.

Frames are processed in batches. Each batch runs the lead-in once (down to
the loop size), then every cycle in memory at the loop size. At each save
point a copy is finished back to the source size and written. When a
batch is done, a marker records it, so a stopped or crashed run resumes at
the first unfinished batch (and re-runs only that batch).
"""
from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import __version__
from .colour import Transfer, curves_from_icc, quantize
from .imageio import list_frames, load_frame, probe, save_frame, webp_backend
from .kernels import KernelResizer
from .recipe import Recipe, RecipeError
from .sizes import Size, size_for
from .throttle import THROTTLE


# ------------------------------------------------------------- devices
def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def total_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * 1024 ** 3


# ------------------------------------------------------------- steps
@dataclass
class StepContext:
    src: Size
    transfer: Transfer
    light: str
    rounding: str
    resizer: KernelResizer
    models: object | None  # faceloop.models.ModelManager
    loop: Size | None = None  # size after the lead-in
    precision: str = "fp32"   # AI models: "fp32" | "fp16"
    check: Callable[[], None] | None = None  # called before every step; may raise to stop


def _target(step: dict, src: Size, cur: Size, loop: Size | None = None) -> Size | None:
    w = step.get("width")
    if w is None:
        return None
    if w == "source":
        return src
    if w == "loop":
        if loop is None:
            raise RecipeError("'loop' width used before the loop size is known")
        return loop
    return size_for(int(w), src)


def _kernel(x: torch.Tensor, target: Size, method: str, ctx: StepContext, **kw) -> torch.Tensor:
    if (x.shape[-1], x.shape[-2]) == (target.width, target.height):
        return x
    if method == "dpid":
        from .dpid import dpid
        lam = float(kw.get("lam", 1.0))
        if ctx.light == "linear":
            y = dpid(ctx.transfer.decode(x), target.width, target.height, lam)
            return ctx.transfer.encode(y.clamp(0, 1))
        return dpid(x, target.width, target.height, lam).clamp(0, 1)
    if ctx.light == "linear":
        y = ctx.resizer.resize(ctx.transfer.decode(x), target.width, target.height, method, **kw)
        return ctx.transfer.encode(y.clamp(0, 1))
    return ctx.resizer.resize(x, target.width, target.height, method, **kw).clamp(0, 1)


def apply_step(x: torch.Tensor, step: dict, ctx: StepContext) -> torch.Tensor:
    cur = Size(x.shape[-1], x.shape[-2])
    op = step["op"]
    if op == "resize":
        if step["method"] == "dpid":
            kw = {"lam": step.get("lambda", 1.0)}
        else:
            kw = {k: step[k] for k in ("antialias", "fixed_point", "phase_bits", "coef_bits") if k in step}
        x = _kernel(x, _target(step, ctx.src, cur, ctx.loop), step["method"], ctx, **kw)
    elif op == "model":
        if ctx.models is None:
            raise RecipeError("model step used but no models are available")
        model = ctx.models.get(step["model"], x.device, ctx.precision)
        x = model(x).clamp(0, 1)
        t = _target(step, ctx.src, cur, ctx.loop)
        if t is not None:
            x = _kernel(x, t, "area", ctx)
    elif op == "xbr":
        from .pixelart import xbr
        x = xbr(x, int(step["scale"])).clamp(0, 1)
        t = _target(step, ctx.src, cur, ctx.loop)
        if t is not None:
            x = _kernel(x, t, "area", ctx)
    elif op == "irn":
        from .recipe import irn_id
        m = ctx.models.get_irn(irn_id(step), x.device)
        x = (m.down(x) if step["direction"] == "down" else m.up(x)).clamp(0, 1)
        t = _target(step, ctx.src, cur, ctx.loop)
        if t is not None:
            x = _kernel(x, t, "area", ctx)
    elif op == "chroma":
        from .chroma import subsample
        x = subsample(x, step.get("mode", "420"), step.get("matrix", "601"))
    else:
        raise RecipeError(f"unknown op {op!r}")
    return quantize(x, ctx.rounding, ctx.transfer)


def run_steps(x: torch.Tensor, steps: list[dict], ctx: StepContext, land: Size | None,
              capture: Callable[[int, torch.Tensor], None] | None = None) -> torch.Tensor:
    """Run steps; then area-land on `land` if the last step didn't end there."""
    for i, step in enumerate(steps):
        if ctx.check is not None:
            ctx.check()
        x = apply_step(x, step, ctx)
        if capture:
            capture(i, x)
    if land is not None and (x.shape[-1], x.shape[-2]) != (land.width, land.height):
        x = quantize(_kernel(x, land, "area", ctx), ctx.rounding, ctx.transfer)
        if capture:
            capture(len(steps), x)
    return x


def run_cycle(x: torch.Tensor, steps: list[dict], ctx: StepContext,
              capture: Callable[[int, torch.Tensor], None] | None = None) -> torch.Tensor:
    """A cycle ends wherever its last step ends; the next cycle's first step
    takes it from there (no automatic resize in between)."""
    return run_steps(x, steps, ctx, None, capture)


def run_lead_in(x: torch.Tensor, recipe: Recipe, ctx: StepContext) -> torch.Tensor:
    x = run_steps(x, recipe.lead_in, ctx, None)
    loop = Size(x.shape[-1], x.shape[-2])
    if ctx.loop is not None and ctx.loop != loop:
        raise RecipeError(f"lead-in gave {loop}, expected {ctx.loop}")
    ctx.loop = loop
    return x


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _cycle_throttled(x, steps, ctx, lock, capture, want, cname, batch, on_contact):
    """One cycle for a batch. With a reserve set (see throttle.py) the batch
    runs in slices with idle gaps between them, so the Mac's own use of the
    GPU gets regular turns. Frames are independent, so results are the same."""
    sl = THROTTLE.slice_size()
    if sl is None or x.shape[0] <= sl:
        t1 = time.time()
        with lock:
            if capture:
                capture(-1, x)
            y = run_cycle(x, steps, ctx, capture)
            if sl is not None:
                _sync(x.device)
        THROTTLE.pause_after(time.time() - t1)
        return y
    parts = []
    for j in range(0, x.shape[0], sl):
        xs = x[j:j + sl]
        cap = None
        local = [i - j for i in want if j <= i < j + sl] if capture else []
        if local:
            def cap(step_i, t, _w=local, _j=j):
                on_contact(cname, step_i, [batch[_j + i].name for i in _w], to_uint8(t[_w]))
        t1 = time.time()
        with lock:
            if cap:
                cap(-1, xs)
            parts.append(run_cycle(xs, steps, ctx, cap))
            _sync(x.device)
        THROTTLE.pause_after(time.time() - t1)
    return torch.cat(parts, 0)


def lead_in_chunked(arrs: list[np.ndarray], recipe: Recipe, ctx: StepContext, chunk: int,
                    device: torch.device, lock=None) -> torch.Tensor:
    """Lead-in for a big batch, a few full-size frames at a time, so only the
    small loop-size frames of the whole batch are held on the GPU."""
    lock = lock if lock is not None else contextlib.nullcontext()
    parts = []
    for i in range(0, len(arrs), chunk):
        xs = torch.from_numpy(np.stack(arrs[i:i + chunk])).to(device).permute(0, 3, 1, 2).float() / 255.0
        t1 = time.time()
        with lock:
            parts.append(run_lead_in(xs, recipe, ctx))
            if THROTTLE.fraction:
                _sync(device)
        THROTTLE.pause_after(time.time() - t1)
    return torch.cat(parts, 0)


def finish_chunked(x: torch.Tensor, recipe: Recipe, ctx: StepContext, chunk: int, lock=None) -> np.ndarray:
    lock = lock if lock is not None else contextlib.nullcontext()
    parts = []
    for i in range(0, x.shape[0], chunk):
        t1 = time.time()
        with lock:
            parts.append(to_uint8(run_finish(x[i:i + chunk], recipe, ctx)))
        THROTTLE.pause_after(time.time() - t1)
    return np.concatenate(parts, 0)


def run_finish(x: torch.Tensor, recipe: Recipe, ctx: StepContext) -> torch.Tensor:
    """Take a copy of the loop frames back to the source size."""
    return run_steps(x, recipe.finish, ctx, ctx.src)


# ------------------------------------------------------------- output
def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(N,3,H,W) encoded floats -> (N,H,W,3) uint8."""
    return (x.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@dataclass
class Progress:
    frames_total: int = 0
    frames_done: int = 0
    batch: int = 0
    batches: int = 0
    cycle: int = 0
    cycles: int = 0
    started: float = field(default_factory=time.time)
    message: str = ""
    frame_cycles: int = 0   # frames x cycles actually processed in this run
    images_saved: int = 0   # saved frames written in this run

    def fraction(self) -> float:
        if not self.frames_total or not self.cycles:
            return 0.0
        done = self.frames_done * self.cycles
        cur = 0
        if self.batches:
            remaining = self.frames_total - self.frames_done
            per = min(remaining, max(1, -(-self.frames_total // self.batches)))
            cur = per * self.cycle
        return min(1.0, (done + cur) / (self.frames_total * self.cycles))

    def eta_seconds(self) -> float | None:
        f = self.fraction()
        if f <= 0:
            return None
        el = time.time() - self.started
        return el * (1 - f) / f


class Controls:
    """Pause / stop from another thread (the GUI)."""

    def __init__(self):
        self._run = threading.Event()
        self._run.set()
        self.stop_requested = False

    def pause(self):
        self._run.clear()

    def resume(self):
        self._run.set()

    def stop(self):
        self.stop_requested = True
        self._run.set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def checkpoint(self):
        self._run.wait()
        if self.stop_requested:
            raise StopRequested()


def _fmt_eta(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


class StopRequested(Exception):
    pass


# ------------------------------------------------------------- the pass
@dataclass
class PassResult:
    folder: Path
    frames: int
    seconds: float
    skipped_frames: int
    stopped: bool = False


def _scale_lookup(models):
    def scale_of(st):
        if st.get("op") == "xbr":
            return int(st["scale"])
        info = getattr(models, "catalog", {}).get(st.get("model")) if models is not None else None
        return info.scale if info and info.scale else None
    return scale_of


def step_size(st: dict, cur: Size, src: Size, loop: Size | None, scale_of) -> tuple[Size, list | None, bool]:
    """(size after the step, raw model/xBR output size or None, scale known?)"""
    w = st.get("width")
    tgt = src if w == "source" else (loop if w == "loop" else (size_for(int(w), src) if isinstance(w, int) else None))
    op = st.get("op")
    if op == "irn":
        k = int(st["scale"])
        raw = ([-(-cur.width // k), -(-cur.height // k)] if st.get("direction") == "down"
               else [cur.width * k, cur.height * k])
        return (tgt or Size(*raw)), raw, True
    if op in ("model", "xbr"):
        k = scale_of(st)
        if not k:
            return (tgt or cur), None, False
        raw = [cur.width * k, cur.height * k]
        return (tgt or Size(*raw)), raw, True
    if op == "chroma":
        return cur, None, True
    return (tgt or cur), None, True


MAX_GROWTH = 16  # a pattern may not make frames bigger than 16x the source's pixels


def simulate_pattern(recipe: Recipe, src: Size, models=None, rounds: int = 3, cycles: dict | None = None) -> dict:
    """Follow the frame size through the lead-in and a few rounds of the
    pattern. Returns the loop size, where each cycle starts once things have
    settled, the biggest size seen, and whether the sizes settle at all."""
    scale_of = _scale_lookup(models)
    cycles = cycles if cycles is not None else recipe.cycles
    known = True
    cur = src
    biggest = src.width * src.height
    for st in recipe.lead_in:
        cur, raw, ok = step_size(st, cur, src, None, scale_of)
        known &= ok
        biggest = max(biggest, cur.width * cur.height, raw[0] * raw[1] if raw else 0)
    loop = cur
    ends, starts = [], {}
    cyc_big = loop.width * loop.height  # biggest size inside the cycles only
    for r in range(rounds):
        for k in recipe.pattern:
            if r == rounds - 1 and k not in starts:
                starts[k] = cur
            for st in cycles[k]:
                cur, raw, ok = step_size(st, cur, src, loop, scale_of)
                known &= ok
                cyc_big = max(cyc_big, cur.width * cur.height, raw[0] * raw[1] if raw else 0)
        ends.append(cur)
    biggest = max(biggest, cyc_big)
    stable = (not known) or ends[-1] == ends[-2]
    too_big = biggest > MAX_GROWTH * src.width * src.height
    return {"loop": loop, "starts": starts, "end": cur, "biggest": biggest, "cycle_biggest": cyc_big,
            "stable": stable, "known": known, "too_big": too_big}


def effective_cycles(recipe: Recipe, src: Size, models=None) -> tuple[dict, set]:
    """The cycles as they will run. If the sizes wouldn't settle on their own
    (which only happens when no cycle has a fixed-size step), the first cycle
    of the pattern starts with the lead-in steps, added automatically; that
    one fixed point is enough. Otherwise nothing is added. Returns (cycles,
    names of the cycles that got the lead-in)."""
    if simulate_pattern(recipe, src, models)["stable"]:
        return dict(recipe.cycles), set()
    auto = {recipe.pattern[0]}
    cyc = {k: (list(recipe.lead_in) + list(v) if k in auto else v) for k, v in recipe.cycles.items()}
    return cyc, auto


def check_pattern(recipe: Recipe, src: Size, models=None) -> tuple[dict, set]:
    cycles, auto = effective_cycles(recipe, src, models)
    sim = simulate_pattern(recipe, src, models, cycles=cycles)
    if not sim["stable"]:
        raise RecipeError("the frames keep changing size from one round of cycles to the next; "
                          "give a cycle a step with a fixed size, like the loop size")
    if sim["too_big"]:
        raise RecipeError(f"frames would get bigger than {MAX_GROWTH}x the source's pixels inside the loop")
    return cycles, auto


def _sim(steps: list[dict], cur: Size, src: Size, loop: Size | None, biggest: int) -> tuple[Size, int]:
    for s in steps:
        w = s.get("width")
        if w == "source":
            tgt = src
        elif w == "loop":
            tgt = loop
        elif isinstance(w, int):
            tgt = size_for(w, src)
        else:
            tgt = None
        if s.get("op") == "irn" and s.get("direction") == "down":
            cur = tgt or Size(-(-cur.width // int(s["scale"])), -(-cur.height // int(s["scale"])))
        elif s.get("op") in ("model", "xbr", "irn"):
            k = int(s.get("scale", 4))  # models: assume up to x4 before landing
            biggest = max(biggest, cur.width * cur.height * k * k)
            cur = tgt or Size(cur.width * k, cur.height * k)
        elif tgt is not None:
            cur = tgt
        biggest = max(biggest, cur.width * cur.height)
    return cur, biggest


def plan_sizes(recipe: Recipe, src: Size, models=None) -> dict:
    """Every step's output size, for showing in the GUI before a run.

    Returns {"loop", "lead_in", "cycles", "cycle_starts", "finish", "stable"}.
    Cycles are shown from where they start once the pattern has settled (the
    first cycle of a run starts from the loop size instead). A row is
    {"width", "height", "from", "raw", "scale", "error", "note"}."""
    from .sizes import aspect_error
    scale_of = _scale_lookup(models)

    def run(steps, cur, loop, land, land_label):
        rows = []
        for st in steps:
            before = cur
            cur, raw, ok = step_size(st, cur, src, loop, scale_of)
            rows.append({"width": cur.width, "height": cur.height, "from": [before.width, before.height],
                         "raw": raw, "scale": (int(st["scale"]) if st.get("op") == "irn"
                                               else scale_of(st) if st.get("op") in ("model", "xbr") else None),
                         "dir": st.get("direction"),
                         "error": round(aspect_error(cur, src), 3),
                         "note": "" if ok else "scale unknown until loaded"})
        if land is not None and cur != land:
            rows.append({"width": land.width, "height": land.height, "from": [cur.width, cur.height],
                         "raw": None, "scale": None, "error": round(aspect_error(land, src), 3),
                         "note": land_label})
            cur = land
        return rows, cur

    eff, auto = effective_cycles(recipe, src, models)
    sim = simulate_pattern(recipe, src, models, cycles=eff)
    loop = sim["loop"]
    lead, _ = run(recipe.lead_in, src, None, None, "")
    cycles, cycle_auto = {}, {}
    for k in dict.fromkeys(recipe.pattern):
        if k not in recipe.cycles:
            continue
        rows, _ = run(eff[k], sim["starts"].get(k, loop), loop, None, "")
        n_auto = len(recipe.lead_in) if k in auto else 0
        for r in rows[:n_auto]:
            r["note"] = "lead-in, added automatically so the sizes don't keep growing"
        cycle_auto[k], cycles[k] = rows[:n_auto], rows[n_auto:]
    fin, _ = run(recipe.finish, sim["end"], loop, src, "area lands on the source size")
    return {"loop": loop, "lead_in": lead, "cycles": cycles, "cycle_auto": cycle_auto, "finish": fin,
            "cycle_starts": {k: [v.width, v.height] for k, v in sim["starts"].items()},
            "auto": sorted(auto), "stable": sim["stable"], "too_big": sim["too_big"]}


MAX_BATCH = 256


def batch_sizes(src: Size, recipe: Recipe, ram: int | None = None, models=None) -> tuple[int, int]:
    """(frames per batch, frames per full-size chunk).

    A batch lives on the GPU through all its cycles at the sizes the cycles
    use, so it can be big when those are small. The lead-in and the finish work at
    full size, so they run a few frames (a chunk) at a time."""
    ram = ram or total_ram_bytes()
    _, lead_big = _sim(recipe.lead_in, src, src, None, src.width * src.height)
    cycles, _ = effective_cycles(recipe, src, models)
    sim = simulate_pattern(recipe, src, models, cycles=cycles)
    loop = sim["loop"]
    cyc_big = max(sim["cycle_biggest"] if sim["known"] else 0, loop.width * loop.height)
    for k in set(recipe.pattern):  # unknown model scales: assume up to x4
        _, cyc_big = _sim(cycles[k], sim["starts"].get(k, loop), src, loop, cyc_big)
    _, fin_big = _sim(recipe.finish, sim["end"], src, loop, src.width * src.height)
    f32 = 3 * 4 * 4  # float32 RGB, a few temporaries
    chunk = int(max(1, min(64, ram * 0.15 // (max(lead_big, fin_big) * f32))))
    by_gpu = ram * 0.20 // (cyc_big * f32)
    by_prefetch = ram * 0.10 // (src.width * src.height * 3 * 2)  # uint8 frames: current + next batch
    batch = int(max(1, min(MAX_BATCH, by_gpu, by_prefetch)))
    return batch, chunk


def default_batch_size(src: Size, recipe: Recipe, ram: int | None = None, models=None) -> int:
    return batch_sizes(src, recipe, ram, models)[0]


def source_identity(source: Path) -> dict:
    """Name a source for output folders. A saved loop of an earlier pass
    (a cXXXX folder next to a recipe.json) is named after its origin, e.g.
    'BG_faceSource+8ba5c0200', and keeps the original source as its root."""
    source = Path(source)
    parent_rec = source.parent / "recipe.json"
    if re.fullmatch(r"c\d{4}", source.name) and parent_rec.exists():
        try:
            info = json.loads(parent_rec.read_text(encoding="utf-8"))
            psrc = info.get("source", {})
            plabel = psrc.get("label") or source.parent.parent.name
            root = psrc.get("root") or plabel
            return {"label": f"{plabel}+{info.get('code', '')}{source.name}", "root": root,
                    "chained_from": str(source)}
        except (json.JSONDecodeError, OSError):
            pass
    return {"label": source.name, "root": source.name}


def contact_indices(n_frames: int, source_name: str, k: int = 6) -> list[int]:
    """The same random frames for every pass of one source."""
    rnd = random.Random(f"faceloop:{source_name}")
    k = min(k, n_frames)
    return sorted(rnd.sample(range(n_frames), k))


def run_pass(recipe: Recipe, source: Path, out_root: Path, *,
             models=None, device: torch.device | None = None,
             batch_size: int | None = None, frame_limit: int | None = None,
             progress: Progress | None = None, controls: Controls | None = None,
             on_contact: Callable | None = None, log: Callable[[str], None] | None = None,
             write: bool = True, contact: bool = True, gpu_lock=None) -> PassResult:
    """gpu_lock: held around each GPU stretch (lead-in, one cycle, one finish),
    so another thread (the GUI's estimate) can use the GPU in between."""
    source = Path(source)
    out_root = Path(out_root)
    device = device or pick_device()
    progress = progress or Progress()
    controls = controls or Controls()
    gan = models.gan_ids() if models is not None else set()
    known = models.known_ids() if models is not None else None
    recipe.validate(known)

    frames = list_frames(source)
    if not frames:
        raise RecipeError(f"no PNG/JPEG/WebP frames in {source}")
    if frame_limit:
        frames = frames[:frame_limit]
    w0, h0, icc = probe(frames[0])
    src = Size(w0, h0)
    curves, warn = curves_from_icc(icc)
    transfer = Transfer(curves)

    ident = source_identity(source)
    folder = out_root / ident["label"] / recipe.folder_name(gan)
    recorder = None
    if write and contact and on_contact is None:
        from .contact import ContactRecorder
        recorder = ContactRecorder(folder, recipe)
        on_contact = recorder
    lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
    marker_dir = folder / ".batches"
    if write:
        marker_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        if log:
            log(line)
        if write:
            with open(folder / "log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")

    _, webp_desc = webp_backend()
    if write:
        d = recipe.to_dict()
        d["slug"] = recipe.slug(gan)
        d["engine"] = {
            "faceloop": __version__, "torch": torch.__version__, "device": str(device),
            "webp": webp_desc, "tone_curve": transfer.describe(),
            "models": models.describe_used(recipe) if models is not None else {},
        }
        d["source"] = {"folder": str(source), "frames": len(frames), "size": str(src), **ident}
        (folder / "recipe.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

    _log(f"pass {folder.name} on {ident['label']}: {len(frames)} frames at {src}, device {device}")
    if warn:
        _log(f"warning: {warn}")
    if recipe.output.get("format") == "webp" and recipe.output.get("sharp_yuv", True) and webp_desc.startswith("Pillow"):
        _log(f"warning: {webp_desc}")

    # which frames are already done?
    done: set[str] = set()
    if write:
        for m in marker_dir.glob("batch_*.json"):
            try:
                info = json.loads(m.read_text())
            except json.JSONDecodeError:
                continue
            if info.get("code") == recipe.code() and info.get("iterations", 0) >= recipe.iterations:
                done.update(info.get("frames", []))
    todo = [p for p in frames if p.name not in done]
    skipped = len(frames) - len(todo)
    if skipped:
        _log(f"resuming: {skipped} frames already done")

    cycles, auto = check_pattern(recipe, src, models)
    bs, chunk = batch_sizes(src, recipe, models=models)
    bs = batch_size or bs
    chunk = min(chunk, bs)
    save_pts = set(recipe.save_points())
    contact_idx = set(contact_indices(len(frames), ident["root"]))
    contact_names = {frames[i].name for i in contact_idx}
    last_of_cycle = {k: max(i for i in range(1, recipe.iterations + 1) if recipe.cycle_at(i) == k)
                     for k in set(recipe.pattern) if any(recipe.cycle_at(i) == k for i in range(1, recipe.iterations + 1))}

    ctx = StepContext(src=src, transfer=transfer, light=recipe.light, rounding=recipe.rounding,
                      resizer=KernelResizer(), models=models, precision=recipe.precision)

    def _stop_check():  # Stop takes effect between steps, not only between cycles
        if controls.stop_requested:
            raise StopRequested()
    ctx.check = _stop_check
    if models is not None and recipe.precision == "fp16":
        for mid in sorted(recipe.model_ids()):
            _log(f"{mid} runs in {models.precision_of(mid, device, 'fp16')}")

    progress.frames_total = len(frames)
    progress.frames_done = skipped
    progress.cycles = recipe.iterations
    progress.batches = -(-len(todo) // bs) if todo else 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
    loader = ThreadPoolExecutor(max_workers=4)  # reads the next batch while the GPU works
    pending = []
    stopped = False
    _log(f"batches of {bs} frames (full-size work in chunks of {chunk})")
    if auto:
        _log(f"cycle{'s' if len(auto) > 1 else ''} {', '.join(sorted(auto))} start with the lead-in "
             "(added automatically: without it the frame sizes would keep growing)")
    if recorder is not None:
        recorder.cycles, recorder.auto, recorder.n_lead = cycles, auto, len(recipe.lead_in)

    def prefetch(start):
        return [loader.submit(load_frame, p) for p in todo[start:start + bs]]

    try:
        nxt = prefetch(0) if todo else []
        for bi in range(0, len(todo), bs):
            batch = todo[bi:bi + bs]
            tb = time.time()
            saved_before = progress.images_saved
            progress.batch = bi // bs + 1
            arrs = [f.result()[0] for f in nxt]
            nxt = prefetch(bi + bs)
            for p, a in zip(batch, arrs):
                if a.shape[:2] != (src.height, src.width):
                    raise RecipeError(f"{p.name} is {a.shape[1]}x{a.shape[0]}, expected {src}")
            x = lead_in_chunked(arrs, recipe, ctx, chunk, device, lock)
            del arrs
            want = [i for i, p in enumerate(batch) if p.name in contact_names]

            for ci in range(1, recipe.iterations + 1):
                controls.checkpoint()
                progress.cycle = ci
                cname = recipe.cycle_at(ci)
                capture = None
                if on_contact and want and last_of_cycle.get(cname) == ci:
                    def capture(step_i, t, _c=cname, _w=want, _b=batch):
                        on_contact(_c, step_i, [_b[i].name for i in _w], to_uint8(t[_w]))
                x = _cycle_throttled(x, cycles[cname], ctx, lock, capture, want, cname, batch, on_contact)
                progress.frame_cycles += len(batch)
                if ci in save_pts and write:
                    cdir = folder / f"c{ci:04d}"
                    cdir.mkdir(exist_ok=True)
                    u8 = finish_chunked(x, recipe, ctx, chunk, lock)
                    for p, img in zip(batch, u8):
                        progress.images_saved += 1
                        pending.append(pool.submit(save_frame, cdir / (p.stem + "." + recipe.output.get("format", "webp")),
                                                   img, icc, recipe.output))
            for f in pending:
                f.result()
            pending.clear()
            if write:
                marker = marker_dir / f"batch_{int(time.time() * 1000)}_{bi:06d}.json"
                marker.write_text(json.dumps({"code": recipe.code(), "iterations": recipe.iterations,
                                              "frames": [p.name for p in batch]}))
            if recorder is not None and want:
                recorder.build()
            progress.frames_done += len(batch)
            progress.cycle = 0
            dt = time.time() - tb
            fc = len(batch) * recipe.iterations
            eta = progress.eta_seconds()
            _log(f"batch {progress.batch}/{progress.batches} done: {len(batch)} frames x {recipe.iterations} cycles"
                 f" = {fc:,} frame-cycles in {dt:.1f}s ({dt / fc * 1000:.2f} ms each)"
                 f" · {progress.images_saved - saved_before} images saved"
                 f" · total {progress.frames_done}/{len(frames)} frames, {progress.frame_cycles:,} frame-cycles,"
                 f" {progress.images_saved:,} images · {_fmt_eta(time.time() - t0)} elapsed"
                 + (f" · ~{_fmt_eta(eta)} left" if eta else ""))
    except StopRequested:
        stopped = True
        _log("stopped; the current batch will be redone on resume")
    finally:
        for f in pending:
            try:
                f.result()
            except Exception:  # noqa: BLE001
                pass
        pool.shutdown(wait=True)
        loader.shutdown(wait=False, cancel_futures=True)

    secs = time.time() - t0
    if write and not stopped:
        _update_catalog(out_root, ident["label"], folder, recipe, gan, src, len(frames))
        _log(f"pass complete in {_fmt_eta(secs)}: {progress.frame_cycles:,} frame-cycles, {progress.images_saved:,} images saved")
    return PassResult(folder=folder, frames=len(frames), seconds=secs, skipped_frames=skipped, stopped=stopped)


CATALOG_FIELDS = ["code", "source", "folder", "slug", "name", "pattern", "cycles", "iterations",
                  "save_every", "rounding", "light", "gan", "size", "frames", "date"]


def _update_catalog(out_root: Path, source_label: str, folder: Path, recipe: Recipe,
                    gan: set[str], src: Size, n: int) -> None:
    path = out_root / "catalog.csv"
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r.get("code") == recipe.code() and r.get("source") == source_label)]
    fin = ", ".join(_describe_step(s) for s in recipe.finish) or "area"
    cyc = ("lead-in: " + ", ".join(_describe_step(s) for s in recipe.lead_in) + " | "
           + " | ".join(f"{k}: " + ", ".join(_describe_step(s) for s in recipe.cycles[k])
                        for k in dict.fromkeys(recipe.pattern))
           + f" | finish: {fin}")
    rows.append({
        "code": recipe.code(), "source": source_label, "folder": str(folder.relative_to(out_root)),
        "slug": recipe.slug(gan), "name": recipe.name, "pattern": ",".join(recipe.pattern),
        "cycles": cyc, "iterations": recipe.iterations, "save_every": recipe.save_every,
        "rounding": recipe.rounding, "light": recipe.light,
        "gan": "yes" if recipe.uses_gan(gan) else "no", "size": str(src), "frames": n,
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
    })
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def _describe_step(s: dict) -> str:
    w = s.get("width", "")
    if s["op"] == "resize":
        extra = []
        if s.get("antialias") is False:
            extra.append("no-AA")
        if s.get("fixed_point"):
            extra.append("fixed-point")
        return f"{s['method']}{'(' + ','.join(extra) + ')' if extra else ''}->{w}"
    if s["op"] == "model":
        return f"{s['model']}->{w if w else 'native'}"
    if s["op"] == "xbr":
        return f"xBR x{s['scale']}->{w if w else 'native'}"
    if s["op"] == "chroma":
        return f"chroma {s.get('mode', '420')} ({s.get('matrix', '601')})"
    if s["op"] == "irn":
        v = {"plus": "IRN+", "finetune": "IRN-finetune"}.get(s.get("variant", "standard"), "IRN")
        return f"{v} {s['direction']} x{s['scale']}->{w if w else 'native'}"
    return f"{s['op']}->{w}"
