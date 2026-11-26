"""Runs a pass over one source folder.

Frames are processed in batches. Each batch runs the lead-in once (down to
the loop size), then every cycle in memory at the loop size. At each save
point a copy is finished back to the source size and written. When a
batch is done, a marker records it, so a stopped or crashed run resumes at
the first unfinished batch (and re-runs only that batch).
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import random
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
    if ctx.light == "linear":
        y = ctx.resizer.resize(ctx.transfer.decode(x), target.width, target.height, method, **kw)
        return ctx.transfer.encode(y.clamp(0, 1))
    return ctx.resizer.resize(x, target.width, target.height, method, **kw).clamp(0, 1)


def apply_step(x: torch.Tensor, step: dict, ctx: StepContext) -> torch.Tensor:
    cur = Size(x.shape[-1], x.shape[-2])
    op = step["op"]
    if op == "resize":
        kw = {k: step[k] for k in ("antialias", "fixed_point", "phase_bits", "coef_bits") if k in step}
        x = _kernel(x, _target(step, ctx.src, cur, ctx.loop), step["method"], ctx, **kw)
    elif op == "model":
        if ctx.models is None:
            raise RecipeError("model step used but no models are available")
        model = ctx.models.get(step["model"], x.device)
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
    """A cycle starts and ends at the loop size (the source size if unset)."""
    return run_steps(x, steps, ctx, ctx.loop or ctx.src, capture)


def run_lead_in(x: torch.Tensor, recipe: Recipe, ctx: StepContext) -> torch.Tensor:
    x = run_steps(x, recipe.lead_in, ctx, None)
    loop = Size(x.shape[-1], x.shape[-2])
    if ctx.loop is not None and ctx.loop != loop:
        raise RecipeError(f"lead-in gave {loop}, expected {ctx.loop}")
    ctx.loop = loop
    return x


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
        if s.get("op") in ("model", "xbr"):
            k = int(s.get("scale", 4))  # models: assume up to x4 before landing
            biggest = max(biggest, cur.width * cur.height * k * k)
            cur = tgt or Size(cur.width * k, cur.height * k)
        elif tgt is not None:
            cur = tgt
        biggest = max(biggest, cur.width * cur.height)
    return cur, biggest


def default_batch_size(src: Size, recipe: Recipe, ram: int | None = None) -> int:
    """Frames per batch, from the largest image any step produces."""
    ram = ram or total_ram_bytes()
    loop, biggest = _sim(recipe.lead_in, src, src, None, src.width * src.height)
    for k in set(recipe.pattern):
        _, biggest = _sim(recipe.cycles[k], loop, src, loop, biggest)
    _, biggest = _sim(recipe.finish, loop, src, loop, biggest)
    per_frame = biggest * 3 * 4 * 4  # float32, a few temporaries
    budget = ram * 0.20
    return int(max(1, min(64, budget // per_frame)))


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
             write: bool = True) -> PassResult:
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

    folder = out_root / source.name / recipe.folder_name(gan)
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
        d["source"] = {"folder": str(source), "frames": len(frames), "size": str(src)}
        (folder / "recipe.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

    _log(f"pass {folder.name} on {source.name}: {len(frames)} frames at {src}, device {device}")
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

    bs = batch_size or default_batch_size(src, recipe)
    save_pts = set(recipe.save_points())
    contact_idx = set(contact_indices(len(frames), source.name))
    contact_names = {frames[i].name for i in contact_idx}
    last_of_cycle = {k: max(i for i in range(1, recipe.iterations + 1) if recipe.cycle_at(i) == k)
                     for k in set(recipe.pattern) if any(recipe.cycle_at(i) == k for i in range(1, recipe.iterations + 1))}

    ctx = StepContext(src=src, transfer=transfer, light=recipe.light, rounding=recipe.rounding,
                      resizer=KernelResizer(), models=models)

    progress.frames_total = len(frames)
    progress.frames_done = skipped
    progress.cycles = recipe.iterations
    progress.batches = -(-len(todo) // bs) if todo else 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))
    pending = []
    stopped = False
    try:
        for bi in range(0, len(todo), bs):
            batch = todo[bi:bi + bs]
            progress.batch = bi // bs + 1
            arrs = list(pool.map(lambda p: load_frame(p)[0], batch))
            for p, a in zip(batch, arrs):
                if a.shape[:2] != (src.height, src.width):
                    raise RecipeError(f"{p.name} is {a.shape[1]}x{a.shape[0]}, expected {src}")
            x = torch.from_numpy(np.stack(arrs)).to(device).permute(0, 3, 1, 2).float() / 255.0
            x = run_lead_in(x, recipe, ctx)
            want = [i for i, p in enumerate(batch) if p.name in contact_names]

            for ci in range(1, recipe.iterations + 1):
                controls.checkpoint()
                progress.cycle = ci
                cname = recipe.cycle_at(ci)
                capture = None
                if on_contact and want and last_of_cycle.get(cname) == ci:
                    def capture(step_i, t, _c=cname, _w=want, _b=batch):
                        on_contact(_c, step_i, [_b[i].name for i in _w], to_uint8(t[_w]))
                x = run_cycle(x, recipe.cycles[cname], ctx, capture)
                if ci in save_pts and write:
                    cdir = folder / f"c{ci:04d}"
                    cdir.mkdir(exist_ok=True)
                    u8 = to_uint8(run_finish(x, recipe, ctx))
                    for p, img in zip(batch, u8):
                        pending.append(pool.submit(save_frame, cdir / (p.stem + "." + recipe.output.get("format", "webp")),
                                                   img, icc, recipe.output))
            for f in pending:
                f.result()
            pending.clear()
            if write:
                marker = marker_dir / f"batch_{int(time.time() * 1000)}_{bi:06d}.json"
                marker.write_text(json.dumps({"code": recipe.code(), "iterations": recipe.iterations,
                                              "frames": [p.name for p in batch]}))
            progress.frames_done += len(batch)
            progress.cycle = 0
            _log(f"batch {progress.batch}/{progress.batches} done ({progress.frames_done}/{len(frames)} frames)")
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

    secs = time.time() - t0
    if write and not stopped:
        _update_catalog(out_root, source, folder, recipe, gan, src, len(frames))
        _log(f"pass complete in {secs / 60:.1f} min")
    return PassResult(folder=folder, frames=len(frames), seconds=secs, skipped_frames=skipped, stopped=stopped)


CATALOG_FIELDS = ["code", "source", "folder", "slug", "name", "pattern", "cycles", "iterations",
                  "save_every", "rounding", "light", "gan", "size", "frames", "date"]


def _update_catalog(out_root: Path, source: Path, folder: Path, recipe: Recipe,
                    gan: set[str], src: Size, n: int) -> None:
    path = out_root / "catalog.csv"
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r.get("code") == recipe.code() and r.get("source") == source.name)]
    fin = ", ".join(_describe_step(s) for s in recipe.finish) or "area"
    cyc = ("lead-in: " + ", ".join(_describe_step(s) for s in recipe.lead_in) + " | "
           + " | ".join(f"{k}: " + ", ".join(_describe_step(s) for s in recipe.cycles[k])
                        for k in dict.fromkeys(recipe.pattern))
           + f" | finish: {fin}")
    rows.append({
        "code": recipe.code(), "source": source.name, "folder": str(folder.relative_to(out_root)),
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
    return f"{s['op']}->{w}"
