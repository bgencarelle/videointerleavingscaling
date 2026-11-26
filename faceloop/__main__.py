"""Command line: python -m faceloop <command>

  models list                 show every model, its label and whether it's downloaded
  models fetch [ids|--all]    download (and checksum) models
  selftest                    check the GPU, the WebP writer and every downloaded model
  bench RECIPE --source DIR   time a few frames and estimate the full pass
  run RECIPE --source DIR --out DIR
  gui [--port 8765] [--no-browser]   the local GUI (this Mac only)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _fmt_secs(s: float) -> str:
    s = int(round(s))
    if s < 90:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 90:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def cmd_models(args):
    from .models import ModelManager
    mm = ModelManager()
    if args.action == "list":
        print(f"weights folder: {mm.weights_dir}")
        for fam in ("waifu2x", "realcugan", "spandrel", "dropin"):
            items = sorted((v for v in mm.catalog.values() if v.family == fam), key=lambda v: v.id)
            if not items:
                continue
            print(f"\n{fam}")
            for v in items:
                have = "yes" if mm.is_downloaded(v.id) else " - "
                sc = f"x{v.scale}" if v.scale else "x?"
                print(f"  [{have}] {v.id:28s} {sc:3s} {v.label:10s} {v.description}")
        for m in mm.manual:
            print(f"\nmanual: {m['id']}: {m['description']}")
        return 0
    ids = sorted(mm.catalog) if args.all else args.ids
    if not ids:
        print("name model ids, or use --all")
        return 2
    for mid in ids:
        if mid not in mm.catalog:
            print(f"unknown model {mid}")
            return 2
        if mm.catalog[mid].family == "dropin":
            continue
        t = time.time()
        mm.download(mid)
        print(f"ok  {mid}  ({time.time() - t:.1f}s)")
    return 0


def cmd_selftest(args):
    from .engine import pick_device, total_ram_bytes
    from .imageio import webp_backend
    from .kernels import KernelResizer
    from .models import ModelManager
    import platform
    ok_all = True
    dev = pick_device(args.device)
    print(f"python {platform.python_version()} on {platform.machine()} / {platform.platform()}")
    print(f"torch {torch.__version__}; device: {dev}; RAM {_fmt_bytes(total_ram_bytes())}")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        print("FAIL: this Python is not native arm64 (Rosetta?). Use Homebrew's Python on Apple Silicon.")
        ok_all = False
    if dev.type != "mps" and platform.system() == "Darwin":
        print("FAIL: the Mac GPU (MPS) is not available to PyTorch")
        ok_all = False
    lib, desc = webp_backend()
    print(f"webp writer: {desc}")
    if lib is None:
        ok_all = False
    # kernels on the device agree with the CPU
    r = KernelResizer()
    x = torch.rand(2, 3, 72, 96)
    a = r.resize(r.resize(x, 40, 30, "lanczos"), 96, 72, "bicubic")
    b = r.resize(r.resize(x.to(dev), 40, 30, "lanczos"), 96, 72, "bicubic").cpu()
    d = float((a - b).abs().max())
    print(f"kernels on {dev}: max difference vs CPU {d:.2e} {'ok' if d < 1e-4 else 'FAIL'}")
    ok_all &= d < 1e-4
    mm = ModelManager()
    have = [m for m in sorted(mm.catalog) if mm.is_downloaded(m)]
    print(f"\nmodels downloaded: {len(have)} of {len(mm.catalog)}")
    for mid in have:
        t = time.time()
        ok, msg = mm.check(mid, dev)
        print(f"  {'ok  ' if ok else 'FAIL'} {mid:28s} {time.time() - t:5.2f}s  {'' if ok else msg}")
        ok_all &= ok
    print("\nALL OK" if ok_all else "\nSOME CHECKS FAILED")
    return 0 if ok_all else 1


def _load_recipe(path):
    from .recipe import Recipe
    return Recipe.load(path)


class EstimateCancelled(Exception):
    pass


def bench(recipe, source: Path, frames: int | None = None, cycles: int | None = None, device=None,
          models=None, profile: bool = True, should_stop=None, budget: float = 30.0, on_progress=None):
    """Time a sample of frames through a few cycles; return estimates plus a
    per-step breakdown (each step timed on its own, with the GPU synced).

    Stays short on purpose: a warm-up on 4 frames measures the speed, the
    sample size is chosen so the timed part fits in `budget` seconds, and
    timing stops early when the budget runs out. `should_stop()` is checked
    between steps; when it returns True, EstimateCancelled is raised."""
    t_start = time.time()

    def say(msg):
        if on_progress:
            on_progress(msg)

    def check():
        if should_stop and should_stop():
            raise EstimateCancelled()
    import io
    import numpy as np
    from .colour import Transfer, curves_from_icc
    from .engine import (StepContext, apply_step, batch_sizes, finish_chunked, lead_in_chunked,
                         pick_device, run_cycle, _kernel)
    from .colour import quantize
    from .imageio import list_frames, load_frame, probe, webp_backend
    from .kernels import KernelResizer
    from .contact import step_label
    from .sizes import Size
    device = device or pick_device()
    allf = list_frames(source)
    if not allf:
        raise SystemExit(f"no frames in {source}")
    w, h, icc = probe(allf[0])
    src = Size(w, h)
    bs, chunk = batch_sizes(src, recipe, models=models)
    n_max = max(1, min(frames or 64, bs, len(allf)))
    n_warm = min(4, n_max)
    pick_all = [allf[int(i * (len(allf) - 1) / max(1, n_max - 1))] for i in range(n_max)] if n_max > 1 else allf[:1]
    say(f"reading {n_warm} frames")
    arrs = [load_frame(p)[0] for p in pick_all[:n_warm]]
    ctx = StepContext(src=src, transfer=Transfer(curves_from_icc(icc)[0]), light=recipe.light,
                      rounding=recipe.rounding, resizer=KernelResizer(), models=models,
                      precision=recipe.precision, check=check)
    plen = len(recipe.pattern)
    cycles = cycles or max(plen, min(recipe.iterations, 2 * plen))

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # warm-up on a few frames: loads models, builds matrices, measures speed
    say("loading models and warming up")
    x = lead_in_chunked(arrs, recipe, ctx, chunk, device)
    for k in dict.fromkeys(recipe.pattern):
        check()
        x = run_cycle(x, recipe.cycles[k], ctx)
    sync()
    t = time.time()
    for k in dict.fromkeys(recipe.pattern):
        check()
        x = run_cycle(x, recipe.cycles[k], ctx)
    sync()
    rough = (time.time() - t) / (plen * n_warm)  # seconds per frame-cycle
    # sample size: the timed cycles should take about half the budget
    n = int(max(n_warm, min(n_max, (budget * 0.5) / max(rough * cycles, 1e-6))))
    if n > n_warm:
        say(f"reading {n} frames")
        arrs += [load_frame(p)[0] for p in pick_all[n_warm:n]]
    check()
    t = time.time()
    x0 = lead_in_chunked(arrs[:n], recipe, ctx, chunk, device)
    sync()
    per_lead = (time.time() - t) / n
    x = x0
    t = time.time()
    done = 0
    for i in range(1, cycles + 1):
        check()
        say(f"timing cycle {i} of {cycles} on {n} frames")
        x = run_cycle(x, recipe.cycles[recipe.cycle_at(i)], ctx)
        done = i
        sync()
        if time.time() - t_start > budget and i >= plen:
            break
    per = (time.time() - t) / (done * n)
    cycles = done
    check()
    say("timing the finish")
    k_fin = min(n, chunk, 8)
    t = time.time()
    fin = finish_chunked(x[:k_fin], recipe, ctx, chunk)
    sync()
    per_finish = (time.time() - t) / k_fin

    steps = []
    if profile:
        def timed(label, fn, xin):
            sync()
            t1 = time.time()
            y = fn(xin)
            sync()
            steps.append({"where": label[0], "step": label[1], "ms": (time.time() - t1) / xin.shape[0] * 1000})
            return y
        say("timing each step")
        k_prof = min(n, 8)
        for k in dict.fromkeys(recipe.pattern):
            xx = x0[:k_prof]
            for st in recipe.cycles[k]:
                check()
                xx = timed((f"cycle {k}", step_label(st)), lambda a, st=st: apply_step(a, st, ctx), xx)
        cyc_total = sum(st["ms"] for st in steps) or 1.0
        for st in steps:
            st["share"] = st["ms"] / cyc_total

    lib, _ = webp_backend()
    sizes = []
    for img in fin:
        if recipe.output.get("format", "webp") == "webp" and lib is not None:
            sizes.append(len(lib.encode_rgb(img, float(recipe.output.get("quality", 95)),
                                            bool(recipe.output.get("sharp_yuv", True)))))
        else:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "PNG" if recipe.output.get("format") == "png" else "WEBP", quality=95)
            sizes.append(len(buf.getvalue()))
    avg = sum(sizes) / len(sizes)
    total_frames = len(allf)
    saves = len(recipe.save_points())
    return {
        "seconds_taken": time.time() - t_start,
        "device": str(device), "frames_timed": n, "cycles_timed": cycles, "batch_size": bs, "chunk": chunk,
        "sec_per_frame_cycle": per, "sec_per_frame_finish": per_finish, "sec_per_frame_lead": per_lead,
        "frames": total_frames, "iterations": recipe.iterations, "loop": str(ctx.loop),
        "est_seconds": (per * recipe.iterations + per_finish * saves + per_lead) * total_frames,
        "est_bytes": avg * total_frames * saves, "saves": saves, "size": str(src), "steps": steps,
        "precision": recipe.precision,
    }


def cmd_bench(args):
    from .models import ModelManager
    from .engine import pick_device
    r = _load_recipe(args.recipe)
    if args.precision:
        r.precision = args.precision
    mm = ModelManager()
    r.validate(mm.known_ids())
    res = bench(r, Path(args.source), frames=args.frames, cycles=args.cycles,
                device=pick_device(args.device), models=mm)
    print(f"device {res['device']}, source {res['frames']} frames at {res['size']}, loop size {res['loop']}")
    print(f"timed {res['frames_timed']} frames x {res['cycles_timed']} cycles: "
          f"{res['sec_per_frame_cycle'] * 1000:.1f} ms per frame per cycle, "
          f"finish {res['sec_per_frame_finish'] * 1000:.1f} ms per frame per save")
    print(f"estimated pass: {_fmt_secs(res['est_seconds'])} for {res['iterations']} cycles, "
          f"disk {_fmt_bytes(res['est_bytes'])} ({res['saves']} saved loops)")
    print(f"batches of {res['batch_size']} frames, full-size work in chunks of {res['chunk']}, "
          f"models in {res['precision']}")
    if res["steps"]:
        print("\nwhere one cycle's time goes (each step timed on its own):")
        for st in res["steps"]:
            bar = "#" * int(round(st["share"] * 40))
            print(f"  {st['where']:9s} {st['step'][:42]:42s} {st['ms']:8.2f} ms/frame {st['share'] * 100:5.1f}% {bar}")
    return 0


def cmd_run(args):
    from .engine import pick_device, run_pass
    from .models import ModelManager
    r = _load_recipe(args.recipe)
    mm = ModelManager()
    t = time.time()
    res = run_pass(r, Path(args.source), Path(args.out), models=mm, device=pick_device(args.device),
                   batch_size=args.batch, frame_limit=args.limit, log=print)
    print(f"done: {res.folder} ({_fmt_secs(time.time() - t)})")
    return 0


def cmd_gui(args):
    from .gui import App, serve
    from .engine import pick_device
    serve(port=args.port, open_browser=not args.no_browser, app=App(device=pick_device(args.device)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="faceloop")
    p.add_argument("--device", default=None, help="mps / cpu (default: best available)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models")
    m.add_argument("action", choices=["list", "fetch"])
    m.add_argument("ids", nargs="*")
    m.add_argument("--all", action="store_true")
    sub.add_parser("selftest")
    b = sub.add_parser("bench")
    b.add_argument("recipe")
    b.add_argument("--source", required=True)
    b.add_argument("--frames", type=int, default=None, help="frames to time (default: one full batch)")
    b.add_argument("--cycles", type=int, default=None)
    b.add_argument("--precision", choices=["fp32", "fp16"], default=None,
                   help="override the recipe's model precision, to compare")
    r = sub.add_parser("run")
    r.add_argument("recipe")
    r.add_argument("--source", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=None)
    r.add_argument("--limit", type=int, default=None, help="only the first N frames (for trying things)")
    g = sub.add_parser("gui")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    return {"models": cmd_models, "selftest": cmd_selftest, "bench": cmd_bench, "run": cmd_run, "gui": cmd_gui}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
