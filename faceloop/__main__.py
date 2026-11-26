"""Command line: python -m faceloop <command>

  models list                 show every model, its label and whether it's downloaded
  models fetch [ids|--all]    download (and checksum) models
  selftest                    check the GPU, the WebP writer and every downloaded model
  bench RECIPE --source DIR   time a few frames and estimate the full pass
  run RECIPE --source DIR --out DIR
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


def bench(recipe, source: Path, frames: int = 4, cycles: int | None = None, device=None, models=None):
    """Time `frames` frames through a few cycles; return a dict of estimates."""
    import numpy as np
    from .colour import Transfer, curves_from_icc
    from .engine import StepContext, default_batch_size, pick_device, run_cycle, to_uint8
    from .imageio import list_frames, load_frame, probe, webp_backend
    from .kernels import KernelResizer
    from .sizes import Size
    import io
    device = device or pick_device()
    allf = list_frames(source)
    if not allf:
        raise SystemExit(f"no frames in {source}")
    w, h, icc = probe(allf[0])
    src = Size(w, h)
    bs = default_batch_size(src, recipe)
    n = max(1, min(frames, bs, len(allf)))
    pick = [allf[int(i * (len(allf) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 else allf[:1]
    x0 = torch.from_numpy(np.stack([load_frame(p)[0] for p in pick])).to(device).permute(0, 3, 1, 2).float() / 255
    ctx = StepContext(src=src, transfer=Transfer(curves_from_icc(icc)[0]), light=recipe.light,
                      rounding=recipe.rounding, resizer=KernelResizer(), models=models)
    plen = len(recipe.pattern)
    cycles = cycles or max(plen, min(recipe.iterations, 2 * plen))

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # warm-up: one of each cycle (loads models, builds matrices)
    x = x0
    for k in dict.fromkeys(recipe.pattern):
        x = run_cycle(x, recipe.cycles[k], ctx)
    sync()
    x = x0
    t = time.time()
    for i in range(1, cycles + 1):
        x = run_cycle(x, recipe.cycles[recipe.cycle_at(i)], ctx)
    sync()
    per = (time.time() - t) / (cycles * n)
    # output size estimate
    lib, _ = webp_backend()
    sizes = []
    for img in to_uint8(x):
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
        "device": str(device), "frames_timed": n, "cycles_timed": cycles, "batch_size": bs,
        "sec_per_frame_cycle": per, "frames": total_frames, "iterations": recipe.iterations,
        "est_seconds": per * total_frames * recipe.iterations,
        "est_bytes": avg * total_frames * saves, "saves": saves, "size": str(src),
    }


def cmd_bench(args):
    from .models import ModelManager
    from .engine import pick_device
    r = _load_recipe(args.recipe)
    mm = ModelManager()
    r.validate(mm.known_ids())
    res = bench(r, Path(args.source), frames=args.frames, cycles=args.cycles,
                device=pick_device(args.device), models=mm)
    print(f"device {res['device']}, source {res['frames']} frames at {res['size']}")
    print(f"timed {res['frames_timed']} frames x {res['cycles_timed']} cycles: "
          f"{res['sec_per_frame_cycle'] * 1000:.1f} ms per frame per cycle")
    print(f"estimated pass: {_fmt_secs(res['est_seconds'])} for {res['iterations']} cycles, "
          f"disk {_fmt_bytes(res['est_bytes'])} ({res['saves']} saved loops)")
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
    b.add_argument("--frames", type=int, default=4)
    b.add_argument("--cycles", type=int, default=None)
    r = sub.add_parser("run")
    r.add_argument("recipe")
    r.add_argument("--source", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=None)
    r.add_argument("--limit", type=int, default=None, help="only the first N frames (for trying things)")
    args = p.parse_args(argv)
    return {"models": cmd_models, "selftest": cmd_selftest, "bench": cmd_bench, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
