# faceloop

Repeatedly downscale and upscale an image sequence, on the Apple Silicon GPU.
Every decision is recorded in [SPEC.md](SPEC.md).

## The GUI

Double-click `faceloop.command` in Finder, or run:

```bash
.venv/bin/python -m faceloop gui
```

It opens `http://127.0.0.1:8765` in your browser. The page is served only to
this Mac. Close it with Ctrl-C in the Terminal window. Queued and stopped
passes are still there next time.

- **Build:** choose the source and output folders, then build the pass:
  - lead-in steps (once, down to the loop size)
  - one or more cycles of steps (each step is a method plus a size)
  - the pattern (e.g. `A, B`), the number of cycles, and how often to save
  - the finish (a copy at each save, landed on the source size or left where it ends)

  Every step shows the size it produces, and the folder name updates as you
  go. Save and load your own presets. **Recent** lists finished passes
  (from `catalog.csv`, newest first), and Load puts that exact recipe back
  in Build. "Estimate & queue…" times one batch on
  the GPU, shows hours and disk space, and asks before adding the pass to
  the queue.
- **Queue:** passes run one after another, with progress and time left.
  - "Leave for the Mac" (Off / 10% / 25%) keeps the Mac responsive while a
    pass runs, by idling the GPU in short gaps and using fewer CPU threads.
    The output doesn't change, only the speed.
  - faceloop always leaves 4 GB of memory free, so the Mac doesn't start
    swapping.
  - Passes run cycle by cycle: all frames go through cycle 1, then all
    through cycle 2, and so on, so the progress reads "cycle 37 of 200" and
    a saved loop is complete as soon as its cycle is done. Frames between
    cycles are kept in the pass folder's `.work/` (raw, deleted at the end).
  - Pause/Resume holds without losing anything; Stop and Continue picks up
    in the same cycle where it stopped.
  - Stop ends after the current step; Continue picks up where it stopped
    (finished frames are skipped).
- **Results:** every pass in an output folder, filterable. Each card shows a
  contact sheet per cycle and the last saved frame.
  - "Load into Build" copies a pass's recipe back to the Build tab, ready to
    tweak.
  - The saved-loop buttons (c25 ▸, c200 ▸, ...) make that loop the source of
    a new pass, so passes chain.
- **Log:** everything in one place: each pass's log lines and recipe,
  estimates with their timings, full errors, and anything printed in the
  terminal, kept in `state/log.txt`. "Copy latest" copies this Mac's setup
  plus everything since the last pass or estimate started, ready to paste
  when asking for help; "only errors and warnings" filters it.
  - A chained pass is filed under where it came from, e.g.
    `BG_faceSource+8ba5c0200/...`, and its contact sheets use the same
    frames as the original source.
  - Each link in a chain adds one more WebP q95 save. Measured on 810x1080
    frames, that's about 53-57 dB per save for soft loop-like frames,
    invisible next to what one loop cycle does.

**Contact sheets:** each pass has a `contact_A.png` (B, ...) for the last run
of each cycle. Rows are the same randomly chosen frames for every pass of a
source; columns are the cycle's start and every step, each scaled with
nearest so small sizes show their real pixels.

The Terminal commands below still work; recipes are the same JSON the GUI
saves as presets (in `presets/`).

## Setup (once)

Needs [Homebrew](https://brew.sh).

```bash
./setup.sh
.venv/bin/python -m faceloop models fetch --all   # about 430 MB, checksummed
.venv/bin/python -m faceloop selftest
```

`selftest` checks that Python is native arm64, that PyTorch sees the Mac GPU,
that the WebP writer can use sharp_yuv, that resizing on the GPU matches the
CPU, and that every downloaded model runs natively on the GPU. Each model is
listed as ok or FAIL.

## Try it

```bash
# how long would this pass take on all frames? (times one full batch)
.venv/bin/python -m faceloop bench examples/demo.json --source /path/to/frames

# run it on the first 20 frames only
.venv/bin/python -m faceloop run examples/demo.json --source /path/to/frames --out /path/to/runs --limit 20

# the whole folder
.venv/bin/python -m faceloop run examples/demo.json --source /path/to/frames --out /path/to/runs
```

Stopping with Ctrl-C is safe. Running the same command again skips finished
frames.

## Recipes

A recipe is a JSON file with three parts:

- **lead-in:** runs once, from the source down to the loop size.
- **cycles:** alternate in `pattern` order, starting at the loop size. Each
  cycle continues from wherever the previous one ended. `iterations` counts
  single cycles.
- **finish:** at every save point, a copy goes through the finish steps and
  is saved, back at the source size or (if you choose) at whatever size the
  finish ends.

```json
{
  "name": "demo",
  "lead_in": [
    {"op": "resize", "method": "area", "width": 162}
  ],
  "cycles": {
    "A": [
      {"op": "resize", "method": "nearest", "width": 54},
      {"op": "model",  "model": "w2x-cunet-n1",         "width": 108},
      {"op": "model",  "model": "realesr-animevideov3", "width": "loop"}
    ],
    "B": [
      {"op": "resize", "method": "bicubic", "width": 81, "antialias": false},
      {"op": "model",  "model": "rcugan-x2", "width": "loop"}
    ]
  },
  "pattern": ["A", "B"],
  "iterations": 200,
  "save_every": 25,
  "finish": [
    {"op": "resize", "method": "bicubic", "width": "source"}
  ]
}
```

- `method`: `nearest`, `area`, `bilinear`, `bicubic`, `lanczos`
  - `"antialias": false`: shrink without prefiltering
  - `"fixed_point": true`: low-precision maths like old hardware scalers
- `{"op": "xbr", "scale": 2, "width": 160}`: xBR pixel-art upscale (2, 3 or
  4), then area to `width` (optional).
- `{"op": "resize", "method": "dpid", "width": 54, "lambda": 1.0}`: DPID,
  detail-preserving downscaling (keeps small highlights that stand out).
  `lambda` 0 = plain box filter; higher = more emphasis on standout pixels.
- `{"op": "irn", "direction": "down", "scale": 4, "width": 54}`: IRN, an AI
  model trained as a matched shrink (`down`) and enlarge (`up`) pair, then
  area to `width` (optional). `"variant"`: `"standard"` (×2/×4/×8, the
  default), `"plus"` (IRN+ ×4, GAN-trained) or `"finetune"` (×2,
  JPEG-robust).
- `{"op": "chroma", "mode": "420"}`: colour resolution reduced like a video
  codec: `422`, `420`, `411` (DV), `410` (YUV9). Optional `"matrix": "709"`.
  The size doesn't change, so there's no width.
- `width`: pixels, `"source"`, or `"loop"`. Height follows the source aspect;
  for 810x1080 frames, multiples of 3 keep it exact.
- A model step runs the model, then area-resizes to `width`.
- There's no automatic resize between cycles. If no cycle has a step with a
  fixed size, so the frames would keep growing, the first cycle
  automatically starts with the lead-in; Build shows it as a dashed row.
- `finish`: a list of steps run on a copy at each save (e.g. one bicubic
  resize). Area lands them exactly on the source size; `[]` means just area.
- `land_on_source`: `true` (default) or `false`: save at whatever size the
  finish ends (`[]` then saves the loop frames as they are).
- `rounding`: `"srgb"` (default) or `"linear"` (harsh shadow banding).
- `precision`: `"fp32"` or `"fp16"`: AI models in half precision, often
  much faster, with slightly different pixels; needed for IRN on the Neural
  Engine. A recipe file without it means `"fp32"`; new recipes in the GUI
  start at `"fp16"`.
  `bench ... --precision fp16` lets you compare without editing anything.
- `light`: `"linear"` (default) or `"encoded"`.
- `output`: default `{"format": "webp", "quality": 95, "sharp_yuv": true}`;
  `"png"` also works.

## Models

```bash
.venv/bin/python -m faceloop models list
```

The list shows every model id, its factor, and its label (`accuracy` or
`GAN`). Passes that use a GAN-labelled model get `_GAN` in their folder name.

**IRN weights** are only on the authors' Google Drive. Download any of
`IRN_x2.pth`, `IRN_x4.pth`, `IRN_x8.pth`, `IRN+_x4.pth` (GAN-trained) and
`IRN_x2_finetune.pth` (JPEG-robust) from
[their folder](https://drive.google.com/drive/folders/1ym6DvYNQegDrOy_4z733HxrULa1XIN92)
(linked from github.com/pkuxmq/Invertible-Image-Rescaling) and put them in
`weights/irn/`. Then run `selftest`, which checks they load and run.

Content-aware and edge-directed methods (no AI): in a resize step, pick
perceptual or content-adaptive (Kopf); the "edge / pixel ×2" step does MMPX, DCCI, NEDI, ICBI or EEDI3
at ×2/×4/×8. `faceloop/eedi.py` is a port of GPL-2 code and carries that
licence. See SPEC.md for what each does and how it was checked.

IRN is a heavy network on the GPU. With half precision on, the first time a
pass uses it at a frame size it also tries the Neural Engine (via Core ML,
`coremltools` from `requirements.txt`), checks the result matches, and uses
whichever is faster. The log says which and how fast. After pulling this
change, run `.venv/bin/python -m pip install -r requirements.txt` once.

**Your own models:** put a `.pth` or `.safetensors` file in `weights/dropin/`.
Add a same-name `.json` to label it:

```json
{"label": "accuracy", "name": "HAT SRx4 ImageNet-pretrain"}
```

Unlabelled files count as GAN in folder names until you label them. Model
files are loaded with spandrel's restricted unpickler, so a model file can't
run code.

## Output

```
runs/
  catalog.csv
  <source folder>/
    demo_area162_near54-w2xcunetn1@108-avv3@L+..._bicS_GAN_xxxx/
      c0025/ c0050/ ...     frames, named like the sources, .webp
      recipe.json           every setting + versions + model checksums
      log.txt
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

With `ncnn` installed (in requirements-dev) and the weights downloaded, the
tests also compare the waifu2x port against the original ncnn runtime.
