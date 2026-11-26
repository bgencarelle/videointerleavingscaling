# faceloop

Repeatedly downscale and upscale an image sequence, on the Apple Silicon GPU.
Every decision is recorded in [SPEC.md](SPEC.md).

This first version has the engine and the models, driven from the Terminal.
The browser GUI comes in a later stage.

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
# how long would this pass take on all frames?
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
- **cycles:** alternate in `pattern` order at the loop size. `iterations`
  counts single cycles.
- **finish:** at every save point, a copy is taken back to full size.

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
  "finish": "auto"
}
```

- `method`: `nearest`, `area`, `bilinear`, `bicubic`, `lanczos`
  - `"antialias": false`: shrink without prefiltering
  - `"fixed_point": true`: low-precision maths like old hardware scalers
- `width`: pixels, `"source"`, or `"loop"`. Height follows the source aspect;
  for 810x1080 frames, multiples of 3 keep it exact.
- A model step runs the model, then area-resizes to `width`.
- A cycle that doesn't end at the loop size is landed there with area.
- `finish`: `"auto"` uses RealESRNet x4 while `finish_threshold` (default 2)
  or more is left, then area for the rest. Or give a list of steps. It
  always lands exactly on the source size.
- `rounding`: `"srgb"` (default) or `"linear"` (harsh shadow banding).
- `light`: `"linear"` (default) or `"encoded"`.
- `output`: default `{"format": "webp", "quality": 95, "sharp_yuv": true}`;
  `"png"` also works.

## Models

```bash
.venv/bin/python -m faceloop models list
```

The list shows every model id, its factor, and its label (`accuracy` or
`GAN`). Passes that use a GAN-labelled model get `_GAN` in their folder name.

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
    demo_area162_near54-w2xcunetn1@108-avv3@L+..._esrnet_GAN_xxxx/
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
