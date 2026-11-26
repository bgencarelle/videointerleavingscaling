# faceloop: specification

A native Apple Silicon tool that repeatedly downscales and upscales an image
sequence, driven from a small local browser page. This file records every
decision; it is updated whenever a decision changes.

Status: stages 1-5 are built: engine, models, xBR and colour resolution,
contact sheets, GUI.

## Passes

- **Input:** one folder = one sequence of PNG, JPEG or WebP frames, all with
  the same aspect ratio. Frames are sorted by name (numbers in names sort
  numerically).
- **Colour profile:** the source profile is kept and embedded in the output.
  Its tone curve (rTRC/gTRC/bTRC, `curv` or `para`) is used for linear light.
  Missing or unreadable profile: sRGB is assumed and the run log says so.
- **Pass shape:**
  1. **lead-in:** steps run once, taking the source frames down to the
     *loop size* (the size after the lead-in).
  2. **cycles:** A, B, C, ... alternate in `pattern` order. Every cycle starts
     and ends at the loop size: if its last step doesn't end there, an area
     resize lands it. `iterations` counts single cycles.
  3. **finish:** at each save point (every `save_every` cycles and after the
     last), a *copy* is taken back to the source size and saved. The loop
     carries on from the small frames, untouched.
- **Steps:** each step is a target width plus a method. The height follows
  the source aspect, rounded to the nearest whole pixel. Width is pixels,
  `"source"`, or `"loop"` (`"loop"` isn't allowed in the lead-in, which
  defines the loop size). Only widths that keep the aspect exactly avoid
  rounding: for 810x1080 (3:4) that means multiples of 3.
- **Model steps:** run the model at its fixed factor (x1, x2, x3 or x4), then
  an area resize to the step's width. Without a width the model's own output
  size is kept.
- **Finish:** you choose it, like any other steps (in the GUI). Area lands
  it exactly on the source size if its steps don't end there; an empty
  finish is just that area resize. There is no automatic finish. Example:
  loop 162x216 → bicubic → 810x1080. (Earlier versions had an automatic AI
  finish, SwinIR then RealESRNet. They cost 0.9 s and 0.37 s per frame per
  save on the M6, more than the looping itself.)
- Replaces the first design, where every cycle returned to the source size.
- **Queue:** passes are built by hand and run one after another. No grid.

## Methods

| kind | methods |
|---|---|
| kernels | nearest, area, bilinear, bicubic (Keys a=-0.5, like Pillow), Lanczos-3 |
| old-school variants | `antialias: false` (shrink without prefiltering), `fixed_point: true` (phase quantised to 4 bits, weights to 6 bits, 8-bit intermediate) |
| pixel-art upscaler | xBR x2/x3/x4: Hyllian's xBR-lv2 shader (MIT), ported exactly with its defaults (threshold 0.32, smoothness 0.30, AA blending, corner C); works on stored values like the shader. Then area to the step's width. Tested: exactly symmetric under rotation, mirroring and transposing |
| colour resolution | like video codecs: Y'CbCr (BT.601 default, or 709; full range, 8-bit), colour averaged over blocks (4:2:2, 4:2:0, 4:1:1 = DV, 4:1:0 = YUV9) and brought back with nearest; brightness keeps full resolution; size unchanged |
| models | see below |

### Models (all run natively on the Mac GPU)

Every built-in model comes from an official source, pinned by URL (GitHub
release or a fixed commit) and SHA-256 (`faceloop/models/catalog.json`).

| family | label | how it runs |
|---|---|---|
| waifu2x: cunet, upconv_7 anime, upconv_7 photo; noise none/0-3; cunet denoise-only x1 | accuracy | our PyTorch port of the ncnn files from nihui/waifu2x-ncnn-vulkan |
| Real-CUGAN: se x2/x3/x4, pro x2/x3, nose x2; conservative / no-denoise / denoise levels | GAN | the same port, from nihui/realcugan-ncnn-vulkan |
| RealESRNet x4, SwinIR classical x2/x4 | accuracy | spandrel |
| Real-ESRGAN x2plus / x4plus / x4plus-anime, animevideov3, general-x4v3, general-wdn-x4v3 | GAN | spandrel |
| drop-ins: any spandrel-loadable file in `weights/dropin/` | from a same-name `.json` (`{"label": "GAN"}` or `"accuracy"`); unlabelled files count as GAN in pass names | spandrel |
| HAT | depends on the file | drop-in (official weights are on Google Drive) |

- **Native only:** on the Mac GPU the CPU fallback is never enabled. A model
  that needs an op the GPU lacks fails `selftest` and is unavailable. No Rosetta.
- **Excluded as generative:** face restorers (GFPGAN, CodeFormer,
  RestoreFormer), AuraSR, anything diffusion-based.
- **Verification of the ports:** every waifu2x and Real-CUGAN variant was
  compared with the ncnn runtime on the same input. Maximum difference is about
  1e-6, with a pixel or two per image differing by 1/255 at rounding
  boundaries.
- One deliberate difference: the cunet denoise-only (x1) models are padded to
  a multiple of 4 instead of 2, because with 2 the original program loses
  the last row or column for some frame sizes.
- **Alignment:** the finish runs on a copy, never inside the loop, so any
  small offset it has can't build up across cycles.
- waifu2x and SwinIR were measured at about 0 shift. The GAN
  models change content too much to measure to a fraction of a pixel, but
  30-cycle loops showed no visible drift.

## Image maths

- Frames are float32 tensors of *encoded* values (what a file stores), 0..1.
- **Kernel steps** resize in linear light (decode with the tone curve, resize,
  encode). The recipe field `light: "encoded"` switches this off for a pass.
- **Model steps** see encoded values, which is what they were trained on.
- **Rounding:** every step rounds to 8 bits. Default: in sRGB (encoded)
  values, exactly like saving an 8-bit file. Per-pass switch
  `rounding: "linear"` rounds in linear light (harsh shadow banding).
- **One pixel-alignment rule everywhere:** output pixel i covers input
  `[i*s, (i+1)*s)`, centre `(i+0.5)*s`; input pixel j has its centre at
  j+0.5. Edges replicate.
- **Nearest at even factors:** the output centre falls exactly between two
  input pixels. Any fixed choice shifts the picture half a pixel per resize,
  so the tie alternates left/right by output column/row.
- Tested: 40 round trips 240 → 40 → 240 with every down/up kernel pair keep
  the picture within 1.5 px.

## Engine

- **Batches** live on the GPU at the loop size through all their cycles, so
  they can be big when the loop size is small: up to 256 frames, limited by
  20% of RAM for the loop-size frames and 10% for the uint8 source frames
  (current + prefetched batch). The lead-in and the finish work at full
  size, so they run in chunks (from 15% of RAM, at most 64 frames).
- **Prefetch:** the next batch is read from disk while the GPU works on the
  current one. Tested: one frame per batch and 8 frames per batch give
  byte-identical output.
- **Model precision** (per pass): `precision: "fp16"` runs AI models in
  half precision: float16 where the model supports it (all waifu2x /
  Real-CUGAN ports, ESRGAN, Compact), else bfloat16 (SwinIR), else full.
  - The log says which each model got.
  - The original waifu2x-ncnn-vulkan also used fp16 storage by default.
  - It changes pixels, so it's part of the code; flag `h16`.
- **bench** times one batch, then each cycle step on its own (GPU synced
  around it), and prints where one cycle's time goes. The GUI's estimate
  dialog shows the same table. `--precision fp16` compares without editing
  the recipe.

- Frames are processed in batches. The batch size comes from the largest
  image any step produces, using 20% of RAM. Models run in micro-batches,
  falling back to smaller batches and then tiles if the GPU runs out of
  memory.
- Each batch runs through the whole pass in memory. Saved loops are written
  as the batch reaches them, then a marker records the batch.
- Stop or crash: the next run skips finished frames and redoes only the
  unfinished batch. Pause (GUI) holds between cycles without losing anything.
- Full speed; no notifications.

## Outputs

- `OUT/<source folder name>/<slug>_<code>/c0025/ c0050/ ...`
- The `code` is 4 hex characters made from everything that decides the
  pixels: lead-in, cycles, pattern, finish, rounding, light and output
  settings. Iterations
  and save-every are not part of it, so a longer run of the same recipe
  reuses the folder.
- The `slug` is readable: `<lead-in>_<cycles>_<finish>`. Steps are written
  like `area162`, `near54`, `w2xcunetn1@108`, `avv3@L` (L = loop size,
  S = source); cycles are joined with `+`; the finish is written the same way (`area` if empty).
  Then come the flags `GAN`
  (any GAN-labelled model), `rlin` (linear rounding), `enc` (no linear light).
  Long slugs are shortened with `~` but always keep the flags.
- **Frames:** lossy WebP q95 with sharp_yuv through the native libwebp
  (Homebrew), with the source colour profile embedded (VP8X + ICCP chunks).
  If libwebp can't be loaded, Pillow writes it without sharp_yuv and the log
  says so. PNG is available as `output.format: "png"`.
- **Each pass folder:** `recipe.json` (every setting, plus the faceloop,
  torch and libwebp versions, the tone curve, and every model with its
  checksum) and `log.txt`.
- `OUT/catalog.csv`: one row per source × pass.
- **Contact sheets:** `contact_<cycle>.png` for the last run of each cycle
  (last A, last B, ...). Rows are 6 random frames, the same for every pass
  of a source (seeded by the source folder name). Columns are the cycle
  start, every step, and the landing on the loop size; each is scaled to a
  240 px row with nearest. Captures are kept in `.contact/` across batches,
  and the sheet is rebuilt after each batch that contains one of the frames.

## GUI (this Mac only)

- `python -m faceloop gui` or `faceloop.command`. It's a standard-library
  HTTP server on 127.0.0.1:8765, and it refuses any Host header that isn't
  localhost.
- The page is one HTML file with plain JavaScript and no build step.
- Folder pickers use the macOS "choose folder" dialog via osascript.
- Images are served only from output folders the app knows about.
1. **Build:** folders; lead-in, cycles and finish as step lists (add,
   reorder, remove); pattern; cycles total; save every; rounding; light;
   output format. Live preview of every step's size (with the aspect error
   when a height can't be exact), the loop size, the folder name, the GAN
   flag and the save points. Presets are saved to `presets/<name>.json`.
2. **Estimate & queue:** a background estimate you can cancel, with
   live progress.
   - It warms up on 4 frames to measure the speed, then picks a sample size
     so the timed cycles fit about 15 of its 30-second budget, and stops
     timing early when the budget runs out.
   - It checks for cancel before every step, and it is stopped after 90 s
     whatever happens. Measured: cancel frees the GPU in about 1 s.
   - "Queue without estimate" is always offered.
   - It shares a lock with the running pass, which waits between cycles.
   - It shows time, disk and the per-step table, and queues on confirmation.
   - Stop on a running pass also takes effect between steps now.
3. **Queue:** `state/queue.json`, run one at a time. Pause/resume between
   cycles; stop; continue/run again; reorder; remove. A pass that was
   running when the app quit is queued again and resumes from the
   finished frames.
4. **Results:** every pass under an output folder (complete or not), with
   contact sheets, the last saved frame, a filter, a card size, and "Load
   into Build". Each saved loop has a button that makes it the source of a
   new pass (chaining).

## Chaining passes

- A saved loop (`.../<pass>/c0200`) can be the source of another pass.
- Its output is filed under `<parent label>+<parent code><cXXXX>`, e.g.
  `BG_faceSource+8ba5c0200`, which chains further
  (`...+8ba5c0200+1f2ac0050`).
- `recipe.json` records `source.label`, `source.root` (the original source
  folder name) and `source.chained_from`.
- Contact frames are seeded by the root, so every pass in a chain shows the
  same frames.
- **Generational loss** of one extra WebP q95 + sharp_yuv save, measured at
  810x1080:
  - soft, loop-like frames: 52.8-56.6 dB after 1 save, 45.8-56.2 dB after 10
  - sharp test card (worst case): 48.9 dB after 1 save, 40.4 dB after 10

  Chaining from the saved WebPs is accepted as part of the look.

## Install and delivery

- Homebrew Python + venv; Homebrew `webp`; PyTorch (arm64, MPS); spandrel.
- First a zip, then `git format-patch` patches by Ben Gencarelle.

## Build stages

1. Core engine: steps, kernels, colour maths, recipe format, tests. **done**
2. Models: ports, catalogue, downloads, drop-ins, GPU self-test, benchmark. **done**
3. xBR and the colour-resolution step. **done** (Analog and the other
   pixel-art scalers were dropped: only existing algorithms.)
4. Contact sheets. **done**
5. GUI, with the estimate before queueing. **done**
