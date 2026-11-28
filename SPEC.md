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
  2. **cycles:** A, B, C, ... alternate in `pattern` order. `iterations`
     counts single cycles.
     - The first cycle starts at the loop size. After that, each cycle ends
       wherever its last step ends, and the next cycle's first step takes it
       from there; there's no automatic resize in between.
     - The sizes must settle into a repeating pattern. They always do if any
       cycle has a step with a fixed size. If none does (e.g. only ×2 and
       colour steps), the first cycle of the pattern starts with the
       lead-in steps, added automatically: that one fixed point is enough,
       and nothing is added otherwise.
     - Build shows these as dashed rows ("lead-in, added automatically"),
       the run log says so, and contact sheet columns are marked
       "(lead-in)".
     - Patterns bigger than 16× the source's pixels are refused.
     - (Until version 5 every cycle was landed back on the loop size with
       area. That threw away the last step's detail and repeated the resize
       the next cycle does anyway.)
  3. **finish:** at each save point (every `save_every` cycles and after the
     last), a *copy* is taken from wherever the cycles left it through the
     finish steps and saved. The loop carries on, untouched.
- **Steps:** each step is a target width plus a method. The height follows
  the source aspect, rounded to the nearest whole pixel. Width is pixels,
  `"source"`, or `"loop"` (`"loop"` isn't allowed in the lead-in, which
  defines the loop size). Only widths that keep the aspect exactly avoid
  rounding: for 810x1080 (3:4) that means multiples of 3.
- **Model steps:** run the model at its fixed factor (x1, x2, x3 or x4), then
  an area resize to the step's width. Without a width the model's own output
  size is kept.
- **Finish:** you choose it, like any other steps (in the GUI). "Ends at"
  (`land_on_source`): the source size (default; area lands it exactly if
  its steps don't end there, an empty finish is just that area resize) or
  wherever the last step ends (saved at that size; an empty finish saves the
  loop frames as they are; the slug gets `free`). Example:
  loop 162x216 → bicubic → 810x1080. (Earlier versions had an automatic AI
  finish, SwinIR then RealESRNet. They cost 0.9 s and 0.37 s per frame per
  save on the M6, more than the looping itself.)
- Replaces the first design, where every cycle returned to the source size.
- **Queue:** passes are built by hand and run one after another. No grid.

## Methods

| kind | methods |
|---|---|
| kernels | nearest, area, bilinear, bicubic (Keys a=-0.5, like Pillow), Lanczos-3 |
| more kernels | Lanczos 2/4/8 lobes; the BC-spline family: Mitchell-Netravali (1/3, 1/3), Catmull-Rom (0, 1/2), B-spline (1, 0), Hermite (0, 0); Spline16/36/64 (Panorama Tools / AviSynth formulas); Gaussian (σ in input pixels, default 0.5, support 3σ); Magic Kernel Sharp 2021 (Costella: magic kernel convolved with [−1, 6, −35, 204, −35, 6, −1]/144, support 4.5). All separable, same pixel-centre convention, antialias and fixed-point variants. Tested: rows sum to 1, the piecewise ones are continuous, the interpolating ones carry linear ramps through enlarging |
| round (EWA) filters | elliptical weighted averages like mpv and ImageMagick's distort resize: EWA Lanczos (jinc windowed by jinc, radius 3.2383 = jinc's third zero), EWA Lanczos sharp (mpv's blur 0.9812505644269356), EWA Robidoux and RobidouxSharp (ImageMagick's cylindrical cubics, B/C 0.3782/0.3109 and 0.2620/0.3690). The footprint stretches with the shrink factor on each axis (antialias); no fixed-point. Tested: flat colours stay flat, a dot spreads the same along x and y |
| frequency | dct: the frame's DCT-II cut to (or zero-padded out to) the new size and transformed back, scaled by sqrt(new/old) per axis; identical to scipy's orthonormal dct/idct round trip; mirror edges. fft: the same with the Fourier transform (scipy.signal.resample on its grid; at even shrink sizes the Nyquist bin is split half and half so it stays real off-grid), periodic: edges wrap. wavelet53 / wavelet97: JPEG 2000's LeGall 5/3 and CDF 9/7, whole-sample symmetric extension; one low band per halving, the inverse with empty detail bands per doubling, the rest of the factor by dct, which also undoes the wavelets' sample offset (a bump stays put to 1e-6 px). "antialias": false keeps every input frequency when shrinking. No fixed-point. All are plain matrices per axis, as fast as Lanczos. Note: the 5/3 low band lifts mid frequencies (up to 12.5% per halving), so shrinking with it in a loop grows a fine pattern that ends up saturating; nearest down + any spectral up does the same |
| old-school variants | `antialias: false` (shrink without prefiltering), `fixed_point: true` (phase quantised to 4 bits, weights to 6 bits, 8-bit intermediate) |
| pixel-art upscaler | xBR x2/x3/x4: Hyllian's xBR-lv2 shader (MIT), ported exactly with its defaults (threshold 0.32, smoothness 0.30, AA blending, corner C); works on stored values like the shader. Then area to the step's width. Tested: exactly symmetric under rotation, mirroring and transposing |
| detail-preserving downscale | **DPID** (Weber et al., SIGGRAPH Asia 2016), from the authors' CUDA reference (BSD 3-Clause): area average, then a 3×3-smoothed guide, then input pixels weighted by coverage × (distance to guide)^λ. λ 0 = box filter, default 1. Uses the same light setting as the other plain resizes. Matches a loop-for-loop transcription of the reference to 1e-5 (20 size/λ combinations) |
| learned shrink / enlarge | **IRN** ×2/×4/×8, **IRN+** ×4 (GAN-trained, labelled GAN), **IRN ×2 finetune** (JPEG-robust) (Invertible Rescaling Network, Xiao et al., ECCV 2020), architecture ported from the authors' code (Apache 2.0). It produces the same output as their classes (difference 0). Shrink = forward pass, first 3 channels; enlarge = reverse pass with the latent z = 0 (its mean; the paper samples it randomly), so it's repeatable. Follows the pass's precision switch: half precision is used only if a test run at load (shrink then enlarge) stays within 2/255 on average of full precision, else full precision. The log says which. **Engines:** the first time a pass shrinks or enlarges frames of a given size, IRN tries each way it can run on this Mac on up to 4 of those frames, checks each against PyTorch's own result (in full precision within 0.25/255 on average; in half precision within 6/255, only a guard against broken output: the Neural Engine's enlarge is about 2.3/255 off, and this project takes fast + noisy) and keeps the fastest (needs to be 5% faster to replace PyTorch): PyTorch on the GPU, and with coremltools on macOS and half precision, the network converted to Core ML for the Neural Engine (it has no float32; Core ML's GPU and mixed paths were tried and dropped: never faster than PyTorch, and their own buffers ran a 16 GB Mac out of memory). PyTorch's cached GPU memory is released before and between tests, and if even the PyTorch test runs out of memory IRN stays on PyTorch without testing. Converted networks are cached in `weights/irn/coreml/` (one per weights file, direction, frame size and precision; a few seconds each the first time). The pass log and the estimate list what was tried and what won, with times and differences. The winner is remembered in `weights/irn/coreml/engines.json` (per weights file, direction, size, precision, device and torch/coremltools versions), so later runs build only it and check it on one frame against PyTorch; delete that file to make IRN test again. Compiled Core ML networks (`.mlmodelc`) are kept next to the packages so they aren't recompiled each run. That setup time doesn't count toward the estimate's time limit. The Haar stage is written as pixel unshuffle plus a 1×1 mixing convolution (same sums, difference 0 against the authors' grouped convolution) so it traces at a fixed size. PyTorch runs up to 8 frames at a time; if the GPU runs out of memory the group shrinks, then frames are cut into overlapping tiles as a last resort (tiles can leave faint seams, so they're only used when nothing else fits). Weights are only on the authors' Google Drive: you put `IRN_x2.pth`, `IRN_x4.pth`, `IRN_x8.pth`, `IRN+_x4.pth` and `IRN_x2_finetune.pth` (whichever you want) in `weights/irn/` yourself. Not used: `IRN_x3` (different first stage), `IRN_color` (not resizing), `CRM_*` (JPEG modules). Label: accuracy |
| content-aware downscale | **perceptual** (Öztireli & Gross, SIGGRAPH 2015): their Algorithm 1 (box average, then a local-contrast correction over 2×2 patches; `patch` 2/3/4); matches a literal transcription of the algorithm to 1e-5 for whole-number factors, other factors use the area average for its first two lines. Same light setting as the plain resizes. **kopf** content-adaptive downscaling (Kopf, Shamir & Peers, SIGGRAPH Asia 2013) from the authors' supplementary pseudocode: one bilateral kernel (spatial Gaussian × CIELAB colour Gaussian) per output pixel, EM with their correction step (neighbour smoothing and a box around the pixel centre, singular values clamped to [0.05, 0.1], colour variance ×1.1 for dominant or staircasing kernels). Their units are ambiguous; we read positions in input pixels and the constants (clamp, directional variance 0.2, edge strength 0.08, 25°) in output-pixel units, so it behaves the same at any ratio. Normalisations in logs (no underflow). Stops when no kernel colour moves more than 0.002 (usually 100-150 rounds) or after `iterations` (default 200). Always in CIELAB from linear light |
| edge / pixel ×2 (op `edi`) | ×2, ×4 or ×8 (repeated ×2), then optionally area to a width. **mmpx** (McGuire & Gagiu, JCGT 2021) from the authors' MIT C reference, matching it exactly on 30 random pixel-art images (exact 8-bit colour rules; on photos it's mostly nearest). **dcci** (Zhou, Shen & Dong 2012): 1.15 threshold, exponent 5, cubic (−1, 9, 9, −1)/16, edge strengths from luma. **nedi** (Li & Orchard 2001): 4-neighbour linear prediction with weights fitted on a 4×4 low-res window (geometric duality), plain average where local variance < 64 (0..255²), weights from luma. **icbi** (Giachetti & Asuni 2011): FCBI then greedy ±δ refinement with the published code's defaults (20 rounds, TM 100, TC 50, α 1, β −1); FCBI matches the reference exactly away from the edge, the refinement updates all pixels at once instead of in raster order. **eedi3** (Kevin Stone's EEDI3 via the VapourSynth port, GPL-2+, so `faceloop/eedi.py` is GPL-2+): rows then columns, filter defaults (α 0.2, β 0.25, γ 20, nrad 2, mdis 20, cost3, ucubic, vcheck 2); matches the filter's float path on 16 crops/settings (one pixel differs, a float tie). **nnedi3** is the VapourSynth nnedi3 filter itself (`faceloop/native/nnedi3.cpp`, unchanged, GPL-2+; NEON code on Apple Silicon), driven through a stand-in for VapourSynth so its own create and getFrame run: field 1 with dh (originals on even rows), rows then columns, per 8-bit plane on the CPU, prescreener 2 (its 8-bit default), etype 0; `quality` fast = its defaults (nsize 6, nns 1, qual 1), best = nnedi3_rpow2's (nsize 0, nns 3, qual 2); weights downloaded once to `weights/nnedi3/` (sha256 checked). **eedi2** is the VapourSynth EEDI2 filter itself (`faceloop/native/EEDI2.cpp`, unchanged, GPL-2+), compiled the first time it's used with the system `c++` (Xcode command line tools) and run per 8-bit plane on the CPU through a small shim that stands in for VapourSynth; defaults mthresh 10, lthresh 20, vthresh 20, estr 2, dstr 4, maxd 24, nt 50, pp 1; needs frames of at least 8×7. The lattice methods keep originals at even positions, half a pixel off; `center` (default on) shifts back with Lanczos-3 like rpow2 scripts, off lets it creep |
| emulator pixel-art scalers (op `edi`, `faceloop/retro.py`) | **scale2x** / **scale3x** (Mazzoleni's AdvanceMAME rules, exact 8-bit colour compares, every output pixel a copy; ×2/×4/×8 and ×3/×9). **2xsai**, **super2xsai**, **supereagle** (Derek Liauw Kie Fa), ported from ScummVM's `graphics/scaler/sai.cpp` (GPL-3.0+, so retro.py is too) with its integer blends, including the 6:1:1 mix that reads the second colour's low bits twice; bit-exact against ScummVM's own template code compiled on 30 random images (palettes and noise), pinned in the tests by checksum. They move the picture up-left (2xSaI ½,½ new pixels; Super 2xSaI ½ across, 1 down; Super Eagle 1,1), so `center` (default on) moves it back after every ×2: whole pixels by moving the frame, halves with Lanczos-3. **hqx** is the hqx library itself (Cameron Zemek's C hq2x/hq3x/hq4x, LGPL-2.1, `faceloop/native/hqx/`, unchanged), compiled the first time, per frame on the CPU; ×2/×3/×4. **superxbr**: passes 0-2 of libretro's `super-xbr.slangp` (Hyllian, MIT, slang-shaders 02bf134), preset settings (mode 1, edge strength 3, weight 0 in pass 0, 1 after); matches a literal per-pixel transcription of the three shaders (texture coordinates, nearest sampling, clamp to edge) to 1e-15 (on flat two-colour art the edge tests can tie exactly, and a tie goes either way with the order of float additions, as on a GPU); the preset's display passes (jinc2 to the screen, deblur) are left out. **scalefx**: Sp00kyFox's ScaleFX, the five passes of libretro's `scalefx.slangp` (MIT, same commit) at its defaults (threshold 0.5, filter AA and filter corners on), ×3/×9; every new pixel is a copy of the pixel or of one up to two away; matches the shaders themselves, run with Mesa's llvmpipe (float buffers for passes 0-1, 8-bit after, clamp to edge), on every pixel of 40 images, pinned by checksum. Its tie-breaks aren't mirror-symmetric (about 1-2% of pixels on random palette art change when flipped), as in the original. All keep a blob's centroid within 0.03 px |
| old video (op `video`, keeps the size) | **fields**: keep the top field, the bottom field, or alternate (top on odd cycles, bottom on even), then rebuild the other lines: `double` (repeat; moves the picture half a line once, after which doubling with either field changes nothing, so it never creeps), `linear` or `cubic` (Catmull-Rom) bob between the kept lines, which stay as they were (no movement over 40 alternating cycles). **squash**: shrink one axis (`axis` x/y) to `factor` (0.05-0.99) with `down` (any kernel or dpid) and stretch it straight back with `up` (any kernel); centred kernels, so a blob stays put to 0.05 px over 20 rounds |
| video AI (op `vsr`, `faceloop/models/basicvsr.py`) | **BasicVSR** ×4 (Chan et al., CVPR 2021; OpenMMLab's released weights: `basicvsr-reds4`, `basicvsr-vimeo-bi`, `basicvsr-vimeo-bd`, downloaded and checked by sha256). Plain-loss regression network, not generative. The layers carry mmagic's names, so the checkpoints load as they are; the result equals BasicSR's `basicvsr_arch.py` with the same weights exactly (max difference 0.0). **Frames are only mixed when asked**: `memory` (default 1) is the window length in frames; at 1 every frame goes through alone. With more, windows are counted from the sequence's first frame, cut at the edges of the batch in memory, and nothing carries from one batch to the next, so a resumed run gives the same frames; no frame is read beyond the batch, the sequence is never padded or mirrored, and every output goes back to the frame it came from (one out per one in, same name). Inside a window frames are put in frame-number order from their file names first (the runner lists them in natural order too: f9 before f10). In the lead-in and finish, where batches aren't tracked, every frame is alone. Knobs (memory > 1): `direction` both / forward / backward (the other chain sees only the current frame), `order` normal / reverse / shuffle (seeded by cycle and window), `flow` 0-3 (multiplies SPyNet's motion; 0 = no alignment). fp16 halves everything but SPyNet. On a panning textured test (x4 bicubic shrink): bilinear 24.6 dB, memory 1 26.5, memory 3 27.8, memory 7 29.1 |
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

- **Cycle by cycle:** every frame finishes a cycle before any frame starts
  the next (stage 0 is the lead-in, stage N is cycle N). Between stages the
  frames live in the pass folder's `.work/` as raw 8-bit arrays
  (`stage_NNNN.npy`, a memory-mapped file per stage, plus how many frames
  of it are written). Only the stage being read and the one being written
  are kept; `.work/` is deleted when the pass completes and `.done.json`
  marks it complete. Frames are rounded to 8 bits after every step anyway,
  so this loses nothing (exact in sRGB rounding; in linear rounding the
  same linear 8-bit values are stored and re-encoded). The log says how much
  disk it needs (2 stages × frames × the biggest stage size) and the pass
  refuses to start without it. A saved loop (cXXXX) is complete for every
  frame as soon as its cycle is done.
- **On the GPU,** a stage goes through the frames in batches sized as
  before: up to 256 frames, limited by 20% of RAM at the biggest size a
  cycle reaches. The lead-in and the finish work at full size, so they run
  in chunks (from 15% of RAM, at most 64 frames).
- **Prefetch:** the next batch is read (from the source or `.work/`) while
  the GPU works on the current one. Batch sizes don't change the pixels.
- **Model precision** (per pass): `precision: "fp16"` runs AI models in
  half precision (the GUI's default for new recipes; a recipe file without
  the key stays full precision). Each model gets the first way that passes a test run when
  it loads:
  1. weights converted to float16 (bfloat16 where spandrel says the model
     needs it)
  2. torch.autocast with float16, then bfloat16. Needed for SwinIR, which
     builds float32 attention masks inside.
  3. full precision

  Measured on CPU: about 0.1/255 average difference for the conv models,
  about 0.7/255 for SwinIR under bfloat16 autocast.
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
- xBR, DPID, chroma and plain resizes go through a batch in pieces sized
  for their working memory (2 GB per piece; xBR counts ~24 float copies per
  output pixel, DPID ~32, resizes ~6), halving the piece if the GPU still
  runs out of memory. Same pixels either way: frames are independent.
- Stop or crash: the next run continues in the same cycle from the last
  frames written to `.work/` (and confirmed saved, at a save point). Pause
  (GUI) holds between batches without losing anything. Passes started
  before cycle-by-cycle runs (`.batches/` markers) count as complete if
  every frame was done; otherwise they start again (same pixels).
- **Leave for the Mac** (Queue tab: Off / 10% / 25%; remembered; can be
  changed while a pass runs). See `throttle.py`.
  - GPU: work runs in slices of 32 frames with idle gaps, so that 10% or 25%
    of the time is idle. Output is byte-identical (tested).
  - CPU: fewer PyTorch CPU threads, and the process priority is lowered
    (nice 10). That stays lowered until restart, because macOS doesn't let a
    program raise its own priority again.
  - Estimates include the slow-down.
  - **Memory:** always, PyTorch's GPU memory is capped so 4 GB of unified
    memory stays free (no swapping). This applies to the GUI, `run` and
    `bench`.
- No notifications.

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
   reorder, remove). Sizes are chosen from a menu, never typed:
   - plain resize: the loop size, the source size, a width that divides the
     incoming width exactly (÷2, ÷3 …, marked "height rounded" when only the
     width divides), or ×2/×3/×4 of it
   - xBR and AI models: keep its size, the loop or source size, or an exact
     division of the step's own output width
   Widths are stored as pixels; one that no longer divides (an earlier
   step changed) is shown as "N px wide (not an exact ÷ or ×)".

   Each step's readout spells out the result, e.g. "×4 → 324×432, then
   area ↓ 162×216 (shrinks 2×)" or "(no extra resize)". pattern; cycles total; save every; rounding; light;
   output format. **Recent:** finished passes from `catalog.csv` in every
   output folder used, newest first. Loading one fills Build with that
   pass's exact recipe and leaves the source folder as it is. Live preview
   of every step's size (with the aspect error
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
   - The queue shows elapsed time for a running pass and how long a
     finished one ran; the log lines include elapsed time too.
3. **Queue:** `state/queue.json`, run one at a time. Pause/resume between
   cycles; stop; continue/run again; reorder; remove. A pass that was
   running when the app quit is queued again and resumes from the
   finished frames.
4. **Results:** every pass under an output folder (complete or not), with
   contact sheets, the last saved frame, a filter, a card size, and "Load
   into Build". Each saved loop has a button that makes it the source of a
   new pass (chaining).
5. **Log:** pass log lines (with the pass's slug and recipe), estimates
   (result, engine notes, per-step times), full tracebacks of failures, and
   terminal output, kept in `state/log.txt` (trimmed past 3 MB), with an
   environment block (git commit, macOS, chip, memory, package versions,
   WebP, "Leave for the Mac"). Copy latest / Copy all / errors-only filter.
   A failed pass and a failed estimate link to it.

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
