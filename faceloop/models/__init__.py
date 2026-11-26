"""Model catalogue, downloads, drop-ins and running models on the device.

Every built-in model comes from an official source, pinned by URL and
SHA-256 (see catalog.json). Files live under the weights folder:

    weights/waifu2x/...     waifu2x ncnn files (run by our PyTorch port)
    weights/realcugan/...   Real-CUGAN ncnn files (same)
    weights/spandrel/...    .pth files loaded with spandrel
    weights/dropin/         your own model files; a same-name .json labels them:
                            {"label": "GAN" | "accuracy", "name": "optional"}

Models must run natively on the device. On the Mac GPU (MPS) the CPU
fallback is never enabled, so a model that needs an op the GPU lacks fails
its check and is marked unavailable instead of silently running slowly.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

CATALOG_PATH = Path(__file__).with_name("catalog.json")
DROPIN_EXTS = {".pth", ".pt", ".safetensors", ".ckpt"}


def default_weights_dir() -> Path:
    env = os.environ.get("FACELOOP_WEIGHTS")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "weights"


@dataclass
class ModelInfo:
    id: str
    family: str          # waifu2x | realcugan | spandrel | dropin
    label: str           # accuracy | GAN | unlabelled
    scale: int | None
    description: str
    files: list[dict] = field(default_factory=list)
    args: dict = field(default_factory=dict)

    @property
    def is_gan(self) -> bool:
        # unlabelled drop-ins count as GAN in pass names until you label them
        return self.label != "accuracy"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRunner:
    """Runs a model on (N,3,H,W) encoded floats in 0..1, with memory fallbacks:
    smaller batches first, then tiles."""

    def __init__(self, info: ModelInfo, module: torch.nn.Module, scale: int, device: torch.device,
                 multiple_of: int = 1, minimum: int = 0, dtype: torch.dtype = torch.float32):
        self.info = info
        self.module = module
        self.scale = scale
        self.device = device
        self.multiple_of = multiple_of
        self.minimum = minimum
        self.dtype = dtype
        self.autocast: torch.dtype | None = None  # run under torch.autocast with this dtype
        self.precision_label = "float32"
        self.micro = 16
        self.tile: int | None = None

    def _pad_req(self, x):
        h, w = x.shape[-2:]
        m = self.multiple_of
        th = max(self.minimum, -(-h // m) * m)
        tw = max(self.minimum, -(-w // m) * m)
        if (th, tw) == (h, w):
            return x, h, w
        return F.pad(x, (0, tw - w, 0, th - h), mode="replicate"), h, w

    @torch.no_grad()
    def _run(self, x):
        xp, h, w = self._pad_req(x)
        if self.autocast is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.autocast):
                y = self.module(xp).float()
        else:
            y = self.module(xp.to(self.dtype)).float()
        return y[..., :h * self.scale, :w * self.scale]

    def _run_tiled(self, x, tile, overlap=16):
        n, c, h, w = x.shape
        s = self.scale
        out = torch.empty(n, c, h * s, w * s, device=x.device, dtype=x.dtype)
        step = tile
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1, x1 = min(h, y0 + step), min(w, x0 + step)
                ya, xa = max(0, y0 - overlap), max(0, x0 - overlap)
                yb, xb = min(h, y1 + overlap), min(w, x1 + overlap)
                r = self._run(x[..., ya:yb, xa:xb])
                out[..., y0 * s:y1 * s, x0 * s:x1 * s] = \
                    r[..., (y0 - ya) * s:(y0 - ya + y1 - y0) * s, (x0 - xa) * s:(x0 - xa + x1 - x0) * s]
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        i = 0
        while i < x.shape[0]:
            chunk = x[i:i + self.micro]
            try:
                if self.tile:
                    y = self._run_tiled(chunk, self.tile)
                else:
                    y = self._run(chunk)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                _empty_cache(self.device)
                if self.micro > 1:
                    self.micro //= 2
                elif self.tile is None:
                    self.tile = 256
                elif self.tile > 32:
                    self.tile //= 2
                else:
                    raise
                continue
            outs.append(y.float())
            i += chunk.shape[0]
        return torch.cat(outs, 0)


def _empty_cache(device):
    try:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class ModelManager:
    def __init__(self, weights_dir: Path | None = None, keep_loaded: int = 2):
        self.weights_dir = Path(weights_dir or default_weights_dir())
        data = json.loads(CATALOG_PATH.read_text())
        self.catalog: dict[str, ModelInfo] = {}
        for d in data["models"]:
            self.catalog[d["id"]] = ModelInfo(d["id"], d["family"], d["label"], d["scale"],
                                              d["description"], d["files"], d.get("args", {}))
        self.manual = data.get("manual", [])
        self._loaded: OrderedDict = OrderedDict()
        self._keep = keep_loaded
        self._lock = threading.Lock()
        self._sha_cache: dict[str, str] = {}
        self.refresh_dropins()

    # ---------------------------------------------------------- listing
    def refresh_dropins(self):
        for k in [k for k, v in self.catalog.items() if v.family == "dropin"]:
            del self.catalog[k]
        d = self.weights_dir / "dropin"
        if not d.is_dir():
            return
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in DROPIN_EXTS:
                continue
            meta = {}
            side = p.with_suffix(".json")
            if side.exists():
                try:
                    meta = json.loads(side.read_text())
                except json.JSONDecodeError:
                    meta = {}
            label = meta.get("label", "unlabelled")
            if label not in ("GAN", "accuracy"):
                label = "unlabelled"
            mid = "dropin-" + "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in p.stem)
            self.catalog[mid] = ModelInfo(mid, "dropin", label, None, meta.get("name", p.name),
                                          [{"path": f"dropin/{p.name}"}], {})

    def known_ids(self) -> set[str]:
        return set(self.catalog)

    def gan_ids(self) -> set[str]:
        return {k for k, v in self.catalog.items() if v.is_gan}

    def is_downloaded(self, mid: str) -> bool:
        info = self.catalog[mid]
        return all((self.weights_dir / f["path"]).exists() for f in info.files)

    def describe_used(self, recipe) -> dict:
        out = {}
        for mid in sorted(recipe.model_ids()):
            info = self.catalog.get(mid)
            if not info:
                continue
            files = {}
            for f in info.files:
                p = self.weights_dir / f["path"]
                if f.get("sha256"):
                    files[f["path"]] = f["sha256"]
                elif p.exists():
                    key = str(p)
                    if key not in self._sha_cache:
                        self._sha_cache[key] = sha256_file(p)
                    files[f["path"]] = self._sha_cache[key]
            out[info.id] = {"label": info.label, "family": info.family, "scale": info.scale,
                            "description": info.description, "files": files}
        return out

    # ---------------------------------------------------------- download
    def download(self, mid: str, progress=None) -> None:
        info = self.catalog[mid]
        for f in info.files:
            if "url" not in f:
                continue
            dest = self.weights_dir / f["path"]
            if dest.exists() and sha256_file(dest) == f["sha256"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            h = hashlib.sha256()
            req = urllib.request.Request(f["url"], headers={"User-Agent": "faceloop"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    h.update(b)
                    got += len(b)
                    if progress:
                        progress(mid, f["path"], got, total)
            if h.hexdigest() != f["sha256"]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {f['url']}")
            os.replace(tmp, dest)

    # ---------------------------------------------------------- loading
    def get(self, mid: str, device: torch.device, precision: str = "fp32") -> ModelRunner:
        """precision "fp16" runs the model in half precision: float16 where the
        model supports it, else bfloat16, else full precision (see runner.dtype)."""
        key = (mid, str(device), precision)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
            runner = self._load(mid, device)
            if precision == "fp16":
                self._make_half(runner)
            self._loaded[key] = runner
            while len(self._loaded) > self._keep:
                self._loaded.popitem(last=False)
                _empty_cache(device)
            return runner

    @staticmethod
    def _half_dtype(runner: ModelRunner) -> torch.dtype:
        desc = getattr(runner, "descriptor", None)
        if desc is None:  # our ncnn ports: fine in float16 (the ncnn programs used it too)
            return torch.float16
        if desc.supports_half:
            return torch.float16
        if desc.supports_bfloat16:
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _probe(runner: ModelRunner) -> bool:
        try:
            with torch.no_grad():
                y = runner._run(torch.rand(1, 3, 32, 32, device=runner.device))
            if runner.device.type == "mps":
                torch.mps.synchronize()
            return bool(torch.isfinite(y).all())
        except (RuntimeError, TypeError, NotImplementedError):
            return False

    def _make_half(self, runner: ModelRunner) -> None:
        """Half precision, the first way that works for this model on this device:
        weights converted to float16/bfloat16; else torch.autocast (each op
        picks its type, for models that build float32 tensors inside, like
        SwinIR's attention mask); else full precision."""
        half = self._half_dtype(runner)
        if half != torch.float32:
            runner.module = runner.module.to(half)
            runner.dtype = half
            if self._probe(runner):
                runner.precision_label = str(half).replace("torch.", "")
                return
            runner.module = runner.module.float()
            runner.dtype = torch.float32
        for ac in (torch.float16, torch.bfloat16):
            runner.autocast = ac
            if self._probe(runner):
                runner.precision_label = str(ac).replace("torch.", "") + " (autocast)"
                return
        runner.autocast = None
        runner.precision_label = "float32 (half precision doesn't work for this model here)"

    def precision_of(self, mid: str, device: torch.device, precision: str) -> str:
        if precision != "fp16":
            return "float32"
        return self.get(mid, device, precision).precision_label

    def _load(self, mid: str, device: torch.device) -> ModelRunner:
        if mid not in self.catalog:
            raise KeyError(f"unknown model {mid!r}")
        info = self.catalog[mid]
        if not self.is_downloaded(mid):
            raise FileNotFoundError(f"model {mid} is not downloaded yet")
        if info.family == "waifu2x":
            from .ncnn_port import Waifu2x
            a = info.args
            m = Waifu2x(self.weights_dir / "waifu2x", a["variant"], a["noise"], a["scale"])
            return ModelRunner(info, m.to(device).eval(), a["scale"], device)
        if info.family == "realcugan":
            from .ncnn_port import RealCUGAN
            a = info.args
            m = RealCUGAN(self.weights_dir / "realcugan", a["kind"], a["scale"], a["mode"])
            return ModelRunner(info, m.to(device).eval(), a["scale"], device)
        # spandrel (built-in or drop-in)
        import spandrel
        # SwinIR's code inside spandrel triggers a harmless torch.meshgrid warning
        warnings.filterwarnings("ignore", message=r"torch\.meshgrid: in an upcoming release")
        path = self.weights_dir / info.files[0]["path"]
        desc = spandrel.ModelLoader().load_from_file(str(path))
        if not isinstance(desc, spandrel.ImageModelDescriptor):
            raise ValueError(f"{path.name} is not an image-to-image model")
        if desc.input_channels != 3 or desc.output_channels != 3:
            raise ValueError(f"{path.name} is not an RGB model")
        if desc.size_requirements.square:
            raise ValueError(f"{path.name} needs square inputs; not supported")
        if info.family == "dropin":
            info.scale = desc.scale
            info.args = {"arch": desc.architecture.id}
        desc.to(device).eval()
        req = desc.size_requirements
        runner = ModelRunner(info, desc.model, desc.scale, device,
                             multiple_of=req.multiple_of, minimum=req.minimum)
        runner.descriptor = desc
        return runner

    def get_irn(self, mid: str, device: torch.device):
        """An IRN network (has .down() and .up()), e.g. 'irn-x4' or 'irnplus-x4'."""
        info = self.catalog.get(mid)
        if info is None or info.family != "irn":
            raise KeyError(f"unknown IRN model {mid!r}")
        key = (mid, str(device), "irn")
        with self._lock:
            if key not in self._loaded:
                fn = info.args["file"]
                if not self.is_downloaded(mid):
                    raise FileNotFoundError(f"{mid} weights missing: put {fn} in "
                                            f"{self.weights_dir / 'irn'} (from the authors' Google Drive)")
                from .irn import IRN
                self._loaded[key] = IRN(self.weights_dir / "irn" / fn, info.scale).to(device)
                while len(self._loaded) > self._keep:
                    self._loaded.popitem(last=False)
            self._loaded.move_to_end(key)
            return self._loaded[key]

    def check(self, mid: str, device: torch.device) -> tuple[bool, str]:
        """Load and run a tiny frame natively on `device`."""
        try:
            if self.catalog[mid].family == "irn":
                m = self.get_irn(mid, device)
                x = torch.rand(1, 3, 32, 40, device=device)
                y = m.up(m.down(x))
                ok = y.shape == x.shape and bool(torch.isfinite(y).all())
                return ok, "ok" if ok else f"bad output {tuple(y.shape)}"
            r = self.get(mid, device)
            x = torch.rand(1, 3, 30, 40, device=device)
            y = r(x)
            if device.type == "mps":
                torch.mps.synchronize()
            ok = y.shape[-2:] == (30 * r.scale, 40 * r.scale) and torch.isfinite(y).all().item()
            return bool(ok), "ok" if ok else f"bad output {tuple(y.shape)}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"
