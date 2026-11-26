"""Running a network through Core ML, so it can use the Neural Engine.

PyTorch on a Mac reaches the GPU (MPS) but never the Neural Engine. Core ML
can, and it can also run the GPU its own way. Converting needs a fixed frame
size, so a network is converted once per (network, size, precision) the
first time a pass needs it, and kept in a folder next to the weights; later
runs load it from there.

Nothing here is trusted blindly: whoever uses a converted network checks its
output against PyTorch's on real frames and times it before choosing it
(see IRN._choose).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np
import torch

# No conversion progress bars in the log. tqdm reads this when it's imported,
# so it has to be set before coremltools loads.
os.environ.setdefault("TQDM_DISABLE", "1")

UNITS = {"ne": "CPU_AND_NE", "all": "ALL", "gpu": "CPU_AND_GPU"}
UNIT_LABEL = {"ne": "the Neural Engine", "all": "Core ML (Neural Engine + GPU)", "gpu": "Core ML on the GPU"}
FORMAT_VERSION = 2  # bump when the converted graphs would change

_lock = threading.Lock()  # one conversion at a time


def unavailable_reason() -> str | None:
    """None if Core ML can run here, else why not."""
    if sys.platform != "darwin":
        return "Core ML needs macOS"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools  # noqa: F401
            from coremltools.models import MLModel  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"coremltools isn't installed ({e.__class__.__name__}); run setup.sh again"
    return None


def convert(module: torch.nn.Module, shape: tuple[int, ...], path: Path, precision: str,
            keep_fp32: frozenset = frozenset()) -> Path:
    """Trace `module` at `shape` (on the CPU, float32) and save it as an
    ML program at `path` (.mlpackage) unless it's already there. In half
    precision, ops whose type is in `keep_fp32` stay in float32 (they then
    run on the GPU or CPU, not the Neural Engine)."""
    path = Path(path)
    if path.exists():
        return path
    with _lock:
        if path.exists():
            return path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools as ct
            module = module.eval().float().cpu()
            ex = torch.rand(shape)
            with torch.no_grad():
                module(ex)  # build any cached constants before tracing
                traced = torch.jit.trace(module, ex, check_trace=False)
            io = np.float16 if precision == "fp16" else np.float32
            ml = ct.convert(
                traced,
                inputs=[ct.TensorType(name="x", shape=shape, dtype=io)],
                outputs=[ct.TensorType(name="y", dtype=io)],
                convert_to="mlprogram",
                compute_precision=(ct.precision.FLOAT32 if precision != "fp16" else
                                   ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep_fp32)
                                   if keep_fp32 else ct.precision.FLOAT16),
                minimum_deployment_target=ct.target.macOS14,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=path.parent)) / path.name
        ml.save(str(tmp))
        try:
            os.replace(tmp, path)
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    return path


class CoreMLRunner:
    """Calls a converted network on a torch batch (N,C,H,W); frames go
    through one at a time as a Core ML batch prediction."""

    def __init__(self, path: Path, units: str, precision: str):
        path = Path(path)
        compiled = path.with_suffix(".mlmodelc")  # compiled once, kept next to the package
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools as ct
            cu = getattr(ct.ComputeUnit, UNITS[units])
            self.model = None
            if compiled.exists():
                try:
                    self.model = ct.models.CompiledMLModel(str(compiled), compute_units=cu)
                except Exception:  # noqa: BLE001
                    shutil.rmtree(compiled, ignore_errors=True)
            if self.model is None:
                self.model = ct.models.MLModel(str(path), compute_units=cu)
                try:
                    src = self.model.get_compiled_model_path()
                    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=path.parent)) / compiled.name
                    shutil.copytree(src, tmp)
                    os.replace(tmp, compiled)
                    shutil.rmtree(tmp.parent, ignore_errors=True)
                except Exception:  # noqa: BLE001
                    pass  # only saves compiling next time
        self.np_dtype = np.float16 if precision == "fp16" else np.float32
        self.units = units
        self._batch = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        arr = x.detach().to("cpu", torch.float32).numpy().astype(self.np_dtype)
        feeds = [{"x": np.ascontiguousarray(a[None])} for a in arr]
        outs = None
        if len(feeds) > 1 and self._batch:
            try:
                outs = self.model.predict(feeds)
            except Exception:  # noqa: BLE001  (a compiled model may not take a list)
                self._batch = False
        if outs is None:
            outs = [self.model.predict(f) for f in feeds]
        y = np.concatenate([np.asarray(o["y"]) for o in outs], 0).astype(np.float32)
        return torch.from_numpy(y).to(dev)
