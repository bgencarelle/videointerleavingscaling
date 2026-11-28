"""Compiles the vendored filters in this folder with the system compiler,
once per source version, into _build/."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).parent
_lock = threading.Lock()


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError("these filters need a C/C++ compiler: run 'xcode-select --install' once") from None
    return r.returncode == 0, (r.stderr or r.stdout)[-1500:]


def library(name: str, main: str, deps: list[str], arm_extra: list[str] | None = None,
            c_files: list[str] | None = None) -> Path:
    """Build lib<name> from `main` (a .cpp including the filter); on arm64
    also try the filter's NEON code (`arm_extra`: C files, with
    -D<NAME>_ARM), falling back to plain C if that doesn't build.
    `c_files` are plain C sources compiled and linked in on every machine."""
    with _lock:
        files = ([HERE / main] + [HERE / d for d in deps] + [HERE / a for a in (arm_extra or [])]
                 + [HERE / c for c in (c_files or [])])
        tag = hashlib.sha256(b"".join(p.read_bytes() for p in files) + platform.machine().encode()).hexdigest()[:10]
        ext = "dylib" if sys.platform == "darwin" else "so"
        out = HERE / "_build" / f"lib{name}-{tag}.{ext}"
        if out.exists():
            return out
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_suffix(".tmp")
        base = ["c++", "-O2", "-std=c++14", "-shared", "-fPIC", "-I", str(HERE), "-o", str(tmp), str(HERE / main)]
        errors = []
        for c in c_files or []:
            o = out.parent / (Path(c).stem + f"-{tag}.o")
            good, msg = _run(["cc", "-O2", "-fPIC", "-c", str(HERE / c), "-o", str(o)])
            if not good:
                raise RuntimeError(f"couldn't compile {name} ({c}): {msg}")
            base.append(str(o))
        if arm_extra and platform.machine() in ("arm64", "aarch64"):
            objs = []
            ok = True
            for c in arm_extra:
                o = out.parent / (Path(c).stem + f"-{tag}.o")
                good, msg = _run(["cc", "-O2", "-fPIC", "-c", str(HERE / c), "-o", str(o)])
                if not good:
                    ok = False
                    errors.append(msg)
                    break
                objs.append(str(o))
            if ok:
                good, msg = _run(base + objs + [f"-D{name.upper()}_ARM"])
                if good:
                    tmp.replace(out)
                    return out
                errors.append(msg)
        good, msg = _run(base)
        if not good:
            raise RuntimeError(f"couldn't compile {name}: " + "\n".join(errors + [msg]))
        tmp.replace(out)
        return out
