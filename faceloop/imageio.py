"""Loading source frames and saving results.

WebP is written through the system libwebp (Homebrew on the Mac) so that
`sharp_yuv` can be used; Pillow cannot set it. The colour profile is added
to the WebP container here (a VP8X + ICCP chunk). If libwebp cannot be
loaded, Pillow is used instead and the run log says sharp_yuv was not used.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import io
import os
import re
import struct
import threading
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# --------------------------------------------------------------- listing
def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_frames(folder: Path) -> list[Path]:
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")]
    return sorted(files, key=_natural_key)


def load_frame(path: Path) -> tuple[np.ndarray, bytes | None]:
    """Return (H, W, 3) uint8 RGB and the embedded ICC profile (or None)."""
    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8).copy()
    return arr, icc


def probe(path: Path) -> tuple[int, int, bytes | None]:
    with Image.open(path) as im:
        return im.width, im.height, im.info.get("icc_profile")


# --------------------------------------------------------------- libwebp
_ABI = 0x020F  # libwebp only checks the major byte (0x02)


class _WebPPicture(ctypes.Structure):
    _fields_ = [
        ("use_argb", ctypes.c_int), ("colorspace", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("y", ctypes.c_void_p), ("u", ctypes.c_void_p), ("v", ctypes.c_void_p),
        ("y_stride", ctypes.c_int), ("uv_stride", ctypes.c_int),
        ("a", ctypes.c_void_p), ("a_stride", ctypes.c_int),
        ("pad1", ctypes.c_uint32 * 2),
        ("argb", ctypes.c_void_p), ("argb_stride", ctypes.c_int),
        ("pad2", ctypes.c_uint32 * 3),
        ("writer", ctypes.c_void_p), ("custom_ptr", ctypes.c_void_p),
        ("extra_info_type", ctypes.c_int), ("extra_info", ctypes.c_void_p),
        ("stats", ctypes.c_void_p), ("error_code", ctypes.c_int),
        ("progress_hook", ctypes.c_void_p), ("user_data", ctypes.c_void_p),
        ("pad3", ctypes.c_uint32 * 3),
        ("pad4", ctypes.c_void_p), ("pad5", ctypes.c_void_p),
        ("pad6", ctypes.c_uint32 * 8),
        ("memory_", ctypes.c_void_p), ("memory_argb_", ctypes.c_void_p),
        ("pad7", ctypes.c_void_p * 2),
        ("_safety", ctypes.c_uint8 * 512),  # room if a newer libwebp grew the struct
    ]


class _WebPMemoryWriter(ctypes.Structure):
    _fields_ = [("mem", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("max_size", ctypes.c_size_t), ("pad", ctypes.c_uint32 * 1),
                ("_safety", ctypes.c_uint8 * 64)]


# indices into WebPConfig, which is all 4-byte ints/floats
_CFG_METHOD = 2
_CFG_THREAD_LEVEL = 21
_CFG_SHARP_YUV = 26


def _find_libwebp() -> str | None:
    cands = [os.environ.get("FACELOOP_LIBWEBP"),
             "/opt/homebrew/lib/libwebp.dylib", "/usr/local/lib/libwebp.dylib",
             ctypes.util.find_library("webp"), "libwebp.so.7", "libwebp.so"]
    for c in cands:
        if not c:
            continue
        try:
            ctypes.CDLL(c)
            return c
        except OSError:
            continue
    return None


class LibWebP:
    def __init__(self, path: str):
        lib = ctypes.CDLL(path)
        self.path = path
        self.lib = lib
        lib.WebPGetEncoderVersion.restype = ctypes.c_int
        v = lib.WebPGetEncoderVersion()
        self.version = f"{(v >> 16) & 255}.{(v >> 8) & 255}.{v & 255}"
        lib.WebPConfigInitInternal.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
        lib.WebPValidateConfig.argtypes = [ctypes.c_void_p]
        lib.WebPPictureInitInternal.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.WebPPictureImportRGB.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.WebPMemoryWriterInit.argtypes = [ctypes.c_void_p]
        lib.WebPMemoryWriterClear.argtypes = [ctypes.c_void_p]
        lib.WebPEncode.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.WebPPictureFree.argtypes = [ctypes.c_void_p]
        self._writer_fn = ctypes.cast(lib.WebPMemoryWrite, ctypes.c_void_p).value

    def encode_rgb(self, rgb: np.ndarray, quality: float, sharp_yuv: bool, method: int = 4) -> bytes:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, _ = rgb.shape
        cfg = (ctypes.c_int32 * 128)()
        if not self.lib.WebPConfigInitInternal(cfg, 1, ctypes.c_float(quality), _ABI):  # preset 1 = photo
            raise RuntimeError("WebPConfigInit failed (ABI mismatch?)")
        cfg[_CFG_METHOD] = int(method)
        cfg[_CFG_THREAD_LEVEL] = 0
        cfg[_CFG_SHARP_YUV] = 1 if sharp_yuv else 0
        if not self.lib.WebPValidateConfig(cfg):
            raise RuntimeError("WebPValidateConfig failed")
        pic = _WebPPicture()
        if not self.lib.WebPPictureInitInternal(ctypes.byref(pic), _ABI):
            raise RuntimeError("WebPPictureInit failed (ABI mismatch?)")
        pic.use_argb = 1
        pic.width, pic.height = w, h
        wr = _WebPMemoryWriter()
        self.lib.WebPMemoryWriterInit(ctypes.byref(wr))
        try:
            if not self.lib.WebPPictureImportRGB(ctypes.byref(pic), rgb.ctypes.data, w * 3):
                raise RuntimeError("WebPPictureImportRGB failed")
            pic.writer = self._writer_fn
            pic.custom_ptr = ctypes.cast(ctypes.byref(wr), ctypes.c_void_p).value
            if not self.lib.WebPEncode(cfg, ctypes.byref(pic)):
                raise RuntimeError(f"WebPEncode failed (error {pic.error_code})")
            return ctypes.string_at(wr.mem, wr.size)
        finally:
            self.lib.WebPPictureFree(ctypes.byref(pic))
            self.lib.WebPMemoryWriterClear(ctypes.byref(wr))


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    out = fourcc + struct.pack("<I", len(data)) + data
    return out + (b"\0" if len(data) & 1 else b"")


def add_icc(webp: bytes, icc: bytes, width: int, height: int) -> bytes:
    """Wrap a simple (VP8) WebP in an extended container with an ICC profile."""
    if webp[:4] != b"RIFF" or webp[8:12] != b"WEBP":
        raise ValueError("not a WebP file")
    body = webp[12:]
    if body[:4] == b"VP8X":
        raise ValueError("already extended")
    flags = 0x20  # ICC
    vp8x = struct.pack("<B3x", flags) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    payload = b"WEBP" + _chunk(b"VP8X", vp8x) + _chunk(b"ICCP", icc) + body
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


_lib_lock = threading.Lock()
_lib_state: dict = {}


def webp_backend() -> tuple[LibWebP | None, str]:
    """(libwebp or None, description). Checked once with a real round trip."""
    with _lib_lock:
        if "lib" in _lib_state:
            return _lib_state["lib"], _lib_state["desc"]
        path = _find_libwebp()
        lib, desc = None, "Pillow (libwebp not found: sharp_yuv not used)"
        if path:
            try:
                cand = LibWebP(path)
                test = (np.indices((48, 64)).sum(0) % 2 * 255).astype(np.uint8)
                rgb = np.stack([test, 255 - test, np.full_like(test, 128)], -1)
                data = cand.encode_rgb(rgb, 95, True)
                with Image.open(io.BytesIO(data)) as im:
                    ok = im.size == (64, 48)
                if not ok:
                    raise RuntimeError("round trip size mismatch")
                lib, desc = cand, f"libwebp {cand.version} ({path})"
            except Exception as e:  # noqa: BLE001
                desc = f"Pillow (libwebp at {path} failed self-check: {e}; sharp_yuv not used)"
        _lib_state["lib"], _lib_state["desc"] = lib, desc
        return lib, desc


def save_frame(path: Path, rgb: np.ndarray, icc: bytes | None, output: dict) -> None:
    """Atomic write: temp file in the same folder, then rename."""
    path = Path(path)
    fmt = output.get("format", "webp")
    tmp = path.with_name("." + path.name + ".tmp")
    if fmt == "webp":
        q = float(output.get("quality", 95))
        sharp = bool(output.get("sharp_yuv", True))
        lib, _ = webp_backend()
        if lib is not None:
            data = lib.encode_rgb(rgb, q, sharp)
            if icc:
                data = add_icc(data, icc, rgb.shape[1], rgb.shape[0])
            tmp.write_bytes(data)
        else:
            im = Image.fromarray(rgb, "RGB")
            im.save(tmp, "WEBP", quality=q, method=4, icc_profile=icc or None)
    elif fmt == "png":
        im = Image.fromarray(rgb, "RGB")
        im.save(tmp, "PNG", compress_level=3, icc_profile=icc or None)
    else:
        raise ValueError(f"unknown format {fmt}")
    os.replace(tmp, path)
