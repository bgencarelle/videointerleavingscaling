"""Leaving room for the rest of the Mac while a pass runs.

macOS has no "use at most N% of the GPU" switch, so this uses three plain
techniques:

- **GPU:** a duty cycle. Work is done in slices of frames, and after each
  slice the worker waits long enough that `reserve` of the time is idle. The
  window system gets regular gaps. Results are identical: frames are
  independent.
- **CPU:** fewer PyTorch CPU threads, and a lower process priority (nice).
  macOS lets a program lower its own priority but not raise it again, so the
  priority stays lowered until the app restarts.
- **Memory:** PyTorch on the GPU is capped so that `leave_free` of unified
  memory stays for everything else, so the Mac never starts swapping. This
  applies at start-up whatever the reserve setting.
"""
from __future__ import annotations

import os
import threading
import time

LEVELS = {"off": 0.0, "10": 0.10, "25": 0.25}
SLICE = 32  # frames per GPU stretch when a reserve is set


class Throttle:
    def __init__(self):
        self.level = "off"
        self._lock = threading.Lock()
        self._niced = False
        self._threads0 = None
        self.memory_note = ""

    @property
    def fraction(self) -> float:
        return LEVELS.get(self.level, 0.0)

    def set(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"reserve must be one of {sorted(LEVELS)}")
        import torch
        with self._lock:
            self.level = level
            if self._threads0 is None:
                self._threads0 = torch.get_num_threads()
            if level == "off":
                torch.set_num_threads(self._threads0)
                return
            cores = os.cpu_count() or 4
            keep = max(1, cores - (2 if level == "10" else max(2, cores // 3)))
            torch.set_num_threads(min(self._threads0, keep))
            if not self._niced:
                try:
                    os.setpriority(os.PRIO_PROCESS, 0, 10)
                    self._niced = True
                except (OSError, AttributeError):
                    pass

    def slice_size(self) -> int | None:
        return SLICE if self.fraction > 0 else None

    def pause_after(self, busy_seconds: float) -> None:
        """Idle so that `fraction` of (busy + idle) is idle."""
        f = self.fraction
        if f > 0 and busy_seconds > 0:
            time.sleep(min(5.0, busy_seconds * f / (1.0 - f)))

    def describe(self) -> dict:
        return {"level": self.level, "levels": list(LEVELS), "cpu_priority_lowered": self._niced,
                "memory": self.memory_note}


THROTTLE = Throttle()


def cap_gpu_memory(leave_free_gb: float = 4.0) -> str:
    """Cap PyTorch's GPU memory so `leave_free_gb` of RAM stays free."""
    import torch
    if not torch.backends.mps.is_available():
        return "memory cap: not on the Mac GPU"
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        allowed = max(1.0, total / 1024 ** 3 - leave_free_gb) * 1024 ** 3
        rec = torch.mps.recommended_max_memory()
        frac = min(2.0, allowed / rec)
        torch.mps.set_per_process_memory_fraction(frac)
        note = (f"GPU memory capped at {allowed / 1024 ** 3:.1f} GB "
                f"({leave_free_gb:g} GB left free of {total / 1024 ** 3:.0f} GB)")
    except Exception as e:  # noqa: BLE001
        note = f"memory cap not set ({type(e).__name__}: {e})"
    THROTTLE.memory_note = note
    return note
