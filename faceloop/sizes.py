"""Frame sizes that keep the source aspect ratio.

Every size in faceloop is chosen by width. The height follows from the
source aspect ratio, rounded to the nearest whole pixel. When the aspect
cannot be hit exactly, the error (in pixels of height) is reported so the
GUI can show it.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


def height_for(width: int, src: Size) -> int:
    """Height for `width` that best keeps the aspect of `src` (at least 1)."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    exact = Fraction(width * src.height, src.width)
    # round half up, deterministic (no banker's rounding)
    h = int(exact + Fraction(1, 2))
    return max(1, h)


def size_for(width: int, src: Size) -> Size:
    if width == src.width:
        return src
    return Size(width, height_for(width, src))


def aspect_error(size: Size, src: Size) -> float:
    """How far `size` is from the exact source aspect, in pixels of height."""
    exact = size.width * src.height / src.width
    return abs(size.height - exact)


def exact_widths(src: Size, max_width: int | None = None) -> list[int]:
    """Widths whose height hits the source aspect exactly."""
    g = Fraction(src.width, src.height)
    unit = g.numerator  # smallest exact width
    max_width = max_width or src.width
    return list(range(unit, max_width + 1, unit))
