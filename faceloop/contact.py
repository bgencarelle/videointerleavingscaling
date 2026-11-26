"""Contact sheets: what each step of a cycle does, on the same frames.

For the last run of each cycle in a pass (last A, last B, ...), the cycle's
input and the result of every step are kept for a few fixed random frames
(the same frames for every pass of a source). They are stored as small PNGs
under <pass>/.contact/ while batches run, and composed into
<pass>/contact_<cycle>.png: one row per frame, one column per step.

Every image is scaled to the same row height with nearest, so tiny sizes
show their real pixels.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROW_H = 240
PAD = 6
LABEL_H = 34
NAME_W = 110
BG = (24, 24, 24)
FG = (230, 230, 230)
DIM = (150, 150, 150)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def step_label(s: dict) -> str:
    w = s.get("width")
    ws = {"source": "source", "loop": "loop", None: ""}.get(w, str(w))
    op = s.get("op")
    if op == "resize":
        extra = []
        if s.get("antialias") is False:
            extra.append("no-AA")
        if s.get("fixed_point"):
            extra.append("fixed")
        if s["method"] == "dpid":
            extra.append(f"λ{s.get('lambda', 1.0):g}")
        return f"{s['method']}{' ' + ' '.join(extra) if extra else ''}->{ws}"
    if op == "model":
        return f"{s['model']}->{ws or 'native'}"
    if op == "xbr":
        return f"xBR x{s['scale']}->{ws or 'native'}"
    if op == "chroma":
        return f"chroma {s.get('mode', '420')}"
    if op == "irn":
        return f"IRN {'shrink' if s['direction'] == 'down' else 'enlarge'} x{s['scale']}->{ws or 'native'}"
    return op or "?"


class ContactRecorder:
    """Pass to run_pass(on_contact=...). Keeps captures on disk across batches."""

    def __init__(self, folder: Path, recipe):
        self.folder = Path(folder)
        self.recipe = recipe
        self.dir = self.folder / ".contact"
        self.cycles = recipe.cycles  # the cycles as run (may start with the lead-in)
        self.auto: set = set()
        self.n_lead = 0

    def __call__(self, cycle: str, step_i: int, names: list[str], imgs: np.ndarray):
        d = self.dir / cycle
        d.mkdir(parents=True, exist_ok=True)
        for name, img in zip(names, imgs):
            Image.fromarray(img).save(d / f"{name}__{step_i + 1:02d}.png", compress_level=1)

    def labels(self, cycle: str) -> list[str]:
        steps = self.cycles[cycle]
        labels = [step_label(s) for s in steps]
        if cycle in self.auto:
            labels = ["(lead-in) " + lab for lab in labels[:self.n_lead]] + labels[self.n_lead:]
        return ["cycle start"] + labels

    def build(self) -> list[Path]:
        out = []
        if not self.dir.is_dir():
            return out
        for cdir in sorted(p for p in self.dir.iterdir() if p.is_dir()):
            sheet = build_sheet(cdir, self.labels(cdir.name) if cdir.name in self.recipe.cycles else None)
            if sheet is not None:
                path = self.folder / f"contact_{cdir.name}.png"
                tmp = path.with_name("." + path.name + ".tmp")
                sheet.save(tmp, "PNG", compress_level=3)
                tmp.replace(path)
                out.append(path)
        return out


def build_sheet(cdir: Path, labels: list[str] | None) -> Image.Image | None:
    files = {}
    for p in cdir.glob("*.png"):
        m = re.match(r"(.+)__(\d+)\.png$", p.name)
        if m:
            files.setdefault(m.group(1), {})[int(m.group(2))] = p
    if not files:
        return None
    names = sorted(files)
    cols = sorted({k for v in files.values() for k in v})
    # column widths from the first frame that has each column
    widths = {}
    for c in cols:
        for n in names:
            if c in files[n]:
                with Image.open(files[n][c]) as im:
                    widths[c] = max(widths.get(c, 0), round(im.width * ROW_H / im.height))
    W = NAME_W + sum(widths[c] + PAD for c in cols) + PAD
    H = LABEL_H + len(names) * (ROW_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)
    f_lab, f_small = _font(13), _font(11)
    x = NAME_W
    for c in cols:
        lab = labels[c] if labels and c < len(labels) else f"step {c}"
        draw.text((x + 2, 4), lab[:40], fill=FG, font=f_lab)
        x += widths[c] + PAD
    for r, n in enumerate(names):
        y = LABEL_H + r * (ROW_H + PAD)
        draw.text((6, y + 4), n[:16], fill=DIM, font=f_small)
        x = NAME_W
        for c in cols:
            p = files[n].get(c)
            if p is not None:
                with Image.open(p) as im:
                    size_txt = f"{im.width}x{im.height}"
                    w = round(im.width * ROW_H / im.height)
                    sheet.paste(im.convert("RGB").resize((w, ROW_H), Image.NEAREST), (x, y))
                draw.text((x + 3, y + ROW_H - 15), size_txt, fill=FG, font=f_small,
                          stroke_width=2, stroke_fill=(0, 0, 0))
            x += widths[c] + PAD
    return sheet
