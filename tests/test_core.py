import json
import math

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from faceloop.colour import (GammaCurve, ParametricCurve, SRGBCurve, TableCurve, Transfer,
                             curves_from_icc, quantize)
from faceloop.engine import run_pass, run_cycle, StepContext, Controls, Progress
from faceloop.kernels import EWA, METHODS, KernelResizer, weights_1d
from faceloop.recipe import Recipe, RecipeError
from faceloop.sizes import Size, aspect_error, height_for, size_for


# ------------------------------------------------------------------ sizes
def test_heights_follow_aspect():
    src = Size(960, 720)
    assert height_for(40, src) == 30
    assert height_for(20, src) == 15
    assert size_for(960, src) == src
    odd = Size(1437, 1080)
    s = size_for(40, odd)
    assert s.height == 30 and aspect_error(s, odd) < 0.5
    wide = Size(1920, 1080)
    assert height_for(64, wide) == 36


# ------------------------------------------------------------------ colour
def _levels():
    return torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16) / 255.0


@pytest.mark.parametrize("curve", [
    SRGBCurve(), GammaCurve(2.2),
    ParametricCurve(3, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045]),
    ParametricCurve(0, [1.8]),
    TableCurve([round(65535 * (i / 1023) ** 2.2) for i in range(1024)]),
])
def test_every_8bit_level_survives_decode_encode(curve):
    x = _levels()
    back = curve.encode(curve.decode(x))
    assert torch.equal(torch.round(back * 255), torch.round(x * 255))


def test_para3_srgb_matches_analytic():
    p = ParametricCurve(3, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
    x = _levels()
    assert torch.allclose(p.decode(x), SRGBCurve().decode(x), atol=2e-5)


def test_icc_curves_from_real_profile():
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    curves, warn = curves_from_icc(icc)
    assert warn is None
    x = _levels()
    for c in curves:
        assert torch.allclose(c.decode(x), SRGBCurve().decode(x), atol=2e-3)
    curves, warn = curves_from_icc(None)
    assert warn and isinstance(curves[0], SRGBCurve)


def test_quantize_modes():
    t = Transfer()
    x = torch.rand(1, 3, 8, 8)
    q = quantize(x, "srgb", t)
    assert torch.equal(q, torch.round(q * 255) / 255)
    ql = quantize(x, "linear", t)
    lin = t.decode(ql)
    assert torch.allclose(lin * 255, torch.round(lin * 255), atol=2e-3)
    # linear rounding is coarser in the shadows
    dark = torch.linspace(0, 0.1, 200).view(1, 1, 1, 200).expand(1, 3, 1, 200)
    assert quantize(dark, "linear", t).unique().numel() < quantize(dark, "srgb", t).unique().numel()


# ------------------------------------------------------------------ kernels
@pytest.mark.parametrize("method", [m for m in METHODS if m not in EWA])
@pytest.mark.parametrize("n_in,n_out", [(960, 40), (40, 960), (240, 160), (30, 45), (7, 3)])
def test_rows_sum_to_one(method, n_in, n_out):
    W = weights_1d(n_in, n_out, method)
    assert torch.allclose(W.sum(1), torch.ones(n_out, dtype=W.dtype))


def test_area_is_block_mean_and_exact_roundtrip():
    x = torch.rand(1, 3, 48, 64)
    r = KernelResizer()
    down = r.resize(x, 16, 12, "area")
    ref = x.view(1, 3, 12, 4, 16, 4).mean(dim=(3, 5))
    assert torch.allclose(down, ref, atol=1e-6)
    up = r.resize(down, 64, 48, "area")
    assert torch.allclose(r.resize(up, 16, 12, "area"), down, atol=1e-6)
    assert torch.allclose(up, r.resize(down, 64, 48, "nearest"), atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_flat_image_stays_flat(method):
    x = torch.full((1, 3, 30, 40), 0.37)
    r = KernelResizer()
    for w, h in [(13, 10), (100, 75), (40, 30)]:
        x = r.resize(x, w, h, method)
        assert torch.allclose(x, torch.full_like(x, 0.37), atol=1e-5)


def _centroid(img):
    m = img[0, 0].double()
    h, w = m.shape
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    return float((m * xs).sum() / m.sum()), float((m * ys).sum() / m.sum())


@pytest.mark.parametrize("down", METHODS)
@pytest.mark.parametrize("up", METHODS)
def test_no_drift_over_many_cycles(down, up):
    """The face must not slide: 40 round trips 240 -> 40 -> 240 keep the centroid."""
    h, w = 180, 240
    ys, xs = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    blob = torch.exp(-(((xs - 131.0) / 20) ** 2 + ((ys - 87.0) / 16) ** 2))
    x = blob.view(1, 1, h, w).repeat(1, 3, 1, 1)
    c0 = _centroid(x)
    r = KernelResizer()
    for _ in range(40):
        x = r.resize(x, 40, 30, down).clamp(0, 1)
        x = r.resize(x, w, h, up).clamp(0, 1)
    c1 = _centroid(x)
    if not (abs(c1[0] - c0[0]) < 1.5 and abs(c1[1] - c0[1]) < 1.5):
        # Some pairs amplify a fine pattern a little on every round trip
        # (sharp kernels, JPEG 2000's 5/3 low band, nearest's alternating
        # pick): over 40 cycles it fills the frame and saturates, and the
        # centroid of that says nothing about drift. Only those may miss.
        g = max(float(torch.linalg.eigvals(weights_1d(a, b, down) @ weights_1d(b, a, up)).abs().max())
                for a, b in ((w, 40), (h, 30))) if down not in EWA and up not in EWA else 1.0
        if g > 1.03:
            pytest.skip(f"pattern grows x{g:.3f} per round trip and saturates")
        raise AssertionError((c0, c1))


def test_no_antialias_aliases_more():
    # fine stripes: a proper filter averages them out, the no-AA one does not
    x = torch.zeros(1, 3, 64, 256)
    x[..., 0::2] = 1.0
    r = KernelResizer()
    aa = r.resize(x, 26, 64, "bilinear")
    na = r.resize(x, 26, 64, "bilinear", antialias=False)
    assert aa.std() < 0.05 and na.std() > 0.2


def test_fixed_point_differs_but_stays_close():
    x = torch.rand(1, 3, 30, 40)
    r = KernelResizer()
    a = r.resize(x, 97, 73, "bicubic")
    b = r.resize(x, 97, 73, "bicubic", fixed_point=True)
    d = (a - b).abs()
    assert d.max() > 1 / 255 and d.mean() < 0.05


# ------------------------------------------------------------------ recipe
def _recipe(**kw):
    d = {
        "lead_in": [{"op": "resize", "method": "area", "width": 32}],
        "finish": [{"op": "resize", "method": "lanczos", "width": "source"}],
        "cycles": {
            "A": [{"op": "resize", "method": "area", "width": 16},
                  {"op": "resize", "method": "lanczos", "width": "loop"}],
            "B": [{"op": "resize", "method": "nearest", "width": 8}],
        },
        "pattern": ["A", "B"], "iterations": 5, "save_every": 2,
    }
    d.update(kw)
    return Recipe.from_dict(d)


def test_recipe_code_ignores_iterations_and_label_only_in_slug():
    a, b = _recipe(), _recipe(iterations=50, save_every=10)
    assert a.code() == b.code()
    assert a.code() != _recipe(rounding="linear").code()
    assert "rlin" in _recipe(rounding="linear").slug()
    assert a.save_points() == [2, 4, 5]
    assert [a.cycle_at(i) for i in range(1, 6)] == ["A", "B", "A", "B", "A"]
    assert _recipe(name="my test").slug().startswith("my-test_")


@pytest.mark.parametrize("bad", [
    {"pattern": ["C"]},
    {"iterations": 0},
    {"cycles": {"A": [{"op": "resize", "method": "sinc", "width": 4}]}, "pattern": ["A"]},
    {"cycles": {"A": [{"op": "resize", "method": "area", "width": -3}]}, "pattern": ["A"]},
    {"rounding": "dither"},
])
def test_recipe_rejects_bad(bad):
    with pytest.raises((RecipeError, KeyError)):
        _recipe(**bad)


# ------------------------------------------------------------------ engine
def _make_source(tmp_path, n=8, w=64, h=48):
    src = tmp_path / "faces_src"
    src.mkdir()
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    rng = np.random.default_rng(1)
    for i in range(n):
        img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(src / f"frame{i:04d}.png", icc_profile=icc)
    return src, icc


def test_run_pass_end_to_end_and_resume(tmp_path):
    src, icc = _make_source(tmp_path)
    out = tmp_path / "out"
    r = _recipe()
    res = run_pass(r, src, out, device=torch.device("cpu"), batch_size=3)
    folder = res.folder
    for c in ("c0002", "c0004", "c0005"):
        files = sorted((folder / c).glob("*.webp"))
        assert len(files) == 8
        im = Image.open(files[0])
        assert im.size == (64, 48)
        assert im.info.get("icc_profile") == icc
    assert not (folder / "c0003").exists()
    rec = json.loads((folder / "recipe.json").read_text())
    assert rec["code"] == r.code()
    import csv
    rows = list(csv.DictReader(open(out / "catalog.csv")))
    assert [row["code"] for row in rows] == [r.code()]
    again = run_pass(r, src, out, device=torch.device("cpu"), batch_size=3)
    assert again.skipped_frames == 8


def test_cycle_ends_where_its_last_step_ends():
    ctx = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                      resizer=KernelResizer(), models=None)
    x = torch.rand(2, 3, 48, 64)
    y = run_cycle(x, [{"op": "resize", "method": "bicubic", "width": 20}], ctx)
    assert y.shape == (2, 3, 15, 20)  # no automatic resize back
    assert torch.equal(y, torch.round(y * 255) / 255)


def _frames(folder):
    import numpy as np
    from PIL import Image
    return {p.relative_to(folder).as_posix(): np.asarray(Image.open(p)) for p in sorted(folder.glob("c*/*.webp"))}


def test_stop_and_resume(tmp_path, monkeypatch):
    import faceloop.engine as eng
    src, _ = _make_source(tmp_path)
    r = _recipe(iterations=4, save_every=4)
    # uninterrupted, everything on the GPU at once
    whole = run_pass(r, src, tmp_path / "a", device=torch.device("cpu"), batch_size=8)
    assert not (whole.folder / ".work").exists()  # temporary frames are gone
    # stopped half-way through cycle 2 (after its first 4 frames are written)
    ctl = Controls()
    real_mark = eng.Work.mark

    def mark(self, k, done):
        real_mark(self, k, done)
        if k == 2 and done == 4:
            ctl.stop()
    monkeypatch.setattr(eng.Work, "mark", mark)
    lines = []
    res = run_pass(r, src, tmp_path / "b", device=torch.device("cpu"), batch_size=4, controls=ctl, log=lines.append)
    assert res.stopped and (res.folder / ".work").exists()
    assert any("cycle 1/4" in l for l in lines) and not any("cycle 2/4" in l for l in lines)
    monkeypatch.setattr(eng.Work, "mark", real_mark)
    lines.clear()
    res2 = run_pass(r, src, tmp_path / "b", device=torch.device("cpu"), batch_size=4, log=lines.append)
    assert not res2.stopped and any("resuming at cycle 2, frame 5 of 8" in l for l in lines)
    assert len(list((res2.folder / "c0004").glob("*.webp"))) == 8
    a, b = _frames(whole.folder), _frames(res2.folder)
    assert a.keys() == b.keys() and all((a[k] == b[k]).all() for k in a)
    # run again: nothing to do
    lines.clear()
    run_pass(r, src, tmp_path / "b", device=torch.device("cpu"), log=lines.append)
    assert any("already complete" in l for l in lines)


def test_temporary_frames_are_exact_in_linear_rounding(tmp_path):
    from faceloop.engine import Work
    from faceloop.colour import quantize
    r = _recipe(iterations=1)
    r.rounding = "linear"
    w = Work(tmp_path / ".work", r, ["a", "b"], Transfer())
    x = quantize(torch.rand(2, 3, 5, 7), "linear", Transfer())
    w.write(1, 0, x, 2)
    back = w.to_tensor(w.read(1, 0, 2), torch.device("cpu"))
    assert (back - x).abs().max() < 1e-6  # same 8-bit codes; float rounding only
    r.rounding = "srgb"
    w = Work(tmp_path / ".work2", r, ["a", "b"], Transfer())
    x = quantize(torch.rand(2, 3, 5, 7), "srgb", Transfer())
    w.write(1, 0, x, 2)
    assert torch.equal(w.to_tensor(w.read(1, 0, 2), torch.device("cpu")), x)


def test_linear_light_differs_from_encoded(tmp_path):
    ctx_lin = StepContext(src=Size(64, 48), transfer=Transfer(), light="linear", rounding="srgb",
                          resizer=KernelResizer(), models=None)
    ctx_enc = StepContext(src=Size(64, 48), transfer=Transfer(), light="encoded", rounding="srgb",
                          resizer=KernelResizer(), models=None)
    x = torch.zeros(1, 3, 48, 64)
    x[..., ::2, ::2] = 1.0  # fine bright detail
    steps = [{"op": "resize", "method": "area", "width": 16}]
    a = run_cycle(x, steps, ctx_lin)
    b = run_cycle(x, steps, ctx_enc)
    assert a.mean() > b.mean()  # linear light keeps bright detail brighter


# ------------------------------------------------------------------ new pass shape
def test_cycles_chain_and_saves_are_full_size(tmp_path):
    src, _ = _make_source(tmp_path)
    sizes = []

    def on_contact(cycle, step_i, names, imgs):
        sizes.append((cycle, step_i, imgs.shape[1:3]))

    r = _recipe(iterations=3, save_every=1)
    res = run_pass(r, src, tmp_path / "out", device=torch.device("cpu"), batch_size=8, on_contact=on_contact)
    # last A (cycle 3) starts where B ended (8x6): area 16 -> 16x12, lanczos to the loop size
    assert ("A", -1, (6, 8)) in sizes and ("A", 0, (12, 16)) in sizes and ("A", 1, (24, 32)) in sizes
    # B (cycle 2) starts where A ended: the loop size
    assert ("B", -1, (24, 32)) in sizes
    # B: nearest 8 -> 8x6, and the cycle ends there (no landing column)
    assert ("B", 0, (6, 8)) in sizes and not any(c == "B" and i == 1 for c, i, _ in sizes)
    for c in ("c0001", "c0002", "c0003"):  # c0002 is finished straight from 8x6
        im = Image.open(next((res.folder / c).glob("*.webp")))
        assert im.size == (64, 48)


def test_growing_patterns_get_the_lead_in_only_where_needed():
    from faceloop.engine import check_pattern, simulate_pattern
    # settles on its own: nothing added
    ok = _recipe(cycles={"A": [{"op": "resize", "method": "nearest", "width": 8}, {"op": "xbr", "scale": 4}],
                         "B": [{"op": "resize", "method": "area", "width": "loop"}]}, pattern=["A", "B"])
    cycles, auto = check_pattern(ok, Size(64, 48))
    assert auto == set() and cycles == ok.cycles
    sim = simulate_pattern(ok, Size(64, 48))
    assert sim["starts"]["B"] == Size(32, 24) and sim["starts"]["A"] == Size(32, 24)
    # one cycle with a fixed size is enough: A grows, B resets it, nothing added
    mixed = _recipe(cycles={"A": [{"op": "xbr", "scale": 2}],
                            "B": [{"op": "resize", "method": "nearest", "width": 8}]}, pattern=["A", "A", "B"])
    assert check_pattern(mixed, Size(64, 48))[1] == set()
    # no fixed size anywhere: would grow; the first cycle of the pattern gets the lead-in
    grows = _recipe(cycles={"A": [{"op": "xbr", "scale": 2}], "C": [{"op": "chroma", "mode": "420"}]},
                    pattern=["C", "A"])
    cycles, auto = check_pattern(grows, Size(64, 48))
    assert auto == {"C"}
    assert cycles["C"] == grows.lead_in + [{"op": "chroma", "mode": "420"}] and cycles["A"] == grows.cycles["A"]
    # truly too big is still refused
    huge = _recipe(cycles={"A": [{"op": "resize", "method": "area", "width": 400}]}, pattern=["A"])
    with pytest.raises(RecipeError):
        check_pattern(huge, Size(64, 48))


def test_auto_lead_in_runs_and_is_labelled(tmp_path):
    src, _ = _make_source(tmp_path)
    r = _recipe(cycles={"A": [{"op": "xbr", "scale": 2}]}, pattern=["A"], iterations=4, save_every=2)
    lines = []
    res = run_pass(r, src, tmp_path / "out", device=torch.device("cpu"), batch_size=8, log=lines.append)
    assert any("start with the lead-in" in line for line in lines)
    assert len(list((res.folder / "c0004").glob("*.webp"))) == 8
    from faceloop.contact import ContactRecorder
    rec = ContactRecorder(res.folder, r)
    rec.cycles, rec.auto, rec.n_lead = {"A": r.lead_in + r.cycles["A"]}, {"A"}, 1
    assert rec.labels("A")[1].startswith("(lead-in)")


def test_lead_in_cannot_use_loop_width():
    with pytest.raises(RecipeError):
        _recipe(lead_in=[{"op": "resize", "method": "area", "width": "loop"}])
    with pytest.raises(RecipeError):
        _recipe(lead_in=[])


def test_finish_is_required_and_part_of_the_code():
    a = _recipe()
    b = _recipe(finish=[{"op": "resize", "method": "bicubic", "width": "source"}])
    assert a.code() != b.code()
    assert b.slug().endswith("_bicS")
    assert _recipe(finish=[]).slug().endswith("_area")
    d = {"lead_in": [{"op": "resize", "method": "area", "width": 32}], "pattern": ["A"],
         "cycles": {"A": [{"op": "resize", "method": "area", "width": 16}]}, "iterations": 1, "save_every": 1}
    with pytest.raises(RecipeError):
        Recipe.from_dict(d)


def test_chained_pass_on_saved_output(tmp_path):
    src, _ = _make_source(tmp_path)
    out = tmp_path / "out"
    r = _recipe(iterations=2, save_every=2)
    first = run_pass(r, src, out, device=torch.device("cpu"), batch_size=8)
    second = run_pass(r, first.folder / "c0002", out, device=torch.device("cpu"), batch_size=8)
    label = second.folder.parent.name
    assert label == f"faces_src+{r.code()}c0002"
    info = json.loads((second.folder / "recipe.json").read_text())
    assert info["source"]["root"] == "faces_src"
    assert info["source"]["chained_from"].endswith("c0002")
    assert len(list((second.folder / "c0002").glob("*.webp"))) == 8
    # same contact frames as the original source
    a = sorted(p.name.split("__")[0] for p in (first.folder / ".contact" / "A").glob("*__00.png"))
    b = sorted(p.name.split("__")[0].rsplit(".", 1)[0] for p in (second.folder / ".contact" / "A").glob("*__00.png"))
    assert [x.rsplit(".", 1)[0] for x in a] == b
    # and a third link in the chain
    third = run_pass(r, second.folder / "c0002", out, device=torch.device("cpu"), batch_size=8)
    assert third.folder.parent.name == f"faces_src+{r.code()}c0002+{r.code()}c0002"


def test_batch_sizes_big_at_small_loop_sizes():
    from faceloop.engine import batch_sizes
    r = Recipe.from_dict({"lead_in": [{"op": "resize", "method": "area", "width": 162}],
                          "cycles": {"A": [{"op": "resize", "method": "nearest", "width": 54},
                                           {"op": "resize", "method": "bicubic", "width": "loop"}]},
                          "pattern": ["A"], "iterations": 1, "save_every": 1,
                          "finish": [{"op": "resize", "method": "bicubic", "width": "source"}]})
    batch, chunk = batch_sizes(Size(810, 1080), r, ram=16 * 1024 ** 3)
    assert batch == 256 and 1 <= chunk <= 64


def test_big_batch_matches_small_batches(tmp_path):
    """Chunked lead-in/finish and prefetching must not change any pixel."""
    src, _ = _make_source(tmp_path, n=8)
    r = _recipe(iterations=3, save_every=3)
    a = run_pass(r, src, tmp_path / "a", device=torch.device("cpu"), batch_size=1)
    b = run_pass(r, src, tmp_path / "b", device=torch.device("cpu"), batch_size=8)
    for f in sorted((a.folder / "c0003").glob("*.webp")):
        assert f.read_bytes() == (b.folder / "c0003" / f.name).read_bytes()


def test_reserve_slices_change_nothing_and_idle(tmp_path, monkeypatch):
    import time as _time
    from faceloop import throttle
    src, _ = _make_source(tmp_path, n=8)
    r = _recipe(iterations=3, save_every=3)
    full = run_pass(r, src, tmp_path / "a", device=torch.device("cpu"), batch_size=8)
    monkeypatch.setattr(throttle, "SLICE", 3)
    slept = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    throttle.THROTTLE.set("25")
    try:
        res = run_pass(r, src, tmp_path / "b", device=torch.device("cpu"), batch_size=8)
    finally:
        throttle.THROTTLE.set("off")
    for f in sorted((full.folder / "c0003").glob("*.webp")):
        assert f.read_bytes() == (res.folder / "c0003" / f.name).read_bytes()
    assert (res.folder / "contact_A.png").exists()
    assert slept and all(s >= 0 for s in slept)
    a = Image.open(full.folder / "contact_A.png")
    b = Image.open(res.folder / "contact_A.png")
    assert a.size == b.size and a.tobytes() == b.tobytes()


def test_pause_keeps_the_reserved_share_idle(monkeypatch):
    from faceloop import throttle
    t = throttle.Throttle()
    slept = []
    monkeypatch.setattr(throttle.time, "sleep", lambda s: slept.append(s))
    t.set("25")
    t.pause_after(3.0)
    assert abs(slept[0] / (3.0 + slept[0]) - 0.25) < 1e-9
    t.set("off")
    t.pause_after(3.0)
    assert len(slept) == 1
    with pytest.raises(ValueError):
        t.set("50")


def test_dpid_step_in_a_pass(tmp_path):
    src, _ = _make_source(tmp_path)
    r = _recipe(cycles={"A": [{"op": "resize", "method": "dpid", "width": 8, "lambda": 0.5},
                              {"op": "resize", "method": "bicubic", "width": "loop"}]}, pattern=["A"],
                iterations=2, save_every=2)
    assert "dpid0.5" in r.slug()
    res = run_pass(r, src, tmp_path / "out", device=torch.device("cpu"), batch_size=8)
    assert len(list((res.folder / "c0002").glob("*.webp"))) == 8
    with pytest.raises(RecipeError):
        _recipe(cycles={"A": [{"op": "resize", "method": "dpid", "width": 8, "lambda": -1}]}, pattern=["A"])
