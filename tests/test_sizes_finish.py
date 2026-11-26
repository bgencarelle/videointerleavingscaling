import numpy as np
from PIL import Image

from faceloop.engine import plan_sizes, run_pass, width_choices
from faceloop.recipe import Recipe
from faceloop.sizes import Size

BASE = {"lead_in": [{"op": "resize", "method": "area", "width": 160}],
        "cycles": {"A": [{"op": "resize", "method": "nearest", "width": 40},
                         {"op": "resize", "method": "bicubic", "width": "loop"}]},
        "pattern": ["A"], "iterations": 2, "save_every": 2,
        "finish": [{"op": "resize", "method": "bicubic", "width": 320}]}


def test_width_choices_divide_exactly():
    src = Size(640, 480)
    ch = width_choices({"from": [160, 120], "raw": None}, src)
    divs = [c for c in ch if c["op"] == "/"]
    assert [c["k"] for c in divs][:4] == [2, 4, 5, 8]
    assert all(160 % c["k"] == 0 and c["width"] == 160 // c["k"] for c in divs)
    assert {c["k"] for c in ch if c["op"] == "x"} == {2, 3, 4}
    # 160/3 isn't whole, so it isn't offered
    assert not any(c["k"] == 3 and c["op"] == "/" for c in ch)
    two = next(c for c in divs if c["k"] == 2)
    assert (two["width"], two["height"], two["exact"]) == (80, 60, True)
    # after a model step, choices divide the model's output, and there are no multiples
    ch = width_choices({"from": [40, 30], "raw": [160, 120]}, src)
    assert ch and all(c["op"] == "/" for c in ch) and ch[0]["width"] == 80


def test_plan_offers_choices_and_saved_size():
    src = Size(640, 480)
    p = plan_sizes(Recipe.from_dict(BASE), src)
    assert p["lead_in"][0]["choices"][0]["width"] == 320
    assert p["saved"] == [640, 480]          # lands on the source
    assert p["finish"][-1]["note"] == "area lands on the source size"
    p = plan_sizes(Recipe.from_dict({**BASE, "land_on_source": False}), src)
    assert p["saved"] == [320, 240]          # ends where the finish ends
    assert len(p["finish"]) == 1


def test_landing_off_changes_code_but_default_keeps_old_code():
    a = Recipe.from_dict(BASE)
    b = Recipe.from_dict({**BASE, "land_on_source": True})
    c = Recipe.from_dict({**BASE, "land_on_source": False})
    assert a.code() == b.code() and a.code() != c.code()
    assert "land_on_source" not in a.pixel_identity()
    assert c.slug().endswith("free")
    assert Recipe.from_dict({**BASE, "finish": [], "land_on_source": False}).slug().endswith("_free")


def test_run_saves_at_the_finish_size(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(src / f"f{i:03d}.png")
    rec = {**BASE, "lead_in": [{"op": "resize", "method": "area", "width": 32}],
           "cycles": {"A": [{"op": "resize", "method": "nearest", "width": 16},
                            {"op": "resize", "method": "bicubic", "width": "loop"}]},
           "finish": [], "land_on_source": False}
    res = run_pass(Recipe.from_dict(rec), src, tmp_path / "out", device=None, contact=False)
    files = sorted((res.folder / "c0002").iterdir())
    assert len(files) == 3
    assert Image.open(files[0]).size == (32, 24)
    cat = (tmp_path / "out" / "catalog.csv").read_text()
    assert "32x24" in cat
