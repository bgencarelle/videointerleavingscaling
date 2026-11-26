import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch
from PIL import Image

from faceloop.gui import App, make_handler

RECIPE = {
    "name": "t",
    "lead_in": [{"op": "resize", "method": "area", "width": 24}],
    "cycles": {"A": [{"op": "resize", "method": "nearest", "width": 12},
                     {"op": "xbr", "scale": 2, "width": "loop"}],
               "B": [{"op": "chroma", "mode": "420"},
                     {"op": "resize", "method": "bicubic", "width": 8}]},
    "pattern": ["A", "B"], "iterations": 6, "save_every": 3,
    "finish": [{"op": "resize", "method": "bicubic", "width": "source"}],
}


@pytest.fixture
def server(tmp_path):
    src = tmp_path / "faces"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        Image.fromarray((rng.random((64, 48, 3)) * 255).astype(np.uint8)).save(src / f"f{i:03d}.png")
    app = App(state_dir=tmp_path / "state", device=torch.device("cpu"), presets_dir=tmp_path / "presets")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, 0))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(app, port)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, host=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        if host:
            req.add_header("Host", host)
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if r.headers.get("Content-Type") == "application/json" else raw

    yield call, src, tmp_path / "out", app
    httpd.shutdown()


def _wait(call, jid, states, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = next(j for j in call("/api/queue")["jobs"] if j["id"] == jid)
        if j["status"] in states:
            return j
        time.sleep(0.1)
    raise AssertionError(f"job stuck in {j['status']}")


def _estimate(call, recipe, src, timeout=60):
    eid = call("/api/estimate/start", {"recipe": recipe, "source": str(src)})["id"]
    t = time.time()
    while time.time() - t < timeout:
        st = call(f"/api/estimate/status?id={eid}")
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.1)
    raise AssertionError("estimate never finished")


def test_page_and_preview(server):
    call, src, out, app = server
    assert b"faceloop" in call("/")
    assert any(m["id"] == "w2x-cunet" for m in call("/api/models")["models"])
    info = call("/api/source", {"path": str(src)})
    assert info["frames"] == 10 and (info["width"], info["height"]) == (48, 64)
    p = call("/api/preview", {"recipe": RECIPE, "source": str(src)})
    assert p["ok"] and p["sizes"]["loop"] == [24, 32]
    a = p["sizes"]["cycles"]["A"]
    assert [(r["width"], r["height"]) for r in a] == [(12, 16), (24, 32)]
    b = p["sizes"]["cycles"]["B"]
    assert [(r["width"], r["height"]) for r in b] == [(24, 32), (8, 11)]  # no landing between cycles
    assert p["sizes"]["cycle_starts"] == {"A": [8, 11], "B": [24, 32]}  # A starts where B ended
    assert b[1]["error"] > 0  # 8 wide can't keep 3:4 exactly
    bad = call("/api/preview", {"recipe": {**RECIPE, "pattern": ["C"]}, "source": str(src)})
    assert not bad["ok"] and "C" in bad["errors"][0]
    grows = call("/api/preview", {"recipe": {**RECIPE, "cycles": {"A": [{"op": "xbr", "scale": 2}]}, "pattern": ["A"]},
                                  "source": str(src)})
    assert grows["ok"] and len(grows["sizes"]["cycle_auto"]["A"]) == 1  # the lead-in, added automatically
    assert "automatically" in grows["sizes"]["cycle_auto"]["A"][0]["note"]


def test_presets(server):
    call, *_ = server
    call("/api/presets/save", {"name": "my pass", "recipe": RECIPE})
    assert call("/api/presets")["presets"] == ["my pass"]
    assert call("/api/presets/load", {"name": "my pass"})["cycles"] == RECIPE["cycles"]
    call("/api/presets/delete", {"name": "my pass"})
    assert call("/api/presets")["presets"] == []


def test_estimate_queue_results_and_files(server):
    call, src, out, app = server
    est = _estimate(call, RECIPE, src)
    assert est["status"] == "done", est
    est = est["result"]
    assert est["est_seconds"] > 0 and est["loop"] == "24x32"
    job = call("/api/queue/add", {"recipe": RECIPE, "source": str(src), "out": str(out),
                                  "estimate": {"text": est["est_text"], "disk": est["disk_text"]}})
    j = _wait(call, job["id"], {"done", "failed"})
    assert j["status"] == "done", j["message"]
    res = call("/api/results?out=" + quote(str(out)))
    (p,) = res["passes"]
    assert p["complete"] and p["saves"] == [3, 6] and len(p["contact"]) == 2
    img = call("/api/file?path=" + quote(p["contact"][0]))
    assert img[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(urllib.error.HTTPError):  # outside any output folder
        call("/api/file?path=" + quote(str(src / "f000.png")))
    with pytest.raises(urllib.error.HTTPError):  # wrong Host header (DNS rebinding)
        call("/api/queue", host="evil.example:80")
    # it shows up under Recent, and loads back as the same recipe
    rec = call("/api/recent")["recent"]
    assert rec and rec[0]["code"] == p["code"]
    back = call("/api/recent/load", {"folder": rec[0]["folder"]})
    assert back["cycles"] == RECIPE["cycles"] and back["iterations"] == RECIPE["iterations"]
    with pytest.raises(urllib.error.HTTPError):
        call("/api/recent/load", {"folder": str(src)})
    # run again: everything is skipped
    call("/api/queue/action", {"id": job["id"], "action": "requeue"})
    j = _wait(call, job["id"], {"done", "failed"})
    assert j["status"] == "done"


def test_pause_stop_continue(server):
    call, src, out, app = server
    slow = {**RECIPE, "iterations": 400, "save_every": 400}
    job = call("/api/queue/add", {"recipe": slow, "source": str(src), "out": str(out)})
    _wait(call, job["id"], {"running"})
    call("/api/queue/action", {"id": job["id"], "action": "pause"})
    assert _wait(call, job["id"], {"paused"})["status"] == "paused"
    call("/api/queue/action", {"id": job["id"], "action": "resume"})
    call("/api/queue/action", {"id": job["id"], "action": "stop"})
    assert _wait(call, job["id"], {"stopped", "done"})["status"] in ("stopped", "done")
    call("/api/queue/action", {"id": job["id"], "action": "requeue"})
    assert _wait(call, job["id"], {"done", "failed"}, timeout=120)["status"] == "done"


def test_estimate_cancel_frees_the_gpu(server):
    call, src, out, app = server
    slow = {**RECIPE, "iterations": 100000}
    eid = call("/api/estimate/start", {"recipe": slow, "source": str(src)})["id"]
    time.sleep(0.3)
    call("/api/estimate/cancel", {"id": eid})
    t = time.time()
    while call(f"/api/estimate/status?id={eid}")["status"] not in ("cancelled", "done", "failed"):
        assert time.time() - t < 20, "cancel didn't stop the estimate"
        time.sleep(0.1)
    assert app.gpu_lock.acquire(timeout=5)  # the GPU is free again
    app.gpu_lock.release()
    # and a queued pass runs
    job = call("/api/queue/add", {"recipe": RECIPE, "source": str(src), "out": str(out)})
    assert _wait(call, job["id"], {"done", "failed"})["status"] == "done"


def test_estimate_has_a_hard_limit(server, monkeypatch):
    call, src, out, app = server
    monkeypatch.setattr(App, "ESTIMATE_LIMIT", 0.5)
    st = _estimate(call, {**RECIPE, "iterations": 100000}, src)
    assert st["status"] in ("failed", "done")
    if st["status"] == "failed":
        assert "longer than" in st["error"]


def test_reserve_setting(server):
    call, src, out, app = server
    from faceloop.throttle import THROTTLE
    try:
        r = call("/api/throttle", {"level": "10"})
        assert r["level"] == "10" and call("/api/queue")["throttle"]["level"] == "10"
        assert app.settings.read()["reserve"] == "10"
        with pytest.raises(urllib.error.HTTPError):
            call("/api/throttle", {"level": "99"})
    finally:
        THROTTLE.set("off")
