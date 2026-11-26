"""The local GUI: a small web server on this Mac only, plus a queue worker.

    python -m faceloop gui            # opens http://127.0.0.1:8765 in the browser

Only the Python standard library is used for the server. It listens on
127.0.0.1 only and refuses requests whose Host header isn't localhost, so
other devices and web pages can't reach it.

State (queue, last folders) lives in ./state/, presets in ./presets/.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import mimetypes
import platform
import re
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser

import torch
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import Controls, Progress, StopRequested, pick_device, plan_sizes, run_pass, source_identity
from .imageio import list_frames, probe
from .models import ModelManager
from .recipe import Recipe, RecipeError
from .sizes import Size

REPO = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name("web")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _fmt_secs(s):
    if s is None:
        return ""
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


class JsonFile:
    def __init__(self, path: Path, default):
        self.path = path
        self.default = default
        self.lock = threading.Lock()

    def read(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return json.loads(json.dumps(self.default))

    def write(self, data):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self.path)


# ----------------------------------------------------------------- queue
class Queue:
    """Jobs run one after another. Finished frames are skipped on resume,
    so a job that was running when the app quit simply continues."""

    def __init__(self, app: "App", store: JsonFile):
        self.app = app
        self.store = store
        self.lock = threading.RLock()
        self.jobs: list[dict] = store.read()
        for j in self.jobs:
            if j["status"] in ("running", "paused"):
                j["status"] = "queued"
                j["message"] = "will resume where it stopped"
        self.current: dict | None = None
        self.controls: Controls | None = None
        self.progress: Progress | None = None
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _save(self):
        self.store.write(self.jobs)

    def add(self, recipe: dict, source: str, out: str, estimate: dict | None) -> dict:
        r = Recipe.from_dict(recipe)
        job = {"id": uuid.uuid4().hex[:8], "recipe": r.to_dict(), "source": source, "out": out,
               "status": "queued", "created": _now(), "message": "", "estimate": estimate,
               "folder": None, "started": None, "finished": None}
        with self.lock:
            self.jobs.append(job)
            self._save()
        self.wake.set()
        return job

    def find(self, jid: str) -> dict:
        for j in self.jobs:
            if j["id"] == jid:
                return j
        raise KeyError(jid)

    def action(self, jid: str, what: str):
        with self.lock:
            j = self.find(jid)
            running = self.current is j
            if what == "pause" and running and self.controls:
                self.controls.pause()
                j["status"] = "paused"
            elif what == "resume" and running and self.controls:
                self.controls.resume()
                j["status"] = "running"
            elif what == "stop" and running and self.controls:
                self.controls.stop()
                j["message"] = "stopping after the current step..."
            elif what == "requeue" and j["status"] in ("stopped", "failed", "done"):
                j["status"] = "queued"
                j["message"] = ""
                self.wake.set()
            elif what == "remove" and not running:
                self.jobs.remove(j)
            elif what in ("up", "down") and not running:
                i = self.jobs.index(j)
                k = i - 1 if what == "up" else i + 1
                if 0 <= k < len(self.jobs):
                    self.jobs[i], self.jobs[k] = self.jobs[k], self.jobs[i]
            else:
                raise ValueError(f"can't {what} a {j['status']} job")
            self._save()

    def snapshot(self) -> list[dict]:
        with self.lock:
            out = []
            for j in self.jobs:
                d = {k: j[k] for k in ("id", "status", "source", "out", "created", "message",
                                       "estimate", "folder", "started", "finished")}
                d["name"] = j["recipe"].get("name") or ""
                d["slug"] = j["recipe"].get("slug", "")
                d["iterations"] = j["recipe"]["iterations"]
                if j is self.current and self.progress is not None:
                    p = self.progress
                    d["progress"] = {"fraction": p.fraction(), "eta": _fmt_secs(p.eta_seconds()),
                                     "frames_done": p.frames_done, "frames": p.frames_total,
                                     "batch": p.batch, "batches": p.batches, "cycle": p.cycle,
                                     "frame_cycles": p.frame_cycles, "images": p.images_saved,
                                     "total_frame_cycles": p.frames_total * p.cycles,
                                     "elapsed": _fmt_secs(time.time() - p.started)}
                elif j.get("started") and j.get("finished"):
                    try:
                        took = (_dt.datetime.fromisoformat(j["finished"]) -
                                _dt.datetime.fromisoformat(j["started"])).total_seconds()
                        d["took"] = _fmt_secs(took)
                    except ValueError:
                        pass
                out.append(d)
            return out

    def _next(self):
        with self.lock:
            for j in self.jobs:
                if j["status"] == "queued":
                    return j
        return None

    def _worker(self):
        while True:
            job = self._next()
            if job is None:
                self.wake.wait(timeout=2)
                self.wake.clear()
                continue
            ctl, prog = Controls(), Progress()
            with self.lock:
                self.current, self.controls, self.progress = job, ctl, prog
                job["status"], job["started"], job["message"] = "running", _now(), ""
                self._save()
            try:
                recipe = Recipe.from_dict(job["recipe"])
                res = run_pass(recipe, Path(job["source"]), Path(job["out"]), models=self.app.models,
                               device=self.app.device, progress=prog, controls=ctl,
                               gpu_lock=self.app.gpu_lock,
                               log=lambda line, j=job: j.__setitem__("message", line.split("] ", 1)[-1]))
                with self.lock:
                    job["folder"] = str(res.folder)
                    job["status"] = "stopped" if res.stopped else "done"
                    job["message"] = ("stopped; requeue to continue" if res.stopped
                                      else f"done in {_fmt_secs(res.seconds)}")
            except StopRequested:
                with self.lock:
                    job["status"], job["message"] = "stopped", "stopped; requeue to continue"
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    job["status"], job["message"] = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                with self.lock:
                    job["finished"] = _now()
                    self.current = self.controls = self.progress = None
                    self._save()
                self.app.remember_out(job["out"])


# ----------------------------------------------------------------- app
class App:
    def __init__(self, state_dir: Path | None = None, device=None, presets_dir: Path | None = None):
        self.state_dir = Path(state_dir or REPO / "state")
        self.presets_dir = Path(presets_dir or REPO / "presets")
        self.models = ModelManager()
        self.device = device or pick_device()
        self.gpu_lock = threading.Lock()
        self.settings = JsonFile(self.state_dir / "settings.json", {"source": "", "out": "", "outs": []})
        self.queue = Queue(self, JsonFile(self.state_dir / "queue.json", []))
        self._probe_cache: dict = {}
        self.estimates: dict = {}
        self._est_lock = threading.Lock()

    # ---------------------------------------------------------- helpers
    def remember_out(self, out: str):
        s = self.settings.read()
        outs = [o for o in s.get("outs", []) if o != out]
        s["outs"] = [out] + outs[:19]
        self.settings.write(s)

    def allowed_roots(self) -> list[Path]:
        s = self.settings.read()
        roots = {Path(o) for o in s.get("outs", []) if o}
        roots |= {Path(j["out"]) for j in self.queue.jobs}
        return [r.expanduser().resolve() for r in roots]

    def source_info(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.is_dir():
            raise ValueError("not a folder")
        key = (str(p), p.stat().st_mtime)
        if key not in self._probe_cache:
            frames = list_frames(p)
            if not frames:
                raise ValueError("no PNG, JPEG or WebP frames in this folder")
            w, h, icc = probe(frames[0])
            ident = source_identity(p)
            self._probe_cache[key] = {"path": str(p), "frames": len(frames), "width": w, "height": h,
                                      "profile": bool(icc), "first": frames[0].name, "last": frames[-1].name,
                                      "label": ident["label"], "chained": "chained_from" in ident}
        return self._probe_cache[key]

    # ---------------------------------------------------------- api
    def api_models(self, _q):
        self.models.refresh_dropins()
        rows = []
        for v in sorted(self.models.catalog.values(), key=lambda v: (v.family, v.id)):
            rows.append({"id": v.id, "family": v.family, "label": v.label, "scale": v.scale,
                         "description": v.description, "downloaded": self.models.is_downloaded(v.id)})
        return {"models": rows, "device": str(self.device)}

    def api_settings(self, _q):
        return self.settings.read()

    def api_pick_folder(self, body):
        if platform.system() != "Darwin":
            raise ValueError("the folder picker needs macOS; type the path instead")
        prompt = (body or {}).get("prompt", "Choose a folder").replace('"', "")
        r = subprocess.run(["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"path": None}
        return {"path": r.stdout.strip().rstrip("/")}

    def api_source(self, body):
        info = self.source_info(body["path"])
        s = self.settings.read()
        s["source"] = info["path"]
        self.settings.write(s)
        return info

    def api_preview(self, body):
        out = {"ok": True, "errors": []}
        try:
            r = Recipe.from_dict(body["recipe"])
            r.validate(self.models.known_ids())
        except (RecipeError, KeyError, TypeError, ValueError) as e:
            return {"ok": False, "errors": [str(e)]}
        gan = self.models.gan_ids()
        out.update({"slug": r.slug(gan), "code": r.code(), "folder": r.folder_name(gan),
                    "gan": r.uses_gan(gan), "saves": r.save_points()})
        missing = sorted(m for m in r.model_ids() if not self.models.is_downloaded(m))
        if missing:
            out["ok"] = False
            out["errors"].append("models not downloaded: " + ", ".join(missing))
        src = body.get("source")
        if src:
            try:
                info = self.source_info(src)
                p = plan_sizes(r, Size(info["width"], info["height"]), self.models)
                out["sizes"] = {"loop": [p["loop"].width, p["loop"].height], "lead_in": p["lead_in"],
                                "cycles": p["cycles"], "finish": p["finish"]}
                out["source"] = info
            except ValueError as e:
                out["errors"].append(f"source: {e}")
        return out

    ESTIMATE_LIMIT = 90.0  # seconds; an estimate is cancelled after this

    def api_estimate_start(self, body):
        """Start an estimate in the background; poll /api/estimate/status."""
        from .__main__ import EstimateCancelled, bench
        r = Recipe.from_dict(body["recipe"])
        r.validate(self.models.known_ids())
        src = Path(body["source"]).expanduser()
        self.source_info(str(src))
        eid = uuid.uuid4().hex[:8]
        est = {"id": eid, "status": "waiting", "progress": "waiting for the GPU (a running pass finishes its cycle)",
               "result": None, "error": None, "cancel": False, "started": time.time()}
        with self._est_lock:
            for e in self.estimates.values():  # only one at a time: cancel older ones
                e["cancel"] = True
            self.estimates[eid] = est

        def stop():
            if time.time() - est["started"] > self.ESTIMATE_LIMIT:
                est["error"] = (f"the estimate took longer than {int(self.ESTIMATE_LIMIT)} s and was stopped; "
                                "this pass is very slow per cycle. You can still queue it without an estimate.")
                return True
            return est["cancel"]

        def work():
            try:
                with self.gpu_lock:
                    if stop():
                        raise EstimateCancelled()
                    est["status"] = "running"
                    res = bench(r, src, device=self.device, models=self.models, should_stop=stop,
                                budget=30.0, on_progress=lambda m: est.__setitem__("progress", m))
                res["est_text"] = _fmt_secs(res["est_seconds"])
                gb = res["est_bytes"] / 1024 ** 3
                res["disk_text"] = f"{gb:.1f} GB" if gb >= 1 else f"{res['est_bytes'] / 1024 ** 2:.0f} MB"
                est["result"], est["status"] = res, "done"
            except EstimateCancelled:
                est["status"] = "failed" if est["error"] else "cancelled"
            except Exception as e:  # noqa: BLE001
                est["status"], est["error"] = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                try:
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=work, daemon=True).start()
        return {"id": eid}

    def api_estimate_status(self, q):
        eid = q.get("id", [""])[0]
        e = self.estimates.get(eid)
        if e is None:
            raise KeyError("no such estimate")
        return {k: e[k] for k in ("id", "status", "progress", "result", "error")} | \
            {"elapsed": round(time.time() - e["started"], 1)}

    def api_estimate_cancel(self, body):
        e = self.estimates.get(body.get("id", ""))
        if e is not None:
            e["cancel"] = True
        return {"ok": True}

    def api_queue(self, _q):
        return {"jobs": self.queue.snapshot(), "device": str(self.device)}

    def api_queue_add(self, body):
        src = str(Path(body["source"]).expanduser())
        out = str(Path(body["out"]).expanduser())
        self.source_info(src)
        Path(out).mkdir(parents=True, exist_ok=True)
        s = self.settings.read()
        s["source"], s["out"] = src, out
        self.settings.write(s)
        self.remember_out(out)
        return self.queue.add(body["recipe"], src, out, body.get("estimate"))

    def api_queue_action(self, body):
        self.queue.action(body["id"], body["action"])
        return {"ok": True}

    def api_presets(self, _q):
        self.presets_dir.mkdir(exist_ok=True)
        return {"presets": sorted(p.stem for p in self.presets_dir.glob("*.json"))}

    def api_preset_load(self, body):
        return json.loads((self.presets_dir / (_safe_name(body["name"]) + ".json")).read_text())

    def api_preset_save(self, body):
        Recipe.from_dict(body["recipe"])  # must be valid
        name = _safe_name(body["name"])
        if not name:
            raise ValueError("give the preset a name")
        self.presets_dir.mkdir(exist_ok=True)
        rec = {k: v for k, v in body["recipe"].items() if k not in ("code", "slug", "version")}
        (self.presets_dir / f"{name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return {"name": name}

    def api_preset_delete(self, body):
        (self.presets_dir / (_safe_name(body["name"]) + ".json")).unlink(missing_ok=True)
        return {"ok": True}

    def api_recent(self, _q):
        """Finished passes from catalog.csv in every output folder used, newest first."""
        rows = []
        for root in self.allowed_roots():
            cat = root / "catalog.csv"
            if not cat.exists():
                continue
            try:
                with open(cat, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        folder = root / r.get("folder", "")
                        if not (folder / "recipe.json").exists():
                            continue
                        rows.append({"folder": str(folder), "date": r.get("date", ""), "name": r.get("name", ""),
                                     "slug": r.get("slug", ""), "code": r.get("code", ""),
                                     "source": r.get("source", ""), "iterations": r.get("iterations", ""),
                                     "gan": r.get("gan", ""), "out": str(root)})
            except (OSError, csv.Error):
                continue
        rows.sort(key=lambda r: r["date"], reverse=True)
        return {"recent": rows[:40]}

    def api_recent_load(self, body):
        folder = Path(body["folder"]).expanduser().resolve()
        if not any(folder.is_relative_to(r) for r in self.allowed_roots()):
            raise ValueError("not a pass in a known output folder")
        info = json.loads((folder / "recipe.json").read_text(encoding="utf-8"))
        keep = ("name", "lead_in", "cycles", "pattern", "iterations", "save_every", "finish",
                "rounding", "light", "precision", "output")
        return {k: info[k] for k in keep if k in info}

    def api_results(self, q):
        out = q.get("out", [""])[0] or self.settings.read().get("out", "")
        if not out:
            return {"out": "", "passes": []}
        root = Path(out).expanduser()
        self.remember_out(str(root))
        rows = []
        cat = root / "catalog.csv"
        done = {}
        if cat.exists():
            with open(cat, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    done[r["folder"]] = r
        # also show passes still running or stopped (no catalog row yet)
        for rec in sorted(root.glob("*/*/recipe.json")):
            folder = rec.parent
            rel = str(folder.relative_to(root))
            try:
                info = json.loads(rec.read_text())
            except json.JSONDecodeError:
                continue
            saves = sorted(int(p.name[1:]) for p in folder.glob("c[0-9][0-9][0-9][0-9]") if p.is_dir())
            sample = None
            if saves:
                imgs = sorted((folder / f"c{saves[-1]:04d}").iterdir())
                sample = str(imgs[0]) if imgs else None
            rows.append({
                "folder": str(folder), "rel": rel, "source": folder.parent.name,
                "name": info.get("name", ""), "slug": info.get("slug", ""), "code": info.get("code", ""),
                "complete": rel in done, "date": done.get(rel, {}).get("date", ""),
                "gan": done.get(rel, {}).get("gan", ""), "description": done.get(rel, {}).get("cycles", ""),
                "iterations": info.get("iterations"), "saves": saves,
                "save_dirs": [str(folder / f"c{n:04d}") for n in saves],
                "chained_from": info.get("source", {}).get("chained_from"),
                "contact": [str(p) for p in sorted(folder.glob("contact_*.png"))],
                "sample": sample, "recipe": {k: info.get(k) for k in
                                             ("name", "lead_in", "cycles", "pattern", "iterations",
                                              "save_every", "finish", "rounding", "light", "output")},
                "mtime": rec.stat().st_mtime,
            })
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return {"out": str(root), "passes": rows}

    def file_allowed(self, path: Path) -> bool:
        try:
            p = path.expanduser().resolve()
        except OSError:
            return False
        if p.suffix.lower() not in (".png", ".webp", ".jpg", ".jpeg"):
            return False
        return any(p.is_relative_to(r) for r in self.allowed_roots())


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9 ._-]", "", s or "").strip()[:60]


# ----------------------------------------------------------------- http
GET_ROUTES = {"/api/models": "api_models", "/api/settings": "api_settings", "/api/queue": "api_queue",
              "/api/presets": "api_presets", "/api/results": "api_results",
              "/api/estimate/status": "api_estimate_status", "/api/recent": "api_recent"}
POST_ROUTES = {"/api/pick_folder": "api_pick_folder", "/api/source": "api_source",
               "/api/preview": "api_preview", "/api/estimate/start": "api_estimate_start",
               "/api/estimate/cancel": "api_estimate_cancel", "/api/recent/load": "api_recent_load",
               "/api/queue/add": "api_queue_add", "/api/queue/action": "api_queue_action",
               "/api/presets/load": "api_preset_load", "/api/presets/save": "api_preset_save",
               "/api/presets/delete": "api_preset_delete"}


def make_handler(app: App, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "faceloop"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _send(self, code, body: bytes, ctype: str, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _call(self, name, arg):
            try:
                self._json(200, getattr(app, name)(arg))
            except (RecipeError, ValueError, KeyError, FileNotFoundError) as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if u.path in GET_ROUTES:
                return self._call(GET_ROUTES[u.path], parse_qs(u.query))
            if u.path == "/api/file":
                p = Path(parse_qs(u.query).get("path", [""])[0])
                if not app.file_allowed(p) or not p.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                return self._send(200, p.read_bytes(), ctype, cache=False)
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            if u.path not in POST_ROUTES:
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})
            self._call(POST_ROUTES[u.path], body)

    return Handler


def serve(port: int = 8765, open_browser: bool = True, app: App | None = None):
    app = app or App()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"faceloop GUI on {url}  (device {app.device}; Ctrl-C to quit)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye; queued and stopped jobs will be there next time")
    finally:
        httpd.server_close()
    return app
