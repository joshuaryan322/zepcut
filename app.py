#!/usr/bin/env python3
"""ZepCut Studio - local desktop app for the ZepCut engine.

Serves a single-page UI plus a JSON API on 127.0.0.1 and opens it in an
Edge app-mode window (looks like a native desktop app; no browser chrome).
Everything runs locally: stdlib HTTP server, no extra dependencies.

Usage:  python app.py            (or the "ZepCut Studio" desktop shortcut)
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zepcut
import projects

HOST, PORT = "127.0.0.1", 8765
ROOT = Path(__file__).parent
STATIC = ROOT / "static"
CACHE = ROOT / ".cache"

# editable via the Settings tab - name: (min, max)
EDITABLE = {
    "silence_db": (-60.0, -10.0),
    "min_silence": (0.3, 5.0),
    "max_pause": (0.5, 10.0),
    "kept_pause": (0.2, 3.0),
    "review_threshold": (0, 100),
    "target_i": (-24.0, -9.0),
    "video_qp": (10, 35),
    "watermark_opacity": (0.1, 1.0),
    "intro_min_duration": (0, 3600),
    "min_duration": (5, 600),
}

JOBS = {}            # stem -> {stage, pct, error, done}
JOB_QUEUE = []       # (stem, callable)
JOB_LOCK = threading.Lock()
WATCH_STOP = threading.Event()
WATCH_THREAD = None


def job_worker():
    while True:
        item = None
        with JOB_LOCK:
            if JOB_QUEUE:
                item = JOB_QUEUE.pop(0)
        if item is None:
            time.sleep(0.5)
            continue
        stem, fn = item
        try:
            fn()
            JOBS[stem].update(stage="done", pct=100, done=True)
        except zepcut.ZepCutSkip as exc:
            JOBS[stem].update(stage="skipped", error=str(exc), done=True)
        except Exception as exc:
            JOBS[stem].update(stage="error", error=str(exc), done=True)


def start_job(stem, fn):
    if stem in JOBS and not JOBS[stem].get("done"):
        return False
    JOBS[stem] = {"stage": "queued", "pct": 0, "error": None, "done": False}
    with JOB_LOCK:
        JOB_QUEUE.append((stem, fn))
    return True


def progress_for(stem):
    def cb(stage, pct):
        JOBS[stem].update(stage=stage, pct=round(pct, 1))
    return cb


# compose.py prints "  12,300/23,694  (52%)  eta 7 min"
COMPOSE_PROGRESS = re.compile(r"([\d,]+)/([\d,]+)\s+\((\d+)%\)(?:\s+eta\s+(\d+)\s*min)?")


def external_render(cfg):
    """
    Surface a compose run started outside the app (from a terminal) so the UI
    still shows it. Progress comes from the size of the picture track being
    written, since we have no pipe to read from.
    """
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return None
    line = next((l for l in out.splitlines() if "compose.py" in l), None)
    if not line:
        return None
    m = re.search(r"compose\.py\s+(\d+)\s+\"?([^\"]+)", line)
    if not m:
        return None
    ep, src = int(m.group(1)), m.group(2).strip()
    stem = Path(src).stem
    d = projects.load()
    fl = d.get("projects", {}).get(d.get("active"), {}).get("faceless")
    pct, stage = 0, "transcribing"
    if fl:
        base = Path(fl) / f"_base-ep{ep:02d}.mp4"
        if base.exists():
            mb = base.stat().st_size / 1e6
            pct = min(int(100 * mb / 26), 99)
            stage = f"composing (external) · {pct}%"
    return {"stem": stem, "stage": stage, "pct": pct,
            "error": None, "done": False, "external": True}


def watch_loop(cfg, stop):
    """
    Watch the Videos folder and run the FULL pipeline on anything new.

    zepcut.cmd_watch calls process_file directly, which bypasses the job queue -
    so it shows no progress in the UI and never composes. This runs the same
    pipeline the Edit button does, so the progress bar and the faceless stage
    work whether a job was started by hand or by the watcher.
    """
    seen = set(output_stems(cfg))
    videos = Path(cfg["videos_dir"])
    sizes = {}
    while not stop.is_set():
        try:
            for f in sorted(videos.iterdir()):
                if stop.is_set():
                    break
                if not f.is_file() or f.suffix.lower() not in zepcut.VIDEO_EXTS:
                    continue
                if f.stem in seen or f.stem in JOBS:
                    continue
                sz = f.stat().st_size
                # only act once the file has stopped growing - OBS is still
                # writing while you record
                if sizes.get(f.stem) != sz:
                    sizes[f.stem] = sz
                    continue
                seen.add(f.stem)
                zepcut.say(f"watch: picked up {f.name}")
                start_job(f.stem, make_pipeline(cfg, f.stem, f))
        except Exception as exc:
            zepcut.say(f"watch: {exc}")
        stop.wait(cfg.get("watch_poll", 5.0))


def make_pipeline(cfg, stem, src):
    """
    The whole job as one unit: cut, then build the picture, then file it.

    This is what autopilot.exe used to do in a separate console window. Doing
    it here means one watcher, one progress bar, and no second process that can
    be orphaned and start a duplicate run.
    """
    def run():
        cb = progress_for(stem)
        zepcut.process_file(cfg, src, overwrite=True, progress=cb)
        d = projects.load()
        if d.get("mode") != "faceless" or not d.get("graphics_enabled", True):
            return
        ep = projects.episode_of(stem, d)
        if ep is None:
            JOBS[stem].update(stage="cut only - no EPnn in filename")
            return
        cut = find_output(cfg, stem)
        if not cut:
            return
        made = run_compose(stem, ep, cut, cb)
        if made:
            fin = Path(cfg["videos_dir"]) / "FINISHED"
            fin.mkdir(exist_ok=True)
            name = stem if re.match(rf"EP0?{ep}\b", stem, re.I) else f"EP{ep:02d}-{stem}"
            shutil.move(str(made), str(fin / f"{name}.mp4"))
            JOBS[stem].update(stage="finished", pct=100)
    return run


def run_compose(stem, ep, cut_path, cb):
    """
    Build the faceless picture, reporting into the same job progress the UI
    already draws a bar from. Runs compose.py as a subprocess so it stays
    usable on its own, and streams its stdout to pick the percentage out.
    """
    d = projects.load()
    p = d.get("projects", {}).get(d.get("active"), {})
    fl = p.get("faceless")
    if not fl or not Path(fl).is_dir():
        return None
    script = Path(fl) / "compose.py"
    if not script.exists():
        return None

    cb("composing", 0)
    proc = subprocess.Popen(
        # -u: unbuffered. Without it Python holds stdout in a buffer when it
        # is a pipe, so the progress bar sits at 0% for the whole render and
        # then jumps to done - which looks identical to being stuck.
        [sys.executable, "-u", str(script), str(ep), str(cut_path)],
        cwd=fl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    last = ""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        last = line
        m = COMPOSE_PROGRESS.search(line)
        if m:
            pct = int(m.group(3))
            eta = m.group(4)
            cb(f"composing{' · ' + eta + ' min left' if eta else ''}", pct)
        elif "transcrib" in line.lower():
            cb("transcribing", 0)
        elif "figures placed" in line.lower():
            cb("placing figures", 2)
        elif "encoding final" in line.lower():
            cb("encoding", 97)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"compose failed: {last[:200]}")
    # compose now names its output per source recording, so two runs for the
    # same episode cannot clobber or steal each other's file
    tag = re.sub(r"[^0-9A-Za-z-]+", "", Path(cut_path).stem)[-14:]
    out = Path(fl) / f"EP{ep:02d}-FACELESS-{tag}.mp4"
    if out.exists():
        return out
    legacy = Path(fl) / f"EP{ep:02d}-FACELESS.mp4"
    return legacy if legacy.exists() else None


def scan_library(cfg):
    videos = Path(cfg["videos_dir"])
    ready, review = videos / "Ready", videos / "Review"
    out = []
    if not videos.exists():
        return out
    done = {}
    for status, d in (("ready", ready), ("review", review)):
        if d.exists():
            for f in d.glob("*.mp4"):
                done[f.stem] = status
    for f in sorted(videos.iterdir(), key=lambda f: f.stat().st_mtime,
                    reverse=True):
        if not f.is_file() or f.suffix.lower() not in zepcut.VIDEO_EXTS:
            continue
        stem = f.stem
        status = done.get(stem, "unprocessed")
        job = JOBS.get(stem)
        if job and not job.get("done"):
            status = "processing"
        score = None
        rep = (ready if status == "ready" else review) / f"{stem}.report.txt"
        if status in ("ready", "review") and rep.exists():
            m = re.search(r"Confidence: (\d+)/100",
                          rep.read_text(encoding="utf-8", errors="replace"))
            if m:
                score = int(m.group(1))
        out.append({
            "name": f.name, "stem": stem,
            "size_mb": round(f.stat().st_size / 1e6, 1),
            "mtime": int(f.stat().st_mtime),
            "status": status, "score": score,
        })
    return out


def find_output(cfg, stem):
    videos = Path(cfg["videos_dir"])
    for d in (videos / "Ready", videos / "Review"):
        if (d / f"{stem}.mp4").exists():
            return d / f"{stem}.mp4"
    return None


def source_for(cfg, stem):
    videos = Path(cfg["videos_dir"])
    for ext in zepcut.VIDEO_EXTS:
        if (videos / f"{stem}{ext}").exists():
            return videos / f"{stem}{ext}"
    return None


def make_snippet(cfg, stem, idx):
    """Cut a small preview around cut #idx from the ORIGINAL recording so the
    user can see exactly what would be removed (plus 1.5s context each side)."""
    src = source_for(cfg, stem)
    cuts_file = Path(cfg["videos_dir"]) / "Review" / f"{stem}.cuts.json"
    if not src or not cuts_file.exists():
        raise zepcut.ZepCutSkip("source or cuts.json missing")
    cuts = json.loads(cuts_file.read_text(encoding="utf-8"))["cuts"]
    if not 0 <= idx < len(cuts):
        raise zepcut.ZepCutSkip("bad cut index")
    c = cuts[idx]
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"snippet-{stem}-{idx}.mp4"
    if out.exists():
        return out
    start = max(0.0, c["start"] - 1.5)
    dur = (c["end"] - c["start"]) + 3.0
    r = zepcut.run([cfg["ffmpeg"], "-hide_banner", "-y",
                    "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                    "-vf", "scale=640:-2", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "24",
                    "-c:a", "aac", "-b:a", "96k", str(out)])
    if r.returncode != 0 or not out.exists():
        raise zepcut.ZepCutSkip(f"snippet failed:\n{r.stderr[-300:]}")
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    # ------------------------------------------------------------ helpers --
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def send_file(self, path, ctype):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_video(self, path):
        """Serve video with Range support so <video> can seek."""
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionAbortedError, BrokenPipeError):
                    return
                remaining -= len(chunk)

    # ------------------------------------------------------------- routes --
    def do_GET(self):
        cfg = zepcut.load_config()
        path = urllib.parse.unquote(self.path.split("?")[0])
        try:
            if path == "/" or path == "/index.html":
                self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
            elif path == "/api/projects":
                d = projects.load()
                key, p = projects.active(d)
                self.send_json({
                    "active": key,
                    "graphics_enabled": d.get("graphics_enabled", False),
                    "projects": {k: v["name"] for k, v in d["projects"].items()},
                    "project": p,
                    "episodes": projects.episodes(d),
                })
            elif path == "/api/state":
                _jobs = dict(JOBS)
                _lib = scan_library(cfg)
                _ext = external_render(cfg)
                if _ext:
                    _jobs[_ext["stem"]] = _ext
                    for row in _lib:
                        if row["stem"] == _ext["stem"]:
                            row["status"] = "processing"
                            break
                self.send_json({
                    "library": _lib,
                    "jobs": _jobs,
                    "watch": WATCH_THREAD is not None and WATCH_THREAD.is_alive(),
                    "config": {k: cfg[k] for k in EDITABLE},
                    "assets": {k: zepcut.assets(cfg)[k].exists()
                               for k in ("intro", "outro", "watermark")},
                })
            elif path.startswith("/api/report/"):
                stem = path.rsplit("/", 1)[1]
                videos = Path(cfg["videos_dir"])
                for d in (videos / "Ready", videos / "Review"):
                    rep = d / f"{stem}.report.txt"
                    if rep.exists():
                        self.send_json({"report": rep.read_text(
                            encoding="utf-8", errors="replace")})
                        return
                self.send_json({"error": "no report"}, 404)
            elif path.startswith("/api/cuts/"):
                stem = path.rsplit("/", 1)[1]
                f = Path(cfg["videos_dir"]) / "Review" / f"{stem}.cuts.json"
                if f.exists():
                    self.send_json(json.loads(f.read_text(encoding="utf-8")))
                else:
                    self.send_json({"error": "no cuts.json"}, 404)
            elif path.startswith("/video/"):
                _, _, where, name = path.split("/", 3)
                base = {"ready": Path(cfg["videos_dir"]) / "Ready",
                        "review": Path(cfg["videos_dir"]) / "Review",
                        "cache": CACHE}.get(where)
                target = (base / name) if base else None
                if target and target.exists() and target.suffix == ".mp4" \
                        and target.parent == base:
                    self.send_video(target)
                else:
                    self.send_json({"error": "not found"}, 404)
            else:
                self.send_json({"error": "not found"}, 404)
        except (ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as exc:
            try:
                self.send_json({"error": str(exc)}, 500)
            except Exception:
                pass

    def do_POST(self):
        global WATCH_THREAD
        cfg = zepcut.load_config()
        path = self.path
        try:
            body = self.read_body()
            if path == "/api/edit":
                stem = Path(body["file"]).stem
                src = source_for(cfg, stem)
                if not src:
                    self.send_json({"error": "source not found"}, 404)
                    return
                ok = start_job(stem, make_pipeline(cfg, stem, src))
                self.send_json({"started": ok})
            elif path == "/api/projects":
                d = projects.load()
                if "active" in body and body["active"] in d["projects"]:
                    d["active"] = body["active"]
                if "graphics_enabled" in body:
                    d["graphics_enabled"] = bool(body["graphics_enabled"])
                projects.save(d)
                self.send_json({"ok": True, "active": d["active"],
                                "graphics_enabled": d["graphics_enabled"]})
            elif path == "/api/replan":
                # regenerate plan-epNN.json for the active project
                key, p = projects.active()
                if not p:
                    self.send_json({"error": "no active project"}, 400); return
                plan = Path(p["autoedit"]) / "plan.py"
                if not plan.exists():
                    self.send_json({"error": f"plan.py not found at {plan}"}, 404); return
                r = subprocess.run([sys.executable, str(plan),
                                    str(body.get("episode", "all"))],
                                   cwd=str(plan.parent), capture_output=True, text=True)
                self.send_json({"ok": r.returncode == 0,
                                "out": (r.stdout or "") + (r.stderr or "")})
            elif path == "/api/watch":
                if body.get("on") and (WATCH_THREAD is None
                                       or not WATCH_THREAD.is_alive()):
                    WATCH_STOP.clear()
                    WATCH_THREAD = threading.Thread(
                        target=watch_loop,
                        args=(cfg, WATCH_STOP), daemon=True)
                    WATCH_THREAD.start()
                elif not body.get("on"):
                    WATCH_STOP.set()
                self.send_json({"watch": bool(body.get("on"))})
            elif path == "/api/snippet":
                out = make_snippet(cfg, body["stem"], int(body["index"]))
                self.send_json({"url": f"/video/cache/{out.name}"})
            elif path == "/api/apply":
                stem = body["stem"]
                cuts_file = Path(cfg["videos_dir"]) / "Review" / f"{stem}.cuts.json"
                data = json.loads(cuts_file.read_text(encoding="utf-8"))
                keep = [data["cuts"][i] for i in body["keep"]
                        if 0 <= i < len(data["cuts"])]
                src = data["source"]
                total = len(data["cuts"])
                ok = start_job(stem, lambda: zepcut.apply_reviewed(
                    cfg, src, keep, total, progress=progress_for(stem)))
                self.send_json({"started": ok})
            elif path == "/api/config":
                current = {}
                if zepcut.CONFIG_PATH.exists():
                    current = json.loads(
                        zepcut.CONFIG_PATH.read_text(encoding="utf-8"))
                for k, v in body.items():
                    if k not in EDITABLE:
                        continue
                    lo, hi = EDITABLE[k]
                    try:
                        v = type(zepcut.DEFAULTS[k])(v)
                    except (TypeError, ValueError):
                        continue
                    current[k] = min(max(v, lo), hi)
                zepcut.CONFIG_PATH.write_text(
                    json.dumps(current, indent=4), encoding="utf-8")
                self.send_json({"saved": True})
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            try:
                self.send_json({"error": str(exc)}, 500)
            except Exception:
                pass


def find_edge():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return None


def main():
    threading.Thread(target=job_worker, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{HOST}:{PORT}"
    zepcut.say(f"ZepCut Studio serving on {url}")
    edge = find_edge()
    if edge:
        # app-mode window: no tabs/address bar - closes -> app exits.
        #
        # --user-data-dir is REQUIRED, not cosmetic. Without it, launching Edge
        # while Edge is already running hands the window to the existing
        # process and the one we spawned exits straight away. proc.wait() then
        # returns in milliseconds, main() falls through, and the daemon server
        # threads die with it - leaving the window showing ERR_CONNECTION_REFUSED.
        # A private profile dir forces a separate process we can actually wait on.
        profile = Path(__file__).with_name(".edge-profile")
        started = time.time()
        proc = subprocess.Popen([edge, f"--app={url}",
                                 f"--user-data-dir={profile}",
                                 "--no-first-run", "--no-default-browser-check",
                                 "--window-size=1180,780"])
        proc.wait()
        # Belt and braces: if the window still handed off (returned absurdly
        # fast), do not exit and kill the server - keep serving instead.
        if time.time() - started < 3:
            zepcut.say("browser handed off to an existing window - "
                       "serving until you close this process")
            while True:
                time.sleep(60)
        # linger so a queued render finishes even if the window closed
        while any(not j.get("done") for j in JOBS.values()):
            time.sleep(2)
    else:
        import webbrowser
        webbrowser.open(url)
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
