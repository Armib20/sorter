"""Diagnostics shared by the whole pipeline. Nothing here sits on the robot's critical path:

  Trace          nested timing spans (pick > move > ..., capture > camera, detect, ...)
  frame_stats    brightness / contrast / sharpness / clipping of the model input (~1 ms)
  drift_warnings compare a frame's stats with the training images' (lighting, blur, glare)
  map_stats      anomaly-map peak, area over threshold, bounding box of the defect region
  render_overlay heatmap + defect box + peak marker + score bar, for humans
  RunRecorder    one folder per run: run.json, items.jsonl, events.jsonl, status.json and
                 per-item images. All encoding and file I/O runs on background threads,
                 so recording never adds latency to the loop it observes.

Run folder (read live by dashboard.py, summarised by evaluate.py):
  logs/<verb>_<time>/run.json        config snapshot, model, devices, versions, git commit
                    /items.jsonl     one line per item (scores, decision, spans, diagnostics)
                    /events.jsonl    run-level info / warnings / errors
                    /status.json     what is happening right now (item, stage, since when)
                    /items/0001_v0_{frame,overlay,thumb}.jpg, 0001_v0_map.png
"""

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import settings

log = logging.getLogger("diag")

STAT_KEYS = ("brightness", "contrast", "sharpness", "clipped")
STAT_HINTS = {
    ("brightness", -1): "darker than training: lighting/exposure changed?",
    ("brightness", 1): "brighter than training: lighting/exposure changed?",
    ("contrast", -1): "flatter than training: lighting or background changed?",
    ("contrast", 1): "more contrast than training: background/shadows changed?",
    ("sharpness", -1): "blurrier than training: motion blur (raise SETTLE_S) or focus",
    ("sharpness", 1): "sharper/noisier than training: different zoom, texture or noise",
    ("clipped", 1): "more clipped pixels than training: glare or over-exposure",
    ("clipped", -1): "fewer clipped pixels than training",
}


# --------------------------------------------------------------------- tracing
class Trace:
    """Nested timing spans for one item. Cheap: two perf_counter calls per span."""

    def __init__(self, listener=None):
        self.t0 = time.perf_counter()
        self.spans = []
        self._depth = 0
        self.listener = listener  # called with (name, attrs) when a span starts

    @contextmanager
    def span(self, name, **attrs):
        start = time.perf_counter()
        rec = {"name": name, "start_ms": round((start - self.t0) * 1000, 2), "depth": self._depth, "attrs": attrs}
        self._depth += 1
        if self.listener:
            self.listener(name, attrs)
        try:
            yield rec["attrs"]  # callers may add result attributes
        except BaseException as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._depth -= 1
            rec["dur_ms"] = round((time.perf_counter() - start) * 1000, 2)
            if not rec["attrs"]:
                del rec["attrs"]
            self.spans.append(rec)

    def elapsed_ms(self):
        return round((time.perf_counter() - self.t0) * 1000, 2)

    def sorted_spans(self):
        return sorted(self.spans, key=lambda s: (s["start_ms"], s["depth"]))

    def top_level_ms(self):
        """{stage: ms} for depth-0 spans (summed if repeated) - the classic per-stage timing."""
        out = {}
        for s in self.spans:
            if s["depth"] == 0:
                out[s["name"]] = round(out.get(s["name"], 0) + s["dur_ms"], 2)
        return out


def maybe_span(trace, name, **attrs):
    return trace.span(name, **attrs) if trace is not None else nullcontext(attrs)


# ------------------------------------------------------------ frame analysis
def frame_stats(x: np.ndarray) -> dict:
    """Image-quality stats of the model input x [1,3,H,W] float in [0,1]."""
    r, g, b = x[0]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return {
        "brightness": round(float(gray.mean()), 4),
        "contrast": round(float(gray.std()), 4),
        "sharpness": round(float(lap.var() * 1000), 4),  # x1000 for readable numbers
        "clipped": round(float(((gray > 0.98) | (gray < 0.02)).mean()), 4),
    }


def summarize_stats(stats_list):
    """Training-set reference: {key: {mean, std, min, max}} over many frame_stats dicts."""
    out = {"n": len(stats_list)}
    for k in STAT_KEYS:
        v = np.array([s[k] for s in stats_list], dtype=np.float64)
        out[k] = {"mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5),
                  "min": round(float(v.min()), 5), "max": round(float(v.max()), 5)}
    return out


def zscores(stats, ref):
    """z-score of each stat vs the training reference (std floored so uniform sets don't explode)."""
    if not ref:
        return {}
    out = {}
    for k in STAT_KEYS:
        if k not in ref or k not in stats:
            continue
        m, s = ref[k]["mean"], ref[k]["std"]
        floor = max(0.05 * abs(m), 0.01 if k == "clipped" else 1e-6)
        out[k] = round((stats[k] - m) / max(s, floor), 2)
    return out


def drift_warnings(stats, ref, limit=None):
    limit = limit or settings.DRIFT_Z
    warns = []
    for k, z in zscores(stats, ref).items():
        if abs(z) >= limit:
            hint = STAT_HINTS.get((k, 1 if z > 0 else -1), "")
            warns.append(f"{k} {stats[k]:.3g} vs train {ref[k]['mean']:.3g}±{ref[k]['std']:.2g} (z={z:+.1f}): {hint}")
    return warns


def map_stats(amap, threshold=0.5) -> dict:
    """Where and how big the anomaly is. amap is normalized: >= 0.5 means over the fitted threshold."""
    if amap is None:
        return {}
    h, w = amap.shape
    peak = int(np.argmax(amap))
    py, px = divmod(peak, w)
    over = amap >= threshold
    out = {"peak": round(float(amap.flat[peak]), 4), "peak_xy": [round(px / w, 3), round(py / h, 3)],
           "mean": round(float(amap.mean()), 4), "area": round(float(over.mean()), 4), "bbox": None}
    if over.any():
        ys, xs = np.nonzero(over)
        out["bbox"] = [round(xs.min() / w, 3), round(ys.min() / h, 3), round((xs.max() + 1) / w, 3),
                       round((ys.max() + 1) / h, 3)]
    return out


# ------------------------------------------------------------------ overlays
def _jet_lut():
    x = np.linspace(0, 1, 256)
    lut = np.stack([np.clip(1.5 - np.abs(4 * x - k), 0, 1) for k in (3, 2, 1)], axis=1)
    return (lut * 255).astype(np.uint8)


JET = _jet_lut()


def render_overlay(frame: Image.Image, amap, info: dict) -> Image.Image:
    """Frame + heatmap (stretched around the threshold) + defect box + peak + header with score bar."""
    img = frame.convert("RGB")
    W, H = img.size
    thr = settings.THRESHOLD
    if amap is not None:
        lo = max(0.0, thr - 0.2)
        a = np.clip((np.asarray(amap, dtype=np.float32) - lo) / (1 - lo), 0, 1)
        a8 = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR))
        heat = JET[a8].astype(np.float32)
        alpha = (0.12 + 0.5 * a8 / 255.0)[..., None]
        img = Image.fromarray((np.asarray(img, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    m = info.get("map") or {}
    lw = max(2, W // 250)
    if m.get("bbox"):
        x0, y0, x1, y1 = m["bbox"]
        draw.rectangle((x0 * W, y0 * H, x1 * W, y1 * H), outline=(255, 255, 255), width=lw)
    if m.get("peak_xy"):
        px, py = m["peak_xy"][0] * W, m["peak_xy"][1] * H
        r = max(8, W // 60)
        draw.line((px - r, py, px + r, py), fill=(255, 255, 255), width=lw)
        draw.line((px, py - r, px, py + r), fill=(255, 255, 255), width=lw)
    # header: text + score bar with threshold tick
    fs = max(14, W // 32)
    bar_h = fs + 16
    draw.rectangle((0, 0, W, bar_h + 8), fill=(0, 0, 0))
    score = info.get("score", 0.0)
    color = (255, 80, 80) if score >= thr else (80, 220, 120)
    draw.text((8, 4), info.get("title", f"score {score:.3f}"), fill=color, font_size=fs)
    bx0, bx1, by = W * 0.62, W - 10, bar_h // 2 + 2
    draw.rectangle((bx0, by - 5, bx1, by + 5), outline=(120, 120, 120))
    draw.rectangle((bx0, by - 5, bx0 + (bx1 - bx0) * min(1, max(0, score)), by + 5), fill=color)
    tx = bx0 + (bx1 - bx0) * thr
    draw.line((tx, by - 10, tx, by + 10), fill=(255, 255, 0), width=2)
    return img


# ------------------------------------------------------------------ run meta
def _versions():
    from importlib import metadata

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ("openvino", "openvino-genai", "viam-sdk", "anomalib", "numpy", "pillow"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def _git():
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=3).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=root, capture_output=True,
                                    text=True, timeout=3).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {}


def settings_snapshot():
    out = {}
    for k in dir(settings):
        if k.isupper():
            v = getattr(settings, k)
            out[k] = str(v) if isinstance(v, Path) else v
    return json.loads(json.dumps(out, default=str))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ------------------------------------------------------------------ recorder
def _background_priority():
    """Thread initializer: make recorder threads yield to inference (efficiency cores / nice 19)."""
    try:
        if sys.platform == "darwin":
            import ctypes
            lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
            lib.pthread_set_qos_class_self_np(0x09, 0)  # QOS_CLASS_BACKGROUND
        elif hasattr(os, "setpriority"):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)  # Linux: per-thread nice
    except Exception:  # noqa: BLE001
        pass


class RunRecorder:
    """Writes one run folder. Every write happens on a background thread."""

    def __init__(self, verb, root=None, artifacts=True, meta=None):
        root = Path(root or settings.LOG_DIR)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = root / f"{verb}_{stamp}"
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = root / f"{verb}_{stamp}_{n}"
        (self.dir / "items").mkdir(parents=True)
        self.artifacts = artifacts
        self._io = ThreadPoolExecutor(1, thread_name_prefix="recorder",          # items/events, in order
                                      initializer=_background_priority)
        self._st = ThreadPoolExecutor(1, thread_name_prefix="recorder-status")  # never blocked by images
        # heavy writes (render/encode) wait for quiet windows: the loop marks inference as busy()
        self._quiet = threading.Event()
        self._quiet.set()
        self._busy = 0
        self._busy_gen = 0
        self._closing = False
        self._busy_lock = threading.Lock()
        self._status = {"state": "starting", "verb": verb}
        run = {"verb": verb, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
               "host": platform.node(), "pid": os.getpid(), "versions": _versions(), "git": _git(),
               "artifacts": artifacts, "settings": settings_snapshot(), **(meta or {})}
        self.update_meta(run)
        self.status(state="running")

    @property
    def items_path(self):
        return self.dir / "items.jsonl"

    def update_meta(self, extra: dict):
        path = self.dir / "run.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(extra)
        self._atomic_write(path, json.dumps(current, indent=2, default=_jsonable))

    # -- status: what is happening right now (dashboard "live" pill)
    def status(self, **fields):
        self._status.update(fields, updated=time.time())
        snap = dict(self._status)
        self._st.submit(self._safe, self._atomic_write, self.dir / "status.json", json.dumps(snap, default=_jsonable))

    def stage_listener(self, item):
        def on_stage(name, attrs):
            self.status(state="running", item=item, stage=name, stage_attrs=attrs, stage_started=time.time())
        return on_stage

    @contextmanager
    def busy(self):
        """Mark latency-critical work (inference, VLM): image encoding pauses until it ends."""
        with self._busy_lock:
            self._busy += 1
            self._busy_gen += 1
            self._quiet.clear()
        try:
            yield
        finally:
            with self._busy_lock:
                self._busy -= 1
                if self._busy == 0:
                    self._quiet.set()

    MIN_QUIET_S = 0.03

    def _wait_quiet(self):
        """Block until the loop has been out of busy() for MIN_QUIET_S. During arm motion that is
        immediate; in back-to-back offline loops rendering simply waits for close()."""
        while not self._closing:
            self._quiet.wait()
            gen = self._busy_gen
            time.sleep(self.MIN_QUIET_S)
            if self._quiet.is_set() and gen == self._busy_gen:
                return

    # -- events: run-level log
    def event(self, level, msg, **data):
        getattr(log, level if level in ("info", "warning", "error") else "info")(msg)
        rec = {"ts": time.strftime("%H:%M:%S"), "t": time.time(), "level": level, "msg": msg, **data}
        self._io.submit(self._safe, self._append, self.dir / "events.jsonl", rec)

    # -- items
    def item(self, rec: dict, views=()):
        """views: [{"frame": PIL, "amap": ndarray|None, "info": {...}}] -> images saved, then the JSON line."""
        self._io.submit(self._safe, self._write_item, rec, list(views))

    def _write_item(self, rec, views):
        if self.artifacts and views:
            arts = []
            for v, view in enumerate(views):
                base = f"items/{rec['item']:04d}_v{v}"
                frame = view["frame"].convert("RGB")
                self._wait_quiet()
                frame.save(self.dir / f"{base}_frame.jpg", quality=88)
                info = dict(view.get("info") or {})
                small = frame.copy()
                small.thumbnail((800, 800))  # overlay at <=800 px: plenty to read, keeps renders short
                self._wait_quiet()
                ov = render_overlay(small, view.get("amap"), info)
                ov.save(self.dir / f"{base}_overlay.jpg", quality=85)
                thumb = ov.copy()
                thumb.thumbnail((240, 240))
                thumb.save(self.dir / f"{base}_thumb.jpg", quality=80)
                a = {"frame": f"{base}_frame.jpg", "overlay": f"{base}_overlay.jpg", "thumb": f"{base}_thumb.jpg"}
                if view.get("amap") is not None:
                    m = np.clip(np.asarray(view["amap"], dtype=np.float32), 0, 1)
                    Image.fromarray((m * 255).astype(np.uint8)).save(self.dir / f"{base}_map.png")
                    a["map"] = f"{base}_map.png"
                arts.append(a)
            rec["artifacts"] = arts
        self._append(self.items_path, rec)  # after the images exist: the dashboard never sees broken links

    def close(self, state="done", **fields):
        self._closing = True
        self._quiet.set()
        self.status(state=state, stage=None, stage_attrs=None, **fields)
        self._io.shutdown(wait=True)
        self._st.shutdown(wait=True)

    # -- helpers
    @staticmethod
    def _append(path, rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    @staticmethod
    def _atomic_write(path, text):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def _safe(self, fn, *a):
        try:
            fn(*a)
        except Exception:  # noqa: BLE001  (diagnostics must never take the robot down)
            log.error("recorder write failed:\n%s", traceback.format_exc())
