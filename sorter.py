"""The sorting loop: capture -> score -> decide -> place, fully traced.

Verbs (run in roughly this order tomorrow):
    python sorter.py resources                 # connect, list components, print arm pose/joints
    python sorter.py capture -n 3              # save frames from the camera (no arm motion unless --move)
    python sorter.py detect                    # score live frames (no arm motion), record a run
    python sorter.py detect --image a.png b/   # ... or score files / folders
    python sorter.py sort --dry-run --step     # full loop on the real machine, arm motion only logged
    python sorter.py sort -n 10                # the real thing (Ctrl-C stops the arm)
    python sorter.py offline                   # full loop on a folder of images, robot in dry-run

Common flags: --device NPU --model-path models/x/model.xml --views 2 --decider rule --no-artifacts
Every run writes logs/<verb>_<time>/ (see diagnostics.py). Watch it live with
`python dashboard.py`, summarise it with `python evaluate.py logs/<run>`.
"""

import argparse
import asyncio
import logging
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import settings
from decider import get_decider, is_uncertain
from detector import Detector, test_images
from diagnostics import RunRecorder, Trace
from robot import Robot

log = logging.getLogger("sorter")

STAGE_ABBR = [("pick", "pick"), ("capture", "cap"), ("detect", "det"), ("decide", "dec"), ("recheck", "re"),
              ("place", "place")]


class _NoRecorder:
    @staticmethod
    def busy():
        return nullcontext()


class Sorter:
    def __init__(self, robot: Robot, detector: Detector, decider, recorder: RunRecorder | None = None,
                 pick=True, recheck=True):
        self.robot, self.detector, self.decider = robot, detector, decider
        self.recorder = recorder
        self._busy = (recorder or _NoRecorder).busy  # pauses background image encoding during inference
        self.pick, self.recheck = pick, recheck
        self.count = 0

    def aggregate(self, scores):
        return max(scores) if settings.VIEW_AGGREGATE == "max" else float(np.mean(scores))

    def _decide(self, trace, images, analyses):
        scores = [a.score for a in analyses]
        score = self.aggregate(scores)
        with trace.span("decide", decider=type(self.decider).__name__) as attrs, self._busy():
            action, confidence = self.decider.decide(images[int(np.argmax(scores))], score)
            attrs.update(action=action, confidence=confidence)
        return score, action, confidence

    async def process_item(self, label=None, source=None, views=None):
        """Handle one item end to end. label: ground truth ("good"/"defect") when known."""
        self.count += 1
        n = self.count
        views = views or settings.NUM_VIEWS
        trace = Trace(self.recorder.stage_listener(n) if self.recorder else None)
        self.robot.trace = trace
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": n, "source": str(source) if source else None,
               "label": label, "device": self.detector.device, "model": self.detector.model_path.parent.name,
               "dry_run": self.robot.dry_run, "status": "ok"}
        images, analyses = [], []
        try:
            if self.pick:
                with trace.span("pick") as attrs:
                    rec["grabbed"] = attrs["grabbed"] = await self.robot.pick("pick")
            with trace.span("capture", views=views):
                images = await self.robot.capture(views)
            with trace.span("detect", device=self.detector.device, views=len(images)), self._busy():
                analyses = [self.detector.analyze(img) for img in images]
            score, action, confidence = self._decide(trace, images, analyses)

            extra = settings.VIEW_OFFSETS[views:]
            if self.recheck and is_uncertain(confidence) and extra:  # unsure: look again before committing
                with trace.span("recheck", extra_views=len(extra), first_score=round(score, 4)):
                    more = await self.robot.capture(extra)
                    images += more
                    with self._busy():
                        analyses += [self.detector.analyze(img) for img in more]
                    score, action, confidence = self._decide(trace, images, analyses)

            with trace.span("place", target=action):
                await self.robot.place(action)
            rec.update(score=round(score, 4), action=action, confidence=confidence,
                       reason=getattr(self.decider, "last_reason", ""))
        except BaseException as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            raise
        finally:
            self.robot.trace = None
            self._finish(rec, trace, images, analyses)
        return rec

    def _finish(self, rec, trace, images, analyses):
        ms = trace.top_level_ms()
        ms["total"] = trace.elapsed_ms()
        ms["infer_per_view"] = [round(a.infer_ms, 2) for a in analyses]
        ms["preprocess_per_view"] = [round(a.preprocess_ms, 2) for a in analyses]
        ms["diag_per_view"] = [round(a.diag_ms, 2) for a in analyses]
        warnings = [f"view {i}: {w}" for i, a in enumerate(analyses) for w in a.warnings]
        rec.update(scores=[round(a.score, 4) for a in analyses], views=len(analyses), ms=ms,
                   spans=trace.sorted_spans(), diag=[a.info() for a in analyses], warnings=warnings)
        if self.recorder:
            title = f"#{rec['item']} " + (f"{rec['score']:.3f} -> {rec['action']} ({rec['confidence']:.2f})"
                                          if "action" in rec else "ERROR")
            views = [{"frame": img, "amap": a.anomaly_map,
                      "info": {"score": a.score, "map": a.map, "title": f"{title}  v{i}"}}
                     for i, (img, a) in enumerate(zip(images, analyses))]
            self.recorder.item(rec, views)
            for w in warnings:
                self.recorder.event("warning", f"item {rec['item']} {w}", item=rec["item"])
            if rec["status"] == "error":
                self.recorder.event("error", f"item {rec['item']} {rec['error']}", item=rec["item"],
                                    traceback=rec["traceback"])
            self.recorder.status(state="running", item=rec["item"], stage="idle", stage_started=time.time())
        self._console(rec)

    def _console(self, rec):
        ms = rec["ms"]
        stages = " ".join(f"{abbr} {ms[k]:.0f}" for k, abbr in STAGE_ABBR if k in ms)
        pre, inf = sum(ms["preprocess_per_view"]), sum(ms["infer_per_view"])
        if rec["status"] == "error":
            log.error("item %d ERROR %s | %s | %.0f ms", rec["item"], rec["error"], stages, ms["total"])
            return
        mark = ""
        if rec.get("label"):
            mark = "  OK" if rec["action"] == settings.ACTION_FOR[rec["label"]] else "  WRONG"
        warn = f"  !{len(rec['warnings'])} warning(s)" if rec["warnings"] else ""
        log.info("item %d: %.3f -> %s (%.2f) | %s (pre %.0f inf %.0f) | %.0f ms%s%s", rec["item"], rec["score"],
                 rec["action"], rec["confidence"], stages, pre, inf, ms["total"], mark, warn)


# ------------------------------------------------------------------ verbs
def build_detector(args):
    det = Detector(args.model_path, args.device)
    log.info("%s", det)
    if not det.ref_stats:
        log.warning("model has no training-image stats: drift warnings off "
                    "(python detector.py --calibrate-stats --model-path %s)", det.model_path)
    return det


def run_meta(det, decider, robot, args):
    import openvino as ov

    core = ov.Core()
    devices = {}
    for d in core.available_devices:
        try:
            devices[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:  # noqa: BLE001
            devices[d] = "?"
    return {
        "args": {k: v for k, v in vars(args).items()},
        "detector": {"model_path": str(det.model_path), "device": det.device, "precision": det.precision or "default",
                     "input_hw": list(det.input_hw), "compile_ms": round(det.compile_ms), "metadata": det.meta},
        "decider": {"type": type(decider).__name__, "threshold": settings.THRESHOLD,
                    "uncertain_margin": settings.UNCERTAIN_MARGIN},
        "robot": {"connected": robot.connected, "dry_run": robot.dry_run, "address": robot.creds.get("address", "")},
        "openvino_devices": devices,
    }


def make_recorder(verb, args, det, decider, robot):
    artifacts = settings.SAVE_ARTIFACTS and not getattr(args, "no_artifacts", False)
    rec = RunRecorder(verb, artifacts=artifacts, meta=run_meta(det, decider, robot, args))
    log.info("recording to %s  (live view: python dashboard.py)", rec.dir)
    return rec


def gather_images(paths):
    """Files and folders -> [(path, label or None)]; labels from folder names good/<defect> when present."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += [(q, "defect" if y else "good") for q, y in test_images(p)]
        else:
            out.append((p, None))
    return out


async def verb_resources(args):
    async with Robot(dry_run=True, address=args.address) as robot:
        import json
        print(json.dumps(await robot.describe(), indent=2))


async def verb_capture(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    async with Robot(dry_run=not args.move, address=args.address) as robot:
        for i in range(args.n):
            trace = Trace()
            robot.trace = trace
            frames = await robot.capture(args.views)
            for v, img in enumerate(frames):
                img.save(out / f"frame_{i:03d}_v{v}.png")
            cams = [s.get("attrs", {}) for s in trace.spans if s["name"] == "camera"]
            log.info("capture %d: %d view(s) %s in %.0f ms  camera: %s", i, len(frames), frames[0].size,
                     trace.elapsed_ms(), cams[0] if cams else {})
    print(f"saved to {out}")


async def run_loop(verb, args, robot, items=None, pick=True, recheck=True):
    """Shared driver for detect / sort / offline. items: [(path, label)] for offline frames, else live."""
    det = build_detector(args)
    decider = get_decider(args.decider)
    recorder = make_recorder(verb, args, det, decider, robot)
    sorter = Sorter(robot, det, decider, recorder, pick=pick, recheck=recheck)
    state = "done"
    try:
        if items is not None:
            for path, label in items:
                robot.set_frames([path])  # every view of this item = this file
                await sorter.process_item(label=label, source=path, views=args.views)
        else:
            for i in range(args.n or 10**9):
                if getattr(args, "step", False):
                    recorder.status(state="running", item=i + 1, stage="waiting for operator",
                                    stage_started=time.time())
                    await asyncio.to_thread(input, f"[item {i + 1}] place an item and press Enter (Ctrl-C stops) ")
                await sorter.process_item(views=args.views)
    except (KeyboardInterrupt, asyncio.CancelledError):
        state = "stopped"
        recorder.event("warning", "interrupted by operator: stopping arm")
        await robot.stop()
    except Exception as e:
        state = "error"
        recorder.event("error", f"run aborted: {type(e).__name__}: {e}", traceback=traceback.format_exc())
        await robot.stop()
        raise
    finally:
        recorder.close(state=state)
        print(f"\nrun: {recorder.dir}\nnext: python evaluate.py {recorder.dir}   |   python dashboard.py")
    return recorder.dir


async def verb_detect(args):
    if args.image:
        robot = await Robot(dry_run=True, connect=False).connect()
        logging.getLogger("robot").setLevel(logging.WARNING)
        return await run_loop("detect", args, robot, items=gather_images(args.image), pick=False, recheck=False)
    async with Robot(dry_run=True, address=args.address) as robot:
        return await run_loop("detect", args, robot, pick=False, recheck=False)


async def verb_sort(args):
    async with Robot(dry_run=args.dry_run, address=args.address) as robot:
        return await run_loop("sort", args, robot, pick=not args.no_pick)


async def verb_offline(args):
    items = [(p, "defect" if y else "good") for p, y in test_images(args.images)]
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    robot = await Robot(dry_run=True, connect=False).connect()
    logging.getLogger("robot").setLevel(logging.WARNING)  # hide per-move dry-run chatter
    # offline every "view" is the same file, so re-checking only adds latency
    return await run_loop("offline", args, robot, items=items, recheck=args.recheck)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    settings.check()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    def common(p, model=True):
        p.add_argument("--address", default=None, help="override api-key.json address")
        if model:
            p.add_argument("--model-path", default=None)
            p.add_argument("--device", default=None, help="CPU | GPU | NPU | AUTO")
            p.add_argument("--decider", default=settings.DECIDER, choices=("rule", "vlm", "hybrid"))
            p.add_argument("--no-artifacts", action="store_true", help="don't save per-item images (JSON only)")
        p.add_argument("--views", type=int, default=None, help=f"views per item (default {settings.NUM_VIEWS})")
        return p

    common(sub.add_parser("resources", help="list components and current arm pose"), model=False)
    p = common(sub.add_parser("capture", help="save camera frames"), model=False)
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--out", default="/tmp/frames")
    p.add_argument("--move", action="store_true", help="actually move between views (default: dry run)")
    p = common(sub.add_parser("detect", help="score live frames or files, no arm motion"))
    p.add_argument("--image", nargs="*", default=None, help="files or folders (good/<defect> subfolders give labels)")
    p.add_argument("-n", type=int, default=3)
    p = common(sub.add_parser("sort", help="run the real sorting loop"))
    p.add_argument("-n", type=int, default=None, help="number of items (default: until Ctrl-C)")
    p.add_argument("--dry-run", action="store_true", help="log arm motion instead of moving")
    p.add_argument("--step", action="store_true", help="wait for Enter before each item")
    p.add_argument("--no-pick", action="store_true", help="items already sit at the inspect spot")
    p = common(sub.add_parser("offline", help="run the loop on a folder of images (robot dry-run)"))
    p.add_argument("--images", default=str(settings.TEST_IMAGES), help="folder with good/ and defect subfolders")
    p.add_argument("--limit", type=int, default=None, help="evenly sample this many images")
    p.add_argument("--recheck", action="store_true", help="exercise the uncertain-item recheck path")

    args = ap.parse_args()
    asyncio.run(globals()[f"verb_{args.verb}"](args))


if __name__ == "__main__":
    main()
