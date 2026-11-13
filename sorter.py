"""The sorting loop: capture -> score -> decide -> place, one JSON log line per item.

Verbs (run in roughly this order tomorrow):
    python sorter.py resources                 # connect, list components, print arm pose/joints
    python sorter.py capture -n 3              # save frames from the camera (no arm motion unless --move)
    python sorter.py detect                    # score live frames, print verdict + latency, save heatmaps
    python sorter.py detect --image x.png      # ... or score files
    python sorter.py sort --dry-run            # full loop on the real machine, arm motion only logged
    python sorter.py sort -n 10                # the real thing (Ctrl-C stops the arm)
    python sorter.py offline                   # full loop on a folder of images, robot in dry-run

Common flags: --device NPU --model-path models/x/model.xml --views 2 --decider rule --save-viz
Each run writes logs/<verb>_<time>.jsonl; feed it to evaluate.py.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

import settings
from decider import get_decider, is_uncertain
from detector import Detector, test_images
from robot import Robot

log = logging.getLogger("sorter")


def ms_since(t0):
    return round((time.perf_counter() - t0) * 1000, 2)


def overlay(image: Image.Image, amap, score, text=""):
    """Anomaly heatmap blended over the image, with the score written on it."""
    from PIL import ImageDraw

    from matplotlib import colormaps

    img = image.convert("RGB")
    if amap is not None:
        # stretch [THRESHOLD-0.2, 1] -> [0, 1]; normal regions stay mostly transparent
        lo = settings.THRESHOLD - 0.2
        a = np.clip((np.asarray(amap, dtype=np.float32) - lo) / (1 - lo), 0, 1)
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
        heat = colormaps["jet"](a)[..., :3]
        alpha = (0.15 + 0.5 * a)[..., None]
        blend = np.asarray(img) / 255.0 * (1 - alpha) + heat * alpha
        img = Image.fromarray((blend * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    size = max(16, img.width // 25)
    draw.rectangle((0, 0, img.width, size + 12), fill=(0, 0, 0))
    draw.text((8, 4), f"score {score:.3f}  {text}", fill=(255, 255, 0), font_size=size)
    return img


class Sorter:
    def __init__(self, robot: Robot, detector: Detector, decider, log_path=None, viz_dir=None, pick=True,
                 recheck=True):
        self.robot, self.detector, self.decider = robot, detector, decider
        self.pick, self.recheck = pick, recheck
        self.log_path = Path(log_path) if log_path else None
        self.viz_dir = Path(viz_dir) if viz_dir else None
        for d in (self.log_path.parent if self.log_path else None, self.viz_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def aggregate(self, scores):
        return max(scores) if settings.VIEW_AGGREGATE == "max" else float(np.mean(scores))

    async def process_item(self, label=None, source=None, views=None):
        """Handle one item end to end. label: ground truth (offline) if known."""
        self.count += 1
        views = views or settings.NUM_VIEWS
        t = {}
        t_item = time.perf_counter()

        grabbed = None
        if self.pick:
            t0 = time.perf_counter()
            grabbed = await self.robot.pick("pick")
            t["pick"] = ms_since(t0)

        t0 = time.perf_counter()
        images = await self.robot.capture(views)
        t["capture"] = ms_since(t0)

        t0 = time.perf_counter()
        results = [self.detector.score(img) for img in images]
        t["detect"] = ms_since(t0)
        t["infer_per_view"] = [round(r[2], 2) for r in results]
        scores = [r[0] for r in results]
        score = self.aggregate(scores)

        t0 = time.perf_counter()
        action, confidence = self.decider.decide(images[int(np.argmax(scores))], score)
        t["decide"] = ms_since(t0)

        # Low confidence and spare viewpoints: look again before committing.
        extra = settings.VIEW_OFFSETS[views:]
        if self.recheck and is_uncertain(confidence) and extra:
            t0 = time.perf_counter()
            more = await self.robot.capture(extra)
            more_results = [self.detector.score(img) for img in more]
            images, results = images + more, results + more_results
            scores = [r[0] for r in results]
            score = self.aggregate(scores)
            action, confidence = self.decider.decide(images[int(np.argmax(scores))], score)
            t["recheck"] = ms_since(t0)

        t0 = time.perf_counter()
        await self.robot.place(action)
        t["place"] = ms_since(t0)
        t["total"] = ms_since(t_item)

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "item": self.count, "source": str(source) if source else None,
            "label": label, "scores": [round(s, 4) for s in scores], "score": round(score, 4),
            "action": action, "confidence": confidence, "reason": getattr(self.decider, "last_reason", ""),
            "views": len(images), "grabbed": grabbed, "device": self.detector.device,
            "model": self.detector.model_path.parent.name, "dry_run": self.robot.dry_run, "ms": t,
        }
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if self.viz_dir:
            i = int(np.argmax(scores))
            overlay(images[i], results[i][1], score, action).save(self.viz_dir / f"item_{self.count:04d}_{action}.jpg")
        mark = "" if label is None else ("  OK" if (action == settings.ACTION_FOR[label]) else "  WRONG")
        log.info("item %d: score=%.3f -> %s (conf %.2f) total %.0f ms%s",
                 self.count, score, action, confidence, t["total"], mark)
        return rec


# ------------------------------------------------------------------ verbs
def run_paths(verb, args):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = settings.LOG_DIR / f"{verb}_{stamp}.jsonl"
    viz_dir = settings.LOG_DIR / f"{verb}_{stamp}_viz" if args.save_viz else None
    return log_path, viz_dir


def build_detector(args):
    det = Detector(args.model_path, args.device)
    log.info("%s", det)
    return det


async def verb_resources(args):
    async with Robot(dry_run=True, address=args.address) as robot:
        print(json.dumps(await robot.describe(), indent=2))


async def verb_capture(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    async with Robot(dry_run=not args.move, address=args.address) as robot:
        for i in range(args.n):
            t0 = time.perf_counter()
            frames = await robot.capture(args.views)
            for v, img in enumerate(frames):
                img.save(out / f"frame_{i:03d}_v{v}.png")
            log.info("capture %d: %d view(s) %s in %.0f ms", i, len(frames), frames[0].size, ms_since(t0))
    print(f"saved to {out}")


async def verb_detect(args):
    det = build_detector(args)
    decider = get_decider(args.decider)
    _, viz_dir = run_paths("detect", args)
    viz_dir = viz_dir or settings.LOG_DIR / "detect_latest"
    viz_dir.mkdir(parents=True, exist_ok=True)
    if args.image:
        sources = [Path(p) for p in args.image]
        images = [(p.name, Image.open(p).convert("RGB")) for p in sources]
    else:
        async with Robot(dry_run=True, address=args.address) as robot:
            images = [(f"live_{i}", (await robot.capture(1))[0]) for i in range(args.n)]
    for name, img in images:
        score, amap, ms = det.score(img)
        action, conf = decider.decide(img, score)
        overlay(img, amap, score, action).save(viz_dir / f"{Path(name).stem}.jpg")
        print(f"{name:30} score={score:.3f} -> {action:7} conf={conf:.2f}  {ms:.1f} ms")
    print(f"heatmaps in {viz_dir}")


async def verb_sort(args):
    det = build_detector(args)
    decider = get_decider(args.decider)
    log_path, viz_dir = run_paths("sort", args)
    async with Robot(dry_run=args.dry_run, address=args.address) as robot:
        sorter = Sorter(robot, det, decider, log_path, viz_dir, pick=not args.no_pick)
        try:
            for i in range(args.n or 10**9):
                if args.step:
                    input(f"[item {i + 1}] place an item and press Enter (Ctrl-C to stop) ")
                await sorter.process_item(views=args.views)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.warning("interrupted: stopping arm")
            await robot.stop()
    print(f"log: {log_path}")


async def verb_offline(args):
    det = build_detector(args)
    decider = get_decider(args.decider)
    log_path, viz_dir = run_paths("offline", args)
    items = test_images(args.images)
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    robot = Robot(dry_run=True, connect=False)
    await robot.connect()
    # offline every "view" is the same file, so re-checking only adds latency
    sorter = Sorter(robot, det, decider, log_path, viz_dir, pick=True, recheck=args.recheck)
    logging.getLogger("robot").setLevel(logging.WARNING)  # hide per-move dry-run chatter
    for path, y in items:
        robot.set_frames([path])  # every view of this item = this file
        await sorter.process_item(label="defect" if y else "good", source=path, views=args.views)
    print(f"\nlog: {log_path}\nnext: python evaluate.py {log_path}")
    return log_path


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
            p.add_argument("--save-viz", action="store_true", help="save heatmap overlays per item")
        p.add_argument("--views", type=int, default=None, help=f"views per item (default {settings.NUM_VIEWS})")
        return p

    common(sub.add_parser("resources", help="list components and current arm pose"), model=False)
    p = common(sub.add_parser("capture", help="save camera frames"), model=False)
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--out", default="/tmp/frames")
    p.add_argument("--move", action="store_true", help="actually move between views (default: dry run)")
    p = common(sub.add_parser("detect", help="score live frames or files"))
    p.add_argument("--image", nargs="*", default=None)
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
