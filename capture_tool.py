"""Save camera frames into the dataset layout train_detector.py expects.

    <DATASET_DIR>/<name>/train/good/*.png
    <DATASET_DIR>/<name>/test/good/*.png
    <DATASET_DIR>/<name>/test/<defect label>/*.png

Frames come through Robot.capture(), i.e. the same camera, crop (settings.CROP)
and viewpoints the live loop uses - train on exactly what you will score.

Examples:
    python capture_tool.py parts --label good -n 60 --step      # Enter between items; 1 in 5 -> test/good
    python capture_tool.py parts --label scratch -n 10 --step   # defects always go to test/<label>
    python capture_tool.py parts --label good -n 30 --interval 0.5   # hands-free burst while you move items
    python capture_tool.py parts --label good --views 4 --move -n 15 # arm rotates the held item per view
    python capture_tool.py parts --label good --import ~/phone_pics  # import existing photos
    python capture_tool.py parts --stats                             # counts per folder

Then: python train_detector.py --data datasets/custom/parts
"""

import argparse
import asyncio
import time
from pathlib import Path

from PIL import Image

import settings
from robot import Robot

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def target_dir(root: Path, label: str, split: str, item_index: int, test_every: int) -> Path:
    if label != "good":
        return root / "test" / label
    if split == "auto":
        split = "test" if test_every and item_index % test_every == test_every - 1 else "train"
    return root / split / "good"


def stats(root: Path):
    rows = []
    for d in sorted(p for p in root.glob("*/*") if p.is_dir()):
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in EXTS)
        rows.append((str(d.relative_to(root)), n))
    for name, n in rows:
        print(f"  {name:28} {n:5d}")
    if not rows:
        print(f"  (empty) {root}")
    return rows


def import_folder(src: Path, root: Path, label: str, split: str, test_every: int):
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    for i, f in enumerate(files):
        out = target_dir(root, label, split, i, test_every)
        out.mkdir(parents=True, exist_ok=True)
        img = Image.open(f).convert("RGB")
        img.save(out / f"{label}_import_{i:04d}.png")
    print(f"imported {len(files)} images from {src}")


async def capture(args, root: Path):
    source = None
    if args.source:  # fake camera from a folder (offline testing)
        source = sorted(p for p in Path(args.source).rglob("*") if p.suffix.lower() in EXTS)
    robot = Robot(dry_run=not args.move, address=args.address, connect=source is None, image_source=source)
    await robot.connect()
    try:
        stamp = time.strftime("%H%M%S")
        for i in range(args.n):
            if args.step:
                input(f"[{i + 1}/{args.n}] position a '{args.label}' item, press Enter ")
            frames = await robot.capture(args.views)
            out = target_dir(root, args.label, args.split, i, args.test_every)
            out.mkdir(parents=True, exist_ok=True)
            for v, img in enumerate(frames):
                img.save(out / f"{args.label}_{stamp}_{i:04d}_v{v}.png")
            print(f"  item {i + 1}: {len(frames)} view(s) {frames[0].size} -> {out.relative_to(root)}")
            if args.interval and not args.step:
                await asyncio.sleep(args.interval)
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset name, e.g. parts")
    ap.add_argument("--label", default="good", help="'good' or a defect name (scratch, dent, missing_cap...)")
    ap.add_argument("-n", type=int, default=20, help="number of items (each gives --views frames)")
    ap.add_argument("--split", choices=("auto", "train", "test"), default="auto",
                    help="good images only: auto puts every --test-every'th item in test/good")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--move", action="store_true", help="move the arm between views (default: no motion)")
    ap.add_argument("--step", action="store_true", help="wait for Enter before each item")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between items without --step")
    ap.add_argument("--import", dest="import_dir", default=None, help="import photos from a folder instead")
    ap.add_argument("--source", default=None, help="offline: use a folder of images as the camera")
    ap.add_argument("--address", default=None)
    ap.add_argument("--root", default=str(settings.DATASET_DIR))
    ap.add_argument("--stats", action="store_true", help="just print image counts")
    args = ap.parse_args()

    root = Path(args.root) / args.name
    if args.stats:
        stats(root)
        return
    if args.label != "good" and args.split == "train":
        raise SystemExit("defect images never go in train/ (anomaly models train on good only)")
    if args.import_dir:
        import_folder(Path(args.import_dir), root, args.label, args.split, args.test_every)
    else:
        asyncio.run(capture(args, root))
    print(f"\n{root}:")
    stats(root)
    print(f"\ntrain: python train_detector.py --data {root}")


if __name__ == "__main__":
    main()
