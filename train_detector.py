"""Train an Anomalib anomaly detector, test it, and export it to OpenVINO.

Works on MVTec AD categories and on your own photos.

Dataset layouts (auto-detected from --data):
    MVTec:  <data>/train/good, <data>/test/{good,<defect>...}, <data>/ground_truth/...
    Folder: <data>/train/good, <data>/test/good, <data>/test/<defect>[, <defect2>...]
            Every folder under test/ except "good" is treated as a defect class.
            If there are no defect photos at all, synthetic anomalies are used to
            pick the threshold (works, but real defects give a better threshold).

Examples:
    python train_detector.py                                   # PatchCore on MVTec bottle
    python train_detector.py --data datasets/parts --coreset-ratio 0.02
    python train_detector.py --model padim --data datasets/parts
    python train_detector.py --model efficientad --epochs 20 --data datasets/parts
    python train_detector.py --compression int8_ptq            # smaller/faster (NPU/GPU)

Output: models/<model>_<dataset>/model.xml + model.bin + metadata.json
(point settings.MODEL_PATH at that model.xml, or pass --model-path to the tools).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

MODELS = ("patchcore", "efficientad", "padim")
COMPRESSION = ("none", "fp16", "int8", "int8_ptq")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="patchcore")
    p.add_argument("--data", default="datasets/MVTecAD/bottle", help="dataset folder (see layouts above)")
    p.add_argument("--name", default=None, help="run name (default: <model>_<dataset folder name>)")
    p.add_argument("--image-size", type=int, nargs="+", default=[256], help="H [W] fed to the model")
    p.add_argument("--coreset-ratio", type=float, default=0.1, help="PatchCore memory-bank fraction; lower = faster")
    p.add_argument("--backbone", default=None, help="override backbone (patchcore/padim), e.g. resnet18")
    p.add_argument("--epochs", type=int, default=20, help="EfficientAD epochs (PatchCore/PaDiM always use 1)")
    p.add_argument("--efficientad-size", choices=("small", "medium"), default="small")
    p.add_argument("--val-split", choices=("from_test", "same_as_test", "synthetic"), default=None,
                   help="where the threshold is fitted (default: from_test, or synthetic if no defect photos)")
    p.add_argument("--compression", choices=COMPRESSION, default="none",
                   help="OpenVINO weight compression; int8_ptq calibrates on the data")
    p.add_argument("--accelerator", default="cpu", help="lightning accelerator for training (cpu, xpu, cuda)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--no-test", action="store_true", help="skip engine.test (faster)")
    return p.parse_args()


def build_datamodule(args, data: Path):
    from anomalib.data import Folder, MVTecAD

    common = dict(train_batch_size=args.batch_size, eval_batch_size=args.batch_size, num_workers=args.workers)
    if args.model == "efficientad":
        common["train_batch_size"] = 1  # EfficientAD requires batch size 1

    if (data / "ground_truth").is_dir():
        if args.val_split:
            common["val_split_mode"] = args.val_split
        return MVTecAD(root=str(data.parent), category=data.name, **common), "mvtec"

    train_good = data / "train" / "good"
    test_dir = data / "test"
    if not train_good.is_dir():
        raise SystemExit(f"expected {train_good} with good images")
    abnormal = sorted(str(d.relative_to(data)) for d in test_dir.iterdir()
                      if d.is_dir() and d.name != "good") if test_dir.is_dir() else []
    abnormal = [a for a in abnormal if any((data / a).iterdir())]
    normal_test = data / "test" / "good"
    has_normal_test = normal_test.is_dir() and any(normal_test.iterdir())

    val_split = args.val_split or ("from_test" if abnormal else "synthetic")
    dm = Folder(
        name=data.name,
        root=str(data),
        normal_dir="train/good",
        normal_test_dir="test/good" if has_normal_test else None,
        abnormal_dir=abnormal or None,
        # with no defect photos, build the test set synthetically from good images
        test_split_mode="from_dir" if abnormal else "synthetic",
        val_split_mode=val_split,
        extensions=(".png", ".jpg", ".jpeg", ".bmp"),
        **common,
    )
    print(f"Folder dataset: abnormal={abnormal or 'none (synthetic)'} val_split={val_split}")
    return dm, "folder"


def build_model(args, image_size):
    from anomalib.models import EfficientAd, Padim, Patchcore

    if args.model == "patchcore":
        kw = dict(coreset_sampling_ratio=args.coreset_ratio,
                  pre_processor=Patchcore.configure_pre_processor(image_size=image_size))
        if args.backbone:
            kw["backbone"] = args.backbone
        return Patchcore(**kw)
    if args.model == "padim":
        kw = dict(pre_processor=Padim.configure_pre_processor(image_size=image_size))
        if args.backbone:
            kw["backbone"] = args.backbone
        return Padim(**kw)
    return EfficientAd(model_size=args.efficientad_size,
                       pre_processor=EfficientAd.configure_pre_processor(image_size=image_size))


def main():
    args = parse_args()
    from anomalib.deploy import CompressionType, ExportType
    from anomalib.engine import Engine

    data = Path(args.data).resolve()
    hw = tuple(args.image_size * 2 if len(args.image_size) == 1 else args.image_size[:2])
    name = args.name or f"{args.model}_{data.name}"
    out_dir = Path(args.models_dir) / name

    datamodule, layout = build_datamodule(args, data)
    model = build_model(args, hw)

    engine_kw = dict(accelerator=args.accelerator, devices=1, default_root_dir="./results")
    if args.model == "efficientad":
        engine_kw["max_epochs"] = args.epochs
    engine = Engine(**engine_kw)

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    fit_s = time.time() - t0

    metrics = {}
    if not args.no_test:
        results = engine.test(model=model, datamodule=datamodule)
        metrics = {k: float(v) for k, v in (results[0] if results else {}).items()}

    # Static [1, 3, H, W] input: NPU needs static shapes, and it avoids the dynamo
    # batch-size specialisation issue (anomalib #3883).
    compression = None if args.compression == "none" else CompressionType(args.compression)
    exported = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=out_dir,
        input_size=hw,
        compression_type=compression,
        datamodule=datamodule if args.compression == "int8_ptq" else None,
        onnx_kwargs={"dynamic_shapes": None},
    )
    exported = Path(exported)
    # flatten models/<name>/weights/openvino/model.* -> models/<name>/model.*
    for f in exported.parent.glob("model.*"):
        shutil.move(str(f), out_dir / f.name)
    shutil.rmtree(out_dir / "weights", ignore_errors=True)

    meta = {
        "model": args.model, "dataset": str(data), "layout": layout, "image_size": list(hw),
        "coreset_ratio": args.coreset_ratio if args.model == "patchcore" else None,
        "compression": args.compression, "fit_seconds": round(fit_s, 1), "metrics": metrics,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "pred_score is normalized: 0.5 == threshold fitted on the validation split",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nExported {out_dir / 'model.xml'}\n{json.dumps(meta, indent=2)}")


# Only run when launched directly. Data-loading worker processes import this
# file (spawn on macOS), and without this guard each would start training again.
if __name__ == "__main__":
    main()
