"""OpenVINO anomaly detector: load the exported Anomalib model once, score images fast.

    from detector import Detector
    det = Detector()                       # settings.MODEL_PATH on settings.DEVICE
    score, amap, ms = det.score(pil_image) # score in [0,1], 0.5 == trained threshold

The exported graph contains Anomalib's whole pipeline (resize + ImageNet
normalisation -> model -> min/max score normalisation), so all we do here is
crop/resize to the static input size and hand it a float RGB tensor in [0,1].

CLI:
    python detector.py                         # score a few test images, print timings
    python detector.py --verify                # compare against Anomalib (torch) on all MVTec test images
    python detector.py --device GPU --model-path models/x/model.xml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import settings


def load_image(image, crop=None) -> Image.Image:
    """Accept a path, PIL image, or HxWx3 uint8 RGB array; return an RGB PIL image."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    if crop:
        image = image.crop(tuple(crop))
    return image


class Detector:
    def __init__(self, model_path=None, device=None, warmup=3, crop=None, precision=None):
        import openvino as ov

        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.device = (device or settings.DEVICE).upper()
        self.crop = crop  # usually None: live frames are cropped by Robot.get_image (settings.CROP)
        meta_file = self.model_path.with_name("metadata.json")
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

        self.precision = precision if precision is not None else settings.INFERENCE_PRECISION
        core = ov.Core()
        # One cache folder per device+precision: OpenVINO's CPU cache key ignores
        # INFERENCE_PRECISION_HINT, so a shared folder silently reuses the wrong blob.
        cache = settings.CACHE_DIR / f"{self.device}_{self.precision or 'default'}".replace(":", "-")
        cache.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache)})
        model = core.read_model(self.model_path)

        # Force a static [1, 3, H, W] input: NPU needs static shapes, and some
        # dynamo exports pin the batch to 2 (anomalib #3883).
        inp = model.input(0)
        h, w = self.meta.get("image_size", [None, None])
        ps = inp.get_partial_shape()
        if h is None:
            h = ps[2].get_length() if ps[2].is_static else 256
            w = ps[3].get_length() if ps[3].is_static else 256
        if not ps.is_static or ps[0].get_length() != 1:
            model.reshape({inp.any_name: [1, 3, h, w]})
        self.input_hw = (h, w)

        config = {"PERFORMANCE_HINT": settings.PERFORMANCE_HINT}
        if self.precision and not self.device.startswith("NPU"):  # NPU is f16 only
            config["INFERENCE_PRECISION_HINT"] = self.precision
        t0 = time.perf_counter()
        self.compiled = core.compile_model(model, self.device, config)
        self.compile_ms = (time.perf_counter() - t0) * 1000
        self.request = self.compiled.create_infer_request()
        self.outputs = {o.any_name: o for o in self.compiled.outputs}

        dummy = np.random.rand(1, 3, h, w).astype(np.float32)
        for _ in range(warmup):
            self.request.infer({0: dummy})

    def preprocess(self, image) -> np.ndarray:
        img = load_image(image, self.crop)
        h, w = self.input_hw
        if img.size != (w, h):
            # PIL bilinear is antialiased, matching torchvision Resize(antialias=True)
            img = img.resize((w, h), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def score(self, image):
        """Return (score, anomaly_map HxW float32, milliseconds incl. preprocessing)."""
        t0 = time.perf_counter()
        x = self.preprocess(image)
        self.request.infer({0: x})
        score = float(self.request.get_tensor(self.outputs["pred_score"]).data.reshape(-1)[0])
        amap = None
        if "anomaly_map" in self.outputs:
            amap = self.request.get_tensor(self.outputs["anomaly_map"]).data.reshape(self.input_hw).copy()
        return score, amap, (time.perf_counter() - t0) * 1000

    def __repr__(self):
        prec = self.precision or "default"
        return (f"Detector({self.model_path}, device={self.device}, precision={prec}, "
                f"input={self.input_hw}, compile={self.compile_ms:.0f}ms)")


def test_images(folder):
    """(path, label) pairs from a test/ folder: good -> 0, anything else -> 1."""
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [(p, 0 if p.parent.name == "good" else 1)
            for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]


def find_checkpoint(meta):
    """The Lightning checkpoint the exported model came from (recorded by train_detector.py)."""
    if meta.get("checkpoint") and Path(meta["checkpoint"]).exists():
        return Path(meta["checkpoint"])
    name = {"patchcore": "Patchcore", "padim": "Padim", "efficientad": "EfficientAd"}[meta["model"]]
    cands = sorted(Path("results").glob(f"{name}/**/weights/lightning/model.ckpt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def verify(det, folder, ckpt=None):
    """Score every test image with OpenVINO and with the Anomalib torch model; compare.

    The torch reference follows engine.test: resize with antialias, then forward().
    (The exported graph's own Resize has antialias disabled, so never feed it
    full-resolution frames - Detector.preprocess resizes first.)
    """
    import torch
    from sklearn.metrics import roc_auc_score
    from torchvision.transforms.v2.functional import resize
    from anomalib.models import EfficientAd, Padim, Patchcore

    items = test_images(folder)
    ckpt = Path(ckpt) if ckpt else find_checkpoint(det.meta)
    cls = {"patchcore": Patchcore, "padim": Padim, "efficientad": EfficientAd}[det.meta["model"]]
    print(f"torch reference: {ckpt}")
    ref = cls.load_from_checkpoint(str(ckpt), map_location="cpu").eval()

    rows = []
    with torch.no_grad():
        for path, label in items:
            s_ov, _, ms = det.score(path)
            rgb = np.asarray(load_image(path, det.crop), dtype=np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
            x = resize(x, list(det.input_hw), antialias=True)
            s_pt = float(ref(x).pred_score.reshape(-1)[0])
            rows.append((path, label, s_ov, s_pt, ms))

    labels = np.array([r[1] for r in rows])
    ov_s = np.array([r[2] for r in rows])
    pt_s = np.array([r[3] for r in rows])
    diff = np.abs(ov_s - pt_s)
    agree = ((ov_s >= settings.THRESHOLD) == (pt_s >= settings.THRESHOLD)).mean()
    print(f"\n{len(rows)} images  |score diff| mean={diff.mean():.4f} max={diff.max():.4f}  "
          f"label agreement={agree:.1%}")
    print(f"image AUROC  openvino={roc_auc_score(labels, ov_s):.4f}  torch={roc_auc_score(labels, pt_s):.4f}")
    acc = ((ov_s >= settings.THRESHOLD) == labels).mean()
    print(f"openvino accuracy @ {settings.THRESHOLD}: {acc:.1%}   median latency {np.median([r[4] for r in rows]):.1f} ms")
    worst = np.argsort(-diff)[:3]
    for i in worst:
        print(f"  largest diff: {rows[i][0].parent.name}/{rows[i][0].name} ov={ov_s[i]:.4f} torch={pt_s[i]:.4f}")
    return diff.max(), agree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--images", default=str(settings.TEST_IMAGES), help="test/ folder with good/ and defect subfolders")
    ap.add_argument("--verify", action="store_true", help="compare with the Anomalib torch model")
    ap.add_argument("--ckpt", default=None, help="Lightning checkpoint for --verify (default: from metadata.json)")
    ap.add_argument("--precision", default=None, help="INFERENCE_PRECISION_HINT, e.g. f32 or f16")
    args = ap.parse_args()

    det = Detector(args.model_path, args.device, precision=args.precision)
    print(det)
    if args.verify:
        verify(det, args.images, args.ckpt)
        return
    items = test_images(args.images)
    for path, label in items[:: max(1, len(items) // 8)]:
        s, _, ms = det.score(path)
        print(f"{path.parent.name:>14}/{path.name}  score={s:.3f}  {'DEFECT' if s >= settings.THRESHOLD else 'good':6}  {ms:6.1f} ms")


if __name__ == "__main__":
    main()
