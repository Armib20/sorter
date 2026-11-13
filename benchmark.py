"""Time exported models on every OpenVINO device on this machine (CPU / GPU / NPU / AUTO).

    python benchmark.py                                     # settings.MODEL_PATH on all devices
    python benchmark.py --models models/*/model.xml         # compare models
    python benchmark.py --devices NPU GPU --precisions f16 f32
    python benchmark.py --throughput                        # also async FPS with the THROUGHPUT hint

Per model x device x precision:
  compile_cold  first compile with an empty cache (what a fresh boot costs)
  compile_warm  compile again from the cache (what every later start costs)
  infer p50/p95 synchronous batch-1 latency (the sorter's case)
  e2e p50       Detector.score on a real image incl. resize/normalise
  fps           async throughput (--throughput)
Failures (e.g. an op the NPU can't compile) are reported, not fatal.
Results also go to logs/benchmark_<time>.json.
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from tabulate import tabulate

import settings


def device_names(core):
    out = []
    for d in core.available_devices:
        try:
            out.append((d, core.get_property(d, "FULL_DEVICE_NAME")))
        except Exception:  # noqa: BLE001
            out.append((d, "?"))
    return out


def bench_one(core_factory, xml, device, precision, iters, warmup, image, throughput):
    import openvino as ov

    from detector import Detector

    row = {"model": Path(xml).parent.name, "device": device, "precision": precision or "default"}
    cache = Path(tempfile.mkdtemp(prefix="ovbench_"))
    try:
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if precision and not device.startswith("NPU"):
            config["INFERENCE_PRECISION_HINT"] = precision
        core = core_factory()
        core.set_property({"CACHE_DIR": str(cache)})
        model = core.read_model(xml)
        shape = model.input(0).get_partial_shape()
        if not shape.is_static or shape[0].get_length() != 1:
            model.reshape({model.input(0).any_name: [1, 3, 256, 256]})
        h, w = model.input(0).get_shape()[2:]

        t0 = time.perf_counter()
        compiled = core.compile_model(model, device, config)
        row["compile_cold_ms"] = round((time.perf_counter() - t0) * 1000)
        del compiled
        core2 = core_factory()
        core2.set_property({"CACHE_DIR": str(cache)})
        t0 = time.perf_counter()
        compiled = core2.compile_model(xml if shape.is_static else model, device, config)
        row["compile_warm_ms"] = round((time.perf_counter() - t0) * 1000)
        try:
            t = compiled.get_property("INFERENCE_PRECISION_HINT")
            row["exec_precision"] = getattr(t, "to_string", lambda: str(t))()
        except Exception:  # noqa: BLE001
            row["exec_precision"] = "?"

        req = compiled.create_infer_request()
        x = np.random.rand(1, 3, h, w).astype(np.float32)
        for _ in range(warmup):
            req.infer({0: x})
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            req.infer({0: x})
            times.append((time.perf_counter() - t0) * 1000)
        row["infer_p50_ms"] = round(float(np.median(times)), 2)
        row["infer_p95_ms"] = round(float(np.percentile(times, 95)), 2)

        if image is not None:
            det = Detector(xml, device, warmup=warmup, precision=precision or "")
            e2e = [det.score(image)[2] for _ in range(max(5, iters // 4))]
            row["e2e_p50_ms"] = round(float(np.median(e2e)), 2)
            row["score"] = round(det.score(image)[0], 4)

        if throughput:
            tconf = {"PERFORMANCE_HINT": "THROUGHPUT"}
            if precision and not device.startswith("NPU"):
                tconf["INFERENCE_PRECISION_HINT"] = precision
            tc = core2.compile_model(model, device, tconf)
            queue = ov.AsyncInferQueue(tc)
            n = max(iters, 4 * len(queue))
            t0 = time.perf_counter()
            for _ in range(n):
                queue.start_async({0: x})
            queue.wait_all()
            row["fps"] = round(n / (time.perf_counter() - t0), 1)
            row["nireq"] = len(queue)
        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("Exception from")]
        row["status"] = f"FAILED: {(lines[-1] if lines else repr(e))[:140]}"
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return row


def main():
    import openvino as ov

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[str(settings.MODEL_PATH)])
    ap.add_argument("--devices", nargs="+", default=None, help="default: all available + AUTO")
    ap.add_argument("--precisions", nargs="+", default=[""], help='e.g. f16 f32 ("" = device default)')
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--image", default=None, help="real image for e2e timing (default: first MVTec test image)")
    ap.add_argument("--throughput", action="store_true")
    args = ap.parse_args()

    core = ov.Core()
    devs = device_names(core)
    print(f"OpenVINO {ov.__version__}")
    for d, name in devs:
        print(f"  {d:8} {name}")
    devices = args.devices or [d for d, _ in devs] + (["AUTO"] if len(devs) > 1 else [])

    image = args.image
    if image is None:
        cands = sorted(Path(settings.TEST_IMAGES).rglob("*.png"))
        image = str(cands[0]) if cands else None

    rows = []
    for xml in args.models:
        for device in devices:
            for prec in args.precisions:
                print(f"... {Path(xml).parent.name} on {device} {prec or 'default'}", flush=True)
                rows.append(bench_one(ov.Core, xml, device, prec, args.iters, args.warmup, image, args.throughput))

    cols = ["model", "device", "precision", "exec_precision", "compile_cold_ms", "compile_warm_ms",
            "infer_p50_ms", "infer_p95_ms", "e2e_p50_ms", "fps", "score", "status"]
    cols = [c for c in cols if any(c in r for r in rows)]
    print("\n" + tabulate([[r.get(c, "") for c in cols] for r in rows], headers=cols, tablefmt="simple"))

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.LOG_DIR / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"openvino": ov.__version__, "devices": devs, "rows": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
