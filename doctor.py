"""Preflight check: is everything ready to sort? Run it first on the NUC and whenever
something looks off. Read-only: it never moves the arm.

    python doctor.py              # everything (robot checks when api-key.json / --address exist)
    python doctor.py --no-robot   # skip the Viam machine
    python doctor.py --vlm        # also load the VLM and ask it one question (slow)

Checks: environment and disk, settings consistency, OpenVINO devices, the model
(compiles on settings.DEVICE and scores a sample image), decider/VLM, and the robot
(connection, every configured component exists, camera sources and latency, frame
quality vs the training images, arm pose). Saves logs/doctor_<time>/ with a report.json
and the camera frame + overlay it scored. Exit code 1 if anything failed.
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import settings

ICON = {"ok": "\033[32m✔\033[0m", "warn": "\033[33m⚠\033[0m", "fail": "\033[31m✘\033[0m", "skip": "\033[90m·\033[0m",
        "info": "\033[36mi\033[0m"}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail="", **data):
        self.rows.append({"section": section, "check": name, "status": status, "detail": detail, **data})
        print(f"  {ICON[status]} {name:28} {detail}")

    def section(self, title):
        print(f"\n\033[1m{title}\033[0m")

    def guard(self, section, name):
        """Context manager turning an exception into a 'fail' row instead of a crash."""
        report = self

        class _G:
            def __enter__(self):
                return self

            def __exit__(self, et, e, tb):
                if e is not None and not isinstance(e, (KeyboardInterrupt, SystemExit)):
                    report.add(section, name, "fail", f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                    return True
        return _G()

    def counts(self):
        out = {}
        for r in self.rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out


# ---------------------------------------------------------------- sections
def check_env(rep: Report):
    from importlib import metadata

    rep.section("Environment")
    py = sys.version_info
    rep.add("env", "python", "ok" if (3, 10) <= py[:2] <= (3, 13) else "warn", sys.version.split()[0])
    for pkg, need in [("openvino", True), ("viam-sdk", True), ("numpy", True), ("pillow", True),
                      ("openvino-genai", settings.DECIDER != "rule"), ("anomalib", False), ("torch", False)]:
        try:
            rep.add("env", pkg, "ok", metadata.version(pkg))
        except metadata.PackageNotFoundError:
            rep.add("env", pkg, "fail" if need else "info", "not installed" + ("" if need else " (only needed to train)"))
    free = shutil.disk_usage(settings.ROOT).free / 1e9
    rep.add("env", "disk free", "ok" if free > 5 else "warn" if free > 1 else "fail", f"{free:.1f} GB")
    from diagnostics import _git
    g = _git()
    if g:
        rep.add("env", "git", "warn" if g["dirty"] else "ok", f"{g['commit']}{' (uncommitted changes)' if g['dirty'] else ''}")


def check_settings(rep: Report):
    rep.section("Settings")
    with rep.guard("settings", "settings.check()"):
        settings.check()
        rep.add("settings", "settings.check()", "ok", f"actions {settings.ACTIONS}, views {settings.NUM_VIEWS}")
    missing = [p for p in ("pick", "inspect") if p not in settings.POSES]
    rep.add("settings", "required poses", "fail" if missing else "ok",
            f"missing {missing}" if missing else ", ".join(sorted(settings.POSES)))
    defaults = settings.POSES.get("pick", {}) == {"x": 350, "y": 0, "z": 60, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
    if defaults:
        rep.add("settings", "poses taught", "warn", "POSES look like the placeholders: teach real ones (sorter.py resources)")
    if settings.CROP:
        x0, y0, x1, y1 = settings.CROP
        rep.add("settings", "CROP", "ok" if x1 > x0 and y1 > y0 else "fail", str(settings.CROP))
    rep.add("settings", "threshold", "info", f"{settings.THRESHOLD} ± {settings.UNCERTAIN_MARGIN} uncertain, "
            f"decider {settings.DECIDER}, aggregate {settings.VIEW_AGGREGATE}")


def check_openvino(rep: Report):
    import openvino as ov

    rep.section("OpenVINO devices")
    core = ov.Core()
    devs = core.available_devices
    for d in devs:
        rep.add("openvino", d, "ok", core.get_property(d, "FULL_DEVICE_NAME"))
    for want in ("GPU", "NPU"):
        if not any(d.startswith(want) for d in devs):
            rep.add("openvino", want, "warn", "not available (driver missing, or not this machine)")
    want = settings.DEVICE.split(":")[0].split(".")[0]
    if want not in ("AUTO", "HETERO", "MULTI") and not any(d.startswith(want) for d in devs):
        rep.add("openvino", "settings.DEVICE", "fail", f"{settings.DEVICE} not in {devs}")
    return devs


def check_model(rep: Report):
    rep.section("Detector")
    mp = Path(settings.MODEL_PATH)
    if not mp.exists() or not mp.with_suffix(".bin").exists():
        rep.add("model", "model files", "fail", f"{mp} (+ .bin) not found: train_detector.py, or copy models/<name>/")
        return None
    meta_f = mp.with_name("metadata.json")
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    rep.add("model", "model files", "ok", f"{mp.parent.name}: {meta.get('model', '?')} {meta.get('image_size', '')} "
            f"{(mp.with_suffix('.bin').stat().st_size / 1e6):.0f} MB")
    if not meta:
        rep.add("model", "metadata.json", "warn", "missing: input size guessed, no drift reference")
    m = meta.get("metrics") or {}
    if m:
        rep.add("model", "test metrics", "ok" if m.get("image_AUROC", 0) >= 0.95 else "warn",
                ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    if not meta.get("image_stats"):
        rep.add("model", "training-image stats", "warn",
                f"none: drift warnings off (python detector.py --calibrate-stats --model-path {mp})")
    else:
        rep.add("model", "training-image stats", "ok", f"n={meta['image_stats'].get('n')} from {meta['image_stats'].get('source', '?')}")

    from detector import Detector, test_images
    det = None
    with rep.guard("model", f"compile on {settings.DEVICE}"):
        det = Detector()
        rep.add("model", f"compile on {settings.DEVICE}", "ok",
                f"{det.compile_ms:.0f} ms, input {det.input_hw}, precision {det.precision or 'device default'}")
    if det is None:
        return None
    samples = test_images(settings.TEST_IMAGES) if Path(settings.TEST_IMAGES).exists() else []
    if samples:
        good = [p for p, y in samples if y == 0][:1]
        bad = [p for p, y in samples if y == 1][:1]
        for p in good + bad:
            a = det.analyze(p)
            expect_defect = p in bad
            right = (a.score >= settings.THRESHOLD) == expect_defect
            rep.add("model", f"sample {p.parent.name}", "ok" if right else "warn",
                    f"score {a.score:.3f} in {a.ms:.0f} ms (pre {a.preprocess_ms:.0f} + infer {a.infer_ms:.0f})"
                    + ("" if right else f" -> wrong side of threshold {settings.THRESHOLD}"))
    else:
        rep.add("model", "sample images", "skip", f"{settings.TEST_IMAGES} not found")
    return det


def check_decider(rep: Report, run_vlm=False):
    rep.section("Decider")
    if settings.DECIDER == "rule":
        rep.add("decider", "rule", "ok", f"threshold {settings.THRESHOLD}")
        return
    try:
        import openvino_genai  # noqa: F401
    except ImportError:
        rep.add("decider", "openvino-genai", "fail", "not installed (pip install openvino-genai==<openvino version>)")
        return
    vp = Path(settings.VLM_PATH)
    rep.add("decider", "VLM files", "ok" if (vp / "openvino_language_model.xml").exists() else "fail", str(vp))
    if run_vlm:
        with rep.guard("decider", "VLM query"):
            from decider import VLMDecider
            t0 = time.perf_counter()
            vlm = VLMDecider()
            load = time.perf_counter() - t0
            img = next(iter(sorted(Path(settings.TEST_IMAGES).rglob("*.png"))), None)
            t0 = time.perf_counter()
            action, conf = vlm.decide(img, 0.5)
            rep.add("decider", "VLM query", "ok", f"load {load:.1f}s, answer {action} in {time.perf_counter() - t0:.1f}s "
                    f"on {settings.VLM_DEVICE}: {vlm.last_reason[:60]}")


async def check_robot(rep: Report, robot, det, out_dir: Path):
    """Robot checks on an attached/connected Robot (read-only)."""
    from diagnostics import Trace, render_overlay

    rep.section("Robot")
    names = sorted(f"{r.subtype}:{r.name}" for r in robot.machine.resource_names)
    rep.add("robot", "resources", "info", ", ".join(names)[:160])
    for label, comp, name in [("arm", robot.arm, settings.ARM_NAME), ("gripper", robot.gripper, settings.GRIPPER_NAME),
                              ("camera", robot.camera, settings.CAMERA_NAME)]:
        rep.add("robot", f"{label} '{name}'", "ok" if comp is not None else "fail",
                "found" if comp is not None else f"not on machine: fix settings.{label.upper()}_NAME")
    if settings.USE_MOTION_SERVICE:
        rep.add("robot", f"motion '{settings.MOTION_NAME}'", "ok" if robot.motion is not None else "fail", "")

    if robot.arm is not None:
        with rep.guard("robot", "arm pose"):
            d = await robot.describe()
            rep.add("robot", "arm pose", "ok", json.dumps(d.get("pose")))
            rep.add("robot", "arm joints", "ok", json.dumps(d.get("joints", {}).get("joints")))
            moving = await robot.arm.is_moving()
            rep.add("robot", "arm idle", "ok" if not moving else "warn", "moving!" if moving else "not moving")

    if robot.camera is not None:
        with rep.guard("robot", "camera"):
            lat, frame, attrs = [], None, {}
            for _ in range(3):
                trace = Trace()
                robot.trace = trace
                frame = await robot.get_image()
                cam = next(s for s in trace.spans if s["name"] == "camera")
                lat.append(cam["dur_ms"])
                attrs = cam.get("attrs", {})
            robot.trace = None
            rep.add("robot", "camera sources", "ok", ", ".join(attrs.get("sources", [])))
            if settings.CAMERA_SOURCE and attrs.get("source_fallback"):
                rep.add("robot", "CAMERA_SOURCE", "warn", f"'{settings.CAMERA_SOURCE}' not a source; using '{attrs.get('used')}'")
            rep.add("robot", "camera frame", "ok", f"{attrs.get('used')} {attrs.get('size')} -> {list(frame.size)} after crop, "
                    f"{attrs.get('bytes', 0) / 1e3:.0f} kB, {sum(lat) / len(lat):.0f} ms avg (rpc {attrs.get('rpc_ms')}, decode {attrs.get('decode_ms')})")
            w, h = frame.size
            if det is not None and abs(w / h - det.input_hw[1] / det.input_hw[0]) > 0.15:
                rep.add("robot", "aspect ratio", "warn", f"frame {w}x{h} is squashed to {det.input_hw}: fine if the "
                        "training photos came from this camera too, else set CROP to a square region")
            frame.save(out_dir / "camera.jpg", quality=90)
            if det is not None:
                a = det.analyze(frame)
                render_overlay(frame, a.anomaly_map, {"score": a.score, "map": a.map,
                                                      "title": f"doctor {a.score:.3f}"}).save(out_dir / "camera_overlay.jpg")
                rep.add("robot", "live frame score", "info", f"{a.score:.3f} in {a.ms:.0f} ms (whatever is in view now)")
                if a.warnings:
                    for wmsg in a.warnings:
                        rep.add("robot", "frame vs training", "warn", wmsg)
                elif det.ref_stats:
                    rep.add("robot", "frame vs training", "ok", "brightness/contrast/sharpness/glare match the training images")


async def check_robot_connect(rep: Report, det, out_dir, address=None):
    from robot import Robot, load_credentials

    creds = load_credentials(None, address)
    rep.section("Robot")
    if not creds.get("address"):
        rep.add("robot", "api-key.json", "skip", f"no address ({settings.API_KEY_FILE} or --address): offline only")
        return
    for k in ("api_key", "api_key_id"):
        if not creds.get(k):
            rep.add("robot", "api-key.json", "fail", f"missing {k}")
            return
    robot = Robot(dry_run=True, address=address)
    t0 = time.perf_counter()
    try:
        await robot.connect(timeout=20)
    except Exception as e:  # noqa: BLE001
        rep.add("robot", "connect", "fail", f"{creds['address']}: {type(e).__name__}: {e} "
                "(machine online in the Viam app? same network? key for this machine?)")
        return
    rep.add("robot", "connect", "ok", f"{creds['address']} in {(time.perf_counter() - t0) * 1000:.0f} ms")
    try:
        await check_robot(rep, robot, det, out_dir)
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--address", default=None)
    ap.add_argument("--vlm", action="store_true", help="load the VLM and run one query")
    args = ap.parse_args()

    out_dir = settings.LOG_DIR / f"doctor_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    check_env(rep)
    check_settings(rep)
    with rep.guard("openvino", "openvino"):
        check_openvino(rep)
    det = None
    with rep.guard("model", "detector"):
        det = check_model(rep)
    with rep.guard("decider", "decider"):
        check_decider(rep, args.vlm)
    if not args.no_robot:
        asyncio.run(check_robot_connect(rep, det, out_dir, args.address))

    c = rep.counts()
    (out_dir / "report.json").write_text(json.dumps(rep.rows, indent=2, default=str))
    print(f"\n{c.get('ok', 0)} ok · {c.get('warn', 0)} warnings · {c.get('fail', 0)} failed   -> {out_dir}")
    sys.exit(1 if c.get("fail") else 0)


if __name__ == "__main__":
    main()
