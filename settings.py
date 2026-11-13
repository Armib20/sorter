"""Every constant the pipeline uses, in one place. Edit this file tomorrow.

Anything here can also be overridden with an environment variable of the same
name (e.g. `DEVICE=NPU python sorter.py sort`), handy for quick A/B runs.

Poses: each pose is either
    Cartesian, in mm and orientation-vector degrees, relative to the arm base
    (or to POSE_FRAME when USE_MOTION_SERVICE is True):
        {"x": 300, "y": 0, "z": 200, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
    or joint angles in degrees (what you get from the Viam app's arm card,
    or from `python sorter.py resources --poses`):
        {"joints": [0, -45, 90, 0, 45, 0]}
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Viam connection -------------------------------------------------------
API_KEY_FILE = ROOT / "api-key.json"   # {"address": ..., "api_key": ..., "api_key_id": ...}  (gitignored)
ROBOT_ADDRESS = ""                     # overrides the address in api-key.json when set

# --- Component names (must match the machine config in the Viam app) -------
ARM_NAME = "arm"
GRIPPER_NAME = "gripper"
CAMERA_NAME = "camera"
CAMERA_SOURCE = "color"   # NamedImage source to use from get_images(); "" = first decodable image
MOTION_NAME = "builtin"   # motion service; only used when USE_MOTION_SERVICE
USE_MOTION_SERVICE = False  # False: arm.move_to_position (simple); True: motion.move (collision-aware)
POSE_FRAME = "world"

# --- Poses ------------------------------------------------------------------
APPROACH_MM = 80          # hover this far above pick/place poses before descending (Cartesian only)
POSES = {
    "home":      {"x": 250, "y": 0,    "z": 300, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},
    "pick":      {"x": 350, "y": 0,    "z": 60,  "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},  # where items arrive
    "inspect":   {"x": 300, "y": 150,  "z": 200, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},  # in front of camera
    "box":       {"x": 200, "y": -250, "z": 150, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},  # good items
    "reject":    {"x": 200, "y": 250,  "z": 150, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0},  # defective items
}
# Extra viewpoints for multi-view inspection: offsets applied to POSES["inspect"]
# (dtheta rotates the wrist). The first view is the inspect pose itself.
VIEW_OFFSETS = [
    {},
    {"theta": 90},
    {"theta": 180},
    {"theta": 270},
]
NUM_VIEWS = 1             # how many of VIEW_OFFSETS to capture per item
SETTLE_S = 0.3            # wait after a move before grabbing a frame (motion blur / exposure)

# --- Detector ---------------------------------------------------------------
MODEL_PATH = ROOT / "models" / "patchcore_bottle_c001" / "model.xml"
DEVICE = "CPU"            # CPU | GPU | NPU | AUTO  (see benchmark.py)
PERFORMANCE_HINT = "LATENCY"
# "" = device default (GPU/NPU and ARM CPUs run f16; Intel CPUs f32). "f32" reproduces
# Anomalib's torch scores exactly on CPU/GPU; f16 shifts scores by ~0.01-0.07, so
# calibrate THRESHOLD on the device you deploy on (evaluate.py --suggest-threshold).
INFERENCE_PRECISION = ""
CACHE_DIR = ROOT / "openvino_cache"   # compiled-model cache: makes GPU/NPU startup fast after the first run
CROP = None               # optional (x0, y0, x1, y1) region of interest in camera pixels; applied to live
                          # frames in Robot.get_image, so capture_tool datasets are saved already cropped

# --- Decision ---------------------------------------------------------------
THRESHOLD = 0.5           # on the normalized score; 0.5 == threshold Anomalib fitted at training time
UNCERTAIN_MARGIN = 0.1    # |score - THRESHOLD| below this -> low confidence (Decider may ask for more views)
VIEW_AGGREGATE = "max"    # how per-view scores combine: max (strict) | mean
ACTIONS = ["box", "reject"]           # every action the Decider may return; each must be a key in POSES
ACTION_FOR = {"good": "box", "defect": "reject"}
DECIDER = "rule"          # rule | vlm | hybrid (VLM only for uncertain scores)
VLM_PATH = ROOT / "models" / "vlm-internvl2-1b"   # OpenVINO GenAI VLM folder
VLM_DEVICE = "GPU"        # keep the VLM off the device running the anomaly model
VLM_MODE = "yesno"        # yesno (robust with small VLMs) | choose (pick from ACTIONS; use 4B+)
VLM_DEFECT_HINT = ""      # e.g. "cracks, chips, missing cap, dirt" once the scenario is known

# --- Data / logs ------------------------------------------------------------
DATASET_DIR = ROOT / "datasets" / "custom"   # capture_tool.py writes <DATASET_DIR>/<name>/{train,test}/...
LOG_DIR = ROOT / "logs"
TEST_IMAGES = ROOT / "datasets" / "MVTecAD" / "bottle" / "test"


def _override(name, value):
    raw = os.environ.get(name)
    if raw is None:
        return value
    if isinstance(value, Path):
        return Path(raw)
    if isinstance(value, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(value, (int, float)):
        return type(value)(raw)
    if isinstance(value, (dict, list, tuple)) or value is None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


for _name, _value in list(globals().items()):
    if _name.isupper():
        globals()[_name] = _override(_name, _value)


def check():
    """Fail fast on config mistakes, before the arm moves."""
    missing = [a for a in ACTIONS if a not in POSES]
    assert not missing, f"ACTIONS without a pose in POSES: {missing}"
    assert set(ACTION_FOR.values()) <= set(ACTIONS), "ACTION_FOR maps to an unknown action"
    assert 1 <= NUM_VIEWS <= len(VIEW_OFFSETS), "NUM_VIEWS must be between 1 and len(VIEW_OFFSETS)"
    assert VIEW_AGGREGATE in ("max", "mean")


if __name__ == "__main__":
    check()
    for k, v in sorted(globals().items()):
        if k.isupper():
            print(f"{k} = {v!r}")
