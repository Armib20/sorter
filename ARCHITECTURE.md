# Track 3 architecture

What the station really is (probed read-only on 2026-11-13 ~11:10 through the SDK and app API), what Viam gives us, and the architecture we're building. Team split and timeline: `PLAN.md`.

## 1. The station, as measured

| Thing | Fact | Consequence |
|---|---|---|
| Arm | **`viam:ufactory:xArm850`** (not xArm6), 6 joints, `use_urdfs`, 60 deg/s, `collision_sensitivity` 0. From fragment `SF-xarm850-realsense-gripper` | Arm card / `motion.move` work as for any arm. Collision detection off on the controller: our obstacles matter |
| Gripper | `viam:ufactory:gripper`, frame = arm + 150 mm, rotated 180°. `DoCommand {"get":true}` → `pos` 0–850 (857 = open). **`IsHoldingSomething` works** (meta `position`, `status`) | Grasp verification for recovery is free. `{"set": N}` opens only as wide as needed (faster than full open) |
| `cam` | `viam:camera:realsense` D435, 640×480 (fragment default 1280×720, overridden by a mod), color + aligned depth, `supports_pcd`, intrinsics fx 605 fy 605 cx 338 cy 247. **Frame: child of `arm`**, translation (67.5, −23.1, 17.5), rotated 90° about z | Pixel + depth → world XYZ through the frame system works today. The extrinsic is a nominal fragment value, not a per-station calibration: verify it (§5) |
| Depth | At ~17 cm from the table: 67% valid pixels, bolt top read 12 mm above the table (163 vs 175 mm) | Depth sees fasteners but sits at the D435 minimum range. Survey from ≥ 25 cm |
| Scene | White, uniform table. Wrist view shows soft shadows | Background segmentation by color is easy and robust |
| `overview-cam` | Webcam on a stand, 640×480, side view of the cell; items are tiny in it | Watching / demo video, not inspection (unless we move the stand) |
| Obstacles | `table`, `wall-front`, `wall-side`, `ceiling` exist in fragment `armfarm-workcell-obstacles` but are **disabled by our fragment mods** | `motion.move` currently has nothing stopping it planning through the table. Fix before any planned motion (§5) |
| Motion | `builtin` motion service present | Collision-aware moves once obstacles are on |
| Shell | Fragment `ViamShellDanger` adds the `shell` service | We can run code on the station (`viam machine part shell` / `cp`) |
| Services not yet configured | No `ov-diag`, no mlmodel, no `anomaly` vision service, no VLM | Person C's first hour |
| Home | `home-pose` joints (rad) [0, 0.394, −0.885, 0, −1.292, 0]; `leader-home` saved. Switch index 2 = go, 1 = save current | `home.py` already drives them |
| Latency from the laptop | connect 4 s; simple RPCs 85–230 ms; `cam.get_images` with depth **1.1–1.3 s**; color only 15–85 ms (measured last session) | The judged run should not go laptop → cloud → station for every call (§4) |
| Org data | Key lists the hackathons datasets (`fastener-anomaly` `6af5edb82deac4224da1b4cd`, `fastener-frames` `6aea99cd94463677c522ad27`), but **cannot read their images** | Someone logged in to Viam: `viam dataset export --destination=datasets/fastener-anomaly --dataset-id=6af5edb82deac4224da1b4cd` |

## 2. What Viam gives us for Track 3

| Piece | What it does | Our use |
|---|---|---|
| `viam:viam-openvino` 0.0.6 → **`viam:openvino:mlmodel`** | Runs OpenVINO IR / ONNX on `CPU`/`GPU`/`NPU`, explicit device, `inference_precision`, `cache_dir`, `num_requests`, on-device `preprocess`. Explicit device that's missing = hard failure (good) | `ov-anomaly`: PatchCore on **GPU, f32** |
| → **`viam:openvino:diagnostics`** (`ov-diag`) | `list_devices`, `get_compiled_properties` (proves the execution device + cache hit), `benchmark` (mean/p50/p90/p99 per device), `stats` (rolling live latency) | **The device-placement score (10%)**: one benchmark table per model × device, plus live `stats` during the run |
| `viam-labs:anomalib` 0.1.5 → `viam-labs:anomalib:vision` | `GetClassifications` (`anomalous` conf = score), `GetDetections` (boxes over threshold), `DoCommand heatmap` (score, label, overlay JPEG), `CaptureAllFromCamera`. Default threshold 0.2 | Quick baseline and heatmaps. Our loop calls the mlmodel or the vision service with **our own crop** (`GetDetections(image)`), not the raw camera frame |
| `hackathons:anomalib-train` 0.4.0 | Cloud training (~10 min): `model=patchcore\|padim\|stfpm\|efficient_ad\|fastflow`, `image_size`; ≥10 `normal`, optional ≥4 `anomalous`; exports `model.xml` `[1,3,S,S]` in 0–1 → `pred_score [1]`, `anomaly_map [1,1,S,S]` | The registry path for models. Same export contract as our `train_detector.py`, so local and cloud models are interchangeable |
| Reference package `fastener-patchcore` v1 | PatchCore on 169 good + 83 marked fasteners from one station's wrist cam, 256 px | Day-1 baseline before our own data exists. It learned another station's pose and lighting: expect a shifted threshold |
| Measured | PatchCore 256: **GPU f32 23–31 ms, CPU 75–77 ms, NPU 158 ms** (or no compile: "Custom Operator format"). GPU f16 → empty `anomaly_map` | Placement decided: anomaly model on GPU f32 |
| `viam:openvino-genai` → `vlm` | Qwen2.5-VL etc. on GPU, `DoCommand ask` with `image_b64`, returns answer + seconds | Name the defect / explain, **off the critical path** |
| `viam:obstacles-pointcloud` 0.2.1 (`obstacles-depth`) | Plane removal + clustering → `GetDetections3D` / object point clouds | Optional 3D locator (§3, plan B) |
| `viam:detections-to-segments` 1.4.0 | 2D detector boxes + depth → 3D segments | Only if we add a 2D detector |
| `viam-labs:grasp-pose` 0.1.0 | From `GetDetections3D`: ranked top-down parallel-jaw grasps (world pose, width) + fitted object boxes as obstacles | Optional grasp planner; we compute the same thing from our 2D blob (§3) |
| `erh:vmodutils` | `obstacle`, `obstacle-open-box` (box/bin as obstacles), `arm-position-saver`, `multi-arm-position-switch`, `pc-crop-camera` | Box, reject bin and hold area as `obstacle-open-box` so placing is collision-aware |
| `viam:camera-calibration` | ChArUco `eye_in_hand` hand-eye calibration | Only if the touch test (§5) shows more than ~3 mm error |

## 3. The architecture

**Idea: the arm parks at a survey pose and every decision comes from one wrist-camera frame, scored on a canonical crop of the object. Accelerators run only the models; everything geometric is cheap CPU code. Recovery comes from always re-looking before acting.**

```
                SURVEY pose (arm parked, cam ≥ 25 cm above the intake zone, white table)
                   │ color + depth frame                                   [CPU]
                   ▼
 1 LOCATE      background segmentation (not-white, shadow-tolerant) → blobs
               blob → px box, long-axis yaw (minAreaRect), median depth
               → world XYZ via intrinsics + frame system              [CPU, ~5 ms]
                   │ per object: item #, t_seen
                   ▼
 2 CANONICALIZE crop around blob, rotate long axis horizontal, pad to the model's
               input size (256×256), same code for training and runtime  [CPU, ~2 ms]
                   ▼
 3 SCORE       ov-anomaly PatchCore, GPU f32  ──┐  (optional, parallel) second model
               pred_score + anomaly_map         │   PaDiM/EfficientAD on NPU
                   ▼                            ┘   [GPU ~25 ms]
 4 DECIDE      good / known defect / novel or uncertain        [CPU, ~1 ms]
               known-defect check = tape-blue hue fraction inside the anomaly region
               → box | reject | hold, confidence      → judge log line, t_decided
                   ▼
 5 ACT         motion.move to pre-grasp above the object (t_first_motion)
               re-check it's still there → descend → grab → IsHoldingSomething?
               → place in box / reject / hold (open-box obstacles) → back to SURVEY
                   ▼ (after motion has started; never delays it)
 6 EXPLAIN     VLM on GPU names the defect for the dashboard and the pitch
```

**The decision layer.** The anomaly model can catch defects it has never seen; a known-defect check names the ones it has:

| Anomaly score | Tape check | Destination |
|---|---|---|
| clearly below threshold | — | **box** |
| clearly above | tape-blue present | **reject** (known defect) |
| clearly above | no tape | **hold** (novel defect for a person), or reject if staff say hold means something else |
| near threshold | — | take a second, closer view; still unsure → **hold** |

Confidence = distance from the threshold, scaled by the spread of good-item scores on the validation set. Calibrate on the deploy device (f32 GPU).

**Why canonical crops.** PatchCore compares patches with memory from training. Training on whole frames makes it learn background, position and orientation. Training on object-centered, orientation-normalized crops at a fixed scale (fixed survey height) means a few dozen good examples per fastener type are enough, and the score reacts to the part, not to where it lies. With crops exactly at input size, the exported graph's resize becomes a no-op, so laptop, `ov-anomaly` and the cloud-trained model give the same scores.

**Why the 2D locate is primary.** Fasteners are 5–15 mm tall, near the depth noise at survey distance. On a white table, color segmentation finds every part, gives the grasp yaw directly, and produces the crop for scoring in the same step. Depth is only for Z sanity and the table height. `obstacles-depth` → `grasp-pose` is plan B for B to try in under an hour.

**Recovery (judge moves an item).**
- Positions are never trusted from the first look: re-survey before each grasp (we're already at the survey pose, ~30 ms).
- After grabbing, `IsHoldingSomething`. Empty → back to survey, re-locate, retry (max 2), then log and move on.
- Item identity across looks: nearest blob with matching size/aspect, so the log keeps the same item number.
- The motion service with `stop_on_error`-style handling of our own: any failed move → stop, home, re-survey.

**Multiple views.** Tape can be on the hidden side. The first view decides when confident. The uncertain band triggers a second view: a lower, closer pose, or an oblique pose from the side. Optional: while carrying the part, show it to `overview-cam` for a bottom view. That only helps if the staff confirm the logged decision can be the final one after a second look.

## 4. Where code runs

- **Models: on the station, through Viam services.** That is what `ov-diag` can measure and what the judges expect.
- **Orchestrator (`sorter.py`): develop on the laptop, run the judged run on the station.** From the laptop each RPC costs 85–230 ms and an RGBD frame 1.1–1.3 s, which lands straight in the latency score. On the station these are local calls. The shell service is already configured; `setup_nuc.sh` builds the torch-free runtime there.
- **Stretch:** wrap the loop as a Viam generic service module (start/stop/status DoCommands), so it runs under viam-server with no SSH session.

Latency budget on the station (item seen → arm moves): frame ~30 ms + locate 5 + crop 2 + score 25–30 + decide 1 + motion start ~50–100 ≈ **0.15–0.2 s**. From the laptop: +0.5–1.5 s.

## 5. Must-do safety and calibration before planned motion

1. **Turn the obstacles back on** with a correct table height: remove the `disabled` mods, or set `table-translation` so the box top equals the measured table surface. Add box / reject / hold as `obstacle-open-box`. Until then, only joint moves to taught poses (`home.py`, arm card).
2. **Measure the table height** in the arm frame: jog the gripper tips to touch, read `get_end_position`, minus the 150 mm tool offset already in the frame.
3. **Touch test the camera extrinsic:** locate a part from the survey pose, move the tips 20 mm above the computed XY, measure the error. Correct a constant offset if needed; ChArUco `eye_in_hand` if it varies across the zone.
4. One driver at a time. E-stop within reach.

## 6. The log the judges score

One line per item, written the moment the decision is made, then updated when the motion starts:

```json
{"item": 3, "decision": "reject", "confidence": 0.93, "timestamp": "2026-11-14T15:41:07.412-07:00",
 "t_seen": "...", "t_decided": "...", "t_first_motion": "...", "latency_ms": 184,
 "score": 0.81, "tape_frac": 0.12, "device": "GPU", "model": "fastener-crops-v3", "reason": "anomalous + tape"}
```

Timestamps in ms, with time zone. Keep the rich per-item run folder (`logs/<run>/`) for the dashboard and the pitch.

## 7. Diagnostics for the 10% and the 20%

- `ov-diag benchmark` for each model on CPU / GPU / NPU (iterations 200) → table + `get_compiled_properties` (execution device, cache hit). The reason for each placement is the measured number.
- `ov-diag stats` during the judged run → live p50/p90 on the dashboard.
- Per-item stage spans (already in `diagnostics.py`) → the latency waterfall slide.
- What failed and what we changed: keep a running `NOTES.md` (one line per surprise) as it happens; it's a fifth of the score.

## 8. Build order (maps to PLAN.md people)

1. **C:** `ov-diag` → `list_devices`; `ov-anomaly` with `fastener-patchcore` on GPU f32; `anomaly` vision service; benchmark CPU/GPU/NPU.
2. **B:** obstacles on; table height; survey pose 25–35 cm above the intake zone, plus box / reject / hold poses; first manual grasps of each fastener type.
3. **A:** export the `fastener-anomaly` dataset; locate + canonicalize on those frames and live frames; capture good crops at the survey pose; train (local 1 min, cloud 10 min); threshold on f32.
4. **D:** judge log + ms timestamps + hold; wire LOCATE → SCORE (`ov-anomaly`) → DECIDE in `sorter.py` with motion dry-run; then real motion from B.
5. **All:** fixed-position loop → re-survey and recovery → run on the station → rehearse five-minute runs with a teammate playing judge.

## Open questions for staff (also in PLAN.md)

1. What is the hold area for (uncertain? novel defect?) — decides the table in §3.
2. How items arrive and what "enters view" means; is a pre-pick inspection move part of the "response"?
3. Is the log read from the station; is JSON lines fine; which timestamp do they use for latency?
4. Is running our script on the station through the shell service allowed?
5. Should the obstacles be enabled (and is the table height in the fragment right)?
