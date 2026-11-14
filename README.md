# Spot it, sort it — Track 3 starter

Anomalib anomaly model → OpenVINO → decide → Viam arm sorts good items into the
box and defective ones into the reject bin. Everything is driven by `settings.py`.

**Start here:** `AGENTS.md` has the current state, the doc map and the rules for the shared arm (for people and coding
agents). This README was written the night before the event (M2 laptop, MVTec practice data). Where it disagrees
with `ARCHITECTURE.md` (the Track 3 design and team decisions), `ARCHITECTURE.md` wins.

Fresh clone: `./setup_nuc.sh` (venv + deps, then `fetch_models.sh`), `python tests/run_all.py`, `python doctor.py`.

| file | what it does |
|---|---|
| `settings.py` | every constant: component names, poses, model path, device, threshold, views, actions, VLM. Env vars override (`DEVICE=NPU python sorter.py ...`) |
| `train_detector.py` | train PatchCore / PaDiM / EfficientAD on MVTec or your own `<name>/train/good`, `<name>/test/{good,<defect>}` photos, test, export static-shape OpenVINO IR + `metadata.json`. `--from-ckpt` re-exports without retraining |
| `detector.py` | `Detector`: loads the IR once on CPU/GPU/NPU, warms up, `score(img) -> (score, anomaly_map, ms)`. `--verify` checks it against Anomalib's torch model |
| `decider.py` | `decide(image, score) -> (action, confidence)`: `RuleDecider` (threshold), `VLMDecider` (OpenVINO GenAI), `HybridDecider` (VLM only when the score is uncertain) |
| `robot.py` | async Viam wrapper: connect via `api-key.json`, `pick(pose)`, `place(name)`, `capture(views)`, `dry_run`, offline frames |
| `sorter.py` | the loop + verbs `resources`, `capture`, `detect`, `sort`, `offline`; one JSON line per item with per-stage ms |
| `capture_tool.py` | camera frames → dataset layout (auto train/test split for good, defects to test) |
| `evaluate.py` | logs → confusion matrix, per-defect-type accuracy, median/p95 per stage, threshold suggestion |
| `benchmark.py` | every model × every OpenVINO device × precision: compile cold/warm, latency, e2e, FPS |
| `diagnostics.py` | span tracing, frame-quality drift checks vs training images, anomaly-region stats, overlays, `RunRecorder` (one folder per run, all writes off the hot path) |
| `dashboard.py` | live web dashboard over the run folders (stdlib, no internet); `--snapshot RUN` writes a self-contained `report.html` |
| `doctor.py` | read-only preflight: env, disk, settings, OpenVINO devices, model compile+sample score, VLM, robot components/camera/frame quality/arm pose |
| `setup_nuc.sh` | NUC setup: torch-free runtime (default) or `--train` with CPU torch; ends by running `fetch_models.sh` |
| `fetch_models.sh` | downloads the gitignored models: smoke-test PatchCore from the GitHub release (checksum verified), `--vlm` / `--vlm-4b` from Hugging Face |
| `tests/` | fake Viam machine (real SDK client over gRPC), sorter/dashboard/doctor tests; `python tests/run_all.py` (~10 s) |

## Verified tonight (M2 MacBook, CPU only)

| check | result |
|---|---|
| PatchCore wide_resnet50, coreset 0.01, MVTec bottle | AUROC 1.0; offline loop **83/83 correct**; good ≤ 0.425 < 0.542 ≤ defect |
| coreset 0.1 vs 0.01 | fit 20 min vs **7 min**; infer 835 vs **227 ms** (f32); same AUROC → default 0.01 |
| OpenVINO vs Anomalib torch (f32) | mean \|Δscore\| 0.0015, AUROC identical (PatchCore, PaDiM, EfficientAD all checked) |
| Folder dataset via capture_tool (48 goods, resnet18) | 34 s fit, AUROC 1.0, 12 MB model, 81/83 offline |
| good-only data (synthetic anomalies for threshold) | 79/83 on real defects without ever seeing one |
| CPU f16 latency p50 | PatchCore-r18 73 ms · PaDiM 82 ms · PatchCore-wrn50 153 ms · EfficientAD 249 ms (conv-heavy: expect it to shine on GPU/NPU) |
| robot.py through the real Viam SDK client (fake machine) | pick / capture (depth frame skipped) / place / joint moves ✔ |
| torch-free runtime venv | sorter offline + fake machine run with `requirements-runtime.txt` only |
| VLM yes/no, 4 good + 4 defect | InternVL2-1B: answers follow the prompt, not the image (useless as a detector), 3–4 s · Qwen3-VL-4B: 5/8 (catches gross breaks, misses contamination), ~14 s CPU |

## Diagnostics: seeing exactly what happens

```
python doctor.py                      # preflight; fix every ✘ before touching the arm
python dashboard.py                   # http://localhost:8765, follows the newest run live
python dashboard.py --host 0.0.0.0    # on the NUC, open http://<nuc-ip>:8765 from the laptop (or ssh -L 8765:localhost:8765 nuc)
python dashboard.py --snapshot logs/<run>   # report.html for judges / teammates (works offline, from file://)
```

Every `detect` / `sort` / `offline` run writes `logs/<verb>_<time>/`:
`run.json` (settings, model + training metrics, devices, versions, git commit, command),
`items.jsonl` (one line per item), `events.jsonl` (warnings/errors with tracebacks),
`status.json` (current item + stage, read by the dashboard's live pill), and per item
`items/NNNN_vK_{frame,overlay,thumb}.jpg` + `_map.png`.

Per item you get:
- **Timeline**: nested spans, e.g. `pick > gripper.open, move pick+approach, move pick, gripper.grab, move ...`,
  `capture > camera (sources, rpc ms, decode ms, size)`, `detect`, `decide`, `place > move reject ...`. Errors mark the failing span.
- **Detector breakdown** per view: preprocess / inference / diagnostics ms, score per view, aggregate.
- **Anomaly region**: peak value + location, % area over threshold, bounding box (drawn on the overlay).
- **Frame quality vs training images** (brightness, contrast, sharpness, clipped pixels, as z-scores). Warns with a
  cause when lighting changes, the image blurs (raise `SETTLE_S`), or glare appears, so a sudden false reject comes with
  its explanation. The baseline is stored in `metadata.json` by `train_detector.py`
  (`python detector.py --calibrate-stats` for older models).
- Decision, confidence, reason (rule or VLM text), ground truth + correct/wrong when known.

Dashboard: live stage pill ("item 12 · move → reject · 1.3 s", flags a dead process), KPIs (accuracy, recalls,
p50/p95, items/min, uncertain, warnings), item viewer (overlay / frame / map, per-view tabs, score gauge with
threshold + uncertain band), score-per-item and score-distribution charts (good/defect gap), stacked latency per
item, stage p50/p95 table, confusion matrix + per-defect-type accuracy, gallery with filters (wrong / uncertain /
per action / warnings / errors), events log. Click any point or thumbnail; ←/→ step, `f` follow, `l` layer;
`#item=12` deep links.

**Overhead, measured** (offline, alternating runs with cool-down): detect p50 170/174/169 ms with images vs
171/173/172 ms without. Tracing is two `perf_counter` calls per span, frame/map stats ~1-2 ms (reported separately
as `diag`), and image encoding runs on a background-priority thread that only starts after 30 ms with no inference.
During arm motion that is immediate; in back-to-back offline loops images are written at the end (a few seconds).
`--no-artifacts` turns images off entirely.

Found with it tonight: on the M2 the first ~10 inferences of a process run at ~85 ms, then ~180 ms (boost ends),
whatever the input. Benchmark sustained latency (`benchmark.py` now reports first vs last quarter), not bursts.

## Day-1 run order (written the night before)

Still valid for the local Python path. The Track 3 flow (survey → inspect move → score → grasp, models served by
Viam's OpenVINO services) is in `ARCHITECTURE.md` §3–4.

0. **`python doctor.py`** on the NUC (then again after every config change), **`python dashboard.py --host 0.0.0.0`** in a second terminal.
1. **Kickoff (9:30)**: note the defect scenario, items, the actions they expect ("decide what to do"), and what the provided pipeline already does (component names, poses, how items arrive).
2. **NUC**: `git clone` → `./setup_nuc.sh` (or `--train`) → it prints which of CPU/GPU/NPU exist and fetches the smoke-test model.
3. **Connect**: write `api-key.json`, `python sorter.py resources` → fix `ARM_NAME/GRIPPER_NAME/CAMERA_NAME/CAMERA_SOURCE` in settings. Jog the arm in the Viam app to pick / inspect / box / reject and paste the printed pose (or `joints`) into `POSES`.
4. **Camera check**: `python sorter.py capture -n 3` → look at `/tmp/frames`. Set `CROP` to the item region (big accuracy win: less background). After training, `python sorter.py detect -n 5` shows live scores + drift warnings in the dashboard.
5. **Data**: `python capture_tool.py parts --label good -n 60 --step`, plus a few of each defect: `--label scratch -n 8 --step`. Same pose/lighting as the live loop.
6. **Train** (1 min with resnet18): `python train_detector.py --data datasets/custom/parts --backbone resnet18` → wide_resnet50 later if accuracy needs it.
7. **Check**: `python sorter.py offline --images datasets/custom/parts/test` → look at wrong/uncertain items in the dashboard → `python evaluate.py logs/<run> --suggest-threshold` → set `THRESHOLD`.
8. **Devices**: `python benchmark.py --models models/*/model.xml --throughput` → pick the device split (below), set `DEVICE`, re-run step 7 on that device (f16 shifts scores).
9. **Arm**: `python sorter.py sort --dry-run --step` (camera live, arm logged) → `python sorter.py sort --step -n 3` → `python sorter.py sort`.
10. **Judging prep**: `python evaluate.py logs/sort_* --group device --markdown` → numbers for the design explanation; `python dashboard.py --snapshot logs/<best run>` → report to show; keep the dashboard on a screen during the judged run.

## Device split (the scored part)

- **Anomaly model → GPU, f32** (decided, `ARCHITECTURE.md` §2). Reference numbers for PatchCore 256 on the station:
  GPU f32 23–31 ms, CPU 75–77 ms, NPU 158 ms or no compile ("Custom Operator format"); GPU f16 returns an empty
  `anomaly_map`. Reproduce these with `ov-diag` on our station. The NPU is the candidate for a second model
  (PaDiM / EfficientAD, pure conv/matmul), and only if it compiles and holds its quality.
- **VLM → GPU** (`VLM_DEVICE`), only for uncertain items (`--decider hybrid`) or to name the defect/response for rejected items. Never in the critical path for confident scores.
- **CPU** → Viam client, image decode/crop/resize, decision rule, logging.
- Story for the judges: `ov-diag` benchmark table per model × device as the reason for each placement; per-item latency table per stage from `evaluate.py`; the response (inspect move) starts before scoring, so motion dominates total time.

## Gotchas found tonight (all handled in code)

- Anomalib 2.7 needs `onnxscript` (`anomalib[openvino]`); the "Failed to convert the model to the target version 14" traceback at export is harmless.
- Default dynamo export has a symbolic batch (example batch 2, anomalib #3883) → we export fully static `[1,3,H,W]` (`dynamic_shapes=None`), which is also what the NPU wants.
- The exported graph's Resize has **antialiasing disabled** while training used antialias → never feed raw camera frames to the IR; `Detector.preprocess` resizes first (matches training).
- f16 (GPU, NPU, ARM CPU default) shifts normalized scores by up to ~0.1 vs f32 → calibrate `THRESHOLD` on the deploy device.
- **OpenVINO's CPU model cache ignores `INFERENCE_PRECISION_HINT`** → a shared cache silently reuses the wrong precision; we use one cache dir per device+precision.
- Exported `pred_score` is min-max normalized: 0.5 == the threshold Anomalib fitted on the validation split.
- MVTec's official download link is dead (404); the bottle set came from the HF mirror `TheoM55/mvtec_anomaly_detection`.
- EfficientAD needs imagenette (1.5 GB) + teacher weights: already in `datasets/imagenette` and the anomalib weights dir. Train it with batch 1, many epochs (3 epochs → AUROC 0.94).
- VLMs tile large images (InternVL: up to 12 tiles) → downscale to 448 first: 25 s → 3 s.
- Small VLMs (1B) can't see subtle defects; even 4B misses contamination. The anomaly model is the detector.
  In `--decider hybrid` with InternVL2-1B the VLM flipped 2 correct uncertain *good* calls to reject: only let a
  VLM override the score after `evaluate.py` shows it helps; otherwise use it to name the defect / choose the response.
- Linux `pip install torch` pulls CUDA; `setup_nuc.sh --train` uses CPU wheels. Runtime on the NUC needs no torch at all.

## Large files (not in git)

`.gitignore` keeps these out of the repo. How to get each one:

| path | how to get it |
|---|---|
| `models/patchcore_bottle_c001/` (107 MB) | `./fetch_models.sh` (GitHub release `model-patchcore_bottle_c001`, checksum verified). Smoke-test model, not a fastener model |
| `models/vlm-internvl2-1b/` (0.8 GB), `models/vlm-qwen3-4b/` (3.1 GB) | `./fetch_models.sh --vlm` / `--vlm-4b` (Hugging Face `OpenVINO/*-int4-ov`) |
| `datasets/MVTecAD/bottle` | HF mirror `TheoM55/mvtec_anomaly_detection` (official link is dead). Only to retrain the smoke-test model. CC BY-NC-SA 4.0 |
| `datasets/imagenette` + EfficientAD teacher weights | Only for training EfficientAD; anomalib downloads them |
| `datasets/custom/` (our captures) | `capture_tool.py`. **Not backed up anywhere yet:** decide on a shared home (Viam dataset or shared drive) before collection moves to the station |
| `results/` | Training outputs from `train_detector.py`; regenerate them |
| `openvino_cache/`, `.venv/`, `config_backups/` | Machine-specific; never copy |
| `api-key.json` | Secret; get it from a teammate privately |

Our own trained models: `scp -r models/<name> nuc:~/sorter/models/` or publish through the Viam registry (`ARCHITECTURE.md` §2).
