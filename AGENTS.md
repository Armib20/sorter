# AGENTS.md

Read this first. It's for teammates and their coding agents. Track 3 ("Spot it, sort it"): a Viam xArm850 with a wrist
RealSense finds fasteners, scores them with an OpenVINO anomaly model, and sorts them into the box or the reject bin.

## Docs, and which one wins

| doc | what it is | status |
|---|---|---|
| `ARCHITECTURE.md` | Measured station facts, team decisions, target design (survey → inspect → score → grasp) | **Current design. Wins on conflicts** |
| `PLAN.md` | Scoring, who owns what (A perception, B motion, C on-device, D integration), timeline, staff questions | Current |
| `COLLECTION_MODEL_PLAN.md` | Proposed data collection + model iteration pipeline (specimen IDs, splits, jobs, benchmark) | Proposal, pending team review, not implemented |
| `README.md` | Per-file reference, diagnostics guide, gotchas, night-before results on MVTec | Reference; its run order predates the Track 3 decisions |

## State (2026-11-13 evening)

- **Decided:** two destinations (box, reject; the reject bin is the hold area); the inspection move is the response that
  stops the latency clock; the judged run executes on the station; anomaly model on GPU f32 through Viam's
  `ov-anomaly` / `ov-diag` services.
- **Code** (last change `c4d18f9`) is the generic local Python loop: `sorter.py` with a fixed pick spot and placeholder
  `POSES`. The default model is the MVTec **bottle** smoke-test model, not a fastener model. Survey/locate/inspect, the
  judge log with ms timestamps, and the Viam OpenVINO service path are **not built yet** (see `ARCHITECTURE.md` §8).
- **Tests:** `python tests/run_all.py` → 7/7 pass (fake Viam machine, no hardware).
- **Station:** last `doctor.py` from the laptop against `armfarm04b` (10:44): 28 ok, 6 expected warnings.
- **Where things run:** collection, logging and model runs currently happen on a teammate's M2 laptop (CPU only). They
  are **moving to the Intel station** (Arc GPU + NPU; reached through Viam's shell service). Write new code so it runs
  there: paths relative to the repo root, the torch-free runtime (`requirements-runtime.txt`), device chosen in `settings.py`.

## Setup and checks

```
./setup_nuc.sh            # venv + runtime deps, prints OpenVINO devices, runs ./fetch_models.sh
./setup_nuc.sh --train    # + CPU torch/anomalib for training
./fetch_models.sh --vlm   # optional VLMs (0.8 GB; --vlm-4b for 3.1 GB)
python tests/run_all.py   # ~10 s, offline
python doctor.py          # read-only preflight; needs api-key.json for the robot section
python dashboard.py       # http://localhost:8765, live view of logs/<run>/
```

`api-key.json` is a secret: get it privately from a teammate and never commit it or paste it into chat.

## What's in git and what isn't

- In git: code, docs, `logs/` (run history, shared on purpose), `models/*/metadata.json`.
- Not in git (`.gitignore`): `models/` weights, `datasets/`, `results/`, `openvino_cache/`, `.venv/`, `config_backups/`,
  `api-key.json`, `.pi/`. How to get each one: README "Large files".
- **Our own captures (`datasets/custom/`) have no shared home yet.** Don't assume a teammate has your data.

## Rules for agents

- **Never move the arm or change station/Viam config unless the human you're working with says so for that action.**
  One driver at a time, with someone at the E-stop. The Viam MCP server and agents can move the arm: treat that like a
  person driving it. `doctor.py`, `sorter.py resources` and `--dry-run` are the safe defaults.
- Planned motion (`motion.move`) is unsafe until the table obstacle is re-enabled (`ARCHITECTURE.md` §5).
- Config changes go through person C, with a backup first (`config_backups/`). `settings.py` is owned by D: keep every
  constant there; env vars override.
- Don't tune thresholds on evaluation data, and calibrate on the deploy device (f16 vs f32 shifts scores).
- When a decision or measurement changes, update `ARCHITECTURE.md` (and `PLAN.md` if it affects owners/timeline) in the
  same commit, and refresh "State" above.
