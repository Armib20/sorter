# Track 3 plan: Spot it, sort it

Team of 4. Sources: the event slides, the reference sheet (version 2026-11-13 08:28, the authority where they differ), this repo, last night's notes in `README.md`.

## How we're scored

| Weight | What | What it means for us |
|---|---|---|
| **30%** | Right decision per item: detection precision and recall, **taken from our own log** | One log line per item: number, decision, confidence, timestamp. **No log = zero on this and on latency.** |
| **30%** | Item ends where we decided to send it | Reliable grasp and place of small fasteners |
| **10%** | Latency: item enters view → arm starts its response | Counts correct decisions only. Model warm, arm parked |
| **10%** | Device placement with a measured reason, shown through diagnostics | `ov-diag` service + per-device numbers |
| **20%** | Five-minute design explanation | What it does, why each choice, what failed, what we'd change |

- **The run:** one five-minute run on 6 judge items: good, defective, and **at least one defect type we've never seen**.
- **Three destinations:** box, reject bin, **hold area**.
- **Recovery** (slides): keeps working when a judge moves an item mid-run.

## What we have

- **Station `armfarm04b`** (probed read-only, details in `ARCHITECTURE.md`): **xArm850** (not xArm6) with a two-finger claw `gripper` (`IsHoldingSomething` works; `vacuum-gripper` exists but is disabled); `cam` = RealSense D435 on the wrist, 640×480 color + aligned depth, already placed in the frame system; `overview-cam` = webcam on the stand. A shell service is configured, so code can run on the station. No OpenVINO services yet.
- **Obstacles are off:** the table/walls/ceiling are disabled by our fragment mods, so planned motion can go through the table until we fix it.
- **Reference data:** `fastener-anomaly` dataset (`6af5edb82deac4224da1b4cd`, 169 normal + 83 anomalous). The machine key can't read it; export with `viam dataset export` from a logged-in account.
- **Items:** fasteners; the known defect is blue tape. From the photo analysis:
  - Good parts are also bluish (zinc plating, hue ≈ 200°); tape is ≈ 230° and matte. Only hue separates them.
  - Tape can be tiny or on the hidden side, so multiple views matter.
  - Lock the camera's white balance and exposure if possible.
- **This repo** (detector, decider, robot, sorter loop, dashboard, doctor, benchmark) is tested offline and read-only against this machine (doctor: 0 failures). Gaps:
  1. Inference runs in our own Python process; judges expect it through Viam's OpenVINO services on the station, measured with `ov-diag`.
  2. Poses are placeholders, the model is the MVTec bottle practice model, there is no hold action, and the pick spot is fixed (no localization, so no recovery).
  3. Log timestamps are to the second. Latency needs ms, plus separate "item seen" and "arm started" times.
- **Teleop isn't needed for Track 3.** It only helps jog the arm to save poses; the arm card in the Viam app does that too.
  - The machine log shows the SO-101 leader reporting a servo **voltage** error and **communication timeouts**: likely power/cable, ask staff.
  - The droop is known: in `leader_hold: manual` the leader goes limp and must be held the whole time teleop runs.

## Who does what

| Person | Owns | First hour | By dinner (6 pm) |
|---|---|---|---|
| **A: Perception and model** | Data, training, threshold, unseen defects | Use the shared `fastener-patchcore` model on `cam` to see raw scores on our parts. Capture good fasteners: every type, several orientations, at the exact inspection pose | Own model trained (Viam training job ~10 min, and locally with `train_detector.py` ~1 min). Threshold set on a held-out set. Tested on self-made defects (marker, bent, dirty) as a stand-in for the unseen type |
| **B: Motion and grasping** | Poses, pick/place, finding items, recovery | Save poses: inspect, box, reject, hold, park. Test-grasp each fastener type with the claw | A dependable fixed-position pick → inspect → place loop, then finding items with depth (`obstacles-pointcloud` → `grasp-pose`). Retry when the grasp is empty or the item moved |
| **C: On-device and diagnostics** | Viam services, device placement, latency | Add `ov-diag`, run `list_devices`. Add `ov-anomaly` mlmodel (GPU, f32) + `anomaly` vision service on `cam`. Read the anomalib README for what the service returns | Per-device numbers (anomaly GPU 23 ms vs CPU 75 ms on armfarm04). A job for the NPU (PaDiM/EfficientAD or an item locator). VLM benchmark on GPU. Pre-warm checklist |
| **D: Integration, decisions, story** | `sorter.py` loop, decision policy, log, dashboard, pitch | Ask staff the questions below. Add the hold action, required log fields with ms timestamps, and a "first arm motion" timestamp | Loop calls C's services and B's motion. Policy: confident good → box, confident defect → reject, uncertain / novel → hold (after a second view). Dry-run rehearsal. Draft design explanation |

**Interfaces**, so people work in parallel:
- **A → C:** model package (registry name + version) and a threshold.
- **C → D:** service names and the returned fields (score, boxes, ms).
- **B → D:** `pick()`, `place(dest)`, `locate()`, which fail with clear errors.
- **D** owns `settings.py`. Until a piece is ready, D's loop uses dry-run or the fake machine in `tests/`.

## Timeline

- **Lunch (1 pm):** first untrained end-to-end path: shared model scoring our cam, poses saved, open questions answered.
- **Dinner (6 pm):** own model, fixed-position loop running for real on 6 items, diagnostics numbers.
- **9 pm (hacking ends):** localization and recovery, hold area, latency work. ≥5 full five-minute rehearsals with a teammate playing judge (moves items, adds a new defect).
- **Day 2 morning:** hardening and rehearsals. **Code freeze ≈ 2 pm.** Design explanation, dashboard snapshot, per-device numbers for the 3:30 judging.

## Questions for Viam staff

**Answered:** inspection move counts as the response; reject bin = hold area (two destinations: box, reject); our script may run on the station. Picking is classical perception + motion planning, no RL or demonstrations (see `ARCHITECTURE.md` §3b).

Original list:

1. **What is the hold area for?** Uncertain items? Defects needing a person? Drives the decision policy (30%).
2. **How are items presented?** One at a time or all at once, and where? What counts as "entering view" for latency?
3. **Is the log read from our station, and in what format?** Is JSON lines fine?
4. **Can we run our script on the station computer (SSH), or must it connect remotely?** Remote adds network delay to latency.
5. **Should the table/wall/ceiling obstacles be on?** Our config disables them through a fragment override, so the motion planner doesn't avoid the table.

## Rules while sharing the machine

- **One person or agent moves the arm at a time.** Say "I have the arm"; someone at the E-stop.
- **Config changes go through C**, with a backup first.
- **The Viam MCP server and AI agents can move the arm.** Treat them like a person driving it.

## Biggest risks

1. Claws grasping small round fasteners reliably (30%).
2. The unseen defect type: the model learns from good parts only; the VLM only explains or sends borderline items to hold.
3. Camera pose or lighting drifting from training.
4. Time lost on teleop.
