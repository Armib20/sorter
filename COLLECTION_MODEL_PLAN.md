# Collection and model iteration MVP

Status (2026-11-13 evening): proposed scope, pending team review. Nothing here is implemented yet: no code has changed since commit `c4d18f9`, and the smoke-test model is still the default. Collection, logging and model runs currently happen on a laptop and are moving to the Intel station.

## Decisions and boundaries

Confirmed by user:
- Focus first on visible gross defects, not subtle internal-thread defects.
- Capture item by item while model building proceeds asynchronously.
- Measure anomaly quality and justify CPU/GPU/NPU placement.

Recommended, pending review:
- Capture in verified category batches; retain category labels on every specimen.
- Start with one pooled PatchCore model across supported categories. Mixed-category inference does not initially require a category classifier. Benchmark each category independently; compare specialist models only when evidence warrants them.
- Known supported catalog, isolated parts, repeatable inspection pose/lighting, explicitly supported surfaces. No promise of arbitrary objects or invisible defects.
- Training creates candidates, never silently replaces the running model.
- Human review and explicit activation between runs for MVP; no hot swapping during an item or autonomous promotion.

## Evidence from existing infrastructure

- `capture_tool.py`: camera capture and good/defect folder layout, manual stepping, multiple views. All views from one capture share a split, but no durable specimen identity across captures. Non-good labels mean defect; category and condition must be separate. Frames are already cropped when settings.CROP is set. Capture filenames/import filenames are not sufficient immutable asset identifiers.
- `train_detector.py`: PatchCore/PaDiM/EfficientAD fitting, validation, testing, OpenVINO export, metadata and training-image quality statistics. Default candidate names can collide; results root is shared. `--from-ckpt` skips fitting, so it is re-export, not continued training. No scheduler, frozen manifest, explicit specimen-grouped splits, or atomic model publication.
- Installed `anomalib==2.7.0`, `models/image/patchcore/lightning_model.py:230-266`: no optimizer; collect features then coreset subsampling. Initial implementation rebuilds from the accumulated approved normal snapshot, not just the latest batch. True incremental feature caching is deferred until measured rebuild cost warrants it.
- `detector.py`: loads an OpenVINO model once and exposes score/map/timing. Metadata loaded beside model; threshold currently lives separately in settings. Preserve matching resize/preprocessing and deploy-device verification.
- `evaluate.py`: confusion matrix, precision/recall, subtype breakdown, stage timing; AUROC and threshold sweep under `--suggest-threshold`. Threshold selection currently maximizes accuracy on the supplied rows and does not enforce split or model identity. Must not tune thresholds on final evaluation data or pool scores across model versions. Existing metrics reflect actions, not an independent detector-only benchmark.
- `benchmark.py`: device/precision compile timing, sustained inference timing and optional throughput. Useful execution benchmark, not dataset-quality benchmarking.
- `dashboard.py`: read-only observer of run folders, score distributions, errors, warnings and timing. Preserve its independence from capture/control. Add collection/job/model summaries, not an unprotected remote control plane.
- `sorter.py`: one detector for a run, per-view scoring and configured max/mean aggregation. Preserve offline/fake-machine operation. Benchmark aggregation must match runtime; do not count each view as an independent item.
- Current default in `settings.py` is the bottle smoke-test model, not an evidenced fastener model. Local example-model metadata is not evidence of real fastener performance.

## Compute and connection evidence

Sources: the event PDFs `[Edge of Intelligence] Viam Overview.pdf` and `Edge of Intelligence - reference sheet.pdf` (not in the repo; ask Ali for a copy).

The PDFs identify the Intel edge box as an ASUS NUC or AAEON station with Arc iGPU, accessed through the Viam machine in hackathons / Arm Farm. They do not supply SSH credentials or an address for a separate Intel training host.

The Track 3 reference sheet documents cloud training through `hackathons:anomalib-train` version `0.4.0`, container `9accd6e7-2992-47c9-9662-2ca5298fa1f1`, in organization `6d6c7293-bc67-43e4-9c8b-31c76aac27d3`. Submission uses `viam train submit custom from-registry`, unique model versions, and a dataset ID. At least 10 normal images are required; optional anomalous images require at least 4, otherwise evaluation uses synthetic anomalies. Typical job time is about 10 minutes, not an SLA or proof of data sufficiency. Registry output is OpenVINO model.xml/model.bin.

`ARCHITECTURE.md` records shell service availability on armfarm04b via Viam machine-part shell/copy, but this has not been rechecked. Local `viam` executable is absent. First integration task is read-only authentication/permission/resource discovery, not installation or machine configuration changes performed implicitly.

Cloud script split semantics and control over explicit grouped calibration splits are unverified. Before relying on its reported scores, inspect exact script source and job behavior. Do not upload final audit data to training jobs. If calibration split control is unavailable, use a separately controlled local pipeline as the benchmark authority; cloud training metrics are provisional only.

## End-to-end flow

1. Operator selects catalog category, specimen ID, verified condition, defect description if applicable, and capture recipe.
2. Capture required views; store originals and derived crops with unique IDs. Review quality/labels. Unknown or rejected captures cannot enter normal training automatically.
3. Publish completed capture records atomically. A training snapshot contains only approved completed records, fixed label revisions, recipe versions and asset hashes.
4. Queue a build after an explicit batch completion initially. Later enable a debounced threshold of new approved specimens/time. Initial cloud eligibility is at least the documented 10 normals; empirical sufficiency is a separate question.
5. One training job runs at a time by default. Coalesce new arrivals into the next snapshot instead of spawning one job per frame. Recording never waits for training. Persist job ID, state and errors so restarting the coordinator does not duplicate a paid/remote submission.
6. Fit using accumulated approved normal training specimens; export to a unique candidate directory/package. Record source revision, dependency versions, training settings, seed, dataset snapshot, timing and artifact hashes.
7. Calibrate candidate threshold on a separate real calibration set on the target device/precision. Validate required outputs, map shape, finite scores and preprocessing compatibility. Candidate remains provisional if no real calibration evidence exists.
8. Compare to active model on a fixed development benchmark. Show regressions and sample counts by category/defect/surface/session.
9. Explicitly approve activation between runs. Model, threshold, preprocessing/view recipe and device/precision are one versioned configuration. Keep previous complete version for rollback. A failed build/export/evaluation never invalidates the active model.

Initial persistence: filesystem assets plus a small local SQLite catalog/job ledger (standard library), not a distributed queue or new web stack. Database transactions publish completed capture records after atomic file writes. Snapshots materialize a compatible folder view for the existing trainer. New module boundaries are justified by dataset consistency and durable job lifecycle, not generic framework abstractions.

## Data contract and split discipline

Record at minimum:
- category ID and catalog revision; specimen ID; condition and label provenance/reviewer;
- capture/session/view IDs; surface ID; camera/pose/lighting recipe revision;
- original and crop paths/hashes, crop transform, source resolution, timestamp;
- image quality results and review status; dataset role and revision.

Dimensions are measured catalog data, not inferred names from a VLM. Do not put visible labels inside the model crop.

Use four roles:
1. Train: verified-good specimens only.
2. Calibration: separate real good/defective specimens; operating threshold selection.
3. Development benchmark: repeated candidate comparisons; acknowledge this is development data.
4. Final audit: separate specimens and preferably a later session, used after choices are frozen.

Keep every capture/view of a physical specimen in the same role, including before/after deliberately introducing a defect. Reserve specimens before collecting. Do not force arbitrary ratios when stock is small: report actual independent counts and missing coverage. Keep at least one defect family out of calibration/development to test novelty if available. Synthetic anomalies are diagnostics, not proof of real-defect recall.

## Benchmark contract

### Detection quality (no arm movement required)

Score the identical frozen benchmark with each candidate and the active model, using deployed preprocessing, required views and aggregation. Store per-item scores, maps, predictions, labels, timings, model version and threshold provenance.

Primary measures:
- Defect recall / false-accept rate; good-part false-reject rate.
- Precision, confusion matrix, AUROC and average precision where both classes exist.
- Counts and metrics per category, defect family, surface and session, plus macro summaries so large categories cannot hide failures.
- Reacquire/hold rate (at this event, hold = the reject bin: count "uncertain → reject" separately from confident rejects), invalid-image/error rate, and coverage. Do not exclude failures from operational totals or disguise all-hold behavior as high detection recall.
- Confidence intervals based on independent specimens; repeated views do not increase the effective sample size.

Report detector predictions separately from final destination/hold decisions and physical placement success. A hold is not evidence that the defect detector correctly identified a defect. Model anomaly scores/threshold distance are not calibrated defect probabilities.

Threshold objective must be explicit. Proposed priority: minimize missed gross defects subject to an agreed good-part rejection budget. Do not select by overall accuracy alone. A tiny six-item rehearsal is useful operational evidence, not a statistically strong reliability estimate. Numeric acceptance targets remain pending inventory and user agreement.

### Device placement and pipeline cost

- Explicit CPU/GPU/NPU trials where supported: cold/warm compile, sustained p50/p95 inference, preprocessing, memory, output validity and prediction agreement. Report unsupported configurations honestly.
- Reference sheet documents PatchCore GPU f32 ~23 ms and CPU ~75 ms on armfarm04, and NPU compilation failure. GPU f16 reportedly produces empty maps. Reproduce on the actual station/model; do not treat these reference timings as current measurements.
- Start with CPU for capture/geometric processing/orchestration, GPU f32 for PatchCore, cloud for training.
- Test a compact NPU-compatible alternative (e.g. PaDiM/EfficientAD candidate) only after baseline. Compilation and end-to-end quality must pass; using every device is not itself a benefit. A part classifier is another possible NPU workload only if the pooled model proves inadequate.
- Hardware shares memory/power budgets. Measure capture + inference idle versus concurrent upload, candidate compilation/evaluation, and any local training. Serialize device microbenchmarks so contention is not confused with device performance.
- Measure time from capture approval to evaluated candidate: queue + upload + fit + export + download + compile + evaluation. Cloud fitting alone is not iteration latency. Compare one controlled local build if cloud turnaround is too slow.
- For judging, reproduce inference and diagnostics through the Viam service path (`ov-diag`), not solely the standalone Python Detector. Exact service integration remains a separate bounded task.
- Record item-seen, first response motion, decision, and completed placement separately during supervised robot rehearsals.

## Implementation slices and owners

1. **Verified collection and snapshots**
   - Extend `capture_tool.py`; add a cohesive dataset/catalog module for specimen identity, revisions, atomic publication and grouped splits.
   - Preserve existing legacy import/layout through explicit conversion; never invent verified labels for legacy images.
   - Deliverable: offline item-by-item capture produces auditable frozen snapshots while more captures continue.
2. **Quality benchmark**
   - Extend `evaluate.py` and add an offline candidate scoring entrypoint using `Detector`.
   - Keep `benchmark.py` responsible for execution/device costs.
   - Deliverable: identical inputs, reproducible comparisons, calibration-only thresholds, per-category failures.
3. **Asynchronous model jobs**
   - Add one durable job coordinator with concrete cloud submission/poll/download integration; reuse `train_detector.py` for controlled local fallback and isolated output directories.
   - Deliverable: capture overlaps training; restart/retry is safe; incomplete artifacts never appear ready.
4. **Dashboard and activation**
   - Extend read-only dashboard with catalog coverage, pending reviews, snapshot versions, queue state, candidate-vs-active results and time-to-candidate.
   - Explicit CLI activation first; pin configuration at run startup and retain rollback.
5. **Station verification and supervised rehearsal**
   - Verify real compute access/services and device matrix. Prove the close-up capture recipe for selected gross defects. Run independent audit and then controlled robot rehearsal with one operator/E-stop.

Slices 2 and 3 can proceed in parallel after slice 1 freezes the snapshot contract. Offline plumbing need not wait for camera access; real optics and performance claims must wait for observations.

## Verification plan

- Existing offline baseline: `.venv/bin/python tests/run_all.py` (not run during this planning pass).
- Add offline tests: same specimen cannot cross splits; unknown/non-good cannot enter normal training; partial write ignored; capture during snapshot does not change snapshot contents; corrected labels create a new snapshot; duplicate imports/jobs are detected.
- Job tests through a fake submission boundary: completion/failure/restart/cancellation, submission reconciliation, pending snapshots coalesced, no duplicate remote jobs, active model survives all candidate failures.
- Evaluation tests on hand-constructed predictions: threshold fitted only on calibration, per-category counts, view grouping, single-class undefined metrics, error/hold denominators, no cross-version score pooling.
- Artifact tests: unique exports, complete bundle required, recipe mismatch rejected, rollback exact, calibration tied to device/precision and candidate identity.
- Real local smoke: small frozen image snapshot -> fit -> export -> target-runtime score -> reproducible report. No hardware movement.
- Station evidence: actual device/output checks and Viam-service equivalence, then concurrent-load timings. Physical quality/robot success cannot be proven by mocks.

## Still needed from the team

- Verified category inventory, available independent good and defective specimen counts, and supported surfaces.
- Human owner for category/condition verification and physical staging.
- Numeric miss/reject/hold/latency goals after first pilot; no unsupported accuracy guarantees.
- Current Viam access/permissions for cloud training, model packages and station diagnostics. Do not paste credentials into this document or chat.
- Confirmation of inspection views and safety-cleared poses through the existing robot operator; this plan is not authorization to move the arm or alter station configuration.
