# Artifacts

Preserved generated material from previous experiment runs. These files are evidence and provenance, not canonical source code. The active architecture lives in `../ARCHITECTURE.md`.

- `runs/qwen36_27b/`: primary 25-domain, 10-run Qwen 3.6 27B experiment output. The project root has a `scheduler_logs` symlink pointing here for compatibility with existing scripts.
- `runs/legacy/`: empty holding directory; older backup and mid-run scheduler trees were compressed into `archives/scheduler_logs_legacy_20260701.tar.gz`.
- `logs/qwen36_27b/`: compressed root-level per-domain run logs moved out of the source root.
- `logs/batch_execution_circuit.out`: top-level batch execution transcript.
- `evaluations/hero_run_deep_dive_20260703.md`: May-June qwen36 hero-run analysis, including achievements and limitations.
- `evaluations/qwen3_coder_next_5epoch_vs_baselines_20260702.md`: held-out qwen-coder-next gauntlet comparison against refreshed baselines.
- `outputs/hall_of_fame_20260611/`: historical hall-of-fame CSV outputs.
- `models/models_v3_qwen36_mid_run/`: preserved generated model snapshot.
- `legacy/nested_automated_science_snapshot/`: stale nested duplicate loader snapshot.
- `archives/`: preserved zip/export artifacts.

This directory is intentionally ignored by Git because it contains generated evidence and local archives, not source code.
