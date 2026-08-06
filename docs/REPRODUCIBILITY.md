# Reproducibility Guide

## Environment

Use the project virtual environment and run from the repository root:

```bash
.venv/bin/python --version
make test
make clean-caches
```

The local environment snapshot is recorded in
`requirements-freeze.txt`. The project currently relies on the machine-local
M4 Max environment; a containerized or cross-platform lockfile is future work.

## Data

Raw public datasets are deliberately not committed. Each acquired source keeps
its archive, response headers, page or documentation response, local SHA-256,
member inventory, schema audit, and split decision under local `data/real/` and
`artifacts/evaluations/`. The repository retains small manifests and fixtures
where they help explain an adapter.

To reproduce a source-gated application, follow its versioned plan in the local
evaluation artifacts, acquire only the fixed official URL, and run the matching
source-gate script. Mirrors, cleaned reconstructions, derived feature tables,
and post-hoc schema repair are outside the protocol.

## Evaluation Order

The order is part of the result:

1. source provenance and schema;
2. adapter mechanics on injected records;
3. artificial recovery and negative controls;
4. train-only observed baseline and selection controls;
5. one native external operation;
6. adaptive fixed-seed stability accounting.

Observed external outcomes never tune a model or change a threshold. A failed
gate closes that application branch and is retained as a negative result.

## LM Studio Hygiene

Before an eligible generation screen:

```bash
make list-lmstudio-models
```

The native LM Studio status API must report zero loaded instances before launch.
Exactly one generation model may be loaded during a resumable job. The screen
must write a run ID, model ID, prompt/configuration, seed, candidate
fingerprints, finite/bound metrics, timeout status, and final unload evidence.

## Local Backup

Before the public cleanup on 2026-08-06, a complete hard-linked working-tree
snapshot was created at
`../_backups/automated_science_20260806_pre_cleanup`. It includes the local
data and untracked research state. The snapshot is outside the repository and
will remain a recovery point for the project cleanup. The former umbrella
tree and unrelated sibling projects are preserved separately at
`../_backups/workspace_20260806_pre_primary_repo_cleanup` and
`../_backups/workspace_non_primary_local_20260806`.

Run commands from the repository root. `make test` executes the package tests
under `tests/` and compiles the active `core/`, `scripts/`, and `tests/`
trees.
