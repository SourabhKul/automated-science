# Automated Science

Automated Science is a research framework for white-box scientific model
discovery. It combines simulation-based inference, executable candidate models,
diagnostics, held-out evaluation, and model-generating agents in one auditable
loop.

The central idea is simple:

```text
fixed data -> candidate model -> parameter inference -> diagnostics
           -> structural proposal -> held-out validation -> provenance
```

The system treats an LLM as a hypothesis generator, never as the scientific
arbiter. Candidates must compile, remain finite, respect declared bounds, and
improve a frozen evaluation protocol before they can replace an incumbent.

## Project Status

This repository is the primary home for the Automated Science project. The
former workspace umbrella and unrelated sibling projects were removed from
the public tree; their local copies remain in the pre-cleanup backup outside
the repository.

The project contains two related research surfaces:

1. The canonical white-box discovery engine in `core/` and `scripts/`.
2. A source-gated real-data evaluation stream with raw adapters, duplicate
   accounting, artificial recovery controls, and narrow file/subject holdout
   benchmarks.

The main historical proof-of-life is the preserved Qwen 3.6 multi-domain
run. The newer evaluation stream is stricter: several candidate sources are
closed when their provenance, schema, duplicate, or negative-control contract
fails. See [docs/RESULTS.md](docs/RESULTS.md) for a compact evidence summary.

The project name is intentionally still a working title. Repository naming is
separate from the scientific cleanup and will be revisited after the current
architecture is easier to understand.

## Quick Start

The checked-in code is designed for the local Python environment used for this
project. The virtual environment itself is kept outside the repository.

```bash
.venv/bin/python -m compileall -q core scripts tests
make test
make inventory
make architecture
```

For a dry-run of the canonical model-replacement loop:

```bash
make run-domain-dry DOMAIN=ecology FAMILY=qwen3_coder_next
```

For a small local LM Studio smoke test, first ensure the server is reachable
at `http://127.0.0.1:1234/v1` and that no unrelated model is resident:

```bash
make list-lmstudio-models
make smoke-qwen-coder
```

Long-running research jobs are always resumable and record their configuration,
seed, model, source, metrics, and terminal status under local `artifacts/`.

## Repository Map

- `core/`: model interfaces, inference, validation, diagnostics, and real-data
  adapter contracts.
- `scripts/run_domain.py`: canonical single-domain model-replacement runner.
- `scripts/run_gauntlet.py`: one model family across multiple domains.
- `scripts/run_model_matrix.py`: multi-model orchestration with LM Studio
  residency and preflight hygiene.
- `scripts/repo_inventory.py`: canonical-versus-legacy inventory report.
- `data/real/`: manifests, split definitions, and small fixtures. Downloaded
  archives and raw provenance remain local and are ignored by Git.
- `docs/`: project guide, reproducibility notes, and evidence summary.
- `artifacts/`: local research logs, reports, run histories, and preserved
  evaluation evidence. It is intentionally excluded from commits except for
  its README.
- `paper/`: manuscript source and generated tables.
- historical orchestrators and generated traces are preserved in the local
  cleanup backup, not mixed into the public source tree. New work should use
  the config-driven runners.

## Scientific Guardrails

Every new application follows the same progression:

1. Freeze official source provenance and the raw schema.
2. Define the candidate surface, external boundary, and evaluator-owned fields.
3. Run injected-record mechanics and output-isolated synthetic/negative
   controls.
4. Fit only the predeclared observed baseline on train data.
5. Open a native external repeat once, only after every selection gate passes.

Non-finite metrics, violated bounds, inconsistent duplicate labels, failed
negative controls, leakage sentinels, or unstable trajectories close a branch.
They are recorded as results, not repaired away. LLM discovery and ABC-SMC are
eligible only for a source-backed mechanistic plan that explicitly permits
them. Before an LLM screen, LM Studio must be empty and exactly one generation
model may be resident.

## What The Evidence Supports

The repository supports a research prototype and narrow retrospective
benchmarks. It does not yet support claims of autonomous scientific discovery
in the broad sense, deployment readiness, causal inference, clinical or safety
decisions, or generalization beyond each frozen source boundary. See
[CLAIMS.md](CLAIMS.md) and [docs/RESULTS.md](docs/RESULTS.md) before using any
metric in a paper, demo, or public announcement.

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md): canonical discovery loop and interfaces.
- [CLAIMS.md](CLAIMS.md): evidence-backed claim audit.
- [docs/RESULTS.md](docs/RESULTS.md): curated narrow benchmark results.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): setup, data, and run
  hygiene.
- [CODE_AUDIT.md](CODE_AUDIT.md): known correctness and security limitations.
- [PROJECT_OVERVIEW_FOR_STRATEGIC_ASSESSMENT.md](PROJECT_OVERVIEW_FOR_STRATEGIC_ASSESSMENT.md):
  strategic context and open risks.
