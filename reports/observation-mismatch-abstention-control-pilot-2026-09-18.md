# Observation-mismatch and abstention control pilot

Run: `20260918T110738Z` (2026-09-18). This is the one approved, serial,
six-cell known-simulator pilot. It uses the frozen
`deterministic_likelihood_weighted_grid_reference`; it does not call Qwen,
canonical SBI, or BDSS.

## Preflight and frozen protocol

Before execution, local `HEAD` and `origin/main` were both
`029b1e2fb592c774041ce7b30240db2854bce25f`, with no competing observation,
ABC, ODE, Qwen, or domain workloads. The host had 16 CPU cores, 128 GiB
physical memory, and 447 GiB free disk. A read-only GET to
`http://127.0.0.1:8000/v1/models` returned HTTP 200 and advertised
`Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`; no generation request was
made.

The run used the unchanged predeclared cells and defaults:

- protocol `observation_mismatch_abstention_control_v1`;
- time grid `0..8` by `0.25`, with train indices `0:16` (`0..3.75`),
  validation indices `16:24` (`4..5.75`), and sealed indices `24:33`
  (`6..8`);
- declared low-signal future times `[8.5, 9.0, 9.5, 10.0]`;
- Experiment A: `x=exp(-0.65*t)`,
  `y=x/(0.35+x)+Normal(0,0.025^2)`;
- Experiment B: `y=exp(-0.65*t)+Normal(0,0.05^2)`;
- A practical margin `sigma/sqrt(8)` for each experiment;
- data seeds `11`, `29`, and `47` for each experiment, with the predeclared
  fit and predictive seed bases;
- 512 fixed predictive draws per model, no ABC epsilon schedule, and no
  simulator/proposal noise stream.

The command completed all six cells serially in 6.8 seconds, below the
540-second pilot cap. Every cell and model receipt was complete; no cell was
silently substituted or dropped.

## Outcomes

| cell | validation decision | validation improvement | sealed result | measurement rule |
|---|---|---:|---|---|
| A/data11 | selected correct saturating observation | 0.0480835 | correct model RMSE 0.0250309; evaluated once | retained alternatives; no measurement search |
| A/data29 | unresolved / retain both | -0.0004535 | final not run after unresolved selection | both forecasts retained |
| A/data47 | selected correct saturating observation | 0.0309933 | correct model RMSE 0.0323430; evaluated once | retained alternatives; no measurement search |
| B/data11 | unresolved / retain simpler | -0.0012644 | sealed improvement -0.0018021; no promotion | no informative measurement under this protocol |
| B/data29 | unresolved / retain simpler | 0.0011174 | sealed improvement 0.0000039; no promotion | no informative measurement under this protocol |
| B/data47 | unresolved / retain simpler | 0.0182601 (passes margin) | sealed improvement -0.0099252; no promotion | no informative measurement under this protocol |

For B, the fixed practical margin is `0.0176777`. Seed 47 exercises the
required nested abstention branch: validation clears the margin, but the
sealed advantage does not reproduce. The posterior-predictive median
disagreements over the fixed future order were at most `0.0079045`, `0.0023063`,
and `0.0263747` for seeds 11, 29, and 47, respectively, all below `2*sigma =
0.1`; therefore no future measurement was recommended.

Each model receipt preserves observed-scale map configuration, solver
configuration, normalized grid weights, ESS, counters, seeds, and predictive
draws. Development receipts contain train/validation hashes and no sealed
outcome hash. A seed-29 A cell retains both model forecasts and explicitly
records `not_run_selection_unresolved`; B retains both forecasts while using
the sealed suffix only for its frozen reproduction gate.

Raw atomic receipts and the SHA-256 manifest are retained under the ignored
[`artifacts/observation_mismatch_control`](../artifacts/observation_mismatch_control)
directory. The manifest digest is
`1cd0c016cfd4c74ae3f16dcd62c837590e057ab3065370377cd18bbb184dc3f1`.

## Claim boundary

These are deterministic numerical controls for a known generator. They show
the recorded observation-model validity and abstention behavior under this
protocol only. They do not establish real kinetic mechanism identification,
calibrated ABC or model probabilities, uncertainty coverage, novelty, or a
real intervention.
