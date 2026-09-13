# Canonical discovery loop: evaluation-boundary acceptance plan

Primary scientific specification, 2026-09-13. This is the next integration milestone after the isolated inference reference. It is not evidence that the canonical runner has been repaired.

## Scientific contract

Model search is an adaptive learning process. Parameter fitting and structural proposal selection need separate development roles; independent final evidence must remain inaccessible to both. The current suffix may legitimately guide search if it is labeled validation. Renaming a metric alone does not create a final test.

The initial repair should support three explicit trajectory partitions for a narrow synthetic integration task, then extend the same roles to independently sampled trajectories, cells, subjects, and input conditions. A suffix forecast and a new-cell prediction are different estimands and must retain distinct labels.

| Stage | May read | Must not read |
| --- | --- | --- |
| Data preparation | Declared metadata and training observations; validation inputs where protocol permits | Final-test outcomes for scaling, interpolation, state initialization, or feature selection |
| Parameter fitting | Training outcomes, masks, inputs, training-fitted transforms | Validation or final outcomes as fitting targets |
| Candidate proposal/repair/selection | Training and validation diagnostics, prior candidate ledger | Final outcomes, metrics, residuals, or final evaluator failure hints |
| Final evaluator | Frozen selected model, parameters or explicitly frozen refit procedure, protocol and final data | Mutable proposal/selection state or an ability to trigger repair/tuning |

For temporal forecasts, the simulator may know declared future inputs only if the scientific task allows it. It may not initialize from an unseen future outcome. For per-subject/cell inference, any permitted calibration window belongs to an explicitly declared adaptation protocol and is excluded from final scoring.

## Required falsification checks

1. Freeze development data and RNG inputs. Replace sealed final outcomes with extreme sentinel values. The proposal prompts, candidate sequence, fitting inputs, selection decisions, and pre-final artifacts must remain unchanged. Mock generation is acceptable for this deterministic isolation test; retain separate actual-Qwen integration evidence.
2. Keep final outcomes unchanged and alter validation outcomes. Structural selection may change, demonstrating that validation is an active development signal rather than an unused decorative split.
3. Inspect every prompt payload and diagnostic packet: no final metric, full-data residual summary, normalization statistic fitted to final data, or final-ground-truth state may enter proposal context.
4. Complete model search without making final data available to its worker. A separate final-evaluation command should succeed after loading the locked model and protocol. Prefer independent process/input access rather than relying on naming conventions.
5. Persist a final evaluation receipt referencing the protocol, data/split hashes, model/parameter hashes, and results. Repeated evaluation of exactly the same frozen object is a replay, not an independent replication. A revised model or protocol cannot reuse the same outcome as fresh confirmatory evidence.
6. Force non-finite final predictions. Preserve an explicit failed final result; do not repair a candidate against final outcomes or silently fall back to a successful development score.

## Scope and acceptance

Preserve historical results as legacy adaptive validation. New records should explicitly identify metric roles and protocol version. Do not silently relabel old metrics as results of the repaired protocol. The isolated ABC reference may pass its scalar checks while the canonical discovery runner still lacks this separation; report those milestones independently.

A narrow integration is ready when the information-flow tests pass, the actual fixed-Qwen development loop produces a frozen model without final access, and the separate final evaluator produces an honest terminal result. Predictive improvement is not required to verify the boundary; a poor or failed final outcome remains valid evidence.
