# ABC6 synthetic campaign: terminal evidence and replay decision

**Status: primary operational decision for implementation and fake-only review; no campaign claim or launch is authorized.** This supplements the reviewed six-parameter synthetic protocol, receipt-root amendment, and supervised-launch decision. It changes no truth, case, prior, threshold, simulator, ABC kernel, baseline, forecast, score, or compute budget. The reserved `ct-abc6-20260928-v1` ID remains unused.

## Evidence required after the one-use claim

The runner must durably publish a post-reveal score receipt under the pinned receipt root. It must bind the frozen manifest and run IDs, the 48 training fit/baseline status receipts and their summary, the target-free forecast artifact, the separately pinned one-use reveal marker, the A/B/M prospective target hashes, and the score-result/forecast-array hashes. The linked result must retain the frozen protocol's full diagnostics: all failure and cost accounting, parameter and prospective-forecast diagnostics for complete eligible fits, paired S/L contrasts including null pairs, N's no-target/no-score status with structural abstention and ceiling summaries, and M's mismatch residual groups. Null forecasts and unresolved fits remain explicit; a scored subset must never be presented as the whole roster. The targets and full arrays remain in ignored local artifacts. Compact evidence may later be committed only after independent replay.

The watchdog must verify the complete claim-to-status-to-forecast-to-marker-to-score chain through bounded, nonblocking, no-follow descriptor-relative reads. It must publish and fsync a terminal receipt, then read back the canonical bytes through the pinned parent and verify their hash and file identity. A child exit code of zero alone never establishes a completed campaign. If the terminal publication or readback fails, preserve the claim and classify the run as unreplayable; do not start a second child or change the run ID in place.

## Replay classifications and scientific gate

An independent, read-only replay verifier may hash and read already-materialized retained artifacts only; it may not reopen the scorer gate or call the simulator, ABC kernel, Qwen, or prospective-target generator. It reports exactly one operational classification, in this precedence order:

- `unreplayable`: terminal evidence is absent, unreadable, altered, or insufficient to verify its claimed classification after a one-use claim.
- `failed`: a valid terminal receipt verifies a child or stage error before any watchdog budget intervention or deliberate operator stop, with the available evidence and failure point verified.
- `incomplete`: a valid terminal receipt verifies a watchdog budget cap or deliberate operator stop before the full scoring path, with the available partial chain verified. A resulting child nonzero exit caused by that intervention does not turn it into `failed`; an independently recorded earlier error does.
- `complete`: terminal receipt and every required evidence link verify, and the declared scoring path finished without an earlier failure or intervention, even if some fits or forecasts are explicitly unresolved.

Separately, the scientific readiness gate is `pass` only if all 16 declared same-family S/L fits have complete posteriors, finite all-particle forecasts, and independent replay, as required by the frozen protocol. A `complete` operational run can therefore fail scientific readiness. Incomplete wrong-family fits, null forecasts, negative recovery contrasts, and structural abstention remain reported rather than discarded. Neither operational completion nor scientific readiness is a physical discovery claim.

## Review boundary

Independent Luna fake-only review must cover a valid full chain, every missing or altered stage, all 48 status receipts present but some scientifically incomplete, child failure after claim, failure after the reveal marker, score-receipt and terminal publication/readback failures, and replay of complete/incomplete/failed/unreplayable cases. Exercise FIFO, symlink, copied-ancestor, duplicate-claim, and tampered-byte substitutions before and after descriptor opening. No protocol prospective targets may be generated during this review. Final source closure, immutable manifest, separate independent GO, and a fresh unused-run preflight remain required before any actual launch.
