# ABC6 replay source map v1 — primary proposal

**Proposal for independent design review; no public replay or launch GO.**
This map names the committed Stage A sources from which a future terminal-v4
replay must derive expected values for `score-attempt.json`. It does not
authorize a campaign claim, scoring, prospective-target materialization, or
reading real tank development or official test targets. The public replay
entrypoint still requires terminal schema v3; a separate v4 helper only links
launch attestation. Production static source pins remain `None`.

## Fixed anchor and source paths

The reviewed manifest, launch approval and independently built authority
bindings must agree on the run ID, physical project root and the anchored
receipt root. For the reserved control, the only receipt root is
`artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/receipts`
(`core/real_data/cascaded_tanks_abc6_campaign_fit.py:58-62`). It is a
no-follow directory identity, not a string supplied by an attempt receipt.
An inconsistent manifest/approval/root is `unreplayable` before any attempt
comparison. Derive all leaves below from the reviewed source and fixed run
roster, then open each through separately pinned no-follow parent descriptors.

| Source | Derived leaf or relative path | Required direct comparison |
| --- | --- | --- |
| External campaign claim | `artifacts/evaluations/cascaded_tanks_abc6_campaign_fit/claims/ct-abc6-20260928-v1.claim` | Canonical raw bytes and SHA-256 must equal the independently opened receipt-root `campaign.claim` and the attempt's `campaign_claim_sha256`. This is distinct from the watchdog and scoring claims. |
| Stage A summary | `<receipt_root>/campaign.training-summary.json` | Recompute raw digest; require the evidence manifest's `training_summary_filename` and `training_summary_sha256`, watchdog gate `summary_sha256`, and attempt `training_summary_sha256` to match. Validate the exact 24-row roster and claim link. |
| Stage A evidence manifest | `<receipt_root>/campaign.training-evidence-manifest.json` | Recompute raw digest; require watchdog gate `evidence_manifest_sha256` and attempt `training_evidence_manifest_sha256` to match. Validate its own canonical payload digest, claim, summary, roster, case evidence and 48 status links. |
| Status receipts | For each integer `i=0..23`, `<receipt_root>/case-{i:02d}.fit-status.json`, then `<receipt_root>/case-{i:02d}.baseline-status.json` | Recompute each raw digest in this exact fit-then-baseline order; require the evidence manifest's `status_receipts` filename/digest entry, watchdog gate's corresponding filename/path/digest row, and attempt row `{case_index:i, component, sha256}` to match. Validate receipt payload identity/status independently. |
| Target-free forecast | `<receipt_root>/campaign.target-free-forecasts.json` | Recompute raw digest; require watchdog gate `target_free_forecast_sha256` and attempt `forecast_artifact_sha256` to match. Validate its 24-case roster, source hashes, summary/evidence/status links and `target_free: true`. |

The external claim's exact seven keys are `schema_version`, `protocol_id`,
`run_id`, `manifest_sha256`, `receipt_directory`, `claimed_at_utc`, and
`claim_semantics` (`campaign_fit.py:2093-2131`). A claim digest asserted by
itself or by the attempt receipt is insufficient. The evidence manifest has
exactly `schema_version`, `protocol_id`, `run_id`, `manifest_sha256`,
`claim_sha256`, `training_summary_filename`, `training_summary_sha256`,
`roster_sha256`, `ordered_case_identities`, `status_receipts`,
`bundle_artifact`, `case_artifacts`, and `payload_sha256`
(`campaign_fit.py:833-847`). Its bundle and 24 case evidence leaves live
under `training-evidence/`; those subordinate links must also be verified,
not trusted by digest name alone (`campaign_fit.py:2379-2424`).

Each status receipt has exactly `schema_version`, `protocol_id`, `run_id`,
`case_index`, `case_id`, `component`, `status`, and `payload_sha256` in the
fixed 48 order (`core/real_data/cascaded_tanks_abc6_receipt_io.py:17-33,
168-220`; `scripts/watch_cascaded_tanks_abc6.py:4583-4633`). The summary
has exactly 24 `case_statuses`; each row carries index, ID, fit/baseline
status and the two receipt digests (`campaign_fit.py:2333-2363`). Forecast
payload keys and roster/status/source links are defined by the runner at
`scripts/run_cascaded_tanks_abc6_synthetic.py:356-391`; the existing
watchdog verifies that exact schema at `watch_cascaded_tanks_abc6.py:4380-4466`.

## Terminal gate and classification boundary

For terminal schema v4, the normal writer's `fit_phase_gate` has exactly these
17 keys: `status_receipt_count`, `status_receipts`, `summary_path`,
`summary_sha256`, `summary_status`, `receipt_root_identity_valid`,
`file_identity_revalidation_valid`, `status_chain_file_identity_valid`,
`all_48_status_receipts_present_and_linked`,
`all_48_fit_and_baseline_statuses_complete`, `evidence_manifest_sha256`,
`training_evidence_chain_valid`, `target_free_forecast_sha256`,
`target_free_forecast_chain_valid`, `pre_score_artifact_chain_valid`,
`postfit_target_gate_opened`, and `problems`
(`scripts/watch_cascaded_tanks_abc6.py:4926-4950`). Its normal
`status_receipts` rows include case index/ID, component, status, filename,
absolute path and raw SHA-256 (`:4583-4633`). Replay must derive the expected
absolute path from the pinned root, never accept the row's path as an
authority. Require the terminal gate's validated chain booleans and exact
48 status rows before using it as the attempt's expected source; any absent
or contradictory field is `unreplayable` for a campaign-claimed attempt.

The watchdog exception fallback currently omits
`status_chain_file_identity_valid`
(`scripts/watch_cascaded_tanks_abc6.py:3391-3409`).
This proposal does not silently normalize that shape into a successful gate:
it is a terminal error/fallback gate that may accompany an already `capped`
terminal, and its exact future v4 replay classification needs a separate
reviewed rule. The existing no-campaign-claim branch classifies only its
structured cases: verified budget/operator stops or an attested child nonzero
exit. A prechild failure without a structured failure receipt is
`unreplayable` (`core/real_data/cascaded_tanks_abc6_replay.py:1805-1858`).
Under the reviewed
pre-marker v1 design, any **campaign-claimed** run without a valid attempt
receipt remains `unreplayable`; child nonzero exit or a partial fit gate does
not establish that the scorer was never entered.

The future verifier must also build expected `authority_bindings` from the
frozen manifest/review/approval, separately verified watchdog claim and
launch attestation, checked roots, runtime/source/role pins and child process
identity. It must verify the durable grant record against that construction,
then compare the attempt's entire bindings object and grant-record digest.
The attempt, forecast, evidence manifest and terminal may cross-check one
another but cannot self-authorize. Any receipt leaf substitution, symlink,
FIFO, copied ancestor, changed device/inode or ambiguous readback fails
closed under the existing anchored-reader contract.

## Gates before use

An independent Luna max reviewer must check every path and key above against
the exact committed source and flag any omitted field or unsupported chain.
Then freeze a source-backed expected-binding builder and terminal-v4 replay
classification with fake-only adversarial tests and a separately reviewed
versioned marker/terminal link for post-marker results. Complete transitive
source/runtime/role closure, immutable manifest, independent report,
approval and a distinct launch GO are still required. The reserved synthetic
campaign was last observed unused at the 11:34Z monitor and authorization
remains HOLD; no real tank Qwen/ABC or target read follows
from this map.
