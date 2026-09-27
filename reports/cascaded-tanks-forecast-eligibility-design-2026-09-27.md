# Cascaded Tanks target-free forecast gate — design, 2026-09-27

**Status: primary design specification, not a frozen real-data run protocol.**
This gate follows the reviewed training-only fit seam and precedes every
conversion of `yEst[768:1024]`. It does not authorize a Qwen request, a real
ABC fit, development scoring, or official-test access. The historical
multi-record source failure still limits this to methods-control evidence.

## Scientific boundary

The training fit reports whether every candidate **declared in its own
manifest** completed ABC-SMC. That is insufficient for a fair comparison:
the manifest could omit a required alternative, a fit receipt could be
altered, or a particle could fail to forecast. A separate, previously
reviewed comparison declaration must name the exact fit-manifest SHA-256,
source/contract identity, ordered candidate IDs and model families, expected
particle count for each, forecast length/alignment, and simulation limits.
It must be committed and reviewed before a real fit; the gate checks exact
bytes and records the declaration hash. Its existence alone cannot prove
pre-outcome timing, so the run ledger and Git commit are part of the
predeclaration evidence. The final real-data declaration and its numeric
values remain to be frozen.

The gate takes **paths** to the source archive, frozen fit manifest,
completed fit receipt, and comparison declaration. It rejects caller-made
arrays, particles, weights, model lists, and fit objects. Read each JSON
document once into a bounded immutable byte buffer; hash and parse those
same bytes with strict UTF-8 JSON and duplicate-key rejection. The source
archive has the analogous single-snapshot hash/parse rule. It verifies the
fit receipt's content hash and schema; official-source scope; run/protocol
IDs; exact fit-manifest byte hash; source/stage, runtime and relevant code
pins; `all_declared_candidate_fits_complete=true`; and
`development_score_eligible=false`. The ordered candidate declarations and
results must equal both the fit manifest and the independent comparison
declaration exactly, with no missing, duplicate, extra, failed, or reordered
candidate. A manifest-relative `status=complete` cannot override a roster
mismatch. Each terminal population must contain exactly the frozen number
of finite unit-cube particles and unchanged normalized nonnegative ABC
weights; decoded parameter values, family and initial states must agree
with the candidate declaration and fit receipt. Malformed or incomplete
evidence is a terminal gate failure, never a reduced ensemble. A self-hash
does not authenticate the fit: independently reconcile each posterior's
unit parameters, weights and distances with the final recorded ABC
population; recompute physical free values from the manifest's ordered
bounds, merge fixed values, and reject any mismatch even when a modified
receipt has a recomputed self-hash. Give every particle the stable identity
`(candidate_id, final_population_index)`.

Using the pinned adapter, load only training arrays and the input-only
development suffix. Recheck the source/stage identities against the fit
receipt and declaration. The adapter must hash and parse the same bounded
archive snapshot; a path replacement or in-place source change cannot
silently alter the forecast inputs. Convert each complete ABC posterior to
the existing immutable forecast-particle interface and call the reviewed
target-free forecast path for **each** declared ABC candidate. Carry every
particle's state through all 768 training transitions, including input
index 767, then forecast all 256 later inputs from state index 768. Retain
the ABC weights without clipping, dropping, or renormalization. Every
declared particle must yield a complete finite 256-step forecast under the
frozen simulator limits. If any candidate or particle fails, report a
terminal failure and no promotable aggregate forecast for the comparison;
preserve diagnostics and the negative result.

On success, issue a content-hashed **target-free forecast-completeness
receipt** binding comparison declaration, fit manifest and receipt hashes,
the complete pinned archive SHA-256 and source/stage hashes, ordered
candidate and particle identities, unchanged weights, simulator/code/runtime
identities, per-particle forecast hashes, and each candidate's
weighted-median forecast hash. Recheck the fit manifest's code/runtime pins
against the fit receipt and current source adapter, simulator, ABC reference,
fit and forecast implementations; also bind the new gate's own code hash.
The existing forecast receipt's code hash does not cover this gate or fit
bridge. Preserve original weights exactly, including under the forecast
interface's normalization tolerance. The gate receipt must state zero
development/official target materialization. This receipt is not a score
and does **not** set `development_score_eligible=true`: matched baselines,
pre-target structural-separation/abstention criteria, Qwen proposal handling,
and the deferred-target scorer remain separate parts of the future frozen
real-data protocol. A later scorer must require the exact reviewed
forecast-completeness receipt and protocol bundle and claim its one-use
marker before materializing `yEst[768:1024]` from the same pinned source.
The official `uVal/yVal` record remains unavailable until the distinct
locked final stage.

## Synthetic checks before production use

Use only private synthetic archive fixtures and synthetic ABC fit receipts.
The gate must reject a missing, extra or reordered candidate; altered fit
manifest/receipt, source, weight or particle; an incomplete ABC population;
and a single nonfinite or truncated forecast. A hidden-target sentinel
mutation may change the full archive identity but must not change any
permitted training/input-only forecast payload; no target parser may be
called. Test source replacement and same-inode mutation around archive
loading **and** fit-receipt/declaration loading, plus an internally
inconsistent but rehashed fit receipt, candidate order, and the 767-to-768
transition. A complete
synthetic roster should yield deterministic hashes on replay. Independent
Luna review must verify both the fail-closed cases and the absence of
development/official target parsing before this gate is included in any
real-data run.
