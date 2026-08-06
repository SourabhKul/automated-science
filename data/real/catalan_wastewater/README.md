# Catalan Wastewater SARS-CoV-2 Source Record

Status: acquired and inspected; **not approved for Phase 20 adapter implementation or LLM discovery**.

## Canonical Source

- Citation: Catalan Surveillance Network of SARS-CoV-2 in Sewage, Zenodo record [16266942](https://doi.org/10.5281/zenodo.16266942), version `1.139`.
- Published: 2025-07-21.
- License declared by record: Other (Open).
- Downloaded file: `raw/sars-aigues-1.139.zip`.
- Source URL: `https://zenodo.org/api/records/16266942/files/icra/sars-aigues-1.139.zip/content`.

## Integrity

| Digest | Value |
| --- | --- |
| Zenodo-published MD5 | `de14539ec8265e7d2de69e8781ea8612` |
| Locally verified MD5 | `de14539ec8265e7d2de69e8781ea8612` |
| Local SHA-256 | `268898a3674e77034c2b638c363dd79db5ff94d05175414afee0cb6c349fc087` |

## Observed Schema

The fixed archive contains `release_with_detection_limits.csv` with 6,593 rows, 59 site labels, and dates from 2020-07-06 through 2025-06-16. The available fields include sample ID, treatment-plant name, detection limit, N1/N2/IP4/E viral targets in genomic copies per litre, 24-hour flow, rain, notes, and a point-value flag.

N1 is nearly complete (15 missing rows), but 24-hour flow is missing in 513 rows and rain in 901 rows. Only three sites have complete flow and complete rain coverage. The release includes method-change notes for 34 sites, but has no laboratory ID, assay-method ID, or structured method-version field.

## Gate Result

Phase 20 requires eight sites with a shared target/normalization definition, aligned flow/rain context, and explicit laboratory/method tracking so that a covariate-adjusted state-space model can be tested against external sites. This archive does not meet that contract.

Do not normalize or impute the missing covariates, collapse method changes into an untracked time effect, or launch an observational/LLM discovery run. A future wastewater re-entry needs a fixed release with structured assay/method metadata and sufficient complete covariate coverage, or a separately pre-registered target that intentionally excludes those covariates and uses stronger matched time-structure controls.
