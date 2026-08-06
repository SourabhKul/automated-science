# Evidence Summary

This table reports narrow retrospective source-file benchmarks from the local
evaluation stream. Scores are macro-F1 on the frozen internal selection split
and, where authorized, one native external file or cohort. They are not claims
about the underlying real-world object, person, process, or deployment setting.

| Phase | Fixed benchmark | Selection macro-F1 | Native external macro-F1 | Interpretation |
| --- | --- | ---: | ---: | --- |
| 41 | UCI ISOLET speaker-file holdout | 0.9543 | 0.9570 | Completed narrow benchmark |
| 42 | UCI PenDigits writer-file holdout | 0.9598 | 0.9164 | Completed narrow benchmark |
| 44 | UCI Optical Digits writer-file holdout | 0.9728 | 0.9500 | Completed narrow benchmark |
| 46 | UCI Spoken Arabic Digit speaker holdout | 0.6160 | 0.6798 | Completed narrow benchmark |
| 54 | UCI Image Segmentation source-file holdout | 0.8491 | 0.8982 | Completed narrow benchmark |
| 57 | Fashion-MNIST creator train/test | 0.8544 | 0.8452 | Completed narrow benchmark |
| 67 | STL-10 creator train/test | 0.2962 | 0.3307 | Completed, low-performing baseline |
| 70 | CODH Kuzushiji-MNIST creator train/test | 0.8225 | 0.6986 | Completed narrow benchmark |

The current Phase 73 UCI Wine Quality candidate has passed its official source
and injected-mechanics gates but has no observed fit or external score yet.

## How To Read This

The external column is intentionally conservative. It is opened only once,
after the source-train controls pass, and is never used for tuning. Three
identical fixed-seed prediction fingerprints trigger the adaptive stop in the
current benchmark protocol.

Several other phases are recorded as blocked or closed negative. Those failures
are part of the evidence: schema mismatches, inconsistent duplicate labels,
hard prediction bounds, or failed specificity controls prevent promotion.
They must not be hidden by selecting only successful tables.

The detailed reports, raw ledgers, checksums, and seed-level metrics are kept in
the local `artifacts/evaluations/` tree and are excluded from the public source
commit to keep the repository reviewable and to avoid publishing downloaded
dataset contents.
