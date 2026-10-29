# Metadata helper inspection

Primary freeze at the October 29 17:55Z wake. This is a new selection after
the completed roster, text recovery and role-source stages. Reuse their
receipts; do not repeat those runs or the old Dataset 1 checks.

Run ID: `yeast-diauxic-mapping-helpers-20261029-v1`.
Source: `/Users/sourabh/Downloads/MATLAB+analysis+code.zip`, 6,885,800 bytes,
MD5 `6316037d5b4fd43a16394b6899082406`, SHA-256
`28d5a982ad2dcac338f6a9575308e1d256d0209599475cfe04f257a22e150084`.
Use the existing validated roster, receipt SHA-256
`412d6c625e0983f388f9a9110cdce68568549eca969bda458ba35377d053dfd9`.

Read only these complete source-text members, all in
`MATLAB analysis code/Common MATLAB Functions/`:

- `load96wmeta.m`, 611 bytes.
- `structify96wmeta.m`, 861 bytes.
- `subtractbg.m`, 215 bytes.

The question is how metadata cells become a well matrix and how the
background function transforms its input. Names alone do not prove that
these helpers identify biological blanks or aliases. Read source text as
evidence; never invoke it or follow its references. Do not infer blank
coordinates from empty labels or use old fixed arguments as a validated
measurement calibration. No observation rows, derived arrays, plots or
binary MAT files may be opened. Stop interpretation if an unexpected
observation table or manual outcome assignment appears.

Use a small clone of the independently reviewed source-text reader, changing
only run/destination, protocol binding and the three roster tuples. Recheck
source and roster hashes. Use size-plus-one bounded reads, exact declared
size checks and full CRC verification. Save member SHA-256, CRC, high-byte
and NUL counts, physical ASCII CR/LF line counts and exact text in one ignored
exclusive receipt. Latin-1 is only a lossless byte view with source encoding
unknown; keep JSON ASCII escaped and do not normalize labels.

Limits: one process, no network, 60 seconds processing wall limit, 512 MiB
sampled RSS, 8 KiB per member, 16 KiB total selected-member text bytes
(expected 1,687), 64 KiB receipt and 256 KiB new artifacts. A compact failure
receipt may finalize after a processing timeout; do no scans in that path.
Record that limitation. Stop on mismatch or read/resource failure; do not
drop failed members or retry automatically. Preserve existing bytes.

Independent Luna max design and exact-byte review, primary GO, and a fresh
job/Git/output-absence preflight precede one execution. No tests, product
code, package installation, MATLAB execution, model call or server change.
No fitting is authorized. Primary will assess the returned evidence and
choose the next scientific design. Keep B2a and all unrelated work intact.

## Parser and label dependencies

The helper read completed once at 17:59Z, receipt SHA-256
`3a422ad8bfee4a542143f8051dc435a7716e8a23011b864e7f9747b404f6497b`.
Its launch protocol hash was
`588083deca215f393c4c0dfb6db4c957068f0883dcbf223162a4407d26770252`.
It verified 1,687 bytes and three CRCs. It calls two source dependencies whose
behavior affects matrix construction and label grouping. This addendum
authorizes preparing one new reader; it does not authorize a helper rerun.

Run `yeast-diauxic-mapping-parser-20261029-v1` may read only these complete
source members under `MATLAB analysis code/Common MATLAB Functions/`:

- `csv2cell.m`, 1,295 bytes.
- `underscorify.m`, 906 bytes.

Use the same bound archive and existing roster, with the same limits and
reader pattern above; selected total is 2,201 bytes. Freeze exact roster
tuples in the command. Independent design/exact-byte GO and primary GO with
fresh preflight must precede execution. No actual CSV, observation, array,
plot or referenced file may be opened; source text only. Do not execute or
adopt label normalization as a biological alias map. Record source parser
rules separately from any newly validated reader. No additional source
dependency is authorized by this addendum. Preserve all earlier receipts.
