# One fixed header record (reduced from two)

Primary decision at the October 23 06:21Z wake. The code package remains
absent. One narrow check of the existing source is useful: does an export
identify its fields or sample labels without opening measurement rows?
The earlier inventory did not answer this because its selector required a
single CSV in the entire archive. Preserve that stop and all original bytes.
This is a new metadata stage, not a rerun or a real-data fitting protocol.

Run ID: `yeast-diauxic-two-header-20261023-v1`. Use the verified existing
`/Users/sourabh/Downloads/Dataset+1.zip`, SHA-256
`2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab`.
The reviewed roster already binds the archive path. Select exactly:

- `Dataset 1/glucose + galactose/OD600_shake_iw_plate1_279105.csv`.

This is the first roster entry for the mixed-sugar plate1-name prefix.
The original design selected two members. Static review required fixes to
CR/LF handling and canonical output. Primary reduced this optional check to
one member to keep the 60-line scope; no prior draft was executed. Selection
uses names only, not values or a biological grouping inference. No other
member, mapping file, growth-rate table or dilution
series may be read. No extraction, acquisition, network, MATLAB execution,
plots, fits, model calls, forecasts or scores.

One Luna max author prepares one ignored command of at most 60 nonblank
lines. A different Luna max reviewer checks the exact command and this
protocol before one execution with primary GO. Recheck archive hash and
the prior roster receipt hash before reading. The roster receipt SHA-256 is
`32424810cc9a3b7b1f9035f98512b6fb6d8d253eb3b2dad29fb401bb18b07f6e`.

Read at most the first physical line of the selected member, capped at
4,096 bytes. No second line, multiline CSV record, numeric conversion,
observation-row iteration or full-member CRC scan. If a line exceeds the
cap, decoding fails, quotes require another line, or fields are numeric or
ambiguous, emit only a status and member name, plus provenance and byte
counts. Do not print, save or hash
unrecognized first-record contents. To emit a header, require at least one
frozen recognized field name (time, well, sample, plate or barcode, with
explicit unit/ID variants in the reviewed command) and no purely numeric
fields. Emit only canonical recognized groups and positions, never raw
field text. Reject ambiguity rather than infer meaning. Transient first-line
bytes may be buffered to recognize the header; report this exact scope.
Stop on the first CR or LF at the caller boundary, with no subsequent byte
read. ZIP-library buffering does not authorize interpreting later bytes.
No strain/control map or scientific split follows from a generic header.

Limits: one process, 30 seconds wall including preflight, 256 MiB sampled
RSS (not a continuous hard maximum), 16 KiB exclusive ignored receipt.
Use a new output path; do not overwrite or silently retry. Report command,
protocol, roster and source hashes, canonical groups only if recognized,
bytes read, status, elapsed time, sampled memory and CRC not checked. An
output-cap fallback must retain these hashes, member name/status/bytes and
CRC flag, with no field text.
The independent reviewer checks the result under the same field boundary.
If no mapping is revealed, retain the existing code-package dependency and
quiet notification policy. Do not rewrite launch plans or claim a new result.
