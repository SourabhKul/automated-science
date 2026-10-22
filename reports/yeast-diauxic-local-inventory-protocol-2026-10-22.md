# Local yeast source inventory

Primary decision, October 22, 2026. The user supplied both Dryad files and
asked to continue. This replaces the access HOLD for this local metadata
stage only. Preserve the October 21 protocol and denied-request history.
No real fitting, growth-value parsing, forecast, score or LLM call is allowed.

Use the existing files in Downloads without copying or moving them:

- `Dataset+1.zip`: 1,594,286 bytes, SHA-256
  `2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab`.
- `README_for_Dataset+1.txt`: 4,777 bytes, SHA-256
  `f91c6d3ae0651919dda79a383fe9947922af67781f109a2bec02dcc0eb722b7b`.

Run ID: `yeast-diauxic-local-inventory-20261022-v1`. First inspect active jobs,
existing same-ID output, Git and storage. Do not overwrite an existing attempt.
One Luna max author prepares one readable ignored command of at most 120
nonblank lines. A different Luna max reviewer checks the exact command and
this protocol before execution. No product code, dependency or test changes.

Recheck source sizes and hashes before interpreting text. Read the external
README, list ZIP names, sizes and CRC, and inspect only the first logical CSV
header record. Inspect small metadata or mapping text only if it is clearly
distinct from growth observations. No numeric observation rows, plots or
workbooks. No ZIP extraction to disk. Validate the full member roster before
opening any member: reject absolute/traversal paths, backslash paths, symlinks
and duplicate normalized names. Verify CRC for any fully opened member.
For a header-only stream, state explicitly that the remaining member bytes
and its full CRC were not checked. Archive SHA binds the source bytes.

Limits: no network, one process, 120 seconds wall, 512 MiB process RSS,
1,000 members, 32 MiB total declared expansion, 64 KiB per interpreted text
member or CSV header record, 1 MiB inventory output. Stop on malformed or
ambiguous records, resource failure or hash mismatch. Keep failure evidence.
Use strict text decoding and bounded header reading. Do not infer groups from
outcome values. A large observation member may supply a bounded header;
never read its rows to finish a checksum.

Save an exclusive ignored receipt with command/protocol/source hashes,
member roster, interpreted text and headers, exact scope, costs and failures.
Record observed resource use and any cap not enforced. The independent
reviewer checks the result from the same permitted metadata. Primary then
decides which plate, strain, alias and control mappings are supported and
whether further metadata are needed. Filenames alone do not establish
biological independence or a valid split. Two plates from shared inoculation
remain paired technical assays. Any real fit needs a separate explicit
estimand, model/measurement specification, split, budget and review.
