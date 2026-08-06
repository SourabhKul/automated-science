#!/usr/bin/env python3
"""Fixed official UCI PenDigits archive and raw-schema gate only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARCHIVE = Path("data/real/uci_pendigits/raw/uci_pendigits_81.zip")
OUT_DIR = Path("artifacts/evaluations/phase42_uci_pendigits_writer_independent_classification_archive_gate_20260726")
MANIFEST = Path("data/real/uci_pendigits/manifest.json")
SPLIT = Path("data/real/uci_pendigits/source_file_split.json")
MEMBERS = ("pendigits-orig.names", "pendigits-orig.tes.Z", "pendigits-orig.tra.Z", "pendigits.names", "pendigits.tes", "pendigits.tra")
DATA_MEMBERS = {"pendigits.tra": 7494, "pendigits.tes": 3498}
LABELS = set(range(10))
SOURCE_SPLIT = {"source_train": "pendigits.tra", "external": "pendigits.tes", "writer_design": "UCI documents 30 writers in source train and other 14 writers in source test"}

def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def _optional_hash(path: Path) -> str | None:
    return _sha256(path) if path.exists() else None

def _inspect(archive: ZipFile, member: str) -> dict[str, object]:
    rows=0; blank=0; width=0; nonfinite=0; noninteger=0; out_of_range=0; invalid_labels=0; duplicate=0; fingerprints=set(); class_counts={label:0 for label in sorted(LABELS)}
    for line in archive.read(member).decode("ascii").splitlines():
        raw=line.strip()
        if not raw: blank += 1; continue
        rows += 1; values=np.fromstring(raw, dtype=float, sep=",")
        if values.shape != (17,): width += 1; continue
        if not np.all(np.isfinite(values)): nonfinite += int(np.count_nonzero(~np.isfinite(values))); continue
        if not np.all(values == np.floor(values)): noninteger += 1; continue
        features, label=values[:-1], int(values[-1])
        if np.any(features < 0) or np.any(features > 100): out_of_range += 1
        if label not in LABELS: invalid_labels += 1
        else: class_counts[label] += 1
        fingerprint=hashlib.sha256(raw.encode("ascii")).hexdigest(); duplicate += int(fingerprint in fingerprints); fingerprints.add(fingerprint)
    return {"rows":rows,"blank_rows":blank,"fields":17,"width_mismatches":width,"nonfinite_cells":nonfinite,"noninteger_rows":noninteger,"out_of_feature_range_rows":out_of_range,"invalid_labels":invalid_labels,"class_counts":{str(k):v for k,v in class_counts.items()},"complete_row_duplicates":duplicate,"unique_complete_rows":len(fingerprints),"first_line_is_header":False}

def run(archive_path: Path=ARCHIVE, output_dir: Path=OUT_DIR) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        inventory=tuple(archive.namelist()); ledgers={member:_inspect(archive, member) for member in DATA_MEMBERS}; names_hash=hashlib.sha256(archive.read("pendigits.names")).hexdigest()
    checks={"exact_official_member_inventory":inventory == MEMBERS,"fixed_rows":{member:ledgers[member]["rows"] == count for member,count in DATA_MEMBERS.items()},"finite_integer_17_field_schema":{member:ledger["blank_rows"] == 0 and ledger["width_mismatches"] == 0 and ledger["nonfinite_cells"] == 0 and ledger["noninteger_rows"] == 0 for member,ledger in ledgers.items()},"feature_range_0_to_100":{member:ledger["out_of_feature_range_rows"] == 0 for member,ledger in ledgers.items()},"labels_0_to_9":{member:ledger["invalid_labels"] == 0 and set(int(label) for label,count in ledger["class_counts"].items() if count) == LABELS for member,ledger in ledgers.items()},"frozen_writer_file_boundary":SOURCE_SPLIT["source_train"] == "pendigits.tra" and SOURCE_SPLIT["external"] == "pendigits.tes"}
    passed=all(value if isinstance(value,bool) else all(value.values()) for value in checks.values())
    provenance=Path("data/real/uci_pendigits/source_provenance")
    result={"phase":42,"status":"passed_official_source_gate_only" if passed else "blocked_official_source_gate","source":{"dataset_page":"https://archive.ics.uci.edu/dataset/81/pen-based%2Brecognition%2Bof%2Bhandwritten%2Bdigits","archive_url":"https://archive.ics.uci.edu/static/public/81/pen%2Bbased%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip","doi":"10.24432/C5MG6K","license":"CC-BY-4.0","archive_bytes":archive_path.stat().st_size,"archive_sha256":_sha256(archive_path),"published_checksum":None,"archive_headers_sha256":_optional_hash(provenance / "archive_headers.txt"),"page_headers_sha256":_optional_hash(provenance / "page_headers.txt"),"page_sha256":_optional_hash(provenance / "dataset_page.html"),"pendigits_names_sha256":names_hash,"members":inventory},"checks":checks,"member_ledgers":ledgers,"source_limitations":{"feature_process":"UCI documents source-owned normalization and spatial resampling to eight coordinate pairs","raw_pressure":"UCI states pressure values were ignored","writer_ids":"individual writer IDs are unavailable within the two release files"},"source_file_split":SOURCE_SPLIT if passed else None,"isolation":{"observed_coordinate_or_label_fitting":False,"external_outcome_scoring":False,"abc_smc_calls":0,"llm_calls":0},"next_gate":"Write the fixed raw-vector adapter, baseline, and output-isolated controls before observed fitting." if passed else "Return to source-backed selection without source repair or reconstructed writer assignment."}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"assessment.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if passed:
        MANIFEST.parent.mkdir(parents=True,exist_ok=True); MANIFEST.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); SPLIT.write_text(json.dumps(SOURCE_SPLIT,indent=2)+"\n")
    return result

def main() -> int:
    print(json.dumps(run(),indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
