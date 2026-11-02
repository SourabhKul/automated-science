#!/usr/bin/env python3
"""Bounded, metadata-only Dryad source gate for the ecology pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "ecology-source-gate-20261102-v2"
DOI = "10.5061/dryad.3kb72"
BASE = "https://datadryad.org"
DATASET_URL = BASE + "/api/v2/datasets/doi%3A10.5061%2Fdryad.3kb72"
EXPECTED_DATE = "2015-10-27"
FILES = ("p-d-h_raw_data3.csv", "p-d-h.serra3.csv")
PROV_ROOT = ROOT / "data/real/ecology/source_provenance"
RAW_ROOT = ROOT / "data/real/ecology/raw"
PROTOCOL = ROOT / "research_lab/protocols/ecology-source-gate-2026-11-02-v2.md"
PLAN = ROOT / "research_lab/plans/ecology-source-gate-v1.md"
PROPOSAL = ROOT / "research_lab/proposals/restart-2026-11-02/ecology-source-gate-v2.json"
REVIEWS = (
    ROOT / "research_lab/reviews/2026-11-02T1806Z/ecology-v2-source-review.json",
    ROOT / "research_lab/reviews/2026-11-02T1806Z/ecology-v2-peer-review.json",
)
SOURCE_CHECK = ROOT / "research_lab/reviews/2026-11-02T1806Z/ecology-v2-source-check-summary.md"
PINNED = {
    PROTOCOL: "f1fdb9c65b7ff8b964ddad898111b6151320742a79d9ff9d4c31cc434cf79919",
    PLAN: "cd35ff8ab2251cbc9837f04652b459e8c752598d25f00447b52a0db28bb14bfd",
    PROPOSAL: "95bb6e330cb3419f8fddea9b8a110e3cf5163e1ee2f8ade6adc1c4a168e84941",
    REVIEWS[0]: "0c4d645bffb713d415a32a8d8ab6daf067253b91396e1dce7527ff00a9130603",
    REVIEWS[1]: "dbeffccd44804c37d3e122d9e89edbf90471c052ff05465ca0cf4e928a6398c4",
    SOURCE_CHECK: "ea2376f220e1123f2fc24037025c034bccdf4dd9e7bb509c4605f3bdb1df5570",
}

# Candidate spellings come from source summaries. They are not verified
# Dryad headers. Exact, case-sensitive matching is mandatory; do not infer.
SCHEMAS = {
    FILES[0]: {
        "day": ("day",), "rep_id": ("repID",),
        "treatment": ("treatment", "trtmt"), "prey": ("prey",),
        "predator": ("predator",), "parasite": ("parasite",),
        "species": ("species",), "sampled_volume": ("vol", "volume"),
        "count": ("count",),
    },
    FILES[1]: {
        "day": ("day",), "rep_id": ("repID",),
        "treatment": ("treatment", "trtmt"), "prey": ("prey",),
        "predator": ("predator",), "parasite": ("parasite",),
        "species": ("species",), "plated_volume": ("Platevol", "plated_volume"),
        "dilution": ("Dilution", "dilution"), "Colonies": ("Colonies",),
    },
}
OPAQUE = {FILES[0]: "count", FILES[1]: "Colonies"}
# Coverage accepts exact lowercase yes/no flags only; no other labels map through.

MAX_META = 128 * 1024
MAX_FILE = 256 * 1024
MAX_BODY = 1024 * 1024
MAX_REQUESTS = 5
MAX_META_GETS = 3
MAX_FILE_GETS = 2
MAX_SECONDS = 90.0
MAX_GET_SECONDS = 15.0
MAX_RSS = 256 * 1024 * 1024
MAX_TRANSACTION = 4 * 1024 * 1024
RECEIPT_RESERVE = 256 * 1024  # Allows one atomic receipt write plus its timed update.
MAX_DATA = 50_000_000_000
FINALIZE_RESERVE_SECONDS = 5.0
CHUNK = 16 * 1024
API_VERSION_RE = re.compile(r"^/api/v2/versions/([0-9]+)(/files)?$")


class GateStop(Exception):
    """A safe, expected stop. Messages must not contain source rows."""


def decimal_day(token: str) -> Decimal:
    if len(token) > 64:
        raise GateStop("day_metadata_invalid")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise GateStop("day_metadata_invalid") from exc
    if not value.is_finite() or value < 0:
        raise GateStop("day_metadata_invalid")
    return Decimal(0) if value == 0 else value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def tree_bytes(paths: tuple[Path, ...]) -> int:
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(folder) / name).is_symlink()]
            for name in names:
                path = Path(folder) / name
                if not path.is_symlink() and path.is_file():
                    total += path.stat().st_size
    return total


def active_bytes() -> tuple[int, int]:
    return tree_bytes((ROOT / "data",)), tree_bytes((ROOT / "artifacts",))


def safe_storage_path(root: Path, path: Path, relative: Path) -> Path:
    repo = root.resolve()
    expected = root / relative
    if path != expected:
        raise GateStop("storage_path_out_of_scope")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GateStop("storage_path_symlink")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise GateStop("storage_path_out_of_scope") from exc
    if path.exists() and not path.is_dir():
        raise GateStop("storage_root_not_directory")
    return resolved


def validate_storage_roots() -> None:
    safe_storage_path(ROOT, PROV_ROOT, Path("data/real/ecology/source_provenance"))
    safe_storage_path(ROOT, RAW_ROOT, Path("data/real/ecology/raw"))


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def check_transaction(prov_dir: Path, raw_dir: Path, pending: int, reserve: int = RECEIPT_RESERVE) -> int:
    projected = tree_bytes((prov_dir, raw_dir)) + pending + reserve
    if projected >= MAX_TRANSACTION:
        raise GateStop("transaction_byte_cap")
    return projected


def write_atomic(path: Path, data: bytes, prov_dir: Path, raw_dir: Path, *, reserve: int = RECEIPT_RESERVE) -> int:
    projected = check_transaction(prov_dir, raw_dir, len(data), reserve)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return projected


def sha256_frozen(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError as exc:
        raise GateStop("frozen_input_missing") from exc


def check_frozen() -> dict[str, str]:
    found = {str(path.relative_to(ROOT)): sha256_frozen(path) for path in PINNED}
    if any(found[str(path.relative_to(ROOT))] != digest for path, digest in PINNED.items()):
        raise GateStop("frozen_input_hash_mismatch")
    return found


def validate_metadata_url(url: str) -> str:
    part = urlsplit(url)
    if part.scheme != "https" or part.hostname != "datadryad.org" or part.port not in (None, 443):
        raise GateStop("metadata_url_origin_rejected")
    if part.username or part.password or part.query or part.fragment:
        raise GateStop("metadata_url_shape_rejected")
    if part.path not in (DATASET_URL.removeprefix(BASE),) and not API_VERSION_RE.fullmatch(part.path):
        raise GateStop("metadata_url_path_rejected")
    return url


def validate_file_url(url: str) -> str:
    part = urlsplit(url)
    if part.scheme != "https" or part.hostname != "datadryad.org" or part.port not in (None, 443):
        raise GateStop("file_url_origin_rejected")
    if part.username or part.password or part.query or part.fragment:
        raise GateStop("file_url_shape_rejected")
    if not re.fullmatch(r"/downloads/file_stream/[1-9][0-9]*", part.path):
        raise GateStop("file_url_path_rejected")
    return url


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@contextmanager
def alarm_limit(seconds: float):
    """Use SIGALRM for a hard main-thread wall bound on this Unix client."""
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        raise GateStop("hard_wall_timer_unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("wall_deadline")

    signal.signal(signal.SIGALRM, expired)
    limit = min(seconds, previous_delay) if previous_delay else seconds
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        signal.setitimer(signal.ITIMER_REAL, max(0.0, previous_delay - elapsed) if previous_delay else 0.0, previous_interval)
        signal.signal(signal.SIGALRM, previous_handler)


class Client:
    def __init__(self, prov_dir: Path, raw_dir: Path, *, opener=None, clock=time.monotonic, started=None):
        self.prov_dir, self.raw_dir = prov_dir, raw_dir
        self.opener = opener or build_opener(ProxyHandler({}), NoRedirect())
        self.clock = clock
        self.started = started if started is not None else clock()
        self.calls = {"metadata": 0, "file": 0}
        self.total_body = 0
        self.rss_samples: list[dict[str, int | float]] = []
        self.request_log: list[dict[str, object]] = []
        self.transaction_peak = 0

    def sample(self) -> None:
        value = rss_bytes()
        self.rss_samples.append({"elapsed_seconds": round(self.clock() - self.started, 6), "rss_high_water_bytes": value})
        if value > MAX_RSS:
            raise GateStop("sampled_rss_cap")

    def fetch(self, url: str, kind: str, target: Path) -> bytes:
        if kind not in self.calls or sum(self.calls.values()) >= MAX_REQUESTS:
            raise GateStop("request_cap")
        url = validate_metadata_url(url) if kind == "metadata" else validate_file_url(url)
        per_kind = MAX_META_GETS if kind == "metadata" else MAX_FILE_GETS
        if self.calls[kind] >= per_kind or self.total_body >= MAX_BODY:
            raise GateStop("request_or_body_cap")
        if MAX_SECONDS - (self.clock() - self.started) <= 0:
            raise GateStop("wall_deadline")
        self.sample()
        cap = MAX_META if kind == "metadata" else MAX_FILE
        self.calls[kind] += 1
        number = sum(self.calls.values())
        part_path = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        body = bytearray()
        entry: dict[str, object] = {"number": number, "kind": kind, "method": "GET", "requested_url": url,
                                    "final_url": None, "status": None, "body_bytes": 0, "body_sha256": None,
                                    "partial_path": None, "error": None}
        self.request_log.append(entry)
        req = Request(url, headers={"Accept": "application/json" if kind == "metadata" else "text/csv,*/*",
                                    "Accept-Encoding": "identity", "X-API-Version": "2.1.0"}, method="GET")
        response = None
        started = self.clock()
        try:
            remaining = max(0.001, MAX_SECONDS - (self.clock() - self.started))
            with alarm_limit(min(MAX_GET_SECONDS, remaining)):
                try:
                    response = self.opener.open(req, timeout=min(MAX_GET_SECONDS, remaining))
                except HTTPError as exc:
                    response = exc
                status = int(response.getcode())
                final_url = response.geturl()
                entry.update({"status": status, "final_url": final_url})
                if final_url != url:
                    raise GateStop("final_url_changed")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > cap:
                    raise GateStop("response_body_cap")
                with part_path.open("xb") as output:
                    while True:
                        remaining_get = MAX_GET_SECONDS - (self.clock() - started)
                        remaining_total = MAX_SECONDS - (self.clock() - self.started)
                        if remaining_get <= 0 or remaining_total <= 0:
                            raise TimeoutError("get_deadline")
                        allowance = min(CHUNK, cap - len(body), MAX_BODY - self.total_body)
                        if allowance <= 0:
                            raise GateStop("response_body_cap")
                        self.transaction_peak = max(self.transaction_peak, check_transaction(self.prov_dir, self.raw_dir, allowance))
                        block = response.read(allowance)
                        if not block:
                            break
                        if len(block) > allowance:
                            raise GateStop("response_body_cap")
                        output.write(block)
                        body.extend(block)
                        digest.update(block)
                        self.total_body += len(block)
                        entry["body_bytes"] = len(body)
                        self.sample()
                        if length is not None and len(body) == int(length):
                            break
                entry["body_sha256"] = digest.hexdigest()
                if status != 200:
                    raise GateStop("http_status_not_200")
                if length is not None and len(body) != int(length):
                    raise GateStop("content_length_mismatch")
                os.replace(part_path, target)
                entry["partial_path"] = None
                entry["saved_path"] = str(target.relative_to(ROOT))
                if self.clock() - started > MAX_GET_SECONDS:
                    raise TimeoutError("get_deadline")
        except Exception as exc:
            entry["error"] = "deadline" if isinstance(exc, TimeoutError) else str(exc) if isinstance(exc, GateStop) else "request_or_storage_failure"
            entry["body_bytes"] = len(body)
            entry["body_sha256"] = digest.hexdigest() if body else None
            entry["partial_path"] = str(part_path.relative_to(ROOT)) if part_path.exists() else None
            raise GateStop(str(entry["error"])) from exc
        finally:
            if response is not None:
                response.close()
        return bytes(body)

    def get_json(self, url: str, target: Path) -> tuple[dict[str, object], bytes]:
        raw = self.fetch(url, "metadata", target)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GateStop("metadata_json_invalid") from exc
        if not isinstance(value, dict):
            raise GateStop("metadata_object_required")
        return value, raw


def link_url(doc: dict[str, object], relation: str) -> str:
    links = doc.get("_links")
    item = links.get(relation) if isinstance(links, dict) else None
    href = item.get("href") if isinstance(item, dict) else None
    if not isinstance(href, str) or not href:
        raise GateStop("required_metadata_link_missing")
    return validate_metadata_url(urljoin(BASE, href))


def dataset_identity(doc: dict[str, object]) -> bool:
    identifier = doc.get("identifier")
    return identifier in (DOI, "doi:" + DOI)


def record_license(doc: dict[str, object]) -> str | None:
    value = doc.get("license")
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("spdx") or value.get("name") or value.get("id")
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise GateStop("license_metadata_invalid")
    return value


def resolve_license(dataset: dict[str, object], version: dict[str, object]) -> tuple[str, str]:
    dataset_license, version_license = record_license(dataset), record_license(version)
    if dataset_license and version_license and dataset_license != version_license:
        raise GateStop("dataset_version_license_conflict")
    if version_license:
        return version_license, "dataset_and_version" if dataset_license else "version"
    if dataset_license:
        return dataset_license, "dataset_fallback"
    raise GateStop("explicit_reuse_license_missing")


def digest_metadata(item: dict[str, object]) -> dict[str, str]:
    found: dict[str, str] = {}
    def add_digest(name: str, value: object) -> None:
        normalized = name.lower().replace("-", "")
        if value is None or value == "":
            return
        if normalized not in ("md5", "sha256") or not isinstance(value, str):
            raise GateStop("unsupported_supplied_digest")
        if normalized in found and found[normalized] != value.lower():
            raise GateStop("conflicting_supplied_digest")
        found[normalized] = value.lower()

    for name in ("md5", "sha256"):
        if name in item:
            add_digest(name, item[name])
    digest, kind = item.get("digest"), item.get("digestType")
    if digest is not None and digest != "":
        if not isinstance(kind, str):
            raise GateStop("unsupported_supplied_digest")
        add_digest(kind, digest)
    checksums = item.get("checksums")
    if isinstance(checksums, dict):
        for key, value in checksums.items():
            add_digest(str(key), value)
    elif checksums not in (None, "", {}):
        raise GateStop("unsupported_supplied_digest")
    for key, value in item.items():
        name = str(key)
        normalized = name.lower().replace("-", "").replace("_", "")
        looks_like_digest = ("digest" in normalized or "checksum" in normalized
                             or normalized.startswith("sha") or normalized.startswith("md5"))
        parsed_exactly = name in ("md5", "sha256", "digest", "digestType", "checksums")
        if looks_like_digest and not parsed_exactly:
            if value not in (None, "", {}):
                raise GateStop("unsupported_supplied_digest")
    for name, value in found.items():
        if not re.fullmatch(r"[0-9a-f]{32}" if name == "md5" else r"[0-9a-f]{64}", value):
            raise GateStop("supplied_digest_invalid")
    return found


def selected_files(doc: dict[str, object]) -> dict[str, dict[str, object]]:
    embedded = doc.get("_embedded")
    rows = embedded.get("stash:files") if isinstance(embedded, dict) else None
    if not isinstance(rows, list):
        raise GateStop("file_list_shape_unknown")
    found: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GateStop("file_metadata_shape_unknown")
        name, file_id, size = row.get("path"), row.get("id"), row.get("size")
        if not isinstance(name, str):
            raise GateStop("file_name_missing")
        if name not in FILES:
            continue
        if name in found or not isinstance(file_id, int) or isinstance(file_id, bool) or file_id <= 0:
            raise GateStop("file_identity_ambiguous")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size > MAX_FILE:
            raise GateStop("file_size_invalid")
        link = row.get("_links", {})
        download = link.get("stash:download", {}) if isinstance(link, dict) else {}
        download_href = download.get("href") if isinstance(download, dict) else None
        if download_href is not None and (not isinstance(download_href, str) or len(download_href) > 4096):
            raise GateStop("api_download_link_metadata_invalid")
        found[name] = {"file_id": file_id, "path": name, "bytes": size,
                       "digests": digest_metadata(row),
                       "api_download_link": download_href}
    if set(found) != set(FILES):
        raise GateStop("selected_file_missing")
    if len({item["file_id"] for item in found.values()}) != len(FILES):
        raise GateStop("file_id_collision")
    return found


def existing_file(meta: dict[str, object], source: dict[str, object]) -> Path | None:
    for receipt_path in PROV_ROOT.glob("*/receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            record = receipt.get("files", {}).get(meta["filename"], {})
            relative = record.get("path")
            if receipt.get("source") != source or record.get("status") != "verified" or not relative:
                continue
            if (record.get("file_id") != meta["file_id"] or record.get("bytes") != meta["bytes"]
                    or record.get("digests") != meta["digests"] or record.get("source_path") != meta["source_path"]):
                continue
            candidate = ROOT / relative
            candidate.resolve().relative_to(RAW_ROOT.resolve())
            if candidate.is_file() and candidate.stat().st_size == meta["bytes"] and sha256_file(candidate) == record.get("sha256"):
                actual = {"md5": hashlib.md5(candidate.read_bytes(), usedforsecurity=False).hexdigest(),
                          "sha256": record["sha256"]}
                if all(actual[key] == value for key, value in meta["digests"].items()):
                    return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            continue
    return None


def audit_header(path: Path, filename: str) -> dict[str, object]:
    spec = SCHEMAS[filename]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise GateStop("header_unreadable") from exc
    header_hash = sha256_bytes(json_bytes(header))
    if len(header) > 64 or any(len(value) > 256 for value in header):
        return {"status": "hold_header_size", "field_count": len(header), "header_sha256": header_hash}
    basis = "provisional candidate spelling from local summaries; not source-verified"
    header_info = {"headers": header, "field_count": len(header), "header_sha256": header_hash,
                   "candidate_roles": list(spec), "mapping_basis_by_role": {role: basis for role in spec}}
    if len(header) != len(set(header)):
        return {**header_info, "status": "hold_duplicate_header"}
    reverse: dict[str, list[str]] = {}
    for field, aliases in spec.items():
        for alias in aliases:
            reverse.setdefault(alias, []).append(field)
    roles: dict[str, int] = {}
    unknown = []
    for index, literal in enumerate(header):
        matches = reverse.get(literal, [])
        if len(matches) != 1:
            unknown.append(index)
        else:
            role = matches[0]
            if role in roles:
                return {**header_info, "status": "hold_ambiguous_role"}
            roles[role] = index
    if unknown or set(roles) != set(spec):
        return {**header_info, "status": "hold_unknown_or_missing_role"}
    return {**header_info, "status": "candidate_headers_mapped", "roles": roles,
            "role_headers": {role: header[index] for role, index in roles.items()}}


def parse_metadata(path: Path, filename: str, header: dict[str, object]) -> list[dict[str, str]]:
    if header.get("status") != "candidate_headers_mapped":
        raise GateStop("data_rows_blocked_by_header")
    roles = header["roles"]
    opaque_index = roles[OPAQUE[filename]]
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            next(reader)
            for row in reader:
                if not row:
                    continue
                if len(row) != len(roles):
                    raise GateStop("row_width_mismatch")
                opaque = row[opaque_index]
                row[opaque_index] = ""
                del opaque
                record: dict[str, str] = {}
                for role, index in roles.items():
                    if role == OPAQUE[filename]:
                        continue
                    value = row[index]
                    if role == "day":
                        decimal_day(value)
                    if role in ("sampled_volume", "plated_volume"):
                        try:
                            number = Decimal(value)
                            if not number.is_finite() or number <= 0:
                                raise ValueError
                        except (ValueError, InvalidOperation) as exc:
                            raise GateStop("volume_metadata_invalid") from exc
                    record[role] = value
                rows.append(record)
                del row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GateStop("csv_metadata_invalid") from exc
    return rows


def build_map(ciliate: list[dict[str, str]], serra: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, dict[str, object]] = {}
    combos: dict[tuple[str, str, str], set[str]] = {}
    serra_groups: dict[str, dict[str, object]] = {}
    serra_combos: dict[tuple[str, str, str], set[str]] = {}
    for rows, groups, arms, exposure_keys in (
        (ciliate, grouped, combos, ("sampled_volume",)),
        (serra, serra_groups, serra_combos, ("plated_volume", "dilution")),
    ):
        for row in rows:
            rep = row.get("rep_id", "")
            flags = (row.get("prey", ""), row.get("predator", ""), row.get("parasite", ""))
            arms.setdefault(flags, set()).add(rep)
            if not rep:
                continue
            group = groups.setdefault(rep, {"treatments": set(), "flags": set(), "days": {}, "exposures": set()})
            group["treatments"].add(row.get("treatment", ""))
            group["flags"].add(flags)
            species = row.get("species", "")
            group["days"].setdefault(species, set()).add(decimal_day(row.get("day", "")))
            group.setdefault("day_tokens", {}).setdefault(species, set()).add(row.get("day", ""))
            for key in exposure_keys:
                if key in row:
                    group["exposures"].add((key, row[key]))
    ciliate_reps, serra_reps = set(grouped), set(serra_groups)
    def combos_out(source):
        return [{"prey": key[0], "predator": key[1], "parasite": key[2], "rep_ids": sorted(value)}
                for key, value in sorted(source.items())]
    def units_out(source):
        mapped = {}
        for rep, group in source.items():
            species_days = {}
            for species, values in group["days"].items():
                ordered = sorted(values)
                species_days[species] = {"distinct_numeric_days": [str(value.normalize()) for value in ordered],
                                         "distinct_day_count": len(ordered),
                                         "original_day_tokens": sorted(group["day_tokens"][species])}
            mapped[rep] = {"treatments": sorted(group["treatments"]),
                           "flag_combinations": [list(item) for item in sorted(group["flags"])],
                           "species_days": species_days, "exposures": sorted([list(x) for x in group["exposures"]]),
                           "planned_serra_missingness": "unknown_no_schedule" if source is serra_groups else None}
        return mapped
    conflicts = {rep: {"treatments": sorted(group["treatments"]),
                       "flag_combinations": [list(item) for item in sorted(group["flags"])]}
                 for rep, group in grouped.items() if len(group["treatments"]) > 1 or len(group["flags"]) > 1}
    serra_conflicts = {rep: {"treatments": sorted(group["treatments"]),
                             "flag_combinations": [list(item) for item in sorted(group["flags"])]}
                       for rep, group in serra_groups.items()
                       if len(group["treatments"]) > 1 or len(group["flags"]) > 1}
    joins = {"join_key_candidate": "literal repID only", "validation": "unresolved",
             "literal_value_overlap": sorted(ciliate_reps & serra_reps),
             "ciliate_without_literal_serra_match": sorted(ciliate_reps - serra_reps),
             "serra_without_literal_ciliate_match": sorted(serra_reps - ciliate_reps),
             "ciliate_rows_without_rep_id": sum(not row.get("rep_id", "") for row in ciliate),
             "serra_rows_without_rep_id": sum(not row.get("rep_id", "") for row in serra)}
    unknown_labels = {}
    for flag in ("prey", "predator", "parasite"):
        observed = {row.get(flag, "") for row in ciliate + serra}
        unknown = sorted(value for value in observed if value not in ("yes", "no"))
        if unknown:
            unknown_labels[flag] = unknown
    if unknown_labels:
        arm_coverage = {"status": "not_evaluated_unknown_labels", "unknown_literal_labels": unknown_labels}
    else:
        arms = {}
        for predator in ("no", "yes"):
            for parasite in ("no", "yes"):
                name = f"predator={predator};parasite={parasite}"
                rows = [row for row in ciliate if (row["prey"], row["predator"], row["parasite"])
                        == ("yes", predator, parasite)]
                reps = sorted({row["rep_id"] for row in rows if row.get("rep_id", "")})
                missing_ids = sum(not row.get("rep_id", "") for row in rows)
                unit_status = {}
                for rep in reps:
                    group = grouped[rep]
                    para_days = len(group["days"].get("PARA", set()))
                    didi_days = len(group["days"].get("DIDI", set()))
                    conflicts_for_rep = rep in conflicts
                    treatment_missing = "" in group["treatments"]
                    para_ok = para_days >= 3
                    didi_required = predator == "yes"
                    didi_ok = not didi_required or didi_days >= 3
                    passed = para_ok and didi_ok and not conflicts_for_rep and not treatment_missing
                    unit_status[rep] = {"para_distinct_days": para_days, "para_pass": para_ok,
                                        "didi_distinct_days": didi_days, "didi_required": didi_required,
                                        "didi_pass": didi_ok, "treatment_or_flag_conflict": conflicts_for_rep,
                                        "treatment_missing": treatment_missing,
                                        "coverage_status": "pass" if passed else "HOLD"}
                arm_pass = len(reps) >= 2 and missing_ids == 0 and all(
                    item["coverage_status"] == "pass" for item in unit_status.values())
                arms[name] = {"prey": "yes", "predator": predator, "parasite": parasite,
                              "unit_count": len(reps), "units": unit_status,
                              "rows_without_rep_id": missing_ids,
                              "coverage_status": "pass" if arm_pass else "HOLD"}
        arm_coverage = {"status": "screening_coverage_pass" if all(
            item["coverage_status"] == "pass" for item in arms.values()) else "HOLD", "arms": arms}
    fit_reasons = ["exposure units", "sampling/removal/replacement", "microcosm independence", "cross-file joins"]
    if unknown_labels:
        fit_reasons.append("label encoding")
    if arm_coverage["status"] != "screening_coverage_pass":
        fit_reasons.append("coverage screen not passed")
    return {"ciliate_rows": ciliate, "serra_rows": serra,
            "raw_flag_combinations": combos_out(combos), "serra_raw_flag_combinations": combos_out(serra_combos),
            "rep_ids": sorted(grouped), "replicate_map": units_out(grouped),
            "serra_unit_map": units_out(serra_groups),
            "treatment_or_flag_conflicts": conflicts, "serra_treatment_or_flag_conflicts": serra_conflicts,
            "serra_join": joins, "serra_observed_coverage": {"by_literal_flag_combination": combos_out(serra_combos),
                                                                 "units": units_out(serra_groups),
                                                                 "planned_missingness": "unknown_no_schedule"},
            "candidate_arm_coverage": arm_coverage,
            "label_encoding": "unresolved_literal_labels_preserved",
            "exposure_units": "unresolved_literal_values_preserved", "fit_readiness": "HOLD",
            "fit_hold_reasons": fit_reasons}


def finalize_receipt(receipt: dict[str, object], client: Client, prov_dir: Path, raw_dir: Path,
                     started: float) -> dict[str, object]:
    out = prov_dir / "receipt.json"
    finalization_started = time.monotonic()
    try:
        remaining = MAX_SECONDS - (time.monotonic() - started)
        with alarm_limit(max(0.001, remaining)):
            data_after = active_bytes()
            if sum(data_after) + RECEIPT_RESERVE >= MAX_DATA:
                receipt["status"] = "held_active_data_cap_after"
                receipt["failure_code"] = "active_data_cap_after"
            completion_status = receipt["status"]
            receipt["resource"].update({
                "data_bytes_after": data_after[0], "artifact_bytes_after": data_after[1],
                "data_after_measurement_point": "after acquisition, before receipt finalization",
                "downloaded_body_bytes": client.total_body,
                "request_counts": client.calls,
                "peak_sampled_rss_bytes": max((item["rss_high_water_bytes"] for item in client.rss_samples), default=0),
                "elapsed_seconds_before_receipt_finalization": round(time.monotonic() - started, 6),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "rss_samples": client.rss_samples,
            })
            receipt["actual_requests"] = len(client.request_log)
            receipt["outcome_retention"] = {
                "interpreted_metadata_records": {"converted": False, "retained": False, "emitted": False},
                "raw_source_files_retained_locally": [
                    name for name, info in receipt["files"].items()
                    if info.get("path") and (ROOT / info["path"]).is_file()
                ],
                "raw_store_path": str(raw_dir.relative_to(ROOT)),
            }
            receipt["frozen_inputs"] = receipt.get("frozen_inputs") or {
                str(path.relative_to(ROOT)): PINNED[path] for path in PINNED
            }
            receipt["resource"]["receipt_finalization_seconds_through_first_write"] = None
            receipt["resource"]["elapsed_seconds_before_last_atomic_receipt_update"] = None
            receipt["resource"]["elapsed_measurement_point"] = (
                "includes first receipt write; excludes the final atomic receipt replacement"
            )
            receipt["status"] = "receipt_finalization_pending"
            receipt["completion_status"] = completion_status
            for index in range(2):
                current = tree_bytes((prov_dir, raw_dir))
                prior_size = out.stat().st_size if out.exists() else 0
                for _ in range(4):
                    body = json_bytes(receipt)
                    retained = current - prior_size + len(body)
                    peak = max(client.transaction_peak, current + len(body))
                    receipt["resource"]["retained_run_bytes"] = retained
                    receipt["resource"]["transaction_peak_bytes"] = peak
                    if len(json_bytes(receipt)) == len(body):
                        break
                body = json_bytes(receipt)
                if len(body) > 128 * 1024:
                    raise GateStop("receipt_size_cap")
                if current + len(body) >= MAX_TRANSACTION:
                    raise GateStop("receipt_transaction_cap")
                client.transaction_peak = max(client.transaction_peak, current + len(body))
                write_atomic(out, body, prov_dir, raw_dir, reserve=0)
                if index == 0:
                    receipt["resource"]["receipt_finalization_seconds_through_first_write"] = round(
                        time.monotonic() - finalization_started, 6
                    )
                    receipt["resource"]["elapsed_seconds_before_last_atomic_receipt_update"] = round(
                        time.monotonic() - started, 6
                    )
                    receipt["resource"]["elapsed_seconds"] = round(time.monotonic() - started, 6)
                    receipt["status"] = completion_status
            receipt["resource"]["elapsed_seconds_after_last_receipt_write"] = round(time.monotonic() - started, 6)
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["failure_code"] = "keyboard_interrupt_during_finalization"
    except Exception as exc:
        receipt["status"] = "held_receipt_finalization"
        receipt["failure_code"] = "wall_deadline" if isinstance(exc, TimeoutError) else (
            str(exc) if isinstance(exc, GateStop) else "receipt_finalization_failure"
        )
    receipt["resource"]["elapsed_seconds_after_finalization_attempt"] = round(time.monotonic() - started, 6)
    return receipt


def exit_code(receipt: dict[str, object]) -> int:
    return 0 if receipt.get("status") == "source_gate_complete_fit_hold" else 1


def run() -> dict[str, object]:
    prov_dir = PROV_ROOT / RUN_ID
    raw_dir = RAW_ROOT / RUN_ID
    validate_storage_roots()
    safe_storage_path(ROOT, prov_dir, Path("data/real/ecology/source_provenance") / RUN_ID)
    safe_storage_path(ROOT, raw_dir, Path("data/real/ecology/raw") / RUN_ID)
    if prov_dir.exists() or raw_dir.exists():
        raise GateStop("run_id_consumed")
    started = time.monotonic()
    PROV_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    validate_storage_roots()
    prov_dir.mkdir(parents=True, exist_ok=False)
    try:
        raw_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        raise GateStop("raw_run_directory_unavailable")
    validate_storage_roots()
    safe_storage_path(ROOT, prov_dir, Path("data/real/ecology/source_provenance") / RUN_ID)
    safe_storage_path(ROOT, raw_dir, Path("data/real/ecology/raw") / RUN_ID)
    client = Client(prov_dir, raw_dir, started=started)
    code_hash = sha256_file(Path(__file__))
    receipt: dict[str, object] = {
        "run_id": RUN_ID, "status": "started", "stage": "source_gate", "fit_readiness": "HOLD",
        "source": None, "files": {}, "requests": client.request_log,
        "frozen_inputs": {}, "code_sha256": code_hash, "map_sha256": None, "map_path": None,
        "limits": {"metadata_gets": MAX_META_GETS, "file_gets": MAX_FILE_GETS, "requests": MAX_REQUESTS,
                   "metadata_bytes_each": MAX_META, "file_bytes_each": MAX_FILE, "all_downloaded_bytes": MAX_BODY,
                   "wall_seconds": MAX_SECONDS, "get_seconds_each": MAX_GET_SECONDS,
                   "sampled_rss_bytes": MAX_RSS, "transaction_bytes": MAX_TRANSACTION},
        "resource": {"data_bytes_before": None, "data_bytes_after": None, "artifact_bytes_before": None,
                     "artifact_bytes_after": None, "retained_run_bytes": None, "transaction_peak_bytes": 0,
                     "data_after_measurement_point": "after acquisition, before final receipt write",
                     "downloaded_body_bytes": 0, "elapsed_seconds": None, "rss_samples": client.rss_samples,
                     "rss_method": "sampled resource.ru_maxrss OS high-water; checkpoint samples, not a cgroup hard kill",
                     "rss_hard_limit": "checked at network/parser checkpoints"},
        "calls": {"model": 0, "simulator": 0, "likelihood": 0, "forecast": 0},
        "missing_costs": ["remote server processing is not observable"],
        "header_roles_verified_by_source": False,
    }
    try:
        with alarm_limit(max(0.001, MAX_SECONDS - FINALIZE_RESERVE_SECONDS - (time.monotonic() - started))):
            data_before = active_bytes()
            receipt["frozen_inputs"] = check_frozen()
            receipt["resource"]["data_bytes_before"], receipt["resource"]["artifact_bytes_before"] = data_before
            if sum(data_before) + MAX_TRANSACTION >= MAX_DATA:
                raise GateStop("active_data_cap_before")
            dataset, _ = client.get_json(DATASET_URL, prov_dir / "metadata-01.json")
            if not dataset_identity(dataset):
                raise GateStop("dataset_identity_mismatch")
            version_url = link_url(dataset, "stash:version")
            version_match = API_VERSION_RE.fullmatch(urlsplit(version_url).path)
            if not version_match or version_match.group(2):
                raise GateStop("version_link_shape_invalid")
            version, _ = client.get_json(version_url, prov_dir / "metadata-02.json")
            version_id = version_match.group(1)
            if str(version.get("id")) != version_id:
                raise GateStop("version_identity_mismatch")
            publication_date = version.get("publicationDate")
            if not isinstance(publication_date, str) or publication_date[:10] != EXPECTED_DATE:
                raise GateStop("version_date_mismatch")
            license_value, license_source = resolve_license(dataset, version)
            files_url = link_url(version, "stash:files")
            files_match = API_VERSION_RE.fullmatch(urlsplit(files_url).path)
            if not files_match or files_match.group(1) != version_id or files_match.group(2) != "/files":
                raise GateStop("file_list_link_shape_invalid")
            file_doc, _ = client.get_json(files_url, prov_dir / "metadata-03.json")
            source = {"doi": DOI, "version_id": version_id, "publication_date": publication_date,
                      "license": license_value, "license_source": license_source}
            receipt["source"] = source
            metadata = selected_files(file_doc)
            paths: dict[str, Path] = {}
            for name in FILES:
                item = metadata[name]
                expected = {**item, "filename": name, "source_path": item["path"],
                            "status": "not_requested", "path": None, "sha256": None, "reused": False}
                receipt["files"][name] = expected
                path = existing_file(expected, source)
                if path is None:
                    url = f"{BASE}/downloads/file_stream/{item['file_id']}"
                    target = raw_dir / name
                    try:
                        client.fetch(url, "file", target)
                    except GateStop:
                        request = client.request_log[-1]
                        expected.update({"status": "partial" if request.get("body_bytes") else "failed",
                                         "path": request.get("partial_path"),
                                         "partial_bytes": request.get("body_bytes"),
                                         "partial_sha256": request.get("body_sha256")})
                        raise
                    path = target
                body_sha = sha256_file(path)
                expected.update({"status": "integrity_check", "path": str(path.relative_to(ROOT)),
                                 "sha256": body_sha, "reused": path.parent != raw_dir})
                if path.stat().st_size != item["bytes"]:
                    expected["status"] = "integrity_failed"
                    raise GateStop("file_size_mismatch")
                if "sha256" in item["digests"] and item["digests"]["sha256"] != body_sha:
                    expected["status"] = "integrity_failed"
                    raise GateStop("sha256_mismatch")
                md5 = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
                if "md5" in item["digests"] and item["digests"]["md5"] != md5:
                    expected["status"] = "integrity_failed"
                    raise GateStop("md5_mismatch")
                expected.update({"status": "verified", "path": str(path.relative_to(ROOT)), "sha256": body_sha,
                                 "observed_md5": md5, "reused": path.parent != raw_dir})
                paths[name] = path
            client.sample()
            headers = {name: audit_header(paths[name], name) for name in FILES}
            if any(item["status"] != "candidate_headers_mapped" for item in headers.values()):
                mapped = {"schema_state": "HOLD_candidate_headers_not_uniquely_mapped",
                          "headers": headers, "fit_readiness": "HOLD", "rows_interpreted": False,
                          "fit_hold_reasons": ["candidate CSV header schema unresolved"]}
            else:
                records = {}
                for name in FILES:
                    client.sample()
                    records[name] = parse_metadata(paths[name], name, headers[name])
                    client.sample()
                mapped = build_map(records[FILES[0]], records[FILES[1]])
                mapped.update({"schema_state": "candidate_headers_mapped_not_source_verified",
                               "headers": headers, "rows_interpreted": True})
            mapped_bytes = json_bytes(mapped)
            if len(mapped_bytes) > 1024 * 1024:
                raise GateStop("metadata_map_size_cap")
            map_path = prov_dir / "metadata-map.json"
            client.transaction_peak = max(client.transaction_peak,
                                          write_atomic(map_path, mapped_bytes, prov_dir, raw_dir))
            receipt["map_path"] = str(map_path.relative_to(ROOT))
            receipt["map_sha256"] = sha256_bytes(mapped_bytes)
            receipt["status"] = "source_gate_complete_fit_hold" if mapped.get("rows_interpreted") else "source_gate_hold_schema"
            receipt["header_audit"] = headers
            receipt["opaque_outcomes"] = {
                "interpreted_metadata_records": {"converted": False, "retained": False, "emitted": False},
                "raw_source_bytes_retained_locally": [str(paths[name].relative_to(ROOT)) for name in FILES],
            }
            if time.monotonic() - started > MAX_SECONDS:
                raise GateStop("wall_deadline")
            client.sample()
            if sha256_file(Path(__file__)) != code_hash or check_frozen() != receipt["frozen_inputs"]:
                raise GateStop("frozen_binding_changed")
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["failure_code"] = "keyboard_interrupt"
    except Exception as exc:
        receipt["status"] = "held_failure"
        receipt["failure_code"] = str(exc) if isinstance(exc, GateStop) else "wall_deadline" if isinstance(exc, TimeoutError) else "client_failure"
    return finalize_receipt(receipt, client, prov_dir, raw_dir, started)


if __name__ == "__main__":
    try:
        result = run()
    except KeyboardInterrupt:
        result = {"status": "interrupted", "failure_code": "keyboard_interrupt_before_receipt"}
    except GateStop as exc:
        result = {"status": "held_failure", "failure_code": str(exc)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(exit_code(result))
