#!/usr/bin/env python3
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import dataclasses
import hashlib
import json
import math
import re
import resource
import signal
import stat
import sys
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core.real_data.cascaded_tanks_pattern_search import run_normalized_cube_pattern_search
from core.abc_smc_reference import make_uniform_prior, run_gaussian_abc_smc_reference

PROTO = ROOT / "reports/yeast-diauxic-inference-protocol-2026-10-30.md"
PROTO_SHA = "776ebf01439bce3f642fb4dbe3d3e053db8c0138a545ab24930f235af5b2ad16"
POINT_SOURCE = ROOT / "core/real_data/cascaded_tanks_pattern_search.py"
POINT_SOURCE_SHA = "c7800c76c73b5d6063a6e15ea45cd756e7a542b7ec584e99a18108e3ea914458"
ABC_SOURCE = ROOT / "core/abc_smc_reference.py"
ABC_SOURCE_SHA = "48aaa39ee6fb7077de5c3410cc67fc70453000364346524f2ba932923767d2f9"
ARCHIVE = Path("/Users/sourabh/Downloads/Dataset+1.zip")
ARCHIVE_SHA = "2ac81bce8313c140487e51a0ac373239ff96a88a670f8ee5e5037f7d738c28ab"
RAW = ROOT / "artifacts/deferred_work/yeast-diauxic-raw-metadata-20261029-v1/receipt.json"
RAW_SHA = "5af91336cc1bd8ebf88976aee49b38489ef8d0b63458e814f61395623165baf2"
CLOCK = ROOT / "artifacts/deferred_work/yeast-diauxic-time-contract-20261029-v1/receipt.json"
CLOCK_SHA = "d42c4223bae1009f1dc6f1a243718060b2b10127c2374aff6dda9a1701d838fb"
LAYOUT = ROOT / "artifacts/deferred_work/yeast-diauxic-mapping-text-20261029-v2/receipt.json"
LAYOUT_SHA = "a98f8ae8f2794bfa8049d50fa353c1e93358d58eb172026f8d61a047b7adeeac"
OUT_ROOT = ROOT / "artifacts/deferred_work"
TRAIN_ID, EVAL_ID = "yeast-diauxic-pilot-train-20261030-v1", "yeast-diauxic-pilot-eval-20261030-v1"
SRC_SIZE, MAX_MEMBER = 1594286, 128 * 1024
MAX_RECORD, MAX_RECORDS, MAX_TOTAL = 8 * 1024, 256, 16 * 1024**2
MAX_NEW, MAX_RECEIPT, MAX_RSS = 64 * 1024**2, 1024 * 1024, 1024**3
MAX_STORAGE = 50 * 1024**3 - 64 * 1024**2
WELLS, STRAINS = ("B2", "B3", "B4", "B5"), ("DBVPG6765", "SK1", "Y55", "YPS128")
CASES = tuple((w, m) for w in WELLS for m in ("one_phase", "two_phase"))
T0 = datetime.strptime("7/19/2013 5:10:58 PM", "%m/%d/%Y %I:%M:%S %p")
T = 158967 / 3600.0
SIG_E, SIG_P, ELL = 0.005, 0.010, 1.0
BG, FOOT = b"Background information", b"Assay Finished: "
CHANNELS = {"OD600": "OD600", "IW-OD600": "OD600"}
NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
EPS, ATTEMPTS = (16.0, 4.0, 1.5), (4096, 8192, 16384)
BLOCK, POINT_BUDGET = 8192, 4096
CURVE_KEYS = ("curve_calls_point", "curve_calls_finite_difference", "curve_calls_importance",
              "curve_calls_prediction", "curve_calls_abc", "curve_calls_score")
COUNTER_KEYS = ("point_calls", "point_likelihood_calls", "curve_calls_point", "curve_calls_finite_difference",
                "importance_proposal_density_calls", "importance_likelihood_calls", "curve_calls_importance",
                "curve_calls_prediction", "curve_calls_abc", "score_likelihood_calls", "curve_calls_score",
                "simulations", "discrepancy_calls", "abc_proposals")

STATE: dict[str, Any] = {}
T0_MONO = DEADLINE = LIMIT = 0.0
LATCH: dict[str, str] | None = None


class Stop(Exception):
    def __init__(self, status: str, code: str):
        self.status, self.code = status, code


def stop(status: str, code: str) -> None:
    raise Stop(status, code)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rss() -> int:
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform == "darwin" else n * 1024)


def wall_alarm(_sig: int, _frame: Any) -> None:
    global LATCH
    LATCH = LATCH or {"status": "stopped_wall_limit", "code": "processing_wall_limit"}
    stop(LATCH["status"], LATCH["code"])


def resources() -> None:
    global LATCH
    if time.monotonic() > DEADLINE:
        LATCH = LATCH or {"status": "stopped_wall_limit", "code": "processing_wall_limit"}
        stop(LATCH["status"], LATCH["code"])
    STATE["sampled_peak_rss_bytes"] = max(STATE.get("sampled_peak_rss_bytes", 0), rss())
    if STATE["sampled_peak_rss_bytes"] > MAX_RSS:
        LATCH = LATCH or {"status": "stopped_rss_limit", "code": "sampled_rss_cap_exceeded"}
        stop(LATCH["status"], LATCH["code"])


def safe(x: Any) -> Any:
    if dataclasses.is_dataclass(x): return safe(dataclasses.asdict(x))
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (bool, np.bool_)): return bool(x)
    return x


def write_json(path: Path, data: Any, cap: int | None = None) -> None:
    raw = json.dumps(safe(data), ensure_ascii=True, allow_nan=False,
                     sort_keys=True, separators=(",", ":")).encode("ascii")
    if cap is not None and len(raw) > cap: stop("stopped_receipt_limit", "json_artifact_exceeds_cap")
    if STATE.get("out_owned") and path.parent == OUT_ROOT/STATE["run_id"]:
        old = path.stat().st_size if path.exists() else 0
        reserve = 0 if path.name == "receipt.json" else MAX_RECEIPT
        if stage_artifact_bytes(path.parent)-old+len(raw)+reserve > MAX_NEW:
            stop("stopped_artifact_limit", "new_artifact_bytes_including_final_receipt")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_npz(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    if STATE.get("out_owned") and path.parent == OUT_ROOT/STATE["run_id"]:
        old = path.stat().st_size if path.exists() else 0
        if stage_artifact_bytes(path.parent)-old+MAX_RECEIPT > MAX_NEW:
            tmp.unlink(); stop("stopped_artifact_limit", "new_artifact_bytes_including_final_receipt")
    os.replace(tmp, path)


def tree_bytes(root: Path) -> int:
    if not root.exists(): return 0
    if root.is_symlink() or not root.is_dir(): stop("stopped_storage_check", "storage_root_invalid")
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        for name in files:
            p = Path(base) / name
            if stat.S_ISREG(p.lstat().st_mode): total += p.lstat().st_size
    return total


def stage_artifact_bytes(out: Path) -> int:
    return tree_bytes(out) + STATE.get("preexisting_artifact_bytes", 0)


def check_artifact_reserve(out: Path) -> None:
    if stage_artifact_bytes(out) + MAX_RECEIPT > MAX_NEW:
        stop("stopped_artifact_limit", "new_artifact_bytes_including_final_receipt")


def curve_count(counters: dict[str, Any]) -> int:
    return sum(int(c.get(key, 0)) for c in counters.values() for key in CURVE_KEYS)


def storage_snapshot() -> dict[str, Any]:
    roots = {"data": ROOT / "data", "artifacts": ROOT / "artifacts"}
    used = {k: tree_bytes(v) for k, v in roots.items()}
    originals = [ARCHIVE, Path("/Users/sourabh/Downloads/README_for_Dataset+1.txt"),
                 Path("/Users/sourabh/Downloads/MATLAB+analysis+code.zip")]
    sizes = []
    for p in originals:
        if not stat.S_ISREG(p.lstat().st_mode): stop("stopped_storage_check", "download_original_not_regular")
        sizes.append({"path": str(p), "bytes": p.stat().st_size})
    total = sum(used.values()) + sum(v["bytes"] for v in sizes)
    if total + MAX_RECEIPT > MAX_STORAGE: stop("stopped_storage_limit", "storage_reserve_would_be_breached")
    return {**used, "downloads_originals": sizes, "included_total": total, "limit_bytes": MAX_STORAGE}


def read_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for p, digest, code in ((PROTO, PROTO_SHA, "protocol_hash"), (POINT_SOURCE, POINT_SOURCE_SHA, "point_source_hash"),
                            (ABC_SOURCE, ABC_SOURCE_SHA, "abc_source_hash"), (RAW, RAW_SHA, "raw_receipt_hash"),
                            (CLOCK, CLOCK_SHA, "clock_receipt_hash"), (LAYOUT, LAYOUT_SHA, "layout_receipt_hash")):
        if sha(p) != digest: stop("stopped_input_hash", code + "_mismatch")
    raw, clock, layout = (json.loads(p.read_text(encoding="utf-8")) for p in (RAW, CLOCK, LAYOUT))
    if (raw.get("run_id") != "yeast-diauxic-raw-metadata-20261029-v1"
            or raw.get("status") != "complete_metadata_only" or len(raw.get("members", [])) != 542
            or raw.get("completed_member_count") != 542): stop("stopped_input_schema", "raw_receipt_identity")
    if (clock.get("run_id") != "yeast-diauxic-time-contract-20261029-v1"
            or clock.get("status") != "complete_naive_clock_only" or clock.get("member_count") != 542
            or clock.get("input", {}).get("sha256") != RAW_SHA): stop("stopped_input_schema", "clock_receipt_identity")
    layouts = [m for m in layout.get("members", []) if m.get("name", "").endswith("/strains.csv")]
    if len(layouts) != 1 or layouts[0].get("sha256") != "4c4237958818caf5a5b33e760760adc64b074b6b8f8eb07ce0e629ae74b85c0e":
        stop("stopped_input_schema", "layout_identity")
    grid = [line.split(",") for line in layouts[0].get("text", "").splitlines()]
    if len(grid) != 8 or tuple(grid[1][1:5]) != STRAINS: stop("stopped_input_schema", "frozen_well_labels")
    return raw, clock


def ordered(raw: dict[str, Any], clock: dict[str, Any], plate: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = [m for m in raw["members"] if m.get("plate_id") == plate]
    by_suffix = {m["filename_suffix"]: m for m in selected}
    stamps = clock["plates"][plate]
    suffix_text = stamps["timeline_suffixes_6digit"]
    suffixes = [suffix_text[i:i + 6] for i in range(0, len(suffix_text), 6)]
    timestamp_text = stamps["timeline_timestamps"]
    if len(selected) != 271 or len(by_suffix) != 271 or len(suffix_text) != 1626 or len(set(suffixes)) != 271 or len(timestamp_text) != 271:
        stop("stopped_input_schema", "timestamp_member_count")
    members, times = [], []
    for suffix, text in zip(suffixes, timestamp_text, strict=True):
        m = by_suffix.get(suffix)
        name = f"Dataset 1/glucose + galactose/OD600_shake_iw_plate{plate[-1]}_{suffix}.csv"
        if m is None or m.get("name") != name or m.get("assay_finished_timestamp") != text or not m.get("crc_verified"):
            stop("stopped_input_schema", "timestamp_member_binding")
        members.append(m)
        times.append((datetime.strptime(text, "%m/%d/%Y %I:%M:%S %p") - T0).total_seconds() / 3600)
    t = np.asarray(times)
    if np.any(np.diff(t) <= 0): stop("stopped_input_schema", "time_order")
    if plate == "plate1" and not np.isclose(t[-1], T, rtol=0, atol=1e-12): stop("stopped_input_schema", "training_horizon")
    if plate == "plate2" and (not np.isclose(t[0], 113 / 3600, atol=1e-12) or not np.isclose(t[-1], 159152 / 3600, atol=1e-12)):
        stop("stopped_input_schema", "evaluation_horizon")
    return members, t


def record_body(row: bytes) -> bytes:
    return row[:-2] if row.endswith(b"\r\n") else row[:-1] if row.endswith((b"\r", b"\n")) else row


def records(data: bytes) -> list[bytes]:
    rows, start, i = [], 0, 0
    while i < len(data):
        if data[i] not in (10, 13): i += 1; continue
        end = i + 1
        if data[i] == 13 and end < len(data) and data[end] == 10: end += 1
        if end - start > MAX_RECORD: stop("stopped_record_limit", "record_too_large")
        rows.append(data[start:end]); start = i = end
        if len(rows) > MAX_RECORDS: stop("stopped_record_limit", "record_count")
    if start < len(data):
        if len(data) - start > MAX_RECORD: stop("stopped_record_limit", "record_too_large")
        rows.append(data[start:])
    if len(rows) > MAX_RECORDS: stop("stopped_record_limit", "record_count")
    return rows


def first(row: bytes) -> bytes: return record_body(row).split(b",", 1)[0]


def parse_member(data: bytes, expected: dict[str, Any]) -> list[float]:
    if len(data) != expected["declared_bytes"] or len(data) > MAX_MEMBER: stop("stopped_member_size", "member_size")
    if f"{zlib.crc32(data) & 0xffffffff:08x}" != expected["declared_crc32"]: stop("stopped_member_crc", "member_crc")
    rows = records(data)
    marker = next((i for i, r in enumerate(rows) if first(r) == BG), None)
    if marker is None or any(first(r) == FOOT for r in rows[:marker]): stop("stopped_member_schema", "background_marker")
    meta = rows[marker + 1:marker + 4]
    if len(meta) != 3 or any(first(r) in (BG, FOOT) for r in meta): stop("stopped_member_schema", "background_metadata")
    channel = record_body(meta[1]).split(b",")
    if len(channel) < 2 or b'"' in b",".join(channel[:2]) or b"\\" in b",".join(channel[:2]):
        stop("stopped_member_schema", "channel_shape")
    token = channel[1].decode("ascii", "strict")
    if token not in CHANNELS or token != expected["background_channel_source_token"] or CHANNELS[token] != expected["canonical_channel"]:
        stop("stopped_member_schema", "channel_binding")
    start = marker + 4
    matrix = rows[start:start + 8]
    if len(matrix) != 8: stop("stopped_member_schema", "matrix_record_count")
    values = None
    for row_i, row in enumerate(matrix):
        raw = record_body(row)
        if b'"' in raw or b"\\" in raw: stop("stopped_member_schema", "matrix_quote_escape")
        cells = raw.split(b",")
        if len(cells) == 13 and cells[-1] == b"": cells.pop()
        if len(cells) != 12: stop("stopped_member_schema", "matrix_field_count")
        if row_i == 1:
            chosen = cells[1:5]
            if any(not NUMBER.fullmatch(c) for c in chosen): stop("stopped_selected_value", "selected_numeric_syntax")
            values = [float(c.decode("ascii")) for c in chosen]
            if not np.all(np.isfinite(values)): stop("stopped_selected_value", "selected_numeric_nonfinite")
    footer = None
    for row in rows[start + 8:]:
        key = first(row)
        if key == BG: stop("stopped_member_schema", "repeated_background_before_footer")
        if key == FOOT: footer = record_body(row); break
    if footer is None: stop("stopped_member_schema", "footer_missing")
    pos = 0
    for _ in range(4):
        comma = footer.find(b",", pos)
        if comma < 0: stop("stopped_member_schema", "footer_timestamp_field")
        pos = comma + 1
    end = footer.find(b",", pos)
    timestamp = footer[pos:] if end < 0 else footer[pos:end]
    if timestamp.decode("ascii", "strict") != expected["assay_finished_timestamp"]:
        stop("stopped_member_schema", "footer_timestamp_binding")
    assert values is not None
    return values


def read_stage(out: Path, members: list[dict[str, Any]], plate: str, archive: zipfile.ZipFile) -> tuple[np.ndarray, list[str]]:
    selected = set(m["name"] for m in members); by_name = {}
    for info in archive.infolist():
        if info.filename in selected:
            if info.filename in by_name: stop("stopped_member_metadata", "duplicate_zip_name")
            by_name[info.filename] = info
    if set(by_name) != selected: stop("stopped_member_metadata", "missing_zip_member")
    for m in members:
        i = by_name[m["name"]]
        if i.file_size != m["declared_bytes"] or i.CRC != int(m["declared_crc32"], 16) or i.file_size > MAX_MEMBER:
            stop("stopped_member_metadata", "zip_central_directory_mismatch")
    values, names, nbytes = [], [], 0
    for m in members:
        resources()
        STATE["current_member"] = m["name"]
        with archive.open(by_name[m["name"]], "r") as f:
            STATE["members_opened"] += 1
            data = f.read(MAX_MEMBER + 1)
        nbytes += len(data); STATE["total_member_bytes_read"] = nbytes
        STATE["members_read"] += 1
        if nbytes > MAX_TOTAL: stop("stopped_read_limit", "selected_bytes_limit")
        parsed = parse_member(data, m)
        values.append(parsed); names.append(m["name"])
        STATE["members_validated"] += 1; STATE["last_member"] = m["name"]
        if plate == "plate1":
            write_npz(out / "training_observations.partial.npz", y=np.asarray(values), names=np.asarray(names, dtype="U128"))
        check_artifact_reserve(out)
    return np.asarray(values), names


def sigmoid(x: Any) -> np.ndarray:
    z = np.asarray(x, dtype=float); out = np.empty_like(z); pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos])); e = np.exp(z[~pos]); out[~pos] = e / (1 + e)
    return out


def curve(x: np.ndarray, times: np.ndarray, model: str) -> np.ndarray:
    x = np.asarray(x, dtype=float); d = 4 if model == "one_phase" else 7
    if x.shape != (d,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
        stop("stopped_numeric", "curve_point_invalid")
    b, a = -0.05 + 0.30*x[0], 1.5*x[1]
    r1, c1 = 0.01*200**x[2], -0.5*T + 2*T*x[3]
    s1 = sigmoid(r1*(times-c1))
    if model == "one_phase": return b + a*s1
    q, r2 = x[4], 0.01*200**x[5]; c2 = c1 + x[6]*(1.5*T-c1)
    return b + a*((1-q)*s1 + q*sigmoid(r2*(times-c2)))


def gaps(times: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0): stop("stopped_numeric", "time_gap_invalid")
    return dt


def innovations(r: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    if r.shape != times.shape or not np.all(np.isfinite(r)): stop("stopped_numeric", "residual_invalid")
    dt, w, fvar = gaps(times), np.empty_like(r), np.empty_like(r)
    m, p = 0.0, SIG_P**2
    for k, val in enumerate(r):
        f = p + SIG_E**2; v = val-m; w[k], fvar[k] = v/math.sqrt(f), f
        mp, pp = m+(p/f)*v, p*SIG_E**2/f
        if k+1 < len(r):
            phi = math.exp(-dt[k]/ELL); m = phi*mp; p = phi*phi*pp + SIG_P**2*(-math.expm1(-2*dt[k]/ELL))
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(fvar)): stop("stopped_numeric", "innovation_nonfinite")
    return w, fvar


def loglike(y: np.ndarray, pred: np.ndarray, times: np.ndarray) -> float:
    w, f = innovations(y-pred, times)
    val = -0.5*(len(y)*math.log(2*math.pi) + np.log(f).sum() + w@w)
    if not math.isfinite(float(val)): stop("stopped_numeric", "loglike_nonfinite")
    return float(val)


def covariance(times: np.ndarray) -> np.ndarray:
    return SIG_P**2*np.exp(-np.abs(times[:, None]-times[None, :])/ELL) + SIG_E**2*np.eye(len(times))


def simulate_ou(rng: np.random.Generator, times: np.ndarray) -> np.ndarray:
    dt = gaps(times); path = np.empty(len(times)); path[0] = rng.normal(0, SIG_P)
    for k, gap in enumerate(dt, 1):
        phi = math.exp(-gap/ELL)
        path[k] = phi*path[k-1]+rng.normal(0, SIG_P*math.sqrt(-math.expm1(-2*gap/ELL)))
    return path


def latch_failure(current: dict[str, str] | None, status: str, code: str) -> dict[str, str]:
    return current or {"status": status, "code": code}


def outer_failure_detail(latched: dict[str, str] | None, callbacks: list[dict[str, Any]]) -> dict[str, Any]:
    cause = latched or {"status": "stopped_abc_runner", "code": "abc_covariance_or_runner_exception"}
    return {"adapter_stop_cause": cause, "outer_exception_captured": True, "callbacks": callbacks}


def normal_logpdf(z: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    l = np.linalg.cholesky(cov); d = np.linalg.solve(l, z-mu)
    return float(-0.5*(len(z)*math.log(2*math.pi)+2*np.log(np.diag(l)).sum()+d@d))


def logq(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(x >= 1): stop("stopped_numeric", "proposal_saturated")
    jac = np.log(x).sum()+np.log1p(-x).sum()
    logln = normal_logpdf(np.log(x)-np.log1p(-x), mu, cov)-float(jac)
    return float(np.logaddexp(math.log(.8)+logln, math.log(.2)))


def draw_q(rng: np.random.Generator, n: int, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    choose = rng.random(n)<.8; x = rng.random((n, len(mu)))
    if choose.any(): x[choose] = sigmoid(rng.multivariate_normal(mu, cov, int(choose.sum())))
    if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(x >= 1): stop("stopped_numeric", "proposal_saturated")
    return x


def lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not math.isfinite(m): stop("stopped_numeric", "logsumexp_nonfinite")
    return m+math.log(float(np.exp(x-m).sum()))


def norm_weights(logw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    z = lse(logw); w = np.exp(logw-z)
    return w, {"log_normalizer": z-math.log(len(w)), "ess": 1/float(w@w), "maximum_weight": float(w.max())}


def proposal(point: np.ndarray, model: str, y: np.ndarray, times: np.ndarray,
             counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(point, .02, .98); z0 = np.log(clipped)-np.log1p(-clipped); jac = np.empty((len(times), len(point)))
    for j in range(len(point)):
        left, right = z0.copy(), z0.copy(); left[j]-=1e-4; right[j]+=1e-4
        jac[:, j] = (curve(sigmoid(right), times, model)-curve(sigmoid(left), times, model))/(2e-4)
        counts["curve_calls_finite_difference"] += 2
    jw = np.column_stack([innovations(jac[:, j], times)[0] for j in range(len(point))])
    vals, vecs = np.linalg.eigh(jw.T@jw + .25*np.eye(len(point)))
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0): stop("stopped_numeric", "proposal_metric")
    cov = (vecs*np.clip(2/vals, 1e-6, 4))@vecs.T
    if not np.all(np.isfinite(cov)): stop("stopped_numeric", "proposal_covariance")
    return z0, cov


def mean_curve(x: np.ndarray, w: np.ndarray, t: np.ndarray, model: str,
               counts: dict[str, int]) -> np.ndarray:
    out = np.zeros(len(t))
    for i in range(0, len(x), 64):
        resources(); b = x[i:i+64]; curves = np.stack([curve(p, t, model) for p in b])
        counts["curve_calls_prediction"] += len(b); out += w[i:i+len(b)]@curves
    return out


def read_archive() -> zipfile.ZipFile:
    if ARCHIVE.stat().st_size != SRC_SIZE or sha(ARCHIVE) != ARCHIVE_SHA:
        stop("stopped_input_hash", "archive_size_or_hash")
    return zipfile.ZipFile(ARCHIVE, "r")


def ready_freeze(m: dict[str, Any], code_sha: str) -> bool:
    cases = m.get("cases", [])
    return (m.get("status") == "complete_training_frozen" and m.get("complete") is True
            and m.get("protocol_sha256") == PROTO_SHA and m.get("command_sha256") == code_sha
            and m.get("point_source_sha256") == POINT_SOURCE_SHA and m.get("abc_source_sha256") == ABC_SOURCE_SHA
            and m.get("archive_sha256") == ARCHIVE_SHA and m.get("archive_size_bytes") == SRC_SIZE
            and m.get("raw_metadata_receipt_sha256") == RAW_SHA
            and m.get("clock_receipt_sha256") == CLOCK_SHA
            and m.get("layout_receipt_sha256") == LAYOUT_SHA
            and m.get("training_member_count") == 271 and m.get("plate2_members_opened") == 0
            and re.fullmatch(r"[0-9a-f]{64}", m.get("training_final_receipt_sha256", ""))
            and len(cases) == 8
            and all(c.get("index") == j and c.get("well") == WELLS[j//2]
                    and c.get("model") == CASES[j][1] and c.get("status") == "complete"
                    and not c.get("quality_gate_failures")
                    for j, c in enumerate(cases))
            and {o.get("path") for o in m.get("outputs", [])} ==
                {"observations_train.npz", "training_freeze_data.npz", "training_diagnostics.json",
                 "receipt.json"}
            and next((o.get("sha256") for o in m.get("outputs", []) if o.get("path") == "receipt.json"), None)
                == m.get("training_final_receipt_sha256"))


def verify_freeze(expected: str, code_sha: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected or ""): stop("stopped_freeze", "explicit_freeze_hash_required")
    d = OUT_ROOT / TRAIN_ID; path = d / "freeze.json"
    if not path.is_file() or sha(path) != expected: stop("stopped_freeze", "freeze_sha_mismatch")
    m = json.loads(path.read_text(encoding="utf-8"))
    if not ready_freeze(m, code_sha): stop("stopped_freeze", "freeze_incomplete_or_changed")
    for item in m.get("outputs", []):
        p = d / item["path"]
        if not p.is_file() or sha(p) != item["sha256"]: stop("stopped_freeze", "freeze_output_hash_mismatch")
    receipt = json.loads((d/"receipt.json").read_text(encoding="utf-8"))
    if (sha(d/"receipt.json") != m["training_final_receipt_sha256"]
            or receipt.get("run_id") != TRAIN_ID or receipt.get("status") != "complete_training_ready_for_freeze"
            or receipt.get("command_sha256") != code_sha or receipt.get("protocol_sha256") != PROTO_SHA
            or receipt.get("point_source_sha256") != POINT_SOURCE_SHA or receipt.get("abc_source_sha256") != ABC_SOURCE_SHA
            or len(receipt.get("cases", [])) != 8
            or any(c.get("index") != j or c.get("well") != WELLS[j//2] or c.get("model") != CASES[j][1]
                   or c.get("status") != "complete" or c.get("quality_gate_failures")
                   for j, c in enumerate(receipt["cases"]))):
        stop("stopped_freeze", "training_final_receipt_invalid")
    STATE["freeze_sha256"], STATE["freeze_manifest"] = expected, m
    return m


def begin(stage: str, expected_freeze: str | None) -> tuple[Path, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    global T0_MONO, DEADLINE, LIMIT
    rid = TRAIN_ID if stage == "train" else EVAL_ID; out = OUT_ROOT/rid
    if out.exists(): stop("stopped_output_exists", "run_id_already_exists")
    if not OUT_ROOT.is_dir() or OUT_ROOT.is_symlink(): stop("stopped_output_path", "deferred_work_root_invalid")
    out.mkdir(); zero = lambda: {k: 0 for k in COUNTER_KEYS}
    STATE.update(run_id=rid, stage=stage, out_owned=True, status="running",
                 cases=[{"index": j, "well": w, "strain": STRAINS[WELLS.index(w)],
                         "model": m, "phase": "not_started", "status": "not_started",
                         "counters": zero(), "unavailable_scopes": ["case_work", "freeze", "plate2_evaluation"],
                         "unavailable_counter_scopes": ["point", "importance", "abc", "prediction", "score"],
                         "quality_gate_failures": []} for j, (w, m) in enumerate(CASES)],
                 counters={str(j): zero() for j in range(8)}, outputs=[], members_read=0,
                              members_opened=0, members_validated=0, current_member=None,
                              sampled_peak_rss_bytes=rss(), total_member_bytes_read=0)
    STATE["preexisting_artifact_bytes"] = tree_bytes(OUT_ROOT/TRAIN_ID) if stage == "eval" else 0
    check_artifact_reserve(out)
    LIMIT = 600 if stage == "train" else 120
    T0_MONO, DEADLINE = time.monotonic(), time.monotonic()+LIMIT
    signal.signal(signal.SIGALRM, wall_alarm); signal.setitimer(signal.ITIMER_REAL, LIMIT)
    raw, clock = read_inputs(); _, t1 = ordered(raw, clock, "plate1"); _, t2 = ordered(raw, clock, "plate2")
    STATE["storage_before"] = storage_snapshot()
    code_sha = sha(Path(__file__).resolve()); STATE["command_sha256"] = code_sha
    if stage == "eval": verify_freeze(expected_freeze or "", code_sha)
    return out, raw, clock, t1, t2


def save_abc_partial(out: Path, j: int, arrays: dict[str, Any], detail: dict[str, Any]) -> None:
    write_npz(out/f"abc_partial_{j}.npz", **arrays)
    write_json(out/f"abc_partial_{j}.json", detail, MAX_NEW)
    check_artifact_reserve(out)


def train(out: Path, raw: dict[str, Any], clock: dict[str, Any], t1: np.ndarray, t2: np.ndarray) -> None:
    members, _ = ordered(raw, clock, "plate1")
    with read_archive() as archive: y, names = read_stage(out, members, "plate1", archive)
    write_npz(out/"observations_train.npz", y=y, times_hours=t1, names=np.asarray(names, dtype="U128"),
              wells=np.asarray(WELLS), strains=np.asarray(STRAINS))
    arrays: dict[str, Any] = {"prediction_times_hours": t2}
    point_detail, is_detail, abc_detail = {}, {}, {}
    cases = STATE["cases"]
    quality_bad = False
    for j, (well, model) in enumerate(CASES):
        resources(); yj = y[:, WELLS.index(well)]; d = 4 if model == "one_phase" else 7
        case = cases[j]; case["status"] = "running"
        c = STATE["counters"][str(j)]; case["phase"] = "point_search"
        case["unavailable_scopes"] = ["importance_controls", "abc", "freeze", "plate2_evaluation"]
        case["unavailable_counter_scopes"] = ["importance", "abc", "score"]
        case["counters"] = c
        def objective(x: tuple[float, ...]) -> float:
            resources(); c["point_likelihood_calls"] += 1; c["curve_calls_point"] += 1
            return -loglike(yj, curve(np.asarray(x), t1, model), t1)
        result = run_normalized_cube_pattern_search(objective, dimension=d, max_calls=POINT_BUDGET)
        c["point_calls"] = result.callback_calls
        point_detail[str(j)] = safe(result); write_json(out/f"point_{j}.json", point_detail[str(j)])
        if result.best_evaluation is None or result.failure_count: stop("stopped_point_search", "point_callback_failure")
        x0 = np.asarray(result.best_evaluation.coordinates); arrays[f"point_{j}"] = x0
        case["point"] = {"best_objective": result.best_evaluation.objective,
                          "callback_calls": result.callback_calls, "failure_count": result.failure_count,
                          "stop_reason": result.stop_reason}
        case["phase"], case["unavailable_scopes"] = "importance_controls", ["abc", "freeze", "plate2_evaluation"]
        case["unavailable_counter_scopes"] = ["abc", "score"]
        mu, cov = proposal(x0, model, yj, t1, c); arrays[f"proposal_center_{j}"] = mu; arrays[f"proposal_cov_{j}"] = cov
        block_stats, block_means, block_logs = [], [], []
        draws_all, lq_all, ll_all, lw_all = [], [], [], []
        for b in range(2):
            resources(); rng = np.random.default_rng(30100+10*j+b); x = draw_q(rng, BLOCK, mu, cov)
            q = np.asarray([logq(v, mu, cov) for v in x]); c["importance_proposal_density_calls"] += BLOCK
            ll = np.full(BLOCK, np.nan)
            try:
                for i, p in enumerate(x):
                    resources(); pred = curve(p, t1, model); c["curve_calls_importance"] += 1
                    c["importance_likelihood_calls"] += 1; ll[i] = loglike(yj, pred, t1)
                    if not math.isfinite(float(ll[i])): stop("stopped_numeric", "importance_loglike")
            except BaseException:
                write_npz(out/f"importance_partial_{j}_{b}.npz", draws=x, logq=q, loglike=ll,
                          completed_likelihoods=np.asarray(int(np.isfinite(ll).sum()))); raise
            lw = ll-q
            if not np.all(np.isfinite(lw)): stop("stopped_numeric", "importance_logweight")
            w, stats = norm_weights(lw)
            write_npz(out/f"importance_{j}_{b}.npz", draws=x, logq=q, loglike=ll, log_weights=lw, weights=w)
            pm = mean_curve(x, w, t2, model, c)
            block_stats.append({"block": b, "seed": 30100+10*j+b, **stats}); block_means.append(pm); block_logs.append(lw)
            draws_all.append(x); lq_all.append(q); ll_all.append(ll); lw_all.append(lw)
            write_npz(out/f"importance_{j}_{b}.npz", draws=x, logq=q, loglike=ll, log_weights=lw, weights=w, block_mean_plate2=pm)
            if stats["ess"] < 128: case["quality_gate_failures"].append(f"block_{b}_ess")
            if stats["maximum_weight"] > .05: case["quality_gate_failures"].append(f"block_{b}_maximum_weight")
        lw = np.concatenate(block_logs); w, pooled = norm_weights(lw)
        diff = abs(block_stats[0]["log_normalizer"]-block_stats[1]["log_normalizer"])
        rms = float(np.sqrt(np.mean((block_means[0]-block_means[1])**2)))
        if pooled["ess"] < 256: case["quality_gate_failures"].append("pooled_ess")
        if diff > .5: case["quality_gate_failures"].append("block_log_normalizer_difference")
        if rms > .0025: case["quality_gate_failures"].append("plate2_block_curve_rms")
        quality_bad |= bool(case["quality_gate_failures"])
        norm_logs = np.asarray([v["log_normalizer"] for v in block_stats]); norm = np.exp(norm_logs-lse(norm_logs))
        arrays[f"importance_draws_{j}"] = np.concatenate(draws_all); arrays[f"importance_logq_{j}"] = np.concatenate(lq_all)
        arrays[f"importance_loglike_{j}"] = np.concatenate(ll_all); arrays[f"importance_logweights_{j}"] = lw
        arrays[f"importance_weights_{j}"] = w
        arrays[f"prediction_point_{j}"] = curve(x0, t2, model); c["curve_calls_prediction"] += 1
        arrays[f"prediction_importance_{j}"] = norm[0]*block_means[0]+norm[1]*block_means[1]
        is_detail[str(j)] = {"blocks": block_stats, "pooled": pooled,
                             "block_log_normalizer_abs_difference": diff, "block_mean_curve_rms_plate2": rms,
                             "quality_gate_failures": case["quality_gate_failures"]}
        case.update(importance=is_detail[str(j)], status="controls_complete_quality_failed" if case["quality_gate_failures"] else "controls_complete")
        case["phase"] = "reference_controls_complete"
        case["unavailable_scopes"] = ["abc", "freeze", "plate2_evaluation"]
        write_json(out/"controls_progress.json", {"cases": cases, "counters": STATE["counters"], "quality_failed": quality_bad}, MAX_RECEIPT)
        check_artifact_reserve(out)
    if quality_bad:
        write_npz(out/"training_controls_partial.npz", **arrays)
        write_json(out/"training_diagnostics_partial.json", {"point": point_detail, "importance": is_detail})
        STATE["status"], STATE["stop_code"] = "incomplete_reference_quality_gate", "reference_quality_gate_failed_before_abc"
        STATE["unavailable_scopes"] = ["all_abc_cases", "training_freeze", "plate2_evaluation"]
        STATE["output_artifact_bytes"] = stage_artifact_bytes(out)
        STATE["storage_after"] = storage_snapshot(); return
    for j, (well, model) in enumerate(CASES):
        resources(); yj = y[:, WELLS.index(well)]; d = 4 if model == "one_phase" else 7
        STATE["cases"][j].update(status="running", phase="abc_smc")
        STATE["cases"][j]["unavailable_scopes"] = ["freeze", "plate2_evaluation"]
        STATE["cases"][j]["unavailable_counter_scopes"] = ["score"]
        callbacks: list[dict[str, Any]] = []; pop_arrays: dict[str, Any] = {}; latched: dict[str, str] | None = None
        def requested() -> bool:
            nonlocal latched
            if latched is not None: return True
            try: resources()
            except Stop as e: latched = latch_failure(latched, e.status, e.code); return True
            return False
        def simulator(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            nonlocal latched
            try:
                resources(); mean = curve(np.asarray(x), t1, model); STATE["counters"][str(j)]["curve_calls_abc"] += 1
                STATE["counters"][str(j)]["simulations"] += 1
                return mean+simulate_ou(rng, t1)+rng.normal(0, SIG_E, len(t1))
            except Stop as e: latched = latch_failure(latched, e.status, e.code); raise
            except Exception: latched = latch_failure(latched, "stopped_simulator", "simulator_callback_failure"); raise
        def discrepancy(sim: np.ndarray) -> float:
            nonlocal latched
            try:
                STATE["counters"][str(j)]["discrepancy_calls"] += 1
                return float(np.sqrt(np.mean(innovations(np.asarray(sim)-yj, t1)[0]**2)))
            except Stop as e: latched = latch_failure(latched, e.status, e.code); raise
            except Exception: latched = latch_failure(latched, "stopped_discrepancy", "discrepancy_callback_failure"); raise
        def event(kind: str, generation: int, record: dict[str, Any] | None) -> None:
            item: dict[str, Any] = {"event": kind, "generation": generation}
            if record is not None:
                item.update(epsilon=record["epsilon"], diagnostics=record["diagnostics"])
                if kind == "end": STATE["counters"][str(j)]["abc_proposals"] += record["diagnostics"]["proposed"]
                for key in ("accepted_params", "weights", "distances", "proposal_covariance"):
                    val = record.get(key)
                    if val is not None: pop_arrays[f"g{generation}_{key}"] = np.asarray(val)
            callbacks.append(item)
        prior, logprior = make_uniform_prior(np.tile([0., 1.], (d, 1)))
        try:
            result = run_gaussian_abc_smc_reference(prior, simulator, discrepancy, target_samples=64,
                epsilon_schedule=EPS, max_attempts_per_population=ATTEMPTS, bounds=np.tile([0., 1.], (d, 1)),
                prior_logpdf=logprior, covariance_scale=2., lambda_noise=.01, nugget=1e-9,
                seed=31100+j, population_event=event, stop_requested=requested)
        except Stop as e:
            abc_detail[str(j)] = {"adapter_stop_cause": latched or {"status": e.status, "code": e.code}, "callbacks": callbacks}
            save_abc_partial(out, j, pop_arrays, abc_detail[str(j)])
            STATE["cases"][j].update(status=e.status, phase="stopped_abc_smc", stop_code=e.code, unavailable_counter_scopes=["score", "abc_proposals"]); raise
        except Exception:
            abc_detail[str(j)] = outer_failure_detail(latched, callbacks)
            save_abc_partial(out, j, pop_arrays, abc_detail[str(j)])
            cause = abc_detail[str(j)]["adapter_stop_cause"]
            STATE["cases"][j]["unavailable_counter_scopes"] = ["score", "abc_proposals"]
            STATE["cases"][j].update(status=cause["status"], phase="stopped_abc_smc", stop_code=cause["code"])
            stop(cause["status"], cause["code"])
        STATE["counters"][str(j)]["abc_proposals"] = sum(p["diagnostics"]["proposed"] for p in result["populations"])
        if latched or not result["complete"] or any(not p["diagnostics"]["complete"] for p in result["populations"]):
            why = latched or {"status": "stopped_incomplete_abc", "code": "abc_population_incomplete"}
            abc_detail[str(j)] = {"runner_status": result["status"], "runner_reason": result["termination_reason"],
                                  "adapter_stop_cause": why, "callbacks": callbacks}
            save_abc_partial(out, j, pop_arrays, abc_detail[str(j)])
            STATE["cases"][j].update(status=why["status"], phase="stopped_abc_smc", stop_code=why["code"])
            stop(why["status"], why["code"])
        last = result["populations"][-1]; params, weights = np.asarray(last["accepted_params"]), np.asarray(last["weights"])
        arrays[f"abc_params_{j}"], arrays[f"abc_weights_{j}"] = params, weights
        for key, val in pop_arrays.items(): arrays[f"abc_{j}_{key}"] = val
        abc_detail[str(j)] = {"runner_status": result["status"], "termination_reason": result["termination_reason"],
                              "seed": 31100+j, "callbacks": callbacks}
        write_npz(out/f"abc_complete_{j}.npz", params=params, weights=weights, **pop_arrays)
        STATE["cases"][j]["phase"] = "abc_prediction"
        write_json(out/"abc_progress.json", {"cases": STATE["cases"], "callbacks": abc_detail}, MAX_NEW)
        arrays[f"prediction_abc_{j}"] = mean_curve(params, weights, t2, model, STATE["counters"][str(j)])
        write_npz(out/f"abc_complete_{j}.npz", params=params, weights=weights,
                  prediction=arrays[f"prediction_abc_{j}"], **pop_arrays)
        STATE["cases"][j].update(status="complete", abc_seed=31100+j, phase="complete",
                                 unavailable_scopes=["plate2_evaluation"], unavailable_counter_scopes=["score"])
        write_json(out/"abc_progress.json", {"cases": STATE["cases"], "callbacks": abc_detail}, MAX_NEW)
        check_artifact_reserve(out)
    write_npz(out/"training_freeze_data.npz", **arrays)
    write_json(out/"training_diagnostics.json", {"point": point_detail, "importance": is_detail, "abc": abc_detail})
    total_curves = curve_count(STATE["counters"])
    if total_curves > 660000 or sum(c["curve_calls_prediction"] for c in STATE["counters"].values()) > 131592:
        stop("stopped_curve_call_limit", "training_curve_call_cap")
    STATE["combined_curve_calls"] = total_curves
    outputs = []
    for name in ("observations_train.npz", "training_freeze_data.npz", "training_diagnostics.json"):
        p = out/name; outputs.append({"path": name, "sha256": sha(p)})
    STATE["outputs"] = outputs; STATE["status"] = "complete_training_ready_for_freeze"
    STATE["output_artifact_bytes"] = stage_artifact_bytes(out)
    check_artifact_reserve(out)
    resources(); STATE["storage_after"] = storage_snapshot()
    write_receipt(out, "complete_training_ready_for_freeze", None, time.monotonic()-T0_MONO, 0)
    receipt_sha = sha(out/"receipt.json")
    outputs.append({"path": "receipt.json", "sha256": receipt_sha})
    STATE["outputs"], STATE["receipt_sha256"] = outputs, receipt_sha
    STATE["storage_after"] = storage_snapshot()
    STATE["output_artifact_bytes"] = stage_artifact_bytes(out)
    check_artifact_reserve(out); resources()
    STATE["processing_elapsed_seconds"] = time.monotonic()-T0_MONO
    finalization_start = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 0)
    manifest = {"schema": 1, "run_id": TRAIN_ID, "status": "complete_training_frozen", "complete": True,
                "protocol_sha256": PROTO_SHA, "command_sha256": STATE["command_sha256"],
                "point_source_sha256": POINT_SOURCE_SHA, "abc_source_sha256": ABC_SOURCE_SHA,
                "archive_sha256": ARCHIVE_SHA, "archive_size_bytes": SRC_SIZE,
                "raw_metadata_receipt_sha256": RAW_SHA, "clock_receipt_sha256": CLOCK_SHA,
                "layout_receipt_sha256": LAYOUT_SHA, "training_member_count": 271,
                "plate2_members_opened": 0, "cases": cases, "counters": STATE["counters"],
                "combined_curve_calls": total_curves, "outputs": outputs,
                "training_final_receipt_sha256": receipt_sha, "storage_after_receipt": STATE["storage_after"],
                "output_artifact_bytes_before_manifest": STATE["output_artifact_bytes"],
                "numpy_version": np.__version__}
    pending = out/"freeze.json.pending"
    manifest_raw = json.dumps(safe(manifest), ensure_ascii=True, allow_nan=False,
                              sort_keys=True, separators=(",", ":")).encode("ascii")
    if len(manifest_raw) > MAX_RECEIPT: stop("stopped_receipt_limit", "freeze_manifest_exceeds_cap")
    with pending.open("xb") as f: f.write(manifest_raw); f.flush(); os.fsync(f.fileno())
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    os.replace(pending, out/"freeze.json")
    STATE.update(freeze_sha256=manifest_sha, freeze_manifest=manifest,
                 processing_finalization_seconds=time.monotonic()-finalization_start,
                 status="complete_training_frozen", stop_code=None)
    STATE["outputs"].append({"path": "freeze.json", "sha256": manifest_sha})
    STATE["freeze_published"] = True


def score_cloud(y: np.ndarray, times: np.ndarray, x: np.ndarray, weights: np.ndarray,
                model: str, counts: dict[str, int]) -> float:
    logs = np.empty(len(x)); logw = np.log(weights)
    for i, p in enumerate(x):
        resources(); pred = curve(p, times, model); counts["curve_calls_score"] += 1
        counts["score_likelihood_calls"] += 1; logs[i] = logw[i]+loglike(y, pred, times)
    return -lse(logs)/len(y)


def load_frozen_cases(path: Path) -> list[dict[str, np.ndarray]]:
    result = []
    try:
        with np.load(path, allow_pickle=False) as frozen:
            for j, (_, model) in enumerate(CASES):
                d = 4 if model == "one_phase" else 7
                names = (f"point_{j}", f"proposal_center_{j}", f"proposal_cov_{j}",
                         f"importance_draws_{j}", f"importance_logq_{j}", f"importance_loglike_{j}",
                         f"importance_logweights_{j}", f"importance_weights_{j}",
                         f"abc_params_{j}", f"abc_weights_{j}", *(f"prediction_{m}_{j}" for m in ("point", "importance", "abc")))
                a = {name: np.asarray(frozen[name]) for name in names}
                if (a[f"point_{j}"].shape != (d,) or a[f"proposal_center_{j}"].shape != (d,)
                        or a[f"proposal_cov_{j}"].shape != (d, d)
                        or a[f"importance_draws_{j}"].shape != (2*BLOCK, d)
                        or any(a[f"importance_{key}_{j}"].shape != (2*BLOCK,) for key in ("logq", "loglike", "logweights", "weights"))
                        or a[f"abc_params_{j}"].shape != (64, d) or a[f"abc_weights_{j}"].shape != (64,)
                        or any(a[f"prediction_{m}_{j}"].shape != (271,) for m in ("point", "importance", "abc"))):
                    stop("stopped_freeze", "frozen_array_shape")
                finite = all(np.all(np.isfinite(v)) for v in a.values())
                points = (a[f"point_{j}"], a[f"importance_draws_{j}"], a[f"abc_params_{j}"])
                if (not finite or any(np.any(x < 0) or np.any(x > 1) for x in points)
                        or any(np.any(a[f"{key}_{j}"] < 0) or not np.isclose(a[f"{key}_{j}"].sum(), 1, rtol=0, atol=1e-10)
                               for key in ("importance_weights", "abc_weights"))):
                    stop("stopped_freeze", "frozen_array_values")
                result.append(a)
    except Stop: raise
    except Exception: stop("stopped_freeze", "frozen_array_read_failure")
    return result


def evaluate(out: Path, raw: dict[str, Any], clock: dict[str, Any], t2: np.ndarray, freeze_sha: str) -> None:
    data_path = OUT_ROOT/TRAIN_ID/"training_freeze_data.npz"
    verify_freeze(freeze_sha, STATE["command_sha256"])
    frozen_cases = load_frozen_cases(data_path)
    per_case = {}; scores = {}
    members, _ = ordered(raw, clock, "plate2")
    with read_archive() as archive:
        y, _names = read_stage(out, members, "plate2", archive)
        for j, (well, model) in enumerate(CASES):
            resources(); yj = y[:, WELLS.index(well)]; c = STATE["counters"][str(j)]
            c["curve_calls_score"], c["score_likelihood_calls"] = 0, 0
            STATE["cases"][j].update(status="running", phase="plate2_scoring", counters=c,
                                     unavailable_scopes=["plate2_scores"], unavailable_counter_scopes=[])
            frozen = frozen_cases[j]
            draws, impw = frozen[f"importance_draws_{j}"], frozen[f"importance_weights_{j}"]
            abc, abcw = frozen[f"abc_params_{j}"], frozen[f"abc_weights_{j}"]
            preds = [frozen[f"prediction_{m}_{j}"] for m in ("point", "importance", "abc")]
            rows = {}
            for method, pred in zip(("point", "likelihood_weighted", "abc"), preds, strict=True):
                rmse = float(np.sqrt(np.mean((pred-yj)**2)))
                if method == "point": nll = -loglike(yj, pred, t2)/len(yj); c["score_likelihood_calls"] += 1
                elif method == "likelihood_weighted": nll = score_cloud(yj, t2, draws, impw, model, c)
                else: nll = score_cloud(yj, t2, abc, abcw, model, c)
                rows[method] = {"rmse_raw_od": rmse, "negative_log_predictive_density_per_record": float(nll)}
                scores[str(j)] = {"well": well, "strain": STRAINS[WELLS.index(well)], "model": model, "methods": rows}
                STATE["cases"][j].update(methods_complete=list(rows))
                write_json(out/"scores_progress.json", {"scores": scores, "cases": STATE["cases"],
                                                         "freeze_sha256": freeze_sha}, MAX_RECEIPT)
                check_artifact_reserve(out)
                combined = curve_count(STATE["freeze_manifest"]["counters"])+curve_count(STATE["counters"])
                if combined > 660000 or curve_count(STATE["counters"]) > 131592: stop("stopped_curve_call_limit", "combined_curve_call_cap")
            per_case[str(j)] = {"index": j, "well": well, "strain": STRAINS[WELLS.index(well)],
                                "model": model, "status": "complete", "counters": c}
            STATE["cases"][j].update(status="complete", phase="complete", methods=rows,
                                     unavailable_scopes=[], unavailable_counter_scopes=[], counters=c)
            write_json(out/"scores_progress.json", {"scores": scores, "cases": STATE["cases"],
                                                     "freeze_sha256": freeze_sha}, MAX_RECEIPT)
            check_artifact_reserve(out)
    write_json(out/"scores.json", {"scores": scores, "cases": per_case, "freeze_sha256": freeze_sha,
                                    "evaluation_members_read": 271, "plate1_members_opened": 0,
                                    "raw_plate2_values_saved": False}, MAX_RECEIPT)
    total_curves = curve_count(STATE["freeze_manifest"]["counters"]) + curve_count(STATE["counters"])
    if total_curves > 660000 or curve_count(STATE["counters"]) > 131592:
        stop("stopped_curve_call_limit", "combined_curve_call_cap")
    STATE["combined_curve_calls"] = total_curves
    STATE["outputs"] = [{"path": "scores.json", "sha256": sha(out/"scores.json")}]
    STATE["output_artifact_bytes"] = stage_artifact_bytes(out)
    check_artifact_reserve(out)
    STATE["storage_after"] = storage_snapshot(); STATE["status"], STATE["stop_code"] = "complete_evaluation", None


def write_receipt(out: Path, status: str, code: str | None, elapsed: float, finalization: float,
                  filename: str = "receipt.json") -> None:
    receipt = {"schema": 1, "run_id": STATE.get("run_id"), "stage": STATE.get("stage"),
        "status": status, "stop_code": code, "protocol_sha256": PROTO_SHA,
        "command_sha256": STATE.get("command_sha256"), "point_source_sha256": POINT_SOURCE_SHA,
        "abc_source_sha256": ABC_SOURCE_SHA, "archive_sha256": ARCHIVE_SHA,
        "raw_metadata_receipt_sha256": RAW_SHA, "clock_receipt_sha256": CLOCK_SHA,
        "layout_receipt_sha256": LAYOUT_SHA, "storage_scope": "data/, artifacts/, three original Downloads files",
        "storage_before": STATE.get("storage_before"), "storage_after": STATE.get("storage_after"),
        "processing_elapsed_seconds": elapsed, "finalization_excluded_seconds": finalization,
        "sampled_peak_rss_bytes": STATE.get("sampled_peak_rss_bytes"),
        "combined_curve_calls": STATE.get("combined_curve_calls"),
        "limits": {"processing_seconds": LIMIT, "rss_bytes": MAX_RSS, "new_artifacts_bytes": MAX_NEW,
                   "receipt_bytes": MAX_RECEIPT, "storage_included_bytes": MAX_STORAGE},
        "members_opened": STATE.get("members_opened", 0), "members_read": STATE.get("members_read", 0),
        "members_validated": STATE.get("members_validated", 0), "current_member": STATE.get("current_member"),
        "last_member": STATE.get("last_member"),
        "total_member_bytes_read": STATE.get("total_member_bytes_read", 0),
        "cases": STATE.get("cases", []), "counters": STATE.get("counters", {}),
        "output_artifact_bytes": STATE.get("output_artifact_bytes"), "outputs": STATE.get("outputs", [])}
    write_json(out/filename, receipt, MAX_RECEIPT)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("train", "eval"), required=True)
    p.add_argument("--freeze-sha256"); a = p.parse_args()
    if a.stage == "eval" and not a.freeze_sha256: p.error("--freeze-sha256 is required for eval")
    out = OUT_ROOT/(TRAIN_ID if a.stage == "train" else EVAL_ID)
    status, code, elapsed, finalization = "stopped_startup", "startup", 0.0, 0.0
    try:
        out, raw, clock, t1, t2 = begin(a.stage, a.freeze_sha256)
        if a.stage == "train": train(out, raw, clock, t1, t2)
        else:
            evaluate(out, raw, clock, t2, a.freeze_sha256 or "")
            resources()
        elapsed = STATE.get("processing_elapsed_seconds", time.monotonic()-T0_MONO)
        finalization = STATE.get("processing_finalization_seconds", 0.0)
        signal.setitimer(signal.ITIMER_REAL, 0); status, code = STATE["status"], STATE["stop_code"]
    except Stop as e:
        status, code = e.status, e.code
        elapsed = max(0.0, time.monotonic()-T0_MONO) if T0_MONO else 0.0
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception:
        status, code = "stopped_unexpected_error", "unexpected_runtime_exception"
        elapsed = max(0.0, time.monotonic()-T0_MONO) if T0_MONO else 0.0
        signal.setitimer(signal.ITIMER_REAL, 0)
    successful_training = (a.stage == "train" and STATE.get("freeze_published")
                           and status == "complete_training_frozen")
    if STATE.get("out_owned") and not successful_training:
        for case in STATE.get("cases", []):
            if case.get("status") == "running":
                case.update(status=status, phase="stopped_"+case.get("phase", "unknown"), stop_code=code)
            elif case.get("status") == "not_started": case["stop_code"] = code
            if not status.startswith("complete") and case.get("status") != "complete":
                case["stage_stop"] = {"status": status, "code": code}
        base = STATE.get("freeze_manifest", {}).get("counters", {}) if a.stage == "eval" else {}
        STATE["combined_curve_calls"] = curve_count(base)+curve_count(STATE.get("counters", {}))
        STATE["output_artifact_bytes"] = stage_artifact_bytes(out)
        t = time.monotonic()
        try: write_receipt(out, status, code, elapsed, time.monotonic()-t)
        except Exception: status, code = "stopped_receipt_write_failure", "final_receipt_persist_failed"
        finalization = time.monotonic()-t
    result = {"run_id": STATE.get("run_id"), "status": status, "stop_code": code,
              "receipt": str(out/"receipt.json") if STATE.get("out_owned") else None,
              "processing_elapsed_seconds": elapsed, "finalization_excluded_seconds": finalization}
    if successful_training: result["freeze_sha256"] = STATE["freeze_sha256"]
    if STATE.get("out_owned") and (out/"receipt.json").exists():
        result["receipt_sha256"] = STATE.get("receipt_sha256") if successful_training else sha(out/"receipt.json")
    print(json.dumps(result)); return 0 if status.startswith("complete") else 2


if __name__ == "__main__": raise SystemExit(main())
