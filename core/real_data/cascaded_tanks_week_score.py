"""Development-only target scanner and scorer for the 2026 tank study."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from core.real_data import cascaded_tanks_controlled as source

PROTOCOL_ID = "ct_memory_week_dev_v1_20261015"
RUN_ID = "ct-memory-week-dev-20261015-v1"
MANIFEST_SCHEMA = "cascaded-tanks-week-development-manifest-v1"
FORECAST_SCHEMA = "cascaded-tanks-week-development-forecasts-v1"
RECEIPT_SCHEMA = "cascaded-tanks-week-development-score-v1"
ROOT = Path(__file__).resolve().parents[2]
MAX_JSON = 32_000_000
MAX_ARCHIVE = 8_000_000
MODELS = ("S0", "O2", "C2")
SEEDS = (11, 29, 47)
KINDS = ("abc", "pattern")
BASELINES = (("arx", "ARX(2,2)", 0), ("persistence", "persistence", 0))
SETTINGS = {
    "parameter_order": ["a", "c", "p", "x1_0"],
    "bounds": [[0.2, 1.5], [0.2, 1.5], [0.05, 1.2], [0.0, 25.0]],
    "fixed_x2_0": 5.205, "fixed_ceiling": 10.0, "float": "float64",
    "max_magnitude": 1_000_000.0, "max_steps": "requested_trajectory_length",
    "calibration_draws": 256, "quantiles": [0.25, 0.1], "quantile_method": "linear",
    "pcg64_seed_sequence": [20261014, 0, "family_index", "seed", "stream"],
    "abc_particles": 64, "abc_attempts_per_population": 8192,
    "abc_populations": 2, "abc_kernel": {"covariance_scale": 2.0, "lambda_noise": 0.01, "nugget": 1e-9},
    "pattern_search": {"initial_mesh": 0.25, "stop_mesh": 0.0009765625},
    "fit_simulator_call_cap": 299520, "fit_wall_seconds": 3600,
    "rss_bytes": 4294967296, "artifact_bytes": 1073741824,
    "forecast_calls": 600, "forecast_wall_seconds": 120,
    "alignment": "observe y[k] before applying u[k]; carry state through 1024-step replay",
}


class WeekStageError(ValueError):
    """A frozen development-stage input or result failed validation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_regular(path: str | Path, limit: int) -> bytes:
    """Read a stable bounded regular file without following its final symlink."""

    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise WeekStageError("input is not a bounded regular file")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(1_048_576, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise WeekStageError("input exceeds its byte limit")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ) or size != before.st_size:
            raise WeekStageError("input changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise WeekStageError(f"duplicate {label} key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise WeekStageError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise WeekStageError(f"{label} must be a JSON object")
    return value


def code_hashes() -> dict[str, str]:
    paths = {
        "source": "core/real_data/cascaded_tanks_controlled.py",
        "simulator": "core/real_data/cascaded_tanks_models.py",
        "abc_reference": "core/abc_smc_reference.py",
        "pattern_search": "core/real_data/cascaded_tanks_pattern_search.py",
        "runner": "scripts/run_cascaded_tanks_week.py",
        "scorer": "core/real_data/cascaded_tanks_week_score.py",
        "protocol": "reports/cascaded-tanks-week-development-v1-protocol-2026-10-15.md",
    }
    return {name: sha256(read_regular(ROOT / path, 32_000_000)) for name, path in paths.items()}


def git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = read_regular(path, MAX_JSON)
    item = _json_object(raw, "manifest")
    if item.get("schema") != MANIFEST_SCHEMA or item.get("protocol_id") != PROTOCOL_ID:
        raise WeekStageError("unsupported development manifest")
    if item.get("run_id") != RUN_ID:
        raise WeekStageError("manifest run ID differs from the frozen ID")
    if item.get("code_sha256") != code_hashes() or item.get("git_head") != git_head():
        raise WeekStageError("manifest code or Git identity differs")
    if item.get("runtime") != {"python": __import__("platform").python_version(), "numpy": np.__version__}:
        raise WeekStageError("manifest runtime differs from this process")
    src = item.get("source")
    if not isinstance(src, dict) or any((
        src.get("archive_bytes") != source.SOURCE_ARCHIVE_BYTES,
        src.get("archive_sha256") != source.SOURCE_ARCHIVE_SHA256,
        src.get("track_id") != source.TRACK_ID,
        src.get("contract_sha256") != source.CONTRACT_SHA256,
    )):
        raise WeekStageError("manifest does not bind the pinned source")
    if not isinstance(item.get("settings"), dict) or not isinstance(item.get("arms"), list):
        raise WeekStageError("manifest settings or frozen roster is missing")
    if item["settings"] != SETTINGS:
        raise WeekStageError("manifest numeric settings differ from the frozen protocol")
    expected = sorted(expected_roster())
    declared = [(arm.get("kind"), arm.get("model"), arm.get("seed")) for arm in item["arms"] if isinstance(arm, dict)]
    if sorted(declared) != expected or len(declared) != len(item["arms"]):
        raise WeekStageError("manifest arm roster differs from the frozen 20-arm roster")
    return item, sha256(raw)


def verify_source(manifest: dict[str, Any], data: Any) -> None:
    record = manifest["source"]
    observed = {
        "archive_sha256": data.archive_sha256,
        "track_id": data.track_id,
        "contract_sha256": data.contract_sha256,
        "source_visible_sha256": data.source_visible_sha256,
        "training_sha256": data.stage_receipts.training_sha256,
        "forecast_inputs_sha256": data.stage_receipts.forecast_inputs_sha256,
    }
    if any(record.get(key) != value for key, value in observed.items()):
        raise WeekStageError("source adapter receipts differ from the manifest")


def archive_snapshot(path: str | Path) -> bytes:
    raw = read_regular(path, MAX_ARCHIVE)
    if len(raw) != source.SOURCE_ARCHIVE_BYTES or sha256(raw) != source.SOURCE_ARCHIVE_SHA256:
        raise WeekStageError("archive bytes differ from the pinned source")
    return raw


def _scan_columns(member: BinaryIO, member_bytes: int, columns: dict[int, tuple[str, int, int]], *, fixture: bool) -> dict[str, tuple[float, ...]]:
    """Validate the fixed CSV byte grammar and decode only requested cells."""

    header = member.readline(129)
    if header != source.CSV_HEADER_WIRE + b"\n" or (not fixture and member_bytes != source.CSV_MEMBER_BYTES):
        raise WeekStageError("CSV header or member length differs from contract")
    found = {name: [] for name, _, _ in columns.values()}
    seen = len(header)
    for row in range(1024):
        for col in range(5):
            chosen = columns.get(col)
            capture = chosen is not None and chosen[1] <= row < chosen[2]
            validate_interval = col == 4 and row == 0
            buf = bytearray() if capture or validate_interval else None
            length = 0
            while True:
                byte = member.read(1)
                if not byte:
                    raise WeekStageError("CSV ends inside a field")
                seen += 1
                if seen > member_bytes:
                    raise WeekStageError("CSV exceeds member length")
                value = byte[0]
                if value == 44:
                    if col == 4:
                        if row == 0:
                            if not length or _number(bytes(buf), f"Ts[{row}]") != 4.0:
                                raise WeekStageError("first sample interval must be 4 seconds")
                        elif length:
                            raise WeekStageError("sample interval must be empty after row zero")
                    elif not length:
                        raise WeekStageError(f"empty CSV cell at row {row}, column {col}")
                    if capture:
                        if not length:
                            raise WeekStageError(f"selected cell {chosen[0]}[{row}] is empty")
                        found[chosen[0]].append(_number(bytes(buf), f"{chosen[0]}[{row}]"))
                    break
                if value in (10, 13, 34) or value < 32 or value == 127:
                    raise WeekStageError("CSV contains a quote or control byte in a field")
                length += 1
                if length > 128:
                    raise WeekStageError("CSV field exceeds the fixed byte limit")
                if buf is not None:
                    buf.append(value)
        if member.read(1) != b"\n":
            raise WeekStageError("CSV row must end with an empty sixth field")
        seen += 1
    if member.read(1) != b"\n":
        raise WeekStageError("CSV must end with one blank LF line")
    seen += 1
    if seen != member_bytes or member.read(1):
        raise WeekStageError("CSV trailing bytes differ from contract")
    result = {name: tuple(values) for name, values in found.items()}
    for name, start, stop in columns.values():
        if len(result[name]) != stop - start:
            raise WeekStageError(f"selected field {name} has the wrong length")
    return result


def scan_columns(member: BinaryIO, member_bytes: int,
                 columns: dict[int, tuple[str, int, int]]) -> dict[str, tuple[float, ...]]:
    return _scan_columns(member, member_bytes, columns, fixture=False)


def _scan_fixture_for_test(member: BinaryIO, member_bytes: int,
                           columns: dict[int, tuple[str, int, int]]) -> dict[str, tuple[float, ...]]:
    """Explicit synthetic seam; it changes only the fixture byte-length check."""
    return _scan_columns(member, member_bytes, columns, fixture=True)


def _number(raw: bytes, label: str) -> float:
    try:
        value = float(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError, OverflowError) as error:
        raise WeekStageError(f"invalid selected number {label}") from error
    if not math.isfinite(value):
        raise WeekStageError(f"non-finite selected number {label}")
    return value


def parse_development_targets(snapshot: bytes) -> tuple[float, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            info = source._validate_zip_members(archive.infolist(), source.OFFICIAL_ARCHIVE)
            if info.filename != source.CSV_MEMBER:
                raise WeekStageError("CSV member path differs")
            with archive.open(info) as member:
                return scan_columns(member, info.file_size, {2: ("yEst_suffix", 768, 1024)})["yEst_suffix"]
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise WeekStageError("pinned CSV member could not be read") from error


def expected_roster() -> set[tuple[str, str, int]]:
    arms = {(kind, model, seed) for model in MODELS for seed in SEEDS for kind in KINDS}
    return arms | set(BASELINES)


def claim_marker(run_dir: str | Path, stage: str, binding: dict[str, Any]) -> str:
    marker_dir = Path(run_dir) / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes({"stage": stage, "run_id": RUN_ID, **binding})
    path = marker_dir / f"{RUN_ID}.{stage}.claim"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(marker_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256(raw)


def write_terminal(run_dir: str | Path, receipt: dict[str, Any]) -> Path:
    path = Path(run_dir) / "development-score.receipt.json"
    unsigned = dict(receipt)
    receipt["receipt_sha256"] = sha256(canonical_bytes(unsigned))
    raw = canonical_bytes(receipt)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def rmse(prediction: Any, target: Any) -> float:
    pred, obs = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != obs.shape or pred.size == 0 or not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise WeekStageError("RMSE requires matching finite vectors")
    return float(np.sqrt(np.mean(np.square(pred - obs), dtype=np.float64)))


def _validate_forecasts(raw: bytes, manifest_hash: str) -> list[dict[str, Any]]:
    bundle = _json_object(raw, "forecast bundle")
    if bundle.get("schema") != FORECAST_SCHEMA or bundle.get("run_id") != RUN_ID:
        raise WeekStageError("unsupported forecast bundle")
    if bundle.get("manifest_sha256") != manifest_hash:
        raise WeekStageError("forecast bundle is not bound to the manifest")
    arms = bundle.get("arms")
    if not isinstance(arms, list) or len(arms) != len(expected_roster()):
        raise WeekStageError("forecast roster must contain all 20 frozen arms")
    roster = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise WeekStageError("forecast arm must be an object")
        key = (arm.get("kind"), arm.get("model"), arm.get("seed"))
        if key in roster:
            raise WeekStageError("forecast roster has a duplicate arm")
        roster.add(key)
        if arm.get("status") == "complete":
            forecast = arm.get("forecast")
            if not isinstance(forecast, list) or len(forecast) != 256:
                raise WeekStageError("complete arm needs 256 forecast values")
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in forecast):
                raise WeekStageError("forecast values must be finite numbers")
        elif arm.get("status") not in ("failed", "not_run") or not arm.get("failure"):
            raise WeekStageError("failed arms need an explicit failure or not-run reason")
    if roster != expected_roster():
        raise WeekStageError("forecast roster differs from the frozen 20-arm roster")
    return arms


def _decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligibility = {}
    medians = {}
    for model in MODELS:
        family = [row for row in rows if row["kind"] == "abc" and row["model"] == model]
        eligibility[model] = len(family) == 3 and all(row["status"] == "complete" for row in family)
        if eligibility[model]:
            medians[model] = float(np.median([row["rmse"] for row in family]))
    eligible = list(medians)
    locked = "unresolved"
    if len(eligible) >= 2:
        winners = [model for model in eligible if all(model == other or medians[model] <= 0.9 * medians[other] for other in eligible)]
        if len(winners) == 1:
            locked = winners[0]
    baseline = {row["kind"]: row["rmse"] for row in rows if row["kind"] in ("arx", "persistence")}
    useful = None
    if locked != "unresolved" and all(key in baseline and baseline[key] is not None for key in ("arx", "persistence")):
        useful = all(medians[locked] <= 0.9 * baseline[key] for key in ("arx", "persistence"))
    contrasts = {}
    for seed in SEEDS:
        c2 = next(row for row in rows if row["kind"] == "abc" and row["model"] == "C2" and row["seed"] == seed)
        o2 = next(row for row in rows if row["kind"] == "abc" and row["model"] == "O2" and row["seed"] == seed)
        contrasts[str(seed)] = c2["rmse"] - o2["rmse"] if c2["rmse"] is not None and o2["rmse"] is not None else None
    return {"family_eligible": eligibility, "abc_median_rmse": medians, "locked_selection": locked,
            "eligible_family_count": len(eligible), "useful_vs_both_baselines": useful,
            "c2_minus_o2_seed_rmse": contrasts}


def score_development(archive_path: str | Path, manifest_path: str | Path,
                      forecast_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Claim once, read only the development target suffix, and score frozen arms."""

    manifest, manifest_hash = read_manifest(manifest_path)
    data = source.load_development_data(archive_path)
    verify_source(manifest, data)
    forecast_raw = read_regular(forecast_path, MAX_JSON)
    arms = _validate_forecasts(forecast_raw, manifest_hash)
    snapshot = archive_snapshot(archive_path)
    marker_hash = claim_marker(run_dir, "development-score", {
        "manifest_sha256": manifest_hash, "archive_sha256": sha256(snapshot),
        "forecast_sha256": sha256(forecast_raw),
    })
    receipt = {"schema": RECEIPT_SCHEMA, "run_id": RUN_ID, "stage": "development-score",
               "status": "terminal_failure", "marker_sha256": marker_hash,
               "manifest_sha256": manifest_hash, "forecast_sha256": sha256(forecast_raw),
               "archive_sha256": sha256(snapshot), "target_access": {"yEst_suffix": 0}}
    try:
        if not any(arm["status"] == "complete" for arm in arms):
            receipt["failure"] = "no arm has a complete frozen forecast"
        else:
            receipt["target_access"] = {"yEst_suffix": "attempted_after_marker"}
            targets = parse_development_targets(snapshot)
            scores = []
            for arm in arms:
                row = {key: arm[key] for key in ("kind", "model", "seed", "status")}
                row.update({"rmse": rmse(arm["forecast"], targets) if arm["status"] == "complete" else None,
                            "failure": arm.get("failure")})
                scores.append(row)
            receipt.update({"status": "complete", "target_access": {"yEst_suffix": 256},
                            "target_sha256": sha256(np.asarray(targets, dtype="<f8").tobytes()),
                            "arm_scores": scores, **_decisions(scores)})
    except Exception as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"[:256]
    write_terminal(run_dir, receipt)
    return receipt


def reject_deferred_stage(stage: str) -> None:
    if stage in ("refit", "official-score", "test-score"):
        raise WeekStageError(f"{stage} is deferred by {PROTOCOL_ID}; no official-test accessor exists")
    raise WeekStageError(f"unknown stage {stage!r}")


__all__ = ["PROTOCOL_ID", "RUN_ID", "WeekStageError", "archive_snapshot", "canonical_bytes",
           "claim_marker", "code_hashes", "expected_roster", "git_head", "parse_development_targets",
           "read_manifest", "reject_deferred_stage", "rmse", "scan_columns", "score_development",
           "sha256", "verify_source", "write_terminal", "SETTINGS"]
