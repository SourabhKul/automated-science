"""Explicit development/final data boundaries for canonical trajectory runs.

The historical runner has a held-out mode whose suffix is adaptive validation.
This module provides the separate, opt-in protocol used by the canonical
trajectory integration: development search receives only train and validation
partitions, while the sealed final partition is represented by a different
manifest consumed by the frozen evaluator.

This is an API/input boundary. It does not claim to provide an OS sandbox or
to make a final file inaccessible to a process that has otherwise been given
the file path.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


CANONICAL_PROTOCOL_VERSION = "trajectory_train_validation_sealed_v1"
DEVELOPMENT_ROLES = ("train", "validation")
FINAL_ROLE = "final"


def _canonical_json(value: Any) -> str:
    """Return deterministic JSON for receipt hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def sha256_array(value: Any) -> str:
    """Hash an array including dtype and shape, without JSON float coercion."""

    array = np.ascontiguousarray(np.asarray(value))
    header = _canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)}).encode("utf-8")
    return sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_partition_arrays(role: str, time_points: Any, observations: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(time_points)
    values = np.asarray(observations)
    if times.ndim != 1:
        raise ValueError(f"{role} time_points must be one-dimensional, got {times.shape}")
    if values.ndim != 2:
        raise ValueError(f"{role} observations must be two-dimensional, got {values.shape}")
    if len(times) != len(values):
        raise ValueError(f"{role} time/observation length mismatch: {len(times)} != {len(values)}")
    if len(times) < 1:
        raise ValueError(f"{role} partition must contain at least one observation")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{role} time_points must be finite")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError(f"{role} time_points must be strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{role} observations must be finite")
    return times, values


@dataclass(frozen=True)
class TrajectoryPartition:
    """One explicitly named trajectory partition."""

    role: str
    time_points: np.ndarray
    observations: np.ndarray
    observation_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        times, values = _validate_partition_arrays(self.role, self.time_points, self.observations)
        if self.role not in {"train", "validation", FINAL_ROLE}:
            raise ValueError(f"unsupported trajectory role: {self.role}")
        mask = None
        if self.observation_mask is not None:
            mask = np.asarray(self.observation_mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError(f"{self.role} observation_mask shape mismatch: {mask.shape} != {values.shape}")
            if not np.any(mask):
                raise ValueError(f"{self.role} observation_mask must include at least one observed value")
        # Freeze arrays held by this value object so callers cannot mutate a
        # receipt after it has been constructed.
        times = np.array(times, copy=True)
        values = np.array(values, copy=True)
        times.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "time_points", times)
        object.__setattr__(self, "observations", values)
        if mask is not None:
            mask = np.array(mask, copy=True)
            mask.setflags(write=False)
        object.__setattr__(self, "observation_mask", mask)

    @property
    def receipt(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "time_points_sha256": sha256_array(self.time_points),
            "observations_sha256": sha256_array(self.observations),
            "time_points_shape": list(self.time_points.shape),
            "observations_shape": list(self.observations.shape),
            "dtype": str(self.observations.dtype),
        }
        if self.observation_mask is not None:
            result["observation_mask_sha256"] = sha256_array(self.observation_mask)
            result["observed_count"] = int(np.sum(self.observation_mask))
        return result


@dataclass(frozen=True)
class CanonicalTrajectoryProtocol:
    """Three-way trajectory contract used by preparation and final scoring."""

    train: TrajectoryPartition
    validation: TrajectoryPartition
    final: TrajectoryPartition
    protocol_version: str = CANONICAL_PROTOCOL_VERSION
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.train.role != "train" or self.validation.role != "validation" or self.final.role != FINAL_ROLE:
            raise ValueError("canonical protocol requires train, validation, and final roles")
        if self.protocol_version != CANONICAL_PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")

    def development_receipt(self) -> dict[str, Any]:
        """Receipt safe to expose to an adaptive development search."""

        result: dict[str, Any] = {
            "schema_version": 1,
            "protocol_version": self.protocol_version,
            "roles": {
                "train": self.train.receipt,
                "validation": self.validation.receipt,
            },
            "final_outcomes_available_to_search": False,
        }
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        result["development_receipt_sha256"] = sha256_json(result)
        return result

    def final_receipt(self) -> dict[str, Any]:
        """Full protocol receipt for a final-evaluation process only."""

        result = self.development_receipt()
        result["roles"] = {
            "train": self.train.receipt,
            "validation": self.validation.receipt,
            FINAL_ROLE: self.final.receipt,
        }
        result["final_outcomes_available_to_search"] = False
        result["final_receipt_sha256"] = sha256_json(result)
        return result


def split_trajectory(
    time_points: Any,
    observations: Any,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    observation_mask: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalTrajectoryProtocol:
    """Create deterministic contiguous train/validation/final partitions.

    The default 60/20/20 split is explicit and is suitable for the narrow
    synthetic integration. It is a chronological suffix protocol; callers
    must label grouped or independent-unit splits separately.
    """

    times, values = _validate_partition_arrays("full", time_points, observations)
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    n = len(times)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    train_end = min(max(train_end, 1), n - 2)
    validation_end = min(max(validation_end, train_end + 1), n - 1)
    if observation_mask is not None:
        mask = np.asarray(observation_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(f"full observation_mask shape mismatch: {mask.shape} != {values.shape}")
        mask_parts = (
            mask[:train_end],
            mask[train_end:validation_end],
            mask[validation_end:],
        )
    else:
        mask_parts = (None, None, None)
    return CanonicalTrajectoryProtocol(
        train=TrajectoryPartition("train", times[:train_end], values[:train_end], mask_parts[0]),
        validation=TrajectoryPartition("validation", times[train_end:validation_end], values[train_end:validation_end], mask_parts[1]),
        final=TrajectoryPartition(FINAL_ROLE, times[validation_end:], values[validation_end:], mask_parts[2]),
        metadata={
            **(dict(metadata) if metadata else {}),
            "split_kind": "chronological_suffix",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "final_fraction": 1.0 - train_fraction - validation_fraction,
        },
    )


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _save_partition(partition: TrajectoryPartition, base_dir: Path) -> dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = partition.role
    time_path = base_dir / f"{prefix}_time_points.npy"
    observation_path = base_dir / f"{prefix}_observations.npy"
    np.save(time_path, partition.time_points)
    np.save(observation_path, partition.observations)
    result: dict[str, Any] = {
        "time_points_path": _relative_path(time_path, base_dir),
        "observations_path": _relative_path(observation_path, base_dir),
    }
    if partition.observation_mask is not None:
        mask_path = base_dir / f"{prefix}_observation_mask.npy"
        np.save(mask_path, partition.observation_mask)
        result["observation_mask_path"] = _relative_path(mask_path, base_dir)
    return result


def write_protocol_manifests(
    protocol: CanonicalTrajectoryProtocol,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write separate development and sealed manifests.

    The development manifest intentionally has no final path or final hash;
    the final evaluator receives the sealed manifest independently.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    partition_dir = root / "partitions"
    paths = {
        role: _save_partition(partition, partition_dir)
        for role, partition in (("train", protocol.train), ("validation", protocol.validation), (FINAL_ROLE, protocol.final))
    }
    for spec in paths.values():
        for key in ("time_points_path", "observations_path", "observation_mask_path"):
            if key in spec:
                spec[key] = str(Path("partitions") / spec[key])
    development = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "roles": {"train": paths["train"], "validation": paths["validation"]},
        "metadata": {**(dict(protocol.metadata) if protocol.metadata else {}), "final_outcomes_available_to_search": False},
    }
    final = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "roles": {FINAL_ROLE: paths[FINAL_ROLE]},
        "metadata": {**(dict(protocol.metadata) if protocol.metadata else {}), "final_outcomes_available_to_search": False},
    }
    development_path = root / "development_manifest.json"
    final_path = root / "sealed_final_manifest.json"
    development_path.write_text(json.dumps(development, indent=2, sort_keys=True) + "\n")
    # The sealed manifest carries the exact development receipt that the
    # frozen selection must reference. This is a provenance binding only; the
    # development manifest remains the sole input exposed to search.
    development_receipt = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "manifest_sha256": sha256_file(development_path),
        "metadata": development.get("metadata") or {},
        "roles": {
            "train": protocol.train.receipt,
            "validation": protocol.validation.receipt,
        },
    }
    development_receipt["development_receipt_sha256"] = sha256_json(development_receipt)
    final["development_receipt"] = development_receipt
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return development_path, final_path


def _manifest_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_partition_manifest(
    manifest_path: str | Path,
    *,
    required_roles: Iterable[str],
    forbidden_roles: Iterable[str] = (),
) -> dict[str, Any]:
    """Load only declared role inputs and return arrays plus a safe receipt."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text())
    if payload.get("protocol_version") != CANONICAL_PROTOCOL_VERSION:
        raise ValueError("unsupported or missing canonical trajectory protocol version")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("trajectory manifest must contain a roles mapping")
    for role in required_roles:
        if role not in roles:
            raise ValueError(f"trajectory manifest is missing required role: {role}")
    for role in forbidden_roles:
        if role in roles:
            raise ValueError(f"trajectory manifest must not expose role: {role}")
    loaded: dict[str, TrajectoryPartition] = {}
    for role in required_roles:
        spec = roles[role]
        if not isinstance(spec, dict):
            raise ValueError(f"manifest role {role} must be a mapping")
        time_path = _manifest_path(path.parent, spec["time_points_path"])
        observations_path = _manifest_path(path.parent, spec["observations_path"])
        mask_path = spec.get("observation_mask_path")
        mask = np.load(_manifest_path(path.parent, mask_path)).astype(bool) if mask_path else None
        loaded[role] = TrajectoryPartition(
            role,
            np.load(time_path),
            np.load(observations_path),
            mask,
        )
    receipt = {
        "schema_version": 1,
        "protocol_version": payload["protocol_version"],
        "manifest_sha256": sha256_file(path),
        "metadata": payload.get("metadata") or {},
        "roles": {role: partition.receipt for role, partition in loaded.items()},
    }
    receipt["development_receipt_sha256"] = sha256_json(receipt)
    return {
        "protocol_version": payload["protocol_version"],
        "metadata": payload.get("metadata") or {},
        "development_receipt": payload.get("development_receipt"),
        "manifest_path": str(path),
        "roles": loaded,
        "receipt": receipt,
    }


def frozen_parameters_hash(parameters: Mapping[str, Any] | Iterable[Any]) -> str:
    if isinstance(parameters, Mapping):
        value: Any = {str(k): float(v) for k, v in sorted(parameters.items())}
    else:
        value = [float(v) for v in parameters]
    return sha256_json(value)


def parameter_semantics_from_code(code: str) -> list[dict[str, Any]]:
    """Extract frozen parameter names/ranges from validated candidate code."""

    tree = ast.parse(code)
    metadata_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets
        ):
            metadata_node = node
            break
    if metadata_node is None:
        raise ValueError("candidate code is missing metadata")
    value = ast.literal_eval(metadata_node.value)
    if not isinstance(value, list) or not value:
        raise ValueError("candidate metadata must be a non-empty list")
    semantics = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"metadata[{index}] has no parameter name")
        bounds = entry.get("range")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"metadata[{index}] has no two-value range")
        semantics.append({"index": index, "name": entry["name"], "range": [float(bounds[0]), float(bounds[1])]})
    return semantics


def build_final_evaluation_receipt(
    *,
    protocol_receipt: Mapping[str, Any],
    model_code_sha256: str,
    parameters_sha256: str,
    prediction: Any | None,
    metrics: Mapping[str, Any] | None,
    status: str,
    failure: str | None = None,
) -> dict[str, Any]:
    """Build a receipt for a frozen final evaluation, including sealed hashes."""

    safe_metrics = None
    nonfinite_metrics: list[str] = []
    if metrics is not None:
        safe_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                safe_metrics[key] = int(value)
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                scalar = float("nan")
            if np.isfinite(scalar):
                safe_metrics[key] = scalar
            else:
                safe_metrics[key] = None
                nonfinite_metrics.append(str(key))
    if nonfinite_metrics:
        status = "failed"
        detail = "non-finite final metrics: " + ", ".join(nonfinite_metrics)
        failure = f"{failure}; {detail}" if failure else detail

    prediction_nonfinite = prediction is not None and not bool(np.all(np.isfinite(np.asarray(prediction))))
    if prediction_nonfinite:
        status = "failed"
        detail = "non-finite final prediction"
        failure = f"{failure}; {detail}" if failure else detail

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": protocol_receipt.get("protocol_version", CANONICAL_PROTOCOL_VERSION),
        "development_receipt_sha256": protocol_receipt.get("development_receipt_sha256"),
        "final_data": protocol_receipt.get("roles", {}).get(FINAL_ROLE),
        "frozen_model": {
            "model_code_sha256": model_code_sha256,
            "parameters_sha256": parameters_sha256,
        },
        "status": status,
        "metrics": safe_metrics,
        "failure": failure,
    }
    if prediction is not None:
        array = np.asarray(prediction)
        result["prediction"] = {
            "sha256": sha256_array(array),
            "shape": list(array.shape),
            "finite": bool(np.all(np.isfinite(array))),
        }
    else:
        result["prediction"] = None
    result["receipt_sha256"] = sha256_json(result)
    return result


__all__ = [
    "CANONICAL_PROTOCOL_VERSION",
    "CanonicalTrajectoryProtocol",
    "DEVELOPMENT_ROLES",
    "FINAL_ROLE",
    "TrajectoryPartition",
    "build_final_evaluation_receipt",
    "frozen_parameters_hash",
    "load_partition_manifest",
    "parameter_semantics_from_code",
    "sha256_array",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "split_trajectory",
    "write_protocol_manifests",
]
