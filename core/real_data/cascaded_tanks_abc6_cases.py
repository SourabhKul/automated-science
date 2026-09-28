"""Frozen, source-free case roster and post-fit target gate for ABC6.

This module constructs only the declared synthetic training traces.  A case
consumer receives an input/output prefix sized to its fit window, so the short
window API never returns training outputs 78--203.  Prospective outputs have no
builder or public module-level accessor: they are materialized by a gate object
only after every fit and baseline status receipt has been durably written and
revalidated.

This is a data-boundary convention for the reviewed synthetic protocol, not a
security boundary against arbitrary Python code in the same process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from core.real_data.cascaded_tanks_models import (
    TankModel,
    TankParameters,
    TankSimulationFailure,
    TankSimulationLimits,
    TankSimulationSuccess,
    TankState,
    simulate_cascaded_tanks,
)

PROTOCOL_ID: Final = "cascaded_tanks_abc6_synthetic_v1_20260928"
RUN_ID: Final = "ct-abc6-20260928-v1"
PARAMETER_ORDER: Final = ("a", "c", "p", "x1_0", "x2_0", "ceiling")
TRAINING_INPUT_L: Final = (8.0,) * 24 + (0.0,) * 180
TRAINING_INPUT_S: Final = TRAINING_INPUT_L[:78]
PROSPECTIVE_INPUT: Final = (3.0,) * 12 + (8.0,) * 12 + (0.0,) * 36
TRAINING_LENGTH: Final = 204
SHORT_LENGTH: Final = 78
PROSPECTIVE_LENGTH: Final = 60
MEASUREMENT_NOISE_SD: Final = 0.05
MISSING_FEEDTHROUGH: Final = 0.15
CASE_COUNT: Final = 24

CROSSING_PRIOR_BOUNDS: Final = (
    (0.40, 0.60),
    (0.30, 0.50),
    (0.40, 0.60),
    (0.0, 1.0),
    (0.0, 1.0),
    (2.6, 3.6),
)
NO_CROSSING_PRIOR_BOUNDS: Final = (
    (0.40, 0.60),
    (0.30, 0.50),
    (0.40, 0.60),
    (0.0, 1.0),
    (0.0, 1.0),
    (900.0, 1100.0),
)


@dataclass(frozen=True, slots=True)
class ABC6Case:
    """One frozen training fit in the reviewed 24-case order."""

    case_index: int
    case_id: str
    truth_id: str
    input_window: str
    replicate: int | None
    fit_model: TankModel

    @property
    def calibration_seed(self) -> int:
        return 9000 + self.case_index

    @property
    def abc_seed(self) -> int:
        return 10000 + self.case_index

    @property
    def input_length(self) -> int:
        return SHORT_LENGTH if self.input_window == "S" else TRAINING_LENGTH

    @property
    def prior_bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            NO_CROSSING_PRIOR_BOUNDS if self.truth_id == "N" else CROSSING_PRIOR_BOUNDS
        )

    @property
    def prior_bounds_by_parameter(self) -> dict[str, tuple[float, float]]:
        return dict(zip(PARAMETER_ORDER, self.prior_bounds, strict=True))


def _case(
    index: int,
    truth_id: str,
    window: str,
    replicate: int | None,
    fit_model: TankModel,
) -> ABC6Case:
    replicate_label = "noiseless" if replicate is None else f"r{replicate}"
    case_id = (
        f"case-{index:02d}-{truth_id}-{window}-{replicate_label}-{fit_model.value}"
    )
    return ABC6Case(index, case_id, truth_id, window, replicate, fit_model)


CASE_ROSTER: Final[tuple[ABC6Case, ...]] = (
    _case(0, "A", "S", 0, TankModel.O2),
    _case(1, "A", "S", 0, TankModel.C2),
    _case(2, "A", "L", 0, TankModel.O2),
    _case(3, "A", "L", 0, TankModel.C2),
    _case(4, "B", "S", 0, TankModel.C2),
    _case(5, "B", "S", 0, TankModel.O2),
    _case(6, "B", "L", 0, TankModel.C2),
    _case(7, "B", "L", 0, TankModel.O2),
    _case(8, "N", "L", None, TankModel.O2),
    _case(9, "N", "L", None, TankModel.C2),
    _case(10, "M", "L", None, TankModel.C2),
    _case(11, "M", "L", None, TankModel.O2),
    _case(12, "A", "S", 1, TankModel.O2),
    _case(13, "A", "L", 1, TankModel.O2),
    _case(14, "A", "S", 2, TankModel.O2),
    _case(15, "A", "L", 2, TankModel.O2),
    _case(16, "A", "S", 3, TankModel.O2),
    _case(17, "A", "L", 3, TankModel.O2),
    _case(18, "B", "S", 1, TankModel.C2),
    _case(19, "B", "L", 1, TankModel.C2),
    _case(20, "B", "S", 2, TankModel.C2),
    _case(21, "B", "L", 2, TankModel.C2),
    _case(22, "B", "S", 3, TankModel.C2),
    _case(23, "B", "L", 3, TankModel.C2),
)

_TRUTH_PARAMETERS: Final = MappingProxyType(
    {
        "A": (
            TankModel.O2,
            TankParameters(0.50, 0.40, 0.50),
            TankState(0.50, 0.50),
            3.0,
        ),
        "B": (
            TankModel.C2,
            TankParameters(0.54, 0.36, 0.46),
            TankState(0.25, 0.75),
            3.2,
        ),
        "N": (
            TankModel.C2,
            TankParameters(0.50, 0.40, 0.50),
            TankState(0.50, 0.50),
            1000.0,
        ),
    }
)
_FINAL_STATUSES: Final = frozenset({"complete", "incomplete", "unresolved", "failed"})
_RECEIPT_COMPONENTS: Final = ("fit", "baseline")
_RECEIPT_SCHEMA_VERSION: Final = 1
_MAX_STATUS_RECEIPT_BYTES: Final = 4096
_POSTFIT_GATE_SEAL = object()


class ABC6SyntheticSimulationError(RuntimeError):
    """A declared synthetic truth or deferred target could not be simulated."""


@dataclass(frozen=True, slots=True)
class ABC6TrainingCaseData:
    """The only observed arrays passed to one fit consumer."""

    case: ABC6Case
    inputs: tuple[float, ...]
    observed_outputs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.inputs) != self.case.input_length:
            raise ValueError("fit input length does not match the frozen window")
        if len(self.observed_outputs) != self.case.input_length:
            raise ValueError("fit output length does not match the frozen window")


@dataclass(frozen=True, slots=True)
class ABC6TrainingBundle:
    """Prefix-limited fit arrays and private latent states for target generation."""

    case_data: tuple[ABC6TrainingCaseData, ...]
    _terminal_states: tuple[tuple[str, TankState], ...] = field(repr=False)

    def data_for_case(self, case_index: int) -> ABC6TrainingCaseData:
        index = _checked_case_index(case_index)
        return self.case_data[index]

    def _terminal_state(self, truth_id: str) -> TankState:
        if truth_id == "M":
            truth_id = "B"
        for name, state in self._terminal_states:
            if name == truth_id:
                return state
        raise KeyError(f"no synthetic training terminal state for truth {truth_id!r}")


@dataclass(frozen=True, slots=True)
class ABC6ProspectiveTargets:
    """Deferred A/B noiseless targets and M's feedthrough observation."""

    _targets: tuple[tuple[str, tuple[float, ...]], ...] = field(repr=False)
    sha256_by_truth: tuple[tuple[str, str], ...]

    @property
    def truth_ids(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._targets)

    def for_truth(self, truth_id: str) -> tuple[float, ...]:
        for name, values in self._targets:
            if name == truth_id:
                return values
        raise KeyError(f"no prospective target is declared for {truth_id!r}")


def _checked_case_index(case_index: object) -> int:
    if isinstance(case_index, bool) or not isinstance(case_index, int):
        raise TypeError("case_index must be an integer in [0, 24)")
    if not 0 <= case_index < CASE_COUNT:
        raise IndexError("case_index must be in [0, 24)")
    return case_index


def case_by_index(case_index: int) -> ABC6Case:
    """Return one immutable entry from the exact frozen roster."""

    return CASE_ROSTER[_checked_case_index(case_index)]


def _finite_output_tuple(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a numeric sequence")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric sequence") from error
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a one-dimensional numeric sequence")
    result = tuple(float(value) for value in array)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _simulate_training_truth(
    truth_id: str,
    *,
    simulator: Callable[..., TankSimulationSuccess | TankSimulationFailure],
) -> tuple[tuple[float, ...], TankState]:
    model, parameters, initial_state, ceiling = _TRUTH_PARAMETERS[truth_id]
    outcome = simulator(
        TRAINING_INPUT_L,
        parameters,
        initial_state,
        model=model,
        ceiling=ceiling,
        limits=TankSimulationLimits(max_steps=TRAINING_LENGTH),
    )
    if isinstance(outcome, TankSimulationFailure):
        raise ABC6SyntheticSimulationError(
            f"synthetic truth {truth_id} failed at step {outcome.step_index}: "
            f"{outcome.category.value}"
        )
    if not isinstance(outcome, TankSimulationSuccess):
        raise TypeError("synthetic truth simulator returned an unknown outcome")
    outputs = _finite_output_tuple(outcome.observations, label="training truth")
    if len(outputs) != TRAINING_LENGTH:
        raise RuntimeError("synthetic truth simulator returned the wrong trace length")
    return outputs, outcome.terminal_state


TrainingOutputMutator = Callable[[str, tuple[float, ...]], Sequence[float]]


def build_synthetic_training_bundle(
    *,
    simulator: Callable[
        ..., TankSimulationSuccess | TankSimulationFailure
    ] = simulate_cascaded_tanks,
    training_output_mutator: TrainingOutputMutator | None = None,
) -> ABC6TrainingBundle:
    """Build the declared synthetic training observations without future outputs.

    ``training_output_mutator`` is a synthetic sentinel seam used to mutate the
    hidden training suffix while holding a short prefix fixed.  It receives
    only one synthetic 204-step truth path; it has no source-data capability.
    """

    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if training_output_mutator is not None and not callable(training_output_mutator):
        raise TypeError("training_output_mutator must be callable")

    raw_paths: dict[str, tuple[float, ...]] = {}
    terminal_states: list[tuple[str, TankState]] = []
    for truth_id in ("A", "B", "N"):
        outputs, terminal_state = _simulate_training_truth(
            truth_id, simulator=simulator
        )
        if training_output_mutator is not None:
            outputs = _finite_output_tuple(
                training_output_mutator(truth_id, outputs),
                label=f"mutated synthetic training truth {truth_id}",
            )
            if len(outputs) != TRAINING_LENGTH:
                raise ValueError("training output mutation changed trace length")
        raw_paths[truth_id] = outputs
        terminal_states.append((truth_id, terminal_state))

    noisy_paths: dict[tuple[str, int], tuple[float, ...]] = {}
    for truth_id, seed_base in (("A", 8100), ("B", 8200)):
        for replicate in range(4):
            noise = np.random.default_rng(seed_base + replicate).normal(
                0.0, MEASUREMENT_NOISE_SD, size=TRAINING_LENGTH
            )
            noisy_paths[(truth_id, replicate)] = tuple(
                float(observed + added)
                for observed, added in zip(raw_paths[truth_id], noise, strict=True)
            )

    case_data: list[ABC6TrainingCaseData] = []
    for case in CASE_ROSTER:
        if case.truth_id in {"A", "B"}:
            if case.replicate is None:
                raise AssertionError("A/B cases require a noise replicate")
            full_outputs = noisy_paths[(case.truth_id, case.replicate)]
        elif case.truth_id == "N":
            full_outputs = raw_paths["N"]
        elif case.truth_id == "M":
            full_outputs = tuple(
                float(observed + MISSING_FEEDTHROUGH * input_value)
                for observed, input_value in zip(
                    raw_paths["B"], TRAINING_INPUT_L, strict=True
                )
            )
        else:  # pragma: no cover - CASE_ROSTER is a frozen module constant.
            raise AssertionError(f"undeclared truth {case.truth_id!r}")

        fit_inputs = TRAINING_INPUT_S if case.input_window == "S" else TRAINING_INPUT_L
        fit_outputs = full_outputs[: case.input_length]
        case_data.append(
            ABC6TrainingCaseData(
                case=case,
                inputs=tuple(fit_inputs),
                observed_outputs=tuple(fit_outputs),
            )
        )

    return ABC6TrainingBundle(tuple(case_data), tuple(terminal_states))


class ABC6StatusReceiptError(ValueError):
    """A fit/baseline status receipt is missing, malformed, or non-durable."""


def _status_path(receipt_directory: Path, case_index: int, component: str) -> Path:
    return receipt_directory / f"case-{case_index:02d}.{component}-status.json"


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _receipt_contents(case_index: int, component: str, status: str) -> bytes:
    case = case_by_index(case_index)
    body: dict[str, object] = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "case_index": case_index,
        "case_id": case.case_id,
        "component": component,
        "status": status,
    }
    body["payload_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return _canonical_json(body)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_status_receipt(
    receipt_directory: str | os.PathLike[str],
    case_index: int,
    component: str,
    status: str,
) -> Path:
    """Create one immutable status receipt and fsync both file and directory.

    Receipt files use exclusive creation.  A duplicate write fails closed;
    partial/corrupt files also keep the post-fit gate closed and require a new
    run directory rather than an in-place retry.
    """

    index = _checked_case_index(case_index)
    if component not in _RECEIPT_COMPONENTS:
        raise ValueError("component must be 'fit' or 'baseline'")
    if status not in _FINAL_STATUSES:
        raise ValueError("status must be a final fit/baseline status")
    directory = Path(receipt_directory)
    if not directory.is_dir():
        raise FileNotFoundError("receipt directory must already exist")
    payload = _receipt_contents(index, component, status)
    path = _status_path(directory, index, component)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:  # pragma: no cover - guarded by the OS write contract.
                raise OSError("status receipt write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(directory)
    return path


def _read_and_validate_status_receipt(
    path: Path, *, case_index: int, component: str
) -> tuple[str, str]:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ABC6StatusReceiptError(f"missing durable receipt: {path.name}") from error
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_size > _MAX_STATUS_RECEIPT_BYTES
    ):
        raise ABC6StatusReceiptError(
            f"receipt is not a small regular file: {path.name}"
        )
    try:
        raw = path.read_bytes()
        decoded = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ABC6StatusReceiptError(f"receipt is unreadable: {path.name}") from error
    if not isinstance(decoded, dict):
        raise ABC6StatusReceiptError(f"receipt must contain a JSON object: {path.name}")
    if len(raw) > _MAX_STATUS_RECEIPT_BYTES:
        raise ABC6StatusReceiptError(f"receipt is too large: {path.name}")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "run_id",
        "case_index",
        "case_id",
        "component",
        "status",
        "payload_sha256",
    }
    if set(decoded) != expected_keys:
        raise ABC6StatusReceiptError(f"receipt fields do not match schema: {path.name}")
    if (
        type(decoded["schema_version"]) is not int
        or type(decoded["case_index"]) is not int
    ):
        raise ABC6StatusReceiptError(
            f"receipt integer fields have wrong types: {path.name}"
        )
    if any(
        not isinstance(decoded[key], str)
        for key in (
            "protocol_id",
            "run_id",
            "case_id",
            "component",
            "status",
            "payload_sha256",
        )
    ):
        raise ABC6StatusReceiptError(
            f"receipt text fields have wrong types: {path.name}"
        )
    case = case_by_index(case_index)
    expected_values = {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "run_id": RUN_ID,
        "case_index": case_index,
        "case_id": case.case_id,
        "component": component,
    }
    if any(decoded.get(key) != value for key, value in expected_values.items()):
        raise ABC6StatusReceiptError(f"receipt identity mismatch: {path.name}")
    status = decoded.get("status")
    if not isinstance(status, str) or status not in _FINAL_STATUSES:
        raise ABC6StatusReceiptError(f"receipt status is not final: {path.name}")
    payload_hash = decoded.get("payload_sha256")
    body = {key: value for key, value in decoded.items() if key != "payload_sha256"}
    expected_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
    if payload_hash != expected_hash or raw != _canonical_json(decoded):
        raise ABC6StatusReceiptError(f"receipt hash or encoding mismatch: {path.name}")
    return status, hashlib.sha256(raw).hexdigest()


def _verify_all_status_receipts(
    receipt_directory: Path,
) -> tuple[tuple[str, str], ...]:
    if not receipt_directory.is_dir():
        raise ABC6StatusReceiptError("status receipt directory does not exist")
    verified: list[tuple[str, str]] = []
    expected_names: set[str] = set()
    for case in CASE_ROSTER:
        for component in _RECEIPT_COMPONENTS:
            path = _status_path(receipt_directory, case.case_index, component)
            expected_names.add(path.name)
            _status, digest = _read_and_validate_status_receipt(
                path, case_index=case.case_index, component=component
            )
            verified.append((path.name, digest))
    unexpected_status_files = {
        path.name
        for path in receipt_directory.iterdir()
        if path.name.startswith("case-") and "-status.json" in path.name
    } - expected_names
    if unexpected_status_files:
        raise ABC6StatusReceiptError(
            "unexpected fit/baseline status receipt(s): "
            + ", ".join(sorted(unexpected_status_files))
        )
    return tuple(verified)


class DeferredABC6TargetGate:
    """Capability to generate prospective targets after all status receipts."""

    __slots__ = ("_receipt_directory", "_seal", "_verified_receipts")

    def __init__(
        self,
        receipt_directory: Path,
        verified_receipts: tuple[tuple[str, str], ...],
        *,
        _seal: object,
    ) -> None:
        if _seal is not _POSTFIT_GATE_SEAL:
            raise TypeError("use open_deferred_abc6_target_gate to obtain the gate")
        self._receipt_directory = receipt_directory
        self._verified_receipts = verified_receipts
        self._seal = _seal

    def generate_targets(
        self,
        training: ABC6TrainingBundle,
        *,
        simulator: Callable[
            ..., TankSimulationSuccess | TankSimulationFailure
        ] = simulate_cascaded_tanks,
    ) -> ABC6ProspectiveTargets:
        """Generate and hash A/B/M targets after rechecking the 48 receipts."""

        if self._seal is not _POSTFIT_GATE_SEAL:
            raise TypeError("invalid post-fit target gate")
        if not isinstance(training, ABC6TrainingBundle):
            raise TypeError("training must be the synthetic ABC6 training bundle")
        if not callable(simulator):
            raise TypeError("simulator must be callable")
        current = _verify_all_status_receipts(self._receipt_directory)
        if current != self._verified_receipts:
            raise ABC6StatusReceiptError(
                "fit/baseline status receipts changed after opening the target gate"
            )
        return _materialize_prospective_targets(training, simulator=simulator)


def open_deferred_abc6_target_gate(
    receipt_directory: str | os.PathLike[str],
) -> DeferredABC6TargetGate:
    """Open the target gate only after all 24 fit and 24 baseline receipts exist."""

    directory = Path(receipt_directory)
    verified = _verify_all_status_receipts(directory)
    return DeferredABC6TargetGate(
        directory,
        verified,
        _seal=_POSTFIT_GATE_SEAL,
    )


def _simulate_future(
    truth_id: str,
    training: ABC6TrainingBundle,
    *,
    simulator: Callable[..., TankSimulationSuccess | TankSimulationFailure],
) -> tuple[float, ...]:
    model, parameters, _initial_state, ceiling = _TRUTH_PARAMETERS[truth_id]
    outcome = simulator(
        PROSPECTIVE_INPUT,
        parameters,
        training._terminal_state(truth_id),
        model=model,
        ceiling=ceiling,
        limits=TankSimulationLimits(max_steps=PROSPECTIVE_LENGTH),
    )
    if isinstance(outcome, TankSimulationFailure):
        raise ABC6SyntheticSimulationError(
            f"prospective synthetic truth {truth_id} failed at step "
            f"{outcome.step_index}: {outcome.category.value}"
        )
    if not isinstance(outcome, TankSimulationSuccess):
        raise TypeError("prospective truth simulator returned an unknown outcome")
    outputs = _finite_output_tuple(outcome.observations, label="prospective truth")
    if len(outputs) != PROSPECTIVE_LENGTH:
        raise RuntimeError(
            "prospective truth simulator returned the wrong trace length"
        )
    return outputs


def _target_sha256(values: tuple[float, ...]) -> str:
    canonical = np.asarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _materialize_prospective_targets(
    training: ABC6TrainingBundle,
    *,
    simulator: Callable[..., TankSimulationSuccess | TankSimulationFailure],
) -> ABC6ProspectiveTargets:
    """Private target constructor; the public path is the verified gate method."""

    a_target = _simulate_future("A", training, simulator=simulator)
    b_target = _simulate_future("B", training, simulator=simulator)
    m_target = tuple(
        float(output + MISSING_FEEDTHROUGH * input_value)
        for output, input_value in zip(b_target, PROSPECTIVE_INPUT, strict=True)
    )
    targets = (("A", a_target), ("B", b_target), ("M", m_target))
    hashes = tuple((truth_id, _target_sha256(values)) for truth_id, values in targets)
    return ABC6ProspectiveTargets(targets, hashes)


__all__ = (
    "CASE_COUNT",
    "CASE_ROSTER",
    "CROSSING_PRIOR_BOUNDS",
    "MISSING_FEEDTHROUGH",
    "NO_CROSSING_PRIOR_BOUNDS",
    "PARAMETER_ORDER",
    "PROSPECTIVE_INPUT",
    "PROSPECTIVE_LENGTH",
    "PROTOCOL_ID",
    "RUN_ID",
    "SHORT_LENGTH",
    "TRAINING_INPUT_L",
    "TRAINING_INPUT_S",
    "TRAINING_LENGTH",
    "ABC6Case",
    "ABC6ProspectiveTargets",
    "ABC6StatusReceiptError",
    "ABC6SyntheticSimulationError",
    "ABC6TrainingBundle",
    "ABC6TrainingCaseData",
    "DeferredABC6TargetGate",
    "build_synthetic_training_bundle",
    "case_by_index",
    "open_deferred_abc6_target_gate",
    "write_status_receipt",
)
