"""Frozen, source-free case roster and post-fit target gate for ABC6.

This module constructs only the declared synthetic training traces.  A case
consumer receives an input/output prefix sized to its fit window, so the short
window API never returns training outputs 78--203. Prospective outputs have no
public module-level accessor: the gate verifies all fit and baseline receipts,
then requires a single-use scoring handoff backed by the fixed durable reveal
marker before materializing them. This protects the public API in the trusted
local checkout; it does not resist arbitrary Python code edits or direct use of
private module internals.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from core.real_data import cascaded_tanks_abc6_authority as authority_module
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
RECEIPT_ROOT_RELATIVE: Final = (
    "artifacts/cascaded_tanks_abc6_synthetic/ct-abc6-20260928-v1/receipts"
)
_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
_REVEAL_MARKER_PATH: Final = (
    _PROJECT_ROOT
    / "artifacts"
    / "evaluations"
    / "cascaded_tanks_abc6_scoring"
    / "claims"
    / f"{RUN_ID}.claim"
)
_TRAINING_SUMMARY_FILENAME: Final = "campaign.training-summary.json"
_FORECAST_ARTIFACT_FILENAME: Final = "campaign.target-free-forecasts.json"
_SCORING_SOURCE_PATHS: Final = (
    "scripts/run_cascaded_tanks_abc6_synthetic.py",
    "core/real_data/cascaded_tanks_abc6_forecast.py",
    "core/real_data/cascaded_tanks_abc6_scoring.py",
)
_MAX_REVEAL_MARKER_BYTES: Final = 256 * 1024
_MAX_HANDOFF_ARTIFACT_BYTES: Final = 128 * 1024 * 1024

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
_SCORING_HANDOFF_SEAL = object()
_AUTHORIZED_CASE_DATA_SEAL = object()
_AUTHORIZED_CASE_DATA_LOCK = threading.RLock()
_TRAINING_BUNDLE_AUTHORITIES: dict[
    int, tuple["ABC6TrainingBundle", authority_module.ABC6LaunchAuthority]
] = {}
_AUTHORIZED_CASE_DATA_REGISTRY: dict[int, "ABC6AuthorizedCaseData"] = {}


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


class ABC6AuthorizedCaseData:
    """One-use fit capability bound to a permit, bundle, and live authority."""

    __slots__ = (
        "_seal",
        "_authority",
        "_case_index",
        "_bundle",
        "_case_data",
        "_used",
        "_lock",
    )

    def __init__(
        self,
        seal: object,
        authority: authority_module.ABC6LaunchAuthority,
        case_index: int,
        bundle: ABC6TrainingBundle,
        case_data: ABC6TrainingCaseData,
    ) -> None:
        if (
            seal is not _AUTHORIZED_CASE_DATA_SEAL
            or type(authority) is not authority_module.ABC6LaunchAuthority
            or type(bundle) is not ABC6TrainingBundle
            or type(case_data) is not ABC6TrainingCaseData
            or type(case_index) is not int
        ):
            raise TypeError("authorized case data is issued only from a consumed case permit")
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_case_index", case_index)
        object.__setattr__(self, "_bundle", bundle)
        object.__setattr__(self, "_case_data", case_data)
        object.__setattr__(self, "_used", [False])
        object.__setattr__(self, "_lock", threading.Lock())

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("authorized case data is immutable")
        object.__setattr__(self, name, value)

    def __copy__(self) -> "ABC6AuthorizedCaseData":
        copied = object.__new__(type(self))
        for name in self.__slots__:
            object.__setattr__(copied, name, getattr(self, name))
        return copied

    def __deepcopy__(self, _memo: dict[int, object]) -> "ABC6AuthorizedCaseData":
        return self.__copy__()

    def _consume_for_fit(self) -> ABC6TrainingCaseData:
        with self._lock:
            with _AUTHORIZED_CASE_DATA_LOCK:
                if (
                    self._seal is not _AUTHORIZED_CASE_DATA_SEAL
                    or _AUTHORIZED_CASE_DATA_REGISTRY.get(id(self)) is not self
                ):
                    raise ABC6StatusReceiptError(
                        "authorized case data is copied, reconstructed, or unregistered"
                    )
            if self._used[0]:
                raise ABC6StatusReceiptError(
                    "authorized case data fit permit was already consumed"
                )
            # Consume before any validation that could be followed by numerical work.
            self._used[0] = True

        authority = self._authority
        authority._assert_live()
        with authority._state.lock:
            if authority._state.training_bundle is not self._bundle:
                raise ABC6StatusReceiptError(
                    "authorized case data no longer matches the grant's training bundle"
                )
        with _AUTHORIZED_CASE_DATA_LOCK:
            binding = _TRAINING_BUNDLE_AUTHORITIES.get(id(self._bundle))
            if (
                binding is None
                or binding[0] is not self._bundle
                or binding[1] is not authority
            ):
                raise ABC6StatusReceiptError(
                    "authorized case data is bound to a different training authority"
                )
            if (
                type(self._case_index) is not int
                or not 0 <= self._case_index < CASE_COUNT
                or self._bundle.case_data[self._case_index] is not self._case_data
                or self._case_data.case is not case_by_index(self._case_index)
            ):
                raise ABC6StatusReceiptError(
                    "authorized case data no longer references its exact bundle entry"
                )
        return self._case_data


def _bind_training_bundle_to_authority(
    bundle: ABC6TrainingBundle,
    authority: authority_module.ABC6LaunchAuthority,
) -> None:
    """Bind the just-built campaign bundle to its live grant exactly once."""

    if type(bundle) is not ABC6TrainingBundle:
        raise TypeError("training bundle must use the exact frozen ABC6 type")
    if type(authority) is not authority_module.ABC6LaunchAuthority:
        raise TypeError("training bundle requires the received ABC6 launch authority")
    authority._assert_live()
    with authority._state.lock:
        if not authority._state.training_started:
            raise authority_module.ABC6AuthorityError(
                "training bundle cannot precede campaign authorization"
            )
        if authority._state.training_bundle is not None:
            raise authority_module.ABC6AuthorityError(
                "training bundle authority binding is one-use"
            )
        with _AUTHORIZED_CASE_DATA_LOCK:
            previous = _TRAINING_BUNDLE_AUTHORITIES.get(id(bundle))
            if previous is not None:
                raise authority_module.ABC6AuthorityError(
                    "training bundle authority binding is one-use"
                )
            _TRAINING_BUNDLE_AUTHORITIES[id(bundle)] = (bundle, authority)
        authority._state.training_bundle = bundle


def get_training_case_data(
    bundle: ABC6TrainingBundle,
    case_index: int,
    permit: authority_module.ABC6TrainingPermit,
) -> ABC6AuthorizedCaseData:
    """Consume the exact case permit and issue one bundle-bound fit capability."""

    if type(bundle) is not ABC6TrainingBundle:
        raise TypeError("bundle must be the exact frozen ABC6 training bundle")
    index = _checked_case_index(case_index)
    if type(permit) is not authority_module.ABC6TrainingPermit:
        raise authority_module.ABC6AuthorityError(
            "training case requires an exact issued case permit"
        )
    authority = permit._authority
    if type(authority) is not authority_module.ABC6LaunchAuthority:
        raise authority_module.ABC6AuthorityError(
            "training case permit has no live launch authority"
        )
    with authority._state.lock:
        if authority._state.training_bundle is not bundle:
            raise authority_module.ABC6AuthorityError(
                "training case bundle differs from the grant's exact bundle"
            )
    with _AUTHORIZED_CASE_DATA_LOCK:
        binding = _TRAINING_BUNDLE_AUTHORITIES.get(id(bundle))
        if (
            binding is None
            or binding[0] is not bundle
            or binding[1] is not authority
        ):
            raise authority_module.ABC6AuthorityError(
                "training case permit and bundle belong to different launch authorities"
            )
    if permit.case_index != index:
        raise authority_module.ABC6AuthorityError(
            "training case permit has the wrong case index"
        )
    case_data = bundle.data_for_case(index)
    if (
        type(case_data) is not ABC6TrainingCaseData
        or case_data.case is not case_by_index(index)
    ):
        raise authority_module.ABC6AuthorityError(
            "training bundle case entry differs from frozen roster"
        )
    # This must remain a direct call from this frozen role function.
    permit.consume(index)
    authorized = ABC6AuthorizedCaseData(
        _AUTHORIZED_CASE_DATA_SEAL,
        authority,
        index,
        bundle,
        case_data,
    )
    with _AUTHORIZED_CASE_DATA_LOCK:
        _AUTHORIZED_CASE_DATA_REGISTRY[id(authorized)] = authorized
    return authorized


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
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ABC6StatusReceiptError(f"receipt is unreadable: {path.name}") from error
    return _validate_status_receipt_bytes(
        raw, path.name, case_index=case_index, component=component
    )


def _validate_status_receipt_bytes(
    raw: bytes, filename: str, *, case_index: int, component: str
) -> tuple[str, str]:
    try:
        decoded = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ABC6StatusReceiptError(
            f"receipt is unreadable: {filename}"
        ) from error
    if not isinstance(decoded, dict):
        raise ABC6StatusReceiptError(
            f"receipt must contain a JSON object: {filename}"
        )
    if len(raw) > _MAX_STATUS_RECEIPT_BYTES:
        raise ABC6StatusReceiptError(f"receipt is too large: {filename}")
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
        raise ABC6StatusReceiptError(
            f"receipt fields do not match schema: {filename}"
        )
    if (
        type(decoded["schema_version"]) is not int
        or type(decoded["case_index"]) is not int
    ):
        raise ABC6StatusReceiptError(
            f"receipt integer fields have wrong types: {filename}"
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
            f"receipt text fields have wrong types: {filename}"
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
        raise ABC6StatusReceiptError(f"receipt identity mismatch: {filename}")
    status = decoded.get("status")
    if not isinstance(status, str) or status not in _FINAL_STATUSES:
        raise ABC6StatusReceiptError(f"receipt status is not final: {filename}")
    payload_hash = decoded.get("payload_sha256")
    body = {key: value for key, value in decoded.items() if key != "payload_sha256"}
    expected_hash = hashlib.sha256(_canonical_json(body)).hexdigest()
    if payload_hash != expected_hash or raw != _canonical_json(decoded):
        raise ABC6StatusReceiptError(
            f"receipt hash or encoding mismatch: {filename}"
        )
    return status, hashlib.sha256(raw).hexdigest()


class _DirectoryAnchor:
    """Open directory identity held across an authorization handoff."""

    __slots__ = ("path", "descriptor", "device", "inode", "_closed")

    def __init__(self, path: Path, descriptor: int) -> None:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ABC6StatusReceiptError("anchored path is not a directory")
        self.path = path
        self.descriptor = descriptor
        self.device = info.st_dev
        self.inode = info.st_ino
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            os.close(self.descriptor)
            self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _open_execution_directory_anchors(
    receipt_root_identity: object,
) -> tuple[_DirectoryAnchor, _DirectoryAnchor]:
    """Duplicate the typed campaign's checkout and fixed receipt anchors."""

    from core.real_data import cascaded_tanks_abc6_campaign_fit as campaign_fit

    if not isinstance(receipt_root_identity, campaign_fit.ABC6ReceiptRootIdentity):
        raise ABC6StatusReceiptError(
            "target gate requires the typed campaign receipt-root identity"
        )
    identity = receipt_root_identity
    if identity.receipt_root_relative != RECEIPT_ROOT_RELATIVE:
        raise ABC6StatusReceiptError(
            "campaign receipt-root identity does not match the fixed run path"
        )
    try:
        identity.verify()
        root_path = Path(identity.repository_root_realpath)
        if (
            not root_path.is_absolute()
            or str(root_path) != identity.repository_root_realpath
            or _absolute_lexical_path(root_path) != root_path
        ):
            raise ABC6StatusReceiptError(
                "campaign checkout root identity is not a canonical absolute path"
            )
        root_fd = identity.duplicate_repository_root_fd()
        try:
            receipt_fd = identity.duplicate_receipt_root_fd()
        except BaseException:
            os.close(root_fd)
            raise
    except ABC6StatusReceiptError:
        raise
    except Exception as error:
        raise ABC6StatusReceiptError(
            "campaign execution has no live physical receipt-root identity"
        ) from error

    root_anchor: _DirectoryAnchor | None = None
    receipt_anchor: _DirectoryAnchor | None = None
    try:
        root_anchor = _DirectoryAnchor(root_path, root_fd)
        root_fd = -1
        receipt_path = root_path / RECEIPT_ROOT_RELATIVE
        receipt_anchor = _DirectoryAnchor(receipt_path, receipt_fd)
        receipt_fd = -1
        runtime = identity.runtime_identity
        root_info = os.fstat(root_anchor.descriptor)
        receipt_info = os.fstat(receipt_anchor.descriptor)
        if (
            (root_info.st_dev, root_info.st_ino)
            != (
                runtime["repository_root_device"],
                runtime["repository_root_inode"],
            )
            or (receipt_info.st_dev, receipt_info.st_ino)
            != (
                runtime["receipt_root_device"],
                runtime["receipt_root_inode"],
            )
        ):
            raise ABC6StatusReceiptError(
                "campaign execution descriptor identities do not match their receipt"
            )
        current_receipt = _open_relative_directory_anchor(
            root_anchor,
            RECEIPT_ROOT_RELATIVE,
            label="campaign receipt root",
        )
        try:
            if (current_receipt.device, current_receipt.inode) != (
                receipt_anchor.device,
                receipt_anchor.inode,
            ):
                raise ABC6StatusReceiptError(
                    "campaign receipt root is not beneath its pinned checkout root"
                )
        finally:
            current_receipt.close()
        _verify_directory_anchor_path(root_anchor, label="campaign checkout root")
        _verify_directory_anchor_path(
            receipt_anchor, label="campaign receipt root"
        )
        return root_anchor, receipt_anchor
    except BaseException:
        if root_anchor is not None:
            root_anchor.close()
        elif root_fd >= 0:
            os.close(root_fd)
        if receipt_anchor is not None:
            receipt_anchor.close()
        elif receipt_fd >= 0:
            os.close(receipt_fd)
        raise


def _absolute_lexical_path(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_directory_anchor(
    path: str | os.PathLike[str], *, label: str, canonicalize_root: bool = False
) -> _DirectoryAnchor:
    """Open an absolute directory one no-follow component at a time."""

    absolute = _absolute_lexical_path(path)
    if canonicalize_root:
        try:
            leaf_info = absolute.lstat()
            if not stat.S_ISDIR(leaf_info.st_mode):
                raise ABC6StatusReceiptError(
                    f"{label} leaf must be a real directory"
                )
            absolute = absolute.resolve(strict=True)
        except ABC6StatusReceiptError:
            raise
        except OSError as error:
            raise ABC6StatusReceiptError(f"{label} root is missing or unsafe") from error
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        current_fd = os.open(os.sep, flags)
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ABC6StatusReceiptError(f"{label} path is not canonical")
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as error:
                raise ABC6StatusReceiptError(
                    f"{label} ancestry is missing or contains a symlink"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        return _DirectoryAnchor(absolute, current_fd)
    except ABC6StatusReceiptError:
        if "current_fd" in locals():
            os.close(current_fd)
        raise
    except OSError as error:
        if "current_fd" in locals():
            os.close(current_fd)
        raise ABC6StatusReceiptError(
            f"{label} ancestry is missing or contains a symlink"
        ) from error


def _validate_relative_directory_path(value: str, *, label: str) -> tuple[str, ...]:
    if (
        type(value) is not str
        or not value
        or Path(value).is_absolute()
        or "\\" in value
        or str(Path(value)) != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ABC6StatusReceiptError(f"{label} must be a canonical relative path")
    return tuple(value.split("/"))


def _open_relative_directory_anchor(
    root_anchor: _DirectoryAnchor,
    relative_path: str,
    *,
    label: str,
    create: bool = False,
) -> _DirectoryAnchor:
    """Open a directory below an already-pinned physical checkout root."""

    if root_anchor._closed:
        raise ABC6StatusReceiptError("checkout root directory anchor is closed")
    parts = _validate_relative_directory_path(relative_path, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.dup(root_anchor.descriptor)
    try:
        for component in parts:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    os.fsync(current_fd)
                except FileExistsError:
                    # The no-follow open below validates the concurrent entry.
                    pass
                next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        if create:
            os.fsync(current_fd)
        return _DirectoryAnchor(root_anchor.path.joinpath(*parts), current_fd)
    except ABC6StatusReceiptError:
        os.close(current_fd)
        raise
    except OSError as error:
        os.close(current_fd)
        raise ABC6StatusReceiptError(
            f"{label} ancestry is missing or contains a symlink"
        ) from error


def _verify_directory_anchor_path(anchor: _DirectoryAnchor, *, label: str) -> None:
    if anchor._closed:
        raise ABC6StatusReceiptError(f"{label} directory anchor is closed")
    try:
        pinned_info = os.fstat(anchor.descriptor)
    except OSError as error:
        raise ABC6StatusReceiptError(f"{label} directory anchor is invalid") from error
    if (
        not stat.S_ISDIR(pinned_info.st_mode)
        or pinned_info.st_dev != anchor.device
        or pinned_info.st_ino != anchor.inode
    ):
        raise ABC6StatusReceiptError(f"{label} directory identity changed")
    current = _open_directory_anchor(anchor.path, label=label)
    try:
        if current.device != anchor.device or current.inode != anchor.inode:
            raise ABC6StatusReceiptError(f"{label} directory identity changed")
    finally:
        current.close()


def _read_regular_file_at(
    directory_fd: int,
    filename: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if filename in {"", ".", ".."} or Path(filename).name != filename:
        raise ABC6StatusReceiptError(f"{label} filename is not a leaf")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise ABC6StatusReceiptError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ABC6StatusReceiptError(f"{label} exceeds its size limit")
        return b"".join(chunks)
    except ABC6StatusReceiptError:
        raise
    except FileNotFoundError as error:
        if label == "status receipt":
            raise ABC6StatusReceiptError(
                f"missing durable receipt: {filename}"
            ) from error
        raise ABC6StatusReceiptError(f"{label} is missing") from error
    except OSError as error:
        raise ABC6StatusReceiptError(f"{label} is missing or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


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


def _verify_all_status_receipts_at(
    receipt_directory_fd: int,
) -> tuple[tuple[str, str], ...]:
    """Verify the fixed receipt roster relative to a pinned directory fd."""

    verified: list[tuple[str, str]] = []
    expected_names: set[str] = set()
    for case in CASE_ROSTER:
        for component in _RECEIPT_COMPONENTS:
            filename = _status_path(Path("."), case.case_index, component).name
            expected_names.add(filename)
            raw = _read_regular_file_at(
                receipt_directory_fd,
                filename,
                maximum_bytes=_MAX_STATUS_RECEIPT_BYTES,
                label="status receipt",
            )
            _status, digest = _validate_status_receipt_bytes(
                raw,
                filename,
                case_index=case.case_index,
                component=component,
            )
            verified.append((filename, digest))
    try:
        entries = set(os.listdir(receipt_directory_fd))
    except OSError as error:
        raise ABC6StatusReceiptError(
            "cannot enumerate the anchored status receipt directory"
        ) from error
    unexpected = {
        name
        for name in entries
        if name.startswith("case-") and "-status.json" in name
    } - expected_names
    if unexpected:
        raise ABC6StatusReceiptError(
            "unexpected fit/baseline status receipt(s): "
            + ", ".join(sorted(unexpected))
        )
    return tuple(verified)


def _read_fixed_reveal_marker(
    marker_sha256: str,
    marker_anchor: _DirectoryAnchor,
    marker_filename: str,
) -> dict[str, object]:
    raw = _read_regular_file_at(
        marker_anchor.descriptor,
        marker_filename,
        maximum_bytes=_MAX_REVEAL_MARKER_BYTES,
        label="fixed scoring reveal marker",
    )

    if hashlib.sha256(raw).hexdigest() != marker_sha256:
        raise ABC6StatusReceiptError("scoring reveal marker digest changed")

    def reject_duplicate_keys(pairs):
        decoded: dict[str, object] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"duplicate marker key: {key}")
            decoded[key] = value
        return decoded

    try:
        marker = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid marker number: {value}")
            ),
        )
        if type(marker) is not dict:
            raise ValueError("marker must be an object")
        canonical = json.dumps(
            marker,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if canonical != raw:
            raise ValueError("marker is not canonical JSON")
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ABC6StatusReceiptError("scoring reveal marker is malformed") from error
    return marker


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fixed_reveal_marker_path(root_anchor: _DirectoryAnchor) -> Path:
    relative = Path(
        "artifacts/evaluations/cascaded_tanks_abc6_scoring/claims"
    ) / f"{RUN_ID}.claim"
    expected = root_anchor.path / relative
    configured = _absolute_lexical_path(_REVEAL_MARKER_PATH)
    if configured != expected:
        raise ABC6StatusReceiptError(
            "reveal marker path differs from the fixed checkout-root claim path"
        )
    return expected


def _validate_scoring_marker_for_gate(
    marker_sha256: str,
    marker_anchor: _DirectoryAnchor,
    marker_filename: str,
    root_anchor: _DirectoryAnchor,
    receipt_anchor: _DirectoryAnchor,
    source_anchors: tuple[tuple[str, _DirectoryAnchor], ...],
    verified_receipts: tuple[tuple[str, str], ...],
) -> None:
    _verify_directory_anchor_path(root_anchor, label="campaign checkout root")
    _verify_directory_anchor_path(marker_anchor, label="reveal marker parent")
    _verify_directory_anchor_path(receipt_anchor, label="status receipt directory")
    expected_marker = _fixed_reveal_marker_path(root_anchor)
    if marker_anchor.path != expected_marker.parent or marker_filename != expected_marker.name:
        raise ABC6StatusReceiptError(
            "scoring reveal marker is not under the pinned checkout root"
        )
    if receipt_anchor.path != root_anchor.path / RECEIPT_ROOT_RELATIVE:
        raise ABC6StatusReceiptError(
            "status receipt directory is not under the pinned checkout root"
        )
    marker = _read_fixed_reveal_marker(
        marker_sha256, marker_anchor, marker_filename
    )
    expected_keys = {
        "schema",
        "protocol_id",
        "run_id",
        "condition",
        "semantics",
        "forecast_roster_sha256",
        "forecast_artifact_sha256",
        "training_manifest_sha256",
        "training_evidence_manifest_sha256",
        "integrated_source_hashes",
        "status_receipt_sha256",
        "training_summary_sha256",
        "created_at_utc",
    }
    if set(marker) != expected_keys:
        raise ABC6StatusReceiptError("scoring reveal marker has an unexpected schema")
    if (
        marker.get("schema") != "cascaded-tanks-abc6-synthetic-reveal-v1"
        or marker.get("protocol_id") != PROTOCOL_ID
        or marker.get("run_id") != RUN_ID
        or marker.get("condition") != "synthetic-prospective-target-confirmation-v1"
        or marker.get("semantics") != "consumed-on-create; success-or-failure; no-retry"
        or not isinstance(marker.get("created_at_utc"), str)
    ):
        raise ABC6StatusReceiptError("scoring reveal marker identity is invalid")

    marker_receipts = marker.get("status_receipt_sha256")
    expected_serialized_receipts = [list(pair) for pair in verified_receipts]
    if marker_receipts != expected_serialized_receipts:
        raise ABC6StatusReceiptError(
            "scoring reveal marker is not bound to this exact receipt snapshot"
        )

    for field_name in (
        "forecast_roster_sha256",
        "forecast_artifact_sha256",
        "training_manifest_sha256",
        "training_evidence_manifest_sha256",
        "training_summary_sha256",
    ):
        if not _is_lower_sha256(marker.get(field_name)):
            raise ABC6StatusReceiptError(
                f"scoring reveal marker {field_name} is invalid"
            )

    source_hashes = marker.get("integrated_source_hashes")
    if type(source_hashes) is not list or len(source_hashes) != len(
        _SCORING_SOURCE_PATHS
    ):
        raise ABC6StatusReceiptError(
            "scoring reveal marker source roster is incomplete"
        )
    source_anchor_by_path = dict(source_anchors)
    for row, expected_path in zip(
        source_hashes, _SCORING_SOURCE_PATHS, strict=True
    ):
        if (
            type(row) is not list
            or len(row) != 2
            or row[0] != expected_path
            or not _is_lower_sha256(row[1])
        ):
            raise ABC6StatusReceiptError(
                "scoring reveal marker source hashes do not match the integrated roster"
            )
        source_anchor = source_anchor_by_path.get(expected_path)
        if source_anchor is None:
            raise ABC6StatusReceiptError(
                "integrated source directory anchors are incomplete"
            )
        if source_anchor.path != root_anchor.path / Path(expected_path).parent:
            raise ABC6StatusReceiptError(
                "integrated source anchor is not under the pinned checkout root"
            )
        _verify_directory_anchor_path(
            source_anchor, label=f"integrated source parent for {expected_path}"
        )
        source_bytes = _read_regular_file_at(
            source_anchor.descriptor,
            Path(expected_path).name,
            maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
            label="integrated scoring source",
        )
        if hashlib.sha256(source_bytes).hexdigest() != row[1]:
            raise ABC6StatusReceiptError(
                "integrated source changed after the scoring reveal claim"
            )

    summary_bytes = _read_regular_file_at(
        receipt_anchor.descriptor,
        _TRAINING_SUMMARY_FILENAME,
        maximum_bytes=1_000_000,
        label="training summary",
    )
    if hashlib.sha256(summary_bytes).hexdigest() != marker["training_summary_sha256"]:
        raise ABC6StatusReceiptError(
            "scoring reveal marker is not bound to this training summary"
        )
    forecast_bytes = _read_regular_file_at(
        receipt_anchor.descriptor,
        _FORECAST_ARTIFACT_FILENAME,
        maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
        label="target-free forecast artifact",
    )
    if hashlib.sha256(forecast_bytes).hexdigest() != marker[
        "forecast_artifact_sha256"
    ]:
        raise ABC6StatusReceiptError(
            "scoring reveal marker is not bound to this forecast artifact"
        )


class _ABC6ScoringTargetHandoff:
    """Private one-use capability issued only after the scorer claims a marker."""

    __slots__ = (
        "_gate",
        "_marker_sha256",
        "_verified_receipts",
        "_marker_path",
        "_marker_filename",
        "_marker_anchor",
        "_source_anchors",
        "_seal",
        "_used",
        "_materialization_started",
        "_lock",
    )

    def __init__(
        self,
        gate: "DeferredABC6TargetGate",
        marker_sha256: str,
        marker_anchor: _DirectoryAnchor,
        marker_filename: str,
        source_anchors: tuple[tuple[str, _DirectoryAnchor], ...],
        *,
        _seal: object,
    ) -> None:
        if _seal is not _SCORING_HANDOFF_SEAL:
            raise TypeError("scoring handoffs are issued only by the private scorer seam")
        self._gate = gate
        self._marker_sha256 = marker_sha256
        self._verified_receipts = gate._verified_receipts
        self._marker_path = _fixed_reveal_marker_path(gate._root_anchor)
        self._marker_filename = marker_filename
        self._marker_anchor = marker_anchor
        self._source_anchors = source_anchors
        self._seal = _seal
        self._used = False
        self._materialization_started = False
        self._lock = threading.Lock()

    def _close_anchors(self) -> None:
        self._marker_anchor.close()
        for _source_path, source_anchor in self._source_anchors:
            source_anchor.close()
        self._gate._receipt_anchor.close()
        self._gate._root_anchor.close()

    def __del__(self) -> None:
        try:
            self._close_anchors()
        except (AttributeError, OSError):
            pass

    def _consume_for_gate(self, gate: "DeferredABC6TargetGate") -> None:
        with self._lock:
            if self._seal is not _SCORING_HANDOFF_SEAL or self._gate is not gate:
                raise ABC6StatusReceiptError(
                    "scoring reveal handoff is not bound to this target gate"
                )
            if self._used:
                raise ABC6StatusReceiptError("scoring reveal handoff was already used")
            self._used = True

    def _begin_materialization(self, gate: "DeferredABC6TargetGate") -> None:
        with self._lock:
            if (
                self._seal is not _SCORING_HANDOFF_SEAL
                or self._gate is not gate
                or not self._used
                or self._materialization_started
            ):
                raise ABC6StatusReceiptError(
                    "scoring reveal handoff is invalid or already consumed"
                )
            self._materialization_started = True


def _issue_scoring_target_handoff(
    gate: "DeferredABC6TargetGate",
    marker_sha256: str,
    *,
    marker_anchor: _DirectoryAnchor,
) -> _ABC6ScoringTargetHandoff:
    """Private scorer seam; verify its durable marker before minting capability."""

    if not isinstance(gate, DeferredABC6TargetGate) or gate._seal is not _POSTFIT_GATE_SEAL:
        marker_anchor.close()
        raise ABC6StatusReceiptError("scoring handoff requires an opened status gate")
    if not _is_lower_sha256(marker_sha256):
        marker_anchor.close()
        raise ABC6StatusReceiptError("scoring reveal marker digest is invalid")
    with gate._handoff_lock:
        if gate._handoff_issued:
            marker_anchor.close()
            raise ABC6StatusReceiptError(
                "scoring target handoff was already issued; reveal remains consumed"
            )
        gate._handoff_issued = True
    try:
        _verify_directory_anchor_path(
            gate._root_anchor, label="campaign checkout root"
        )
        _verify_directory_anchor_path(
            gate._receipt_anchor, label="status receipt directory"
        )
        expected_marker = _fixed_reveal_marker_path(gate._root_anchor)
        if marker_anchor.path != expected_marker.parent:
            raise ABC6StatusReceiptError(
                "scorer marker directory anchor does not match the pinned checkout root"
            )
        current_receipts = _verify_all_status_receipts_at(
            gate._receipt_anchor.descriptor
        )
        if current_receipts != gate._verified_receipts:
            raise ABC6StatusReceiptError(
                "fit/baseline status receipts changed before scorer handoff"
            )
    except BaseException:
        marker_anchor.close()
        raise
    source_anchors: list[tuple[str, _DirectoryAnchor]] = []
    try:
        anchors_by_parent: dict[Path, _DirectoryAnchor] = {}
        for relative_path in _SCORING_SOURCE_PATHS:
            source_parent_relative = Path(relative_path).parent.as_posix()
            source_parent = gate._root_anchor.path / source_parent_relative
            source_anchor = anchors_by_parent.get(source_parent)
            if source_anchor is None:
                source_anchor = _open_relative_directory_anchor(
                    gate._root_anchor,
                    source_parent_relative,
                    label=f"integrated source parent for {relative_path}",
                )
                anchors_by_parent[source_parent] = source_anchor
            source_anchors.append((relative_path, source_anchor))
        source_anchor_tuple = tuple(source_anchors)
        _validate_scoring_marker_for_gate(
            marker_sha256,
            marker_anchor,
            expected_marker.name,
            gate._root_anchor,
            gate._receipt_anchor,
            source_anchor_tuple,
            gate._verified_receipts,
        )
        return _ABC6ScoringTargetHandoff(
            gate,
            marker_sha256,
            marker_anchor,
            expected_marker.name,
            source_anchor_tuple,
            _seal=_SCORING_HANDOFF_SEAL,
        )
    except BaseException:
        marker_anchor.close()
        for _source_path, source_anchor in source_anchors:
            source_anchor.close()
        raise


def _validate_scoring_target_handoff(
    gate: "DeferredABC6TargetGate", handoff: _ABC6ScoringTargetHandoff
) -> None:
    if (
        handoff._seal is not _SCORING_HANDOFF_SEAL
        or handoff._gate is not gate
        or handoff._verified_receipts != gate._verified_receipts
        or handoff._marker_path != _fixed_reveal_marker_path(gate._root_anchor)
    ):
        raise ABC6StatusReceiptError(
            "scoring reveal handoff does not match this gate and marker path"
        )
    _verify_directory_anchor_path(
        gate._root_anchor, label="campaign checkout root"
    )
    _verify_directory_anchor_path(
        gate._receipt_anchor, label="status receipt directory"
    )
    current_receipts = _verify_all_status_receipts_at(
        gate._receipt_anchor.descriptor
    )
    if current_receipts != gate._verified_receipts:
        raise ABC6StatusReceiptError(
            "fit/baseline status receipts changed after scorer handoff"
        )
    _validate_scoring_marker_for_gate(
        handoff._marker_sha256,
        handoff._marker_anchor,
        handoff._marker_filename,
        gate._root_anchor,
        gate._receipt_anchor,
        handoff._source_anchors,
        gate._verified_receipts,
    )


class DeferredABC6TargetGate:
    """Status-verified gate requiring a one-use, durable scorer handoff."""

    __slots__ = (
        "_root_anchor",
        "_receipt_directory",
        "_receipt_anchor",
        "_seal",
        "_verified_receipts",
        "_handoff_issued",
        "_handoff_lock",
        "_generation_started",
        "_generation_lock",
    )

    def __init__(
        self,
        root_anchor: _DirectoryAnchor,
        receipt_anchor: _DirectoryAnchor,
        verified_receipts: tuple[tuple[str, str], ...],
        *,
        _seal: object,
    ) -> None:
        if _seal is not _POSTFIT_GATE_SEAL:
            raise TypeError("use open_deferred_abc6_target_gate to obtain the gate")
        self._root_anchor = root_anchor
        self._receipt_directory = receipt_anchor.path
        self._receipt_anchor = receipt_anchor
        self._verified_receipts = verified_receipts
        self._seal = _seal
        self._handoff_issued = False
        self._handoff_lock = threading.Lock()
        self._generation_started = False
        self._generation_lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self._receipt_anchor.close()
            self._root_anchor.close()
        except (AttributeError, OSError):
            pass

    def generate_targets(
        self,
        training: ABC6TrainingBundle,
        *,
        simulator: Callable[
            ..., TankSimulationSuccess | TankSimulationFailure
        ] = simulate_cascaded_tanks,
        _scoring_handoff: _ABC6ScoringTargetHandoff | None = None,
    ) -> ABC6ProspectiveTargets:
        """Generate targets only through the private scorer-issued handoff."""

        if self._seal is not _POSTFIT_GATE_SEAL:
            raise TypeError("invalid post-fit target gate")
        if not isinstance(training, ABC6TrainingBundle):
            raise TypeError("training must be the synthetic ABC6 training bundle")
        if not callable(simulator):
            raise TypeError("simulator must be callable")
        if not isinstance(_scoring_handoff, _ABC6ScoringTargetHandoff):
            raise ABC6StatusReceiptError(
                "prospective targets require the scorer's durable reveal handoff"
            )
        with self._generation_lock:
            if self._generation_started:
                raise ABC6StatusReceiptError(
                    "target generation already started; reveal condition is consumed"
                )
            if (
                _scoring_handoff._seal is not _SCORING_HANDOFF_SEAL
                or _scoring_handoff._gate is not self
            ):
                raise ABC6StatusReceiptError(
                    "scoring reveal handoff is not bound to this target gate"
                )
            # Mark the attempt terminal before touching files or calling the
            # simulator, so any uncertainty after the reveal marker is no-retry.
            self._generation_started = True
            _scoring_handoff._consume_for_gate(self)
        try:
            _validate_scoring_target_handoff(self, _scoring_handoff)
            return _materialize_prospective_targets(
                training,
                simulator=simulator,
                _scoring_handoff=_scoring_handoff,
            )
        finally:
            _scoring_handoff._close_anchors()


def open_deferred_abc6_target_gate(
    receipt_root_identity: object,
) -> DeferredABC6TargetGate:
    """Open a target gate from a typed campaign identity after all receipts exist."""

    root_anchor, receipt_anchor = _open_execution_directory_anchors(
        receipt_root_identity
    )
    try:
        verified = _verify_all_status_receipts_at(receipt_anchor.descriptor)
        return DeferredABC6TargetGate(
            root_anchor,
            receipt_anchor,
            verified,
            _seal=_POSTFIT_GATE_SEAL,
        )
    except BaseException:
        root_anchor.close()
        receipt_anchor.close()
        raise


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
    _scoring_handoff: _ABC6ScoringTargetHandoff,
) -> ABC6ProspectiveTargets:
    """Private constructor that consumes the scorer's already-used handoff."""

    if not isinstance(training, ABC6TrainingBundle):
        raise TypeError("training must be the synthetic ABC6 training bundle")
    if not callable(simulator):
        raise TypeError("simulator must be callable")
    if not isinstance(_scoring_handoff, _ABC6ScoringTargetHandoff):
        raise ABC6StatusReceiptError("target constructor requires a scoring handoff")
    gate = _scoring_handoff._gate
    _scoring_handoff._begin_materialization(gate)
    _validate_scoring_target_handoff(gate, _scoring_handoff)
    return _materialize_prospective_targets_impl(training, simulator=simulator)


def _materialize_prospective_targets_for_test(
    training: ABC6TrainingBundle,
    *,
    simulator: Callable[..., TankSimulationSuccess | TankSimulationFailure],
) -> ABC6ProspectiveTargets:
    """Private test-only seam for deterministic fake future simulators.

    Production callers must use the scorer-issued target gate. This seam is
    intentionally absent from ``__all__`` and requires an explicit simulator
    so offline unit tests never fall back to the prospective truth simulator.
    """

    if not isinstance(training, ABC6TrainingBundle):
        raise TypeError("training must be the synthetic ABC6 training bundle")
    if not callable(simulator):
        raise TypeError("simulator must be callable")
    return _materialize_prospective_targets_impl(training, simulator=simulator)


def _materialize_prospective_targets_impl(
    training: ABC6TrainingBundle,
    *,
    simulator: Callable[..., TankSimulationSuccess | TankSimulationFailure],
) -> ABC6ProspectiveTargets:
    """Shared low-level constructor; callers are either gated or test-private."""

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
    "ABC6AuthorizedCaseData",
    "ABC6ProspectiveTargets",
    "ABC6StatusReceiptError",
    "ABC6SyntheticSimulationError",
    "ABC6TrainingBundle",
    "ABC6TrainingCaseData",
    "DeferredABC6TargetGate",
    "build_synthetic_training_bundle",
    "case_by_index",
    "get_training_case_data",
    "open_deferred_abc6_target_gate",
    "write_status_receipt",
)
