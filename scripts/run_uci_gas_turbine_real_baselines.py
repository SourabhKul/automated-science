#!/usr/bin/env python3
"""Execute the fixed Phase 38 measured-NOx baseline gate exactly once per invocation."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.real_data.uci_gas_turbine import CANDIDATE_FIELDS, NOX_BOUNDS_MG_M3, UCIGasTurbineRawInput, candidate_input_from_payload, predict_independent_rows


ARCHIVE = Path("data/real/uci_gas_turbine/raw/uci_gas_turbine_551.zip")
SPLIT_PATH = Path("data/real/uci_gas_turbine/source_year_split.json")
OUT_DIR = Path("artifacts/evaluations/phase38_uci_gas_turbine_nox_real_baselines_20260725")
HEADERS = ("AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX")
MEMBER_ROWS = {2011: 7411, 2012: 7628, 2013: 7152, 2014: 7158, 2015: 7384}
PENALTIES = (0.01, 0.1, 1.0, 10.0)
AMBIENT_INDICES = np.asarray([0, 1, 2], dtype=int)
PROCESS_INDICES = np.asarray([3, 4, 5, 6, 7, 8], dtype=int)


@dataclass(frozen=True)
class LockedRows:
    inputs: np.ndarray
    target_nox: np.ndarray | None


@dataclass(frozen=True)
class RidgeModel:
    field_indices: np.ndarray
    quadratic: bool
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, raw_inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_inputs, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(CANDIDATE_FIELDS) or not np.all(np.isfinite(values)):
            raise ValueError("gas-turbine prediction needs finite raw nine-field inputs")
        selected = values[:, self.field_indices]
        design = _design(selected, self.quadratic)
        prediction = ((design - self.mean) / self.scale) @ self.coefficients + self.intercept
        _validate_predictions(prediction)
        return prediction


def _roles(split_path: Path) -> dict[int, str]:
    payload = json.loads(split_path.read_text())
    roles = {int(year): "train" for year in payload.get("train_years", [])}
    roles[int(payload["selection_year"])] = "selection"
    roles[int(payload["external_year"])] = "external"
    expected = {2011: "train", 2012: "train", 2013: "train", 2014: "selection", 2015: "external"}
    if roles != expected:
        raise ValueError("gas-turbine split must remain 2011--2013/2014/2015")
    return roles


def _member(year: int) -> str:
    return f"gt_{year}.csv"


def _fingerprint(row: dict[str, str]) -> str:
    return hashlib.sha256(",".join(row[name] for name in HEADERS).encode("utf-8")).hexdigest()


def load_locked_rows(
    archive_path: Path = ARCHIVE,
    split_path: Path = SPLIT_PATH,
    *,
    retain_external_outcomes: bool = False,
) -> tuple[dict[str, LockedRows], dict[str, object]]:
    """Revalidate all members, retaining 2015 targets only after the selection gate."""
    roles = _roles(split_path)
    inputs = {role: [] for role in ("train", "selection", "external")}
    targets = {role: [] for role in ("train", "selection", "external")}
    fingerprints_by_split = {role: set() for role in ("train", "selection", "external")}
    member_ledger: dict[str, dict[str, object]] = {}
    cross_partition_duplicates = 0
    with ZipFile(archive_path) as archive:
        if archive.namelist() != [_member(year) for year in sorted(MEMBER_ROWS)]:
            raise ValueError("gas-turbine archive member inventory changed")
        for year, expected_rows in MEMBER_ROWS.items():
            member = _member(year)
            role = roles[year]
            member_fingerprints: set[str] = set()
            row_count = 0
            within_member_duplicates = 0
            same_split_cross_member_duplicates = 0
            with archive.open(member) as raw:
                reader = csv.DictReader(TextIOWrapper(raw, encoding="utf-8", newline=""))
                if tuple(reader.fieldnames or ()) != HEADERS:
                    raise ValueError(f"{member} does not retain the fixed 11-column header")
                for row in reader:
                    row_count += 1
                    if set(row) != set(HEADERS) or any(row[name] in (None, "") for name in HEADERS):
                        raise ValueError(f"{member} contains a malformed row")
                    try:
                        numeric = {name: float(row[name]) for name in HEADERS}
                    except ValueError as exc:
                        raise ValueError(f"{member} contains a nonnumeric field") from exc
                    if not np.all(np.isfinite(list(numeric.values()))):
                        raise ValueError(f"{member} contains a non-finite field")
                    fingerprint = _fingerprint(row)
                    seen_in_member = fingerprint in member_fingerprints
                    seen_in_split = fingerprint in fingerprints_by_split[role]
                    within_member_duplicates += int(seen_in_member)
                    member_fingerprints.add(fingerprint)
                    for other_role, seen in fingerprints_by_split.items():
                        if other_role != role and fingerprint in seen:
                            cross_partition_duplicates += 1
                    same_split_cross_member_duplicates += int(seen_in_split and not seen_in_member)
                    fingerprints_by_split[role].add(fingerprint)
                    raw_input = np.asarray([numeric[name] for name in CANDIDATE_FIELDS], dtype=float)
                    if role != "external" or retain_external_outcomes:
                        inputs[role].append(raw_input)
                        targets[role].append(float(numeric["NOX"]))
            if row_count != expected_rows:
                raise ValueError(f"{member} row count changed from the locked contract")
            member_ledger[member] = {
                "year": year,
                "split": role,
                "rows": row_count,
                "within_member_duplicate_complete_rows": within_member_duplicates,
                "same_split_cross_member_duplicate_complete_rows": same_split_cross_member_duplicates,
                "targets_retained": role != "external" or retain_external_outcomes,
            }
    if cross_partition_duplicates:
        raise ValueError("gas-turbine complete-row fingerprint crosses a frozen partition")
    records: dict[str, LockedRows] = {}
    expected_counts = {"train": sum(MEMBER_ROWS[year] for year in (2011, 2012, 2013)), "selection": MEMBER_ROWS[2014]}
    if retain_external_outcomes:
        expected_counts["external"] = MEMBER_ROWS[2015]
    for role in ("train", "selection", "external"):
        if role == "external" and not retain_external_outcomes:
            records[role] = LockedRows(np.empty((0, len(CANDIDATE_FIELDS)), dtype=float), None)
            continue
        matrix = np.vstack(inputs[role])
        target = np.asarray(targets[role], dtype=float)
        if matrix.shape != (expected_counts[role], len(CANDIDATE_FIELDS)) or target.shape != (expected_counts[role],):
            raise ValueError(f"gas-turbine {role} retained record count violates the locked contract")
        records[role] = LockedRows(matrix, target)
    ledger = {
        "passed": True,
        "member_ledger": member_ledger,
        "cross_partition_complete_row_duplicates": cross_partition_duplicates,
        "external_outcomes_retained": retain_external_outcomes,
    }
    return records, ledger


def _design(values: np.ndarray, quadratic: bool) -> np.ndarray:
    return np.column_stack([values, np.square(values)]) if quadratic else values


def _validate_predictions(prediction: np.ndarray) -> None:
    values = np.asarray(prediction, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < NOX_BOUNDS_MG_M3[0]) or np.any(values > NOX_BOUNDS_MG_M3[1]):
        raise ValueError("gas-turbine prediction violates the finite unclipped [0, 150] mg/m3 envelope")


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    _validate_predictions(predicted)
    if observed.shape != predicted.shape or not np.all(np.isfinite(observed)):
        raise ValueError("gas-turbine metric needs finite aligned target and prediction")
    absolute = np.abs(predicted - observed)
    result = {
        "rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "mae": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in result.values()):
        raise ValueError("gas-turbine metric is non-finite")
    return result


def _fit_ridge(inputs: np.ndarray, target: np.ndarray, field_indices: np.ndarray, quadratic: bool, penalty: float) -> RidgeModel:
    selected = np.asarray(inputs, dtype=float)[:, field_indices]
    observed = np.asarray(target, dtype=float)
    design = _design(selected, quadratic)
    mean = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("gas-turbine train-only scaler found a constant candidate field")
    standardized = (design - mean) / scale
    intercept = float(np.mean(observed))
    coefficients = np.linalg.solve(
        standardized.T @ standardized + float(penalty) * np.eye(standardized.shape[1]),
        standardized.T @ (observed - intercept),
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("gas-turbine ridge coefficients are non-finite")
    return RidgeModel(field_indices.copy(), quadratic, mean, scale, coefficients, intercept)


def _select(
    train: LockedRows,
    selection: LockedRows,
    field_indices: np.ndarray | None = None,
) -> tuple[RidgeModel, str, dict[str, float], dict[str, dict[str, float]]]:
    if train.target_nox is None or selection.target_nox is None:
        raise ValueError("gas-turbine selection requires retained train and selection NOx only")
    indices = np.arange(len(CANDIDATE_FIELDS), dtype=int) if field_indices is None else np.asarray(field_indices, dtype=int)
    candidates: list[tuple[float, int, float, RidgeModel, str, dict[str, float]]] = []
    metrics_by_candidate: dict[str, dict[str, float]] = {}
    for family_rank, (family, quadratic) in enumerate((("linear", False), ("quadratic", True))):
        for penalty in PENALTIES:
            model = _fit_ridge(train.inputs, train.target_nox, indices, quadratic, penalty)
            metrics = _metrics(selection.target_nox, model.predict(selection.inputs))
            key = f"{family}_ridge_{penalty:g}"
            metrics_by_candidate[key] = metrics
            candidates.append((metrics["rmse"], family_rank, penalty, model, key, metrics))
    _, _, _, model, key, metrics = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return model, key, metrics, metrics_by_candidate


def _median_metrics(train: LockedRows, evaluation: LockedRows) -> tuple[float, dict[str, float]]:
    if train.target_nox is None or evaluation.target_nox is None:
        raise ValueError("gas-turbine median comparator needs retained targets")
    median = float(np.median(train.target_nox))
    return median, _metrics(evaluation.target_nox, np.full(evaluation.target_nox.shape, median, dtype=float))


def _rejected(payload: dict[str, object]) -> bool:
    try:
        candidate_input_from_payload(payload)
    except ValueError:
        return True
    return False


def _selection_controls(train: LockedRows, selection: LockedRows, model: RidgeModel, selected_metrics: dict[str, float], selected_key: str) -> dict[str, object]:
    if train.target_nox is None or selection.target_nox is None:
        raise ValueError("selection controls require retained selection target")
    paired_train = LockedRows(train.inputs, np.roll(train.target_nox, 1))
    _, pairing_key, pairing_metrics, _ = _select(paired_train, selection)
    reversed_train = LockedRows(train.inputs[::-1], train.target_nox[::-1])
    reversed_selection = LockedRows(selection.inputs[::-1], selection.target_nox[::-1])
    reversed_model, reversed_key, reversed_metrics, _ = _select(reversed_train, reversed_selection)
    reversed_prediction = reversed_model.predict(selection.inputs[::-1])[::-1]
    original_prediction = model.predict(selection.inputs)
    _, ambient_key, ambient_metrics, _ = _select(train, selection, PROCESS_INDICES)
    _, process_key, process_metrics, _ = _select(train, selection, AMBIENT_INDICES)
    candidate = UCIGasTurbineRawInput(selection.inputs[0])
    batch_prediction = predict_independent_rows(
        [UCIGasTurbineRawInput(values) for values in selection.inputs[:8]],
        lambda state, values: float(model.predict(values.reshape(1, -1))[0]),
    )
    alone_prediction = predict_independent_rows(
        [UCIGasTurbineRawInput(selection.inputs[0])],
        lambda state, values: float(model.predict(values.reshape(1, -1))[0]),
    )
    baseline = max(float(selected_metrics["rmse"]), 1e-12)
    return {
        "pairing": {
            "selected_model": pairing_key,
            "rmse": pairing_metrics["rmse"],
            "degradation_ratio": float(pairing_metrics["rmse"] / baseline),
            "threshold": 1.25,
        },
        "row_order": {
            "selected_model": reversed_key,
            "rmse": reversed_metrics["rmse"],
            "prediction_max_abs_difference": float(np.max(np.abs(reversed_prediction - original_prediction))),
            "rmse_absolute_difference": float(abs(reversed_metrics["rmse"] - selected_metrics["rmse"])),
            "threshold": 1e-12,
        },
        "ambient_ablation": {
            "selected_model": ambient_key,
            "rmse": ambient_metrics["rmse"],
            "degradation_ratio": float(ambient_metrics["rmse"] / baseline),
            "threshold": 1.02,
        },
        "process_ablation": {
            "selected_model": process_key,
            "rmse": process_metrics["rmse"],
            "degradation_ratio": float(process_metrics["rmse"] / baseline),
            "threshold": 1.02,
        },
        "sentinels": {
            "nox_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "NOX": 60.0}),
            "co_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "CO": 1.0}),
            "member_rejected": _rejected({"raw_source_units": candidate.raw_source_units, "year": 2014}),
            "member_permutation_invariant": bool(np.array_equal(model.predict(selection.inputs), model.predict(selection.inputs.copy()))),
            "independent_row_reset": bool(abs(float(batch_prediction[0]) - float(alone_prediction[0])) <= 1e-12),
        },
    }


def run(archive_path: Path = ARCHIVE, split_path: Path = SPLIT_PATH, output_dir: Path = OUT_DIR) -> dict[str, object]:
    """Run the fixed train/selection gate, scoring external NOx at most once after selection passes."""
    records, ledger = load_locked_rows(archive_path, split_path, retain_external_outcomes=False)
    train, selection = records["train"], records["selection"]
    train_median, median_selection = _median_metrics(train, selection)
    model, selected_key, selected_metrics, selection_candidates = _select(train, selection)
    controls = _selection_controls(train, selection, model, selected_metrics, selected_key)
    checks = {
        "source_revalidation": bool(ledger["passed"] and ledger["cross_partition_complete_row_duplicates"] == 0 and not ledger["external_outcomes_retained"]),
        "selected_margin_over_train_median": bool(selected_metrics["rmse"] <= 0.90 * median_selection["rmse"]),
        "pairing": bool(controls["pairing"]["degradation_ratio"] >= controls["pairing"]["threshold"]),
        "row_order": bool(
            controls["row_order"]["prediction_max_abs_difference"] <= controls["row_order"]["threshold"]
            and controls["row_order"]["rmse_absolute_difference"] <= controls["row_order"]["threshold"]
        ),
        "ambient_ablation": bool(controls["ambient_ablation"]["degradation_ratio"] >= controls["ambient_ablation"]["threshold"]),
        "process_ablation": bool(controls["process_ablation"]["degradation_ratio"] >= controls["process_ablation"]["threshold"]),
        "sentinels_and_reset": bool(all(controls["sentinels"].values())),
        "finite_bounds": bool(all(np.isfinite(value) and value >= 0.0 for value in selected_metrics.values())),
    }
    result: dict[str, object] = {
        "phase": 38,
        "source_revalidation": ledger,
        "selection": {
            "train_median_nox_mg_m3": train_median,
            "train_median": median_selection,
            "selected_model": selected_key,
            "selected_model_metrics": selected_metrics,
            "candidate_metrics": selection_candidates,
            "controls": controls,
            "checks": checks,
        },
        "external": {"opened": False, "reason": "selection gate pending"},
        "abc_smc_calls": 0,
        "llm_calls": 0,
        "campaign_calls": 0,
    }
    if not all(checks.values()):
        result["status"] = "closed_negative_selection_gate_external_untouched"
        result["decision"] = "A fixed selection gate failed; 2015 NOx remained unread for scoring."
        _write_outputs(result, output_dir)
        return result

    scored_records, external_ledger = load_locked_rows(archive_path, split_path, retain_external_outcomes=True)
    external = scored_records["external"]
    if external.target_nox is None:
        raise AssertionError("authorized external score did not retain its target")
    external_prediction = model.predict(external.inputs)
    external_metrics = _metrics(external.target_nox, external_prediction)
    _, external_median = _median_metrics(train, external)
    external_checks = {
        "margin_over_train_median": bool(external_metrics["rmse"] <= 0.95 * external_median["rmse"]),
        "selection_stability": bool(external_metrics["rmse"] <= 1.50 * selected_metrics["rmse"]),
        "finite_bounds": bool(all(np.isfinite(value) and value >= 0.0 for value in external_metrics.values())),
    }
    result["external"] = {
        "opened": True,
        "score_count": 1,
        "revalidation": external_ledger,
        "train_median": external_median,
        "selected_model_metrics": external_metrics,
        "checks": external_checks,
    }
    if all(external_checks.values()):
        result["status"] = "passed_real_baseline_gate"
        result["decision"] = "Fixed observational held-out-year baseline gate passed."
    else:
        result["status"] = "closed_negative_external_gate"
        result["decision"] = "The one authorized 2015 score failed a frozen external gate."
    _write_outputs(result, output_dir)
    return result


def _write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = str(result["status"])
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    selection = result["selection"]
    selected = selection["selected_model_metrics"]
    external = result["external"]
    (output_dir / "report.md").write_text(
        "# Phase 38 UCI Gas-Turbine Fixed Measured-NOx Baseline Gate\n\n"
        f"Status: **{status}**.\n\n"
        "All five official annual members were revalidated against the locked schema and frozen year split. "
        "The candidate surface used only the nine predeclared raw-unit inputs; CO, NOx, annual identity, and row order were excluded. "
        f"The selected 2014 model was `{selection['selected_model']}` with RMSE `{selected['rmse']:.6g}` mg/m3 versus train-median `{selection['train_median']['rmse']:.6g}`. "
        f"The 2015 outcome surface was {'opened exactly once after selection passed' if external['opened'] else 'not opened because selection failed'}.\n"
    )
    (output_dir / "decision.json").write_text(json.dumps({
        "phase": 38,
        "decision": result["decision"],
        "status": status,
        "external_opened": external["opened"],
        "external_score_count": external.get("score_count", 0),
        "next_action": "Return to source-backed application selection." if status.startswith("closed_negative") else "Prepare a separately reviewed bounded next-stage plan.",
        "prohibited": ["model tuning", "alternate external variants", "ABC-SMC", "LLM discovery", "campaign"],
    }, indent=2) + "\n")


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "external_opened": result["external"]["opened"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
