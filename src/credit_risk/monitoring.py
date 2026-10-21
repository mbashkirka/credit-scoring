from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd


SEVERITY_ORDER = {"PASS": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass(frozen=True)
class MonitoringThresholds:
    """Configurable operational thresholds for Model V1 monitoring.

    These are project-level heuristics, not regulatory limits. They were introduced
    after the M6 final lockbox evaluation, so M6 can only be used as a historical
    case study. Future production windows may use them prospectively once frozen.
    """

    feature_psi_warning: float = 0.10
    feature_psi_critical: float = 0.25
    missing_rate_delta_warning: float = 0.02
    missing_rate_delta_critical: float = 0.05
    prediction_psi_warning: float = 0.10
    prediction_psi_critical: float = 0.25
    mean_pd_abs_delta_warning: float = 0.005
    mean_pd_abs_delta_critical: float = 0.010
    segment_share_abs_delta_warning: float = 0.02
    segment_share_abs_delta_critical: float = 0.05
    auc_drop_warning: float = 0.015
    auc_drop_critical: float = 0.030
    calibration_bias_abs_warning: float = 0.005
    calibration_bias_abs_critical: float = 0.010
    calibration_slope_abs_from_one_warning: float = 0.10
    calibration_slope_abs_from_one_critical: float = 0.20
    brier_increase_warning: float = 0.002
    brier_increase_critical: float = 0.005
    extra_schema_feature_warning_count: int = 1
    missing_nonstructural_critical_count: int = 1

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NumericReferenceProfile:
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    missing: int
    n: int
    mean: float
    std: float

    def to_dict(self) -> dict[str, object]:
        def encode_number(value: float) -> float | str | None:
            if np.isnan(value):
                return None
            if np.isposinf(value):
                return "inf"
            if np.isneginf(value):
                return "-inf"
            return float(value)

        return {
            "edges": [encode_number(x) for x in self.edges],
            "counts": list(self.counts),
            "missing": self.missing,
            "n": self.n,
            "mean": encode_number(self.mean),
            "std": encode_number(self.std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NumericReferenceProfile":
        def decode_number(value: object) -> float:
            if value is None:
                return float("nan")
            if value == "inf":
                return float("inf")
            if value == "-inf":
                return float("-inf")
            return float(value)

        return cls(
            edges=tuple(decode_number(x) for x in payload["edges"]),
            counts=tuple(int(x) for x in payload["counts"]),
            missing=int(payload["missing"]),
            n=int(payload["n"]),
            mean=decode_number(payload["mean"]),
            std=decode_number(payload["std"]),
        )


@dataclass(frozen=True)
class MonitorAlert:
    category: str
    metric: str
    status: str
    observed: float | int | str
    reference: float | int | str | None
    delta: float | int | None
    warning_threshold: float | int | None
    critical_threshold: float | int | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def quantile_edges(values: np.ndarray, *, n_bins: int = 10) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    valid = _finite(values)
    if valid.size == 0:
        return np.asarray([-np.inf, np.inf], dtype=np.float64)
    q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(valid, q))
    return np.r_[-np.inf, cuts, np.inf].astype(np.float64)


def histogram_counts(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=np.float64)
    valid_mask = np.isfinite(arr)
    counts, _ = np.histogram(arr[valid_mask], bins=np.asarray(edges, dtype=np.float64))
    return counts.astype(np.int64), int((~valid_mask).sum())


def build_numeric_reference_profile(
    values: np.ndarray,
    *,
    n_bins: int = 10,
    edges: np.ndarray | None = None,
) -> NumericReferenceProfile:
    arr = np.asarray(values, dtype=np.float64)
    chosen_edges = quantile_edges(arr, n_bins=n_bins) if edges is None else np.asarray(edges, dtype=float)
    counts, missing = histogram_counts(arr, chosen_edges)
    valid = _finite(arr)
    return NumericReferenceProfile(
        edges=tuple(float(x) for x in chosen_edges),
        counts=tuple(int(x) for x in counts),
        missing=missing,
        n=int(arr.size),
        mean=float(valid.mean()) if valid.size else float("nan"),
        std=float(valid.std(ddof=0)) if valid.size else float("nan"),
    )


def psi_from_counts(
    reference_counts: np.ndarray,
    reference_missing: int,
    current_counts: np.ndarray,
    current_missing: int,
    *,
    epsilon: float = 1e-6,
) -> float:
    ref = np.r_[np.asarray(reference_counts, dtype=np.float64), float(reference_missing)]
    cur = np.r_[np.asarray(current_counts, dtype=np.float64), float(current_missing)]
    if ref.sum() <= 0 or cur.sum() <= 0:
        return float("nan")
    ref /= ref.sum()
    cur /= cur.sum()
    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compare_numeric_profile(
    reference: NumericReferenceProfile,
    current_values: np.ndarray,
) -> dict[str, float | int]:
    edges = np.asarray(reference.edges, dtype=np.float64)
    counts, missing = histogram_counts(current_values, edges)
    arr = np.asarray(current_values, dtype=np.float64)
    valid = _finite(arr)
    current_mean = float(valid.mean()) if valid.size else float("nan")
    current_std = float(valid.std(ddof=0)) if valid.size else float("nan")
    current_missing_rate = missing / len(arr) if len(arr) else float("nan")
    reference_missing_rate = reference.missing / reference.n if reference.n else float("nan")
    return {
        "psi": psi_from_counts(
            np.asarray(reference.counts), reference.missing, counts, missing
        ),
        "reference_n": reference.n,
        "current_n": int(len(arr)),
        "reference_mean": reference.mean,
        "current_mean": current_mean,
        "mean_delta": current_mean - reference.mean,
        "reference_std": reference.std,
        "current_std": current_std,
        "reference_missing_rate": reference_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_delta": current_missing_rate - reference_missing_rate,
    }


def schema_signature(feature_names: Iterable[str]) -> str:
    """Stable SHA-256 signature for an unordered raw feature-name set."""
    names = sorted({str(name) for name in feature_names})
    payload = "\n".join(names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def categorical_shares(values: Iterable[object], labels: Iterable[str]) -> dict[str, float]:
    items = pd.Series(list(values), dtype="object")
    n = len(items)
    if n == 0:
        return {str(label): float("nan") for label in labels}
    return {str(label): float((items == label).mean()) for label in labels}


def compare_categorical_shares(
    reference: dict[str, float],
    current_values: Iterable[object],
) -> pd.DataFrame:
    labels = list(reference)
    current = categorical_shares(current_values, labels)
    rows = []
    for label in labels:
        ref = float(reference[label])
        cur = float(current[label])
        rows.append(
            {
                "label": label,
                "reference_share": ref,
                "current_share": cur,
                "share_delta": cur - ref,
                "abs_share_delta": abs(cur - ref),
            }
        )
    return pd.DataFrame(rows)


def threshold_status(value: float, warning: float, critical: float) -> str:
    if not (0 <= warning <= critical):
        raise ValueError("Thresholds must satisfy 0 <= warning <= critical")
    if not np.isfinite(value):
        return "CRITICAL"
    magnitude = abs(float(value))
    if magnitude >= critical:
        return "CRITICAL"
    if magnitude >= warning:
        return "WARNING"
    return "PASS"


def drop_status(drop: float, warning: float, critical: float) -> str:
    """Status for a positive degradation amount (e.g. reference AUC - current AUC)."""
    if not np.isfinite(drop):
        return "CRITICAL"
    if drop >= critical:
        return "CRITICAL"
    if drop >= warning:
        return "WARNING"
    return "PASS"


def increase_status(increase: float, warning: float, critical: float) -> str:
    return drop_status(increase, warning, critical)


def worst_status(statuses: Iterable[str]) -> str:
    values = list(statuses)
    if not values:
        return "PASS"
    unknown = [status for status in values if status not in SEVERITY_ORDER]
    if unknown:
        raise ValueError(f"Unknown monitoring statuses: {unknown}")
    return max(values, key=lambda status: SEVERITY_ORDER[status])


def feature_alerts(
    feature_drift: pd.DataFrame,
    thresholds: MonitoringThresholds,
) -> list[MonitorAlert]:
    alerts: list[MonitorAlert] = []
    for row in feature_drift.itertuples(index=False):
        psi_status = threshold_status(
            float(row.psi), thresholds.feature_psi_warning, thresholds.feature_psi_critical
        )
        if psi_status != "PASS":
            alerts.append(
                MonitorAlert(
                    category="feature_drift",
                    metric=f"psi::{row.feature}",
                    status=psi_status,
                    observed=float(row.psi),
                    reference=0.0,
                    delta=float(row.psi),
                    warning_threshold=thresholds.feature_psi_warning,
                    critical_threshold=thresholds.feature_psi_critical,
                    message=f"Feature {row.feature} population stability index exceeded threshold.",
                )
            )
        missing_delta = float(row.missing_rate_delta)
        missing_status = threshold_status(
            missing_delta,
            thresholds.missing_rate_delta_warning,
            thresholds.missing_rate_delta_critical,
        )
        if missing_status != "PASS":
            alerts.append(
                MonitorAlert(
                    category="data_quality",
                    metric=f"missing_rate_delta::{row.feature}",
                    status=missing_status,
                    observed=float(row.current_missing_rate),
                    reference=float(row.reference_missing_rate),
                    delta=missing_delta,
                    warning_threshold=thresholds.missing_rate_delta_warning,
                    critical_threshold=thresholds.missing_rate_delta_critical,
                    message=f"Feature {row.feature} missing-rate shift exceeded threshold.",
                )
            )
    return alerts


def schema_alerts(
    schema_frame: pd.DataFrame,
    thresholds: MonitoringThresholds,
    *,
    reference_raw_schema_signatures: Iterable[str] | None = None,
) -> list[MonitorAlert]:
    """Create schema alerts, optionally relative to observed reference raw schemas.

    With a reference baseline, raw features outside the frozen model schema are not
    automatically drift: if the exact raw feature-name set was already observed in
    the reference population, the state is treated as baseline-compatible.
    """
    alerts: list[MonitorAlert] = []
    baseline = (
        {str(value) for value in reference_raw_schema_signatures}
        if reference_raw_schema_signatures is not None
        else None
    )
    if baseline is not None and not baseline:
        raise ValueError("reference_raw_schema_signatures must not be empty")

    for row in schema_frame.itertuples(index=False):
        nonstructural = int(row.missing_nonstructural_count)
        current_signature = getattr(row, "raw_schema_signature", None)
        baseline_match = baseline is not None and str(current_signature) in baseline

        if baseline_match:
            # The complete raw feature-name set is an observed reference state.
            # Therefore its missing/extra pattern relative to the frozen 212-feature
            # serving schema is known baseline behavior, not schema drift.
            continue

        if nonstructural >= thresholds.missing_nonstructural_critical_count:
            alerts.append(
                MonitorAlert(
                    category="schema",
                    metric=f"missing_nonstructural::{row.population}",
                    status="CRITICAL",
                    observed=nonstructural,
                    reference=0,
                    delta=nonstructural,
                    warning_threshold=None,
                    critical_threshold=thresholds.missing_nonstructural_critical_count,
                    message="Frozen Model V1 input schema is missing non-structural features.",
                )
            )
            if baseline is not None:
                continue

        if baseline is not None:
            alerts.append(
                MonitorAlert(
                    category="schema",
                    metric=f"raw_schema_change::{row.population}",
                    status="WARNING" if nonstructural == 0 else "CRITICAL",
                    observed=str(current_signature),
                    reference="observed_reference_schema_set",
                    delta=None,
                    warning_threshold=None,
                    critical_threshold=None,
                    message=(
                        "Current raw feature-name set differs from every schema observed "
                        "in the M10 reference population."
                    ),
                )
            )
            continue

        # Legacy fallback for callers without a reference schema profile.
        extras = int(row.dropped_extra_count)
        if extras >= thresholds.extra_schema_feature_warning_count:
            alerts.append(
                MonitorAlert(
                    category="schema",
                    metric=f"dropped_extra::{row.population}",
                    status="WARNING",
                    observed=extras,
                    reference=0,
                    delta=extras,
                    warning_threshold=thresholds.extra_schema_feature_warning_count,
                    critical_threshold=None,
                    message=(
                        "Current population produced features outside the frozen schema; "
                        "they were dropped as required by Model V1."
                    ),
                )
            )
    return alerts


def alerts_to_frame(alerts: Iterable[MonitorAlert]) -> pd.DataFrame:
    rows = [alert.to_dict() for alert in alerts]
    columns = [
        "category",
        "metric",
        "status",
        "observed",
        "reference",
        "delta",
        "warning_threshold",
        "critical_threshold",
        "message",
    ]
    return pd.DataFrame(rows, columns=columns)
