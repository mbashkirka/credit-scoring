from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureDriftAccumulator:
    feature: str
    edges: np.ndarray
    dev_counts: np.ndarray
    dev_missing: int
    dev_n: int
    dev_mean: float
    dev_std: float
    holdout_counts: np.ndarray
    holdout_missing: int = 0
    holdout_n: int = 0
    holdout_sum: float = 0.0
    holdout_sum_sq: float = 0.0
    holdout_valid_n: int = 0


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(valid, quantiles))
    return np.r_[-np.inf, cuts, np.inf].astype(np.float64)


def build_drift_accumulators(
    development: pd.DataFrame,
    feature_names: list[str],
    *,
    n_bins: int = 10,
) -> dict[str, FeatureDriftAccumulator]:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    accumulators: dict[str, FeatureDriftAccumulator] = {}
    for feature in feature_names:
        values = development[feature].to_numpy(dtype=np.float64, copy=False)
        valid_mask = np.isfinite(values)
        valid = values[valid_mask]
        edges = _bin_edges(values, n_bins)
        counts, _ = np.histogram(valid, bins=edges)
        accumulators[feature] = FeatureDriftAccumulator(
            feature=feature,
            edges=edges,
            dev_counts=counts.astype(np.int64),
            dev_missing=int((~valid_mask).sum()),
            dev_n=int(len(values)),
            dev_mean=float(valid.mean()) if valid.size else float("nan"),
            dev_std=float(valid.std(ddof=0)) if valid.size else float("nan"),
            holdout_counts=np.zeros_like(counts, dtype=np.int64),
        )
    return accumulators


def update_drift_accumulators(
    accumulators: dict[str, FeatureDriftAccumulator],
    holdout: pd.DataFrame,
) -> None:
    for feature, acc in accumulators.items():
        values = holdout[feature].to_numpy(dtype=np.float64, copy=False)
        valid_mask = np.isfinite(values)
        valid = values[valid_mask]
        counts, _ = np.histogram(valid, bins=acc.edges)
        acc.holdout_counts += counts.astype(np.int64)
        acc.holdout_missing += int((~valid_mask).sum())
        acc.holdout_n += int(len(values))
        if valid.size:
            acc.holdout_sum += float(valid.sum())
            acc.holdout_sum_sq += float(np.square(valid).sum())
            acc.holdout_valid_n += int(valid.size)


def _psi_from_counts(
    reference_counts: np.ndarray,
    reference_missing: int,
    current_counts: np.ndarray,
    current_missing: int,
    *,
    epsilon: float = 1e-6,
) -> float:
    ref = np.r_[reference_counts.astype(np.float64), float(reference_missing)]
    cur = np.r_[current_counts.astype(np.float64), float(current_missing)]
    if ref.sum() <= 0 or cur.sum() <= 0:
        return float("nan")
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def summarize_drift(
    accumulators: dict[str, FeatureDriftAccumulator],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for feature, acc in accumulators.items():
        holdout_mean = (
            acc.holdout_sum / acc.holdout_valid_n
            if acc.holdout_valid_n
            else float("nan")
        )
        if acc.holdout_valid_n:
            variance = max(
                0.0,
                acc.holdout_sum_sq / acc.holdout_valid_n - holdout_mean**2,
            )
            holdout_std = float(np.sqrt(variance))
        else:
            holdout_std = float("nan")
        if np.isfinite(acc.dev_std) and acc.dev_std > 1e-12:
            smd = (holdout_mean - acc.dev_mean) / acc.dev_std
        else:
            smd = 0.0 if np.isclose(holdout_mean, acc.dev_mean, equal_nan=True) else float("nan")
        rows.append(
            {
                "feature": feature,
                "psi": _psi_from_counts(
                    acc.dev_counts,
                    acc.dev_missing,
                    acc.holdout_counts,
                    acc.holdout_missing,
                ),
                "dev_missing_rate": acc.dev_missing / acc.dev_n if acc.dev_n else float("nan"),
                "holdout_missing_rate": (
                    acc.holdout_missing / acc.holdout_n if acc.holdout_n else float("nan")
                ),
                "missing_rate_delta": (
                    acc.holdout_missing / acc.holdout_n - acc.dev_missing / acc.dev_n
                    if acc.dev_n and acc.holdout_n
                    else float("nan")
                ),
                "dev_mean": acc.dev_mean,
                "holdout_mean": holdout_mean,
                "dev_std": acc.dev_std,
                "holdout_std": holdout_std,
                "standardized_mean_diff": smd,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
