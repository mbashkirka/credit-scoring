from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


_EPS = 1e-6


def clip_probability(probability: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probability must be one-dimensional.")
    if not np.all(np.isfinite(p)):
        raise ValueError("probability contains non-finite values.")
    return np.clip(p, eps, 1.0 - eps)


def logit(probability: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = clip_probability(probability, eps=eps)
    return np.log(p / (1.0 - p))




@dataclass
class IdentityCalibrator:
    def fit(self, raw_probability: np.ndarray, y_true: np.ndarray) -> "IdentityCalibrator":
        _ = y_true
        clip_probability(raw_probability)
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        return clip_probability(raw_probability)


@dataclass
class PlattCalibrator:
    """Sigmoid calibration fitted on the log-odds of a frozen model score."""

    model: LogisticRegression | None = None

    def fit(self, raw_probability: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        y = np.asarray(y_true, dtype=np.int8)
        z = logit(raw_probability).reshape(-1, 1)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("Calibration target must contain both classes.")
        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(z, y)
        self.model = model
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PlattCalibrator is not fitted.")
        z = logit(raw_probability).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression | None = None

    def fit(self, raw_probability: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        y = np.asarray(y_true, dtype=np.int8)
        p = clip_probability(raw_probability)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("Calibration target must contain both classes.")
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p, y)
        self.model = model
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator is not fitted.")
        return np.asarray(self.model.predict(clip_probability(raw_probability)), dtype=np.float64)


@dataclass(frozen=True)
class CalibrationSlopeIntercept:
    intercept: float
    slope: float


def calibration_slope_intercept(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> CalibrationSlopeIntercept:
    """Fit y ~ logistic(intercept + slope * logit(p)). Ideal is (0, 1)."""
    y = np.asarray(y_true, dtype=np.int8)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Target must contain both classes.")
    z = logit(probability).reshape(-1, 1)
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(z, y)
    return CalibrationSlopeIntercept(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
    )


def quantile_reliability_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probability(probability)
    if len(y) != len(p):
        raise ValueError("y_true and probability lengths differ.")

    # Rank-based bins remain well-defined when isotonic calibration produces ties.
    order = np.argsort(p, kind="mergesort")
    bin_id = np.empty(len(p), dtype=np.int16)
    bin_id[order] = np.minimum(
        n_bins - 1,
        (np.arange(len(p), dtype=np.int64) * n_bins) // max(1, len(p)),
    )

    rows: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = bin_id == b
        if not np.any(mask):
            continue
        observed = float(y[mask].mean())
        predicted = float(p[mask].mean())
        rows.append(
            {
                "bin": b + 1,
                "n": int(mask.sum()),
                "predicted_pd_mean": predicted,
                "observed_default_rate": observed,
                "absolute_gap": abs(predicted - observed),
                "pd_min": float(p[mask].min()),
                "pd_max": float(p[mask].max()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    table = quantile_reliability_table(y_true, probability, n_bins=n_bins)
    if table.empty:
        return float("nan")
    weights = table["n"].to_numpy(dtype=np.float64)
    gaps = table["absolute_gap"].to_numpy(dtype=np.float64)
    return float(np.average(gaps, weights=weights))


def calibration_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probability(probability)
    slope_intercept = calibration_slope_intercept(y, p)
    observed = float(y.mean())
    predicted = float(p.mean())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted_pd": predicted,
        "observed_default_rate": observed,
        "calibration_bias": predicted - observed,
        f"ece_quantile_{n_bins}": expected_calibration_error(y, p, n_bins=n_bins),
        "calibration_intercept": slope_intercept.intercept,
        "calibration_slope": slope_intercept.slope,
    }
