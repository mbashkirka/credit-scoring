from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BinaryClassWeights:
    negative: float
    positive: float

    def as_sklearn_dict(self) -> dict[int, float]:
        return {0: self.negative, 1: self.positive}

    def as_catboost_list(self) -> list[float]:
        return [self.negative, self.positive]


def balanced_binary_class_weights(y: np.ndarray) -> BinaryClassWeights:
    """Return sklearn-style balanced weights for a binary target.

    w_c = n_samples / (2 * n_samples_in_class_c)

    Explicit weights are used in M3 so Logistic Regression and CatBoost receive
    exactly the same class weighting on each training fold.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, got {sorted(classes)}")

    n = len(y)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    return BinaryClassWeights(
        negative=n / (2.0 * n0),
        positive=n / (2.0 * n1),
    )


def make_logistic_pipeline(class_weights: BinaryClassWeights | None) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None if class_weights is None else class_weights.as_sklearn_dict(),
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class FrozenMedianImputer:
    """Lightweight median imputer for large M4 matrices and batch inference."""

    statistics_: np.ndarray

    def __post_init__(self) -> None:
        self.statistics_ = np.asarray(self.statistics_, dtype=np.float32)
        if self.statistics_.ndim != 1:
            raise ValueError("statistics_ must be one-dimensional.")
        if not np.all(np.isfinite(self.statistics_)):
            raise ValueError("statistics_ contains non-finite values.")

    def transform(self, X) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.statistics_):
            raise ValueError("Input feature count does not match frozen medians.")
        result = values.copy()
        missing_rows, missing_cols = np.where(~np.isfinite(result))
        if missing_rows.size:
            result[missing_rows, missing_cols] = self.statistics_[missing_cols]
        return result
