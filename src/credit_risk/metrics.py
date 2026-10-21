from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ranking_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")

    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))

    n_top = max(1, int(np.ceil(len(y_true) * top_fraction)))
    top_pct = int(round(top_fraction * 100))
    top_idx = np.argsort(-y_score)[:n_top]
    positives = int(y_true.sum())
    captured = int(y_true[top_idx].sum())

    capture_rate = float(captured / positives) if positives else float("nan")
    lift = float(capture_rate / (n_top / len(y_true))) if positives else float("nan")
    precision_top = float(y_true[top_idx].mean())

    return {
        "roc_auc": auc,
        "gini": float(2.0 * auc - 1.0),
        "average_precision": ap,
        "ks": ks,
        f"capture_rate_top_{top_pct}pct": capture_rate,
        f"lift_top_{top_pct}pct": lift,
        f"precision_top_{top_pct}pct": precision_top,
    }
