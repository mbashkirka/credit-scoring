from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapDeltaSummary:
    metric: str
    point_delta: float
    ci_low: float
    ci_high: float
    win_probability: float
    n_bootstrap: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "point_delta": self.point_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "win_probability": self.win_probability,
            "n_bootstrap": self.n_bootstrap,
        }


class _WeightedRankingEvaluator:
    """Efficient weighted AUC/AP evaluator for repeated paired bootstraps.

    Score ordering and tie groups are precomputed once. Each bootstrap replicate
    changes only observation multiplicities, which is equivalent to resampling
    rows with replacement while keeping the model scores fixed.
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray):
        y = np.asarray(y_true, dtype=np.int8)
        score = np.asarray(y_score, dtype=np.float64)
        if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
            raise ValueError("y_true and y_score must be one-dimensional arrays of equal length.")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("Expected binary target containing both classes.")
        if not np.all(np.isfinite(score)):
            raise ValueError("y_score contains non-finite values.")

        # Descending score order makes AP natural. Ties are reduced as one threshold.
        self.order = np.argsort(-score, kind="mergesort")
        sorted_score = score[self.order]
        self.sorted_y = y[self.order]
        self.group_starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]

    def auc_ap(self, weights: np.ndarray) -> tuple[float, float]:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or len(w) != len(self.order):
            raise ValueError("weights must match evaluator population length.")
        ws = w[self.order]
        pos_w = ws * self.sorted_y
        neg_w = ws * (1 - self.sorted_y)
        gp = np.add.reduceat(pos_w, self.group_starts)
        gn = np.add.reduceat(neg_w, self.group_starts)

        total_pos = gp.sum()
        total_neg = gn.sum()
        if total_pos <= 0 or total_neg <= 0:
            return float("nan"), float("nan")

        # AUC from descending thresholds: each positive group outranks negatives
        # appearing strictly later, with half credit for score ties.
        neg_after = total_neg - np.cumsum(gn)
        auc = float(np.sum(gp * (neg_after + 0.5 * gn)) / (total_pos * total_neg))

        cum_pos = np.cumsum(gp)
        cum_all = np.cumsum(gp + gn)
        precision = np.divide(cum_pos, cum_all, out=np.zeros_like(cum_pos), where=cum_all > 0)
        ap = float(np.sum((gp / total_pos) * precision))
        return auc, ap


def _validate_paired_inputs(
    y_true: np.ndarray,
    score_reference: np.ndarray,
    score_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8)
    ref = np.asarray(score_reference, dtype=np.float64)
    cand = np.asarray(score_candidate, dtype=np.float64)
    if y.ndim != 1 or ref.ndim != 1 or cand.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")
    if not (len(y) == len(ref) == len(cand)):
        raise ValueError("Paired bootstrap inputs must have equal length.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Expected binary target containing both classes.")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise ValueError("Scores contain non-finite values.")
    return y, ref, cand


def paired_stratified_bootstrap_ranking_delta(
    y_true: np.ndarray,
    score_reference: np.ndarray,
    score_candidate: np.ndarray,
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[BootstrapDeltaSummary, BootstrapDeltaSummary, np.ndarray]:
    """Paired stratified bootstrap for candidate-minus-reference AUC and AP.

    The same bootstrap multiplicities are applied to both model score vectors,
    preserving the paired nature of the comparison. Positive and negative class
    counts are resampled separately, so every replicate preserves class prevalence
    and contains both classes.
    """
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")

    y, ref, cand = _validate_paired_inputs(y_true, score_reference, score_candidate)
    ref_eval = _WeightedRankingEvaluator(y, ref)
    cand_eval = _WeightedRankingEvaluator(y, cand)

    unit_weights = np.ones(len(y), dtype=np.float64)
    ref_auc, ref_ap = ref_eval.auc_ap(unit_weights)
    cand_auc, cand_ap = cand_eval.auc_ap(unit_weights)
    point_auc_delta = cand_auc - ref_auc
    point_ap_delta = cand_ap - ref_ap

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(random_state)
    deltas = np.empty((n_bootstrap, 2), dtype=np.float64)
    weights = np.zeros(len(y), dtype=np.float64)

    for b in range(n_bootstrap):
        weights.fill(0.0)
        pos_draw = rng.integers(0, len(pos_idx), size=len(pos_idx))
        neg_draw = rng.integers(0, len(neg_idx), size=len(neg_idx))
        weights[pos_idx] = np.bincount(pos_draw, minlength=len(pos_idx))
        weights[neg_idx] = np.bincount(neg_draw, minlength=len(neg_idx))

        ref_auc_b, ref_ap_b = ref_eval.auc_ap(weights)
        cand_auc_b, cand_ap_b = cand_eval.auc_ap(weights)
        deltas[b, 0] = cand_auc_b - ref_auc_b
        deltas[b, 1] = cand_ap_b - ref_ap_b

    alpha = 1.0 - confidence
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)

    def summarize(metric: str, point: float, values: np.ndarray) -> BootstrapDeltaSummary:
        lo, hi = np.percentile(values, [lo_q, hi_q])
        win_probability = float(np.mean(values > 0))
        return BootstrapDeltaSummary(
            metric=metric,
            point_delta=float(point),
            ci_low=float(lo),
            ci_high=float(hi),
            win_probability=win_probability,
            n_bootstrap=int(n_bootstrap),
        )

    return (
        summarize("roc_auc", point_auc_delta, deltas[:, 0]),
        summarize("average_precision", point_ap_delta, deltas[:, 1]),
        deltas,
    )


@dataclass(frozen=True)
class BootstrapMetricCI:
    metric: str
    point: float
    ci_low: float
    ci_high: float
    n_bootstrap: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_bootstrap": self.n_bootstrap,
        }


def stratified_bootstrap_probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[list[BootstrapMetricCI], np.ndarray]:
    """Stratified nonparametric bootstrap CIs for AUC/AP/Brier/log-loss.

    Positive and negative sample counts are preserved in each replicate. This is
    inference-only uncertainty quantification: it never refits a model or calibrator.
    """
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probability must be one-dimensional and equally sized.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Expected binary target containing both classes.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and lie in [0, 1].")

    evaluator = _WeightedRankingEvaluator(y, p)
    unit_weights = np.ones(len(y), dtype=np.float64)
    auc, ap = evaluator.auc_ap(unit_weights)
    p_clip = np.clip(p, 1e-15, 1 - 1e-15)
    sq_error = (p - y) ** 2
    log_loss_each = -(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip))
    point = np.array([auc, ap, sq_error.mean(), log_loss_each.mean()], dtype=np.float64)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(random_state)
    draws = np.empty((n_bootstrap, 4), dtype=np.float64)
    weights = np.zeros(len(y), dtype=np.float64)
    for b in range(n_bootstrap):
        weights.fill(0.0)
        pos_draw = rng.integers(0, len(pos_idx), size=len(pos_idx))
        neg_draw = rng.integers(0, len(neg_idx), size=len(neg_idx))
        weights[pos_idx] = np.bincount(pos_draw, minlength=len(pos_idx))
        weights[neg_idx] = np.bincount(neg_draw, minlength=len(neg_idx))
        auc_b, ap_b = evaluator.auc_ap(weights)
        total_w = weights.sum()
        draws[b, 0] = auc_b
        draws[b, 1] = ap_b
        draws[b, 2] = float(np.dot(weights, sq_error) / total_w)
        draws[b, 3] = float(np.dot(weights, log_loss_each) / total_w)

    alpha = 1.0 - confidence
    low_q, high_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    names = ("roc_auc", "average_precision", "brier_score", "log_loss")
    summaries: list[BootstrapMetricCI] = []
    for j, name in enumerate(names):
        lo, hi = np.percentile(draws[:, j], [low_q, high_q])
        summaries.append(
            BootstrapMetricCI(
                metric=name,
                point=float(point[j]),
                ci_low=float(lo),
                ci_high=float(hi),
                n_bootstrap=int(n_bootstrap),
            )
        )
    return summaries, draws
