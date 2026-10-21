import numpy as np

from credit_risk.statistics import (
    paired_stratified_bootstrap_ranking_delta,
    stratified_bootstrap_probability_metrics,
)


def test_paired_bootstrap_detects_better_ranking():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    reference = np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.5, 0.6, 0.7])
    candidate = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    auc, ap, raw = paired_stratified_bootstrap_ranking_delta(
        y, reference, candidate, n_bootstrap=100, random_state=7
    )
    assert auc.point_delta > 0
    assert ap.point_delta > 0
    assert auc.win_probability > 0.8
    assert raw.shape == (100, 2)


def test_paired_bootstrap_identical_scores_has_zero_delta():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
    score = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
    auc, ap, raw = paired_stratified_bootstrap_ranking_delta(
        y, score, score, n_bootstrap=50, random_state=11
    )
    assert np.isclose(auc.point_delta, 0.0)
    assert np.isclose(ap.point_delta, 0.0)
    assert np.allclose(raw, 0.0)


def test_single_model_bootstrap_probability_metrics_have_finite_intervals():
    rng = np.random.default_rng(123)
    y = np.array([0] * 180 + [1] * 20, dtype=np.int8)
    p = np.clip(0.03 + 0.55 * y + rng.normal(0, 0.08, len(y)), 0.001, 0.999)
    summaries, draws = stratified_bootstrap_probability_metrics(
        y, p, n_bootstrap=30, random_state=7
    )
    assert draws.shape == (30, 4)
    assert {s.metric for s in summaries} == {
        "roc_auc", "average_precision", "brier_score", "log_loss"
    }
    for s in summaries:
        assert np.isfinite(s.point)
        assert np.isfinite(s.ci_low)
        assert np.isfinite(s.ci_high)
        assert s.ci_low <= s.ci_high
