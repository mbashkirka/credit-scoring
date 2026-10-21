import numpy as np

from credit_risk.decisioning import (
    DEFAULT_SCENARIOS,
    assign_decisions,
    assign_risk_bands,
    derive_quantile_band_cutpoints,
    evaluate_policy,
    expected_profit_rate,
    pd_to_score,
    risk_band_summary,
)


def _balanced():
    return [s for s in DEFAULT_SCENARIOS if s.name == "balanced"][0]


def test_balanced_thresholds_are_economic_and_ordered():
    s = _balanced()
    assert np.isclose(s.break_even_pd, 0.10 / 0.57)
    assert np.isclose(s.reject_pd_threshold, 0.12)
    assert np.isclose(s.approve_pd_threshold, 0.096)
    assert s.approve_pd_threshold < s.reject_pd_threshold < s.break_even_pd


def test_expected_profit_crosses_zero_at_break_even_without_risk_cap_effect():
    s = _balanced()
    p = np.array([s.break_even_pd])
    assert abs(expected_profit_rate(p, s)[0]) < 1e-10


def test_assign_decisions_uses_approve_review_reject_regions():
    s = _balanced()
    p = np.array([0.02, 0.10, 0.20])
    assert assign_decisions(p, s).tolist() == ["APPROVE", "REVIEW", "REJECT"]


def test_quantile_cutpoints_are_strict_even_with_isotonic_ties():
    p = np.repeat(np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12]), [35, 15, 15, 15, 10, 10])
    cuts = derive_quantile_band_cutpoints(p)
    assert cuts.shape == (4,)
    assert np.all(np.diff(cuts) > 0)
    bands = assign_risk_bands(p, cuts)
    assert set(bands.tolist()) == {"A", "B", "C", "D", "E"}


def test_risk_band_summary_accounts_for_population_and_bads():
    p = np.linspace(0.005, 0.20, 100)
    y = (np.arange(100) % 10 == 0).astype(np.int8)
    cuts = derive_quantile_band_cutpoints(p)
    bands = assign_risk_bands(p, cuts)
    summary = risk_band_summary(y, p, bands)
    assert np.isclose(summary["population_share"].sum(), 1.0)
    assert summary["bad_count"].sum() == int(y.sum())
    assert np.isclose(summary["bad_capture_rate"].sum(), 1.0)


def test_policy_metrics_are_consistent():
    s = _balanced()
    p = np.array([0.01, 0.05, 0.10, 0.20])
    y = np.array([0, 0, 1, 1], dtype=np.int8)
    metrics = evaluate_policy(y, p, s)
    assert np.isclose(metrics["approval_rate"], 0.5)
    assert np.isclose(metrics["review_rate"], 0.25)
    assert np.isclose(metrics["rejection_rate"], 0.25)
    assert np.isclose(
        metrics["approval_rate"] + metrics["review_rate"] + metrics["rejection_rate"],
        1.0,
    )
    assert np.isclose(metrics["default_capture_review_or_reject"], 1.0)


def test_credit_score_is_monotone_decreasing_in_pd():
    p = np.array([0.01, 0.03, 0.10, 0.30])
    score = pd_to_score(p)
    assert np.all(np.diff(score) < 0)
    assert np.all((score >= 300) & (score <= 850))
