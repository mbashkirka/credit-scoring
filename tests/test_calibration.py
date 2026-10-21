import numpy as np

from credit_risk.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_metrics,
    calibration_slope_intercept,
    expected_calibration_error,
    quantile_reliability_table,
)


def test_calibration_metrics_perfectish_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    p = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
    metrics = calibration_metrics(y, p, n_bins=3)
    assert 0 <= metrics["brier_score"] < 0.05
    assert metrics["roc_auc"] == 1.0
    assert np.isfinite(metrics["calibration_intercept"])
    assert np.isfinite(metrics["calibration_slope"])


def test_platt_corrects_logit_shift_on_validation():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=6000)
    true_p = 1 / (1 + np.exp(-(-2.2 + 1.1 * latent)))
    y = rng.binomial(1, true_p).astype(np.int8)
    # Deliberately miscalibrated but rank-preserving score.
    raw = 1 / (1 + np.exp(-(1.5 + 1.8 * latent)))
    fit = np.arange(0, 3000)
    valid = np.arange(3000, 6000)
    calibrator = PlattCalibrator().fit(raw[fit], y[fit])
    raw_brier = calibration_metrics(y[valid], raw[valid], n_bins=10)["brier_score"]
    calibrated = calibrator.predict(raw[valid])
    calibrated_brier = calibration_metrics(y[valid], calibrated, n_bins=10)["brier_score"]
    assert calibrated_brier < raw_brier


def test_isotonic_predictions_are_bounded_and_monotone():
    x = np.linspace(0.01, 0.99, 200)
    y = (x > 0.6).astype(np.int8)
    calibrator = IsotonicCalibrator().fit(x, y)
    pred = calibrator.predict(x)
    assert np.all((pred >= 0) & (pred <= 1))
    assert np.all(np.diff(pred) >= -1e-12)


def test_reliability_table_covers_population():
    y = np.array([0, 1] * 50, dtype=np.int8)
    p = np.linspace(0.01, 0.99, len(y))
    table = quantile_reliability_table(y, p, n_bins=10)
    assert table["n"].sum() == len(y)
    assert len(table) == 10
    assert expected_calibration_error(y, p, n_bins=10) >= 0


def test_calibration_slope_intercept_returns_finite_values():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.7, size=2000)
    y = rng.binomial(1, p).astype(np.int8)
    result = calibration_slope_intercept(y, p)
    assert np.isfinite(result.intercept)
    assert np.isfinite(result.slope)
