import numpy as np
import pandas as pd

from credit_risk.drift import (
    build_drift_accumulators,
    summarize_drift,
    update_drift_accumulators,
)


def test_streaming_psi_is_near_zero_for_identical_population():
    frame = pd.DataFrame({"x": np.arange(100, dtype=float)})
    acc = build_drift_accumulators(frame, ["x"], n_bins=10)
    update_drift_accumulators(acc, frame.iloc[:50])
    update_drift_accumulators(acc, frame.iloc[50:])
    result = summarize_drift(acc).iloc[0]
    assert result["psi"] < 1e-12
    assert abs(result["standardized_mean_diff"]) < 1e-12


def test_streaming_psi_detects_shift():
    dev = pd.DataFrame({"x": np.linspace(0, 1, 1000)})
    holdout = pd.DataFrame({"x": np.linspace(1, 2, 1000)})
    acc = build_drift_accumulators(dev, ["x"], n_bins=10)
    update_drift_accumulators(acc, holdout)
    result = summarize_drift(acc).iloc[0]
    assert result["psi"] > 0.1
    assert result["standardized_mean_diff"] > 1.0
