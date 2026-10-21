import numpy as np
import pandas as pd

from credit_risk.monitoring import (
    MonitoringThresholds,
    alerts_to_frame,
    build_numeric_reference_profile,
    categorical_shares,
    compare_categorical_shares,
    compare_numeric_profile,
    feature_alerts,
    schema_alerts,
    threshold_status,
    worst_status,
)


def test_numeric_profile_identical_population_has_zero_psi():
    values = np.arange(1000, dtype=float)
    ref = build_numeric_reference_profile(values, n_bins=10)
    out = compare_numeric_profile(ref, values)
    assert out["psi"] < 1e-12
    assert abs(out["mean_delta"]) < 1e-12


def test_numeric_profile_detects_large_shift():
    ref = build_numeric_reference_profile(np.linspace(0, 1, 1000), n_bins=10)
    out = compare_numeric_profile(ref, np.linspace(2, 3, 1000))
    assert out["psi"] > 0.25
    assert out["mean_delta"] > 1.0


def test_profile_round_trip():
    values = np.asarray([0.0, 1.0, 2.0, np.nan])
    ref = build_numeric_reference_profile(values, n_bins=3)
    restored = type(ref).from_dict(ref.to_dict())
    assert restored.counts == ref.counts
    assert restored.missing == 1
    assert restored.n == 4


def test_categorical_share_comparison():
    ref = categorical_shares(["A", "A", "B", "B"], ["A", "B"])
    out = compare_categorical_shares(ref, ["A", "A", "A", "B"])
    a = out[out["label"] == "A"].iloc[0]
    assert a["reference_share"] == 0.5
    assert a["current_share"] == 0.75
    assert a["share_delta"] == 0.25


def test_threshold_status_boundaries():
    assert threshold_status(0.09, 0.10, 0.25) == "PASS"
    assert threshold_status(0.10, 0.10, 0.25) == "WARNING"
    assert threshold_status(0.25, 0.10, 0.25) == "CRITICAL"
    assert threshold_status(-0.30, 0.10, 0.25) == "CRITICAL"


def test_feature_alerts_emit_psi_and_missing_alerts():
    drift = pd.DataFrame(
        [
            {
                "feature": "x",
                "psi": 0.30,
                "reference_missing_rate": 0.0,
                "current_missing_rate": 0.10,
                "missing_rate_delta": 0.10,
            }
        ]
    )
    alerts = feature_alerts(drift, MonitoringThresholds())
    frame = alerts_to_frame(alerts)
    assert set(frame["category"]) == {"feature_drift", "data_quality"}
    assert set(frame["status"]) == {"CRITICAL"}


def test_schema_extra_is_warning_and_missing_nonstructural_is_critical():
    schema = pd.DataFrame(
        [
            {
                "population": "window_1",
                "missing_nonstructural_count": 1,
                "dropped_extra_count": 2,
            }
        ]
    )
    frame = alerts_to_frame(schema_alerts(schema, MonitoringThresholds()))
    assert set(frame["status"]) == {"WARNING", "CRITICAL"}


def test_worst_status():
    assert worst_status([]) == "PASS"
    assert worst_status(["PASS", "WARNING"]) == "WARNING"
    assert worst_status(["PASS", "CRITICAL", "WARNING"]) == "CRITICAL"


def test_schema_signature_is_order_independent():
    from credit_risk.monitoring import schema_signature

    assert schema_signature(["b", "a", "c"]) == schema_signature(["c", "b", "a"])


def test_reference_schema_match_suppresses_known_extra_feature_warning():
    from credit_risk.monitoring import schema_signature

    signature = schema_signature(["frozen_a", "frozen_b", "known_extra"])
    schema = pd.DataFrame(
        [
            {
                "population": "window_1",
                "raw_schema_signature": signature,
                "missing_nonstructural_count": 0,
                "dropped_extra_count": 1,
            }
        ]
    )
    alerts = schema_alerts(
        schema,
        MonitoringThresholds(),
        reference_raw_schema_signatures={signature},
    )
    assert alerts == []


def test_reference_schema_mismatch_emits_warning_without_missing_frozen_features():
    from credit_risk.monitoring import schema_signature

    reference_signature = schema_signature(["frozen_a", "frozen_b", "known_extra"])
    current_signature = schema_signature(["frozen_a", "frozen_b", "novel_extra"])
    schema = pd.DataFrame(
        [
            {
                "population": "window_1",
                "raw_schema_signature": current_signature,
                "missing_nonstructural_count": 0,
                "dropped_extra_count": 1,
            }
        ]
    )
    frame = alerts_to_frame(
        schema_alerts(
            schema,
            MonitoringThresholds(),
            reference_raw_schema_signatures={reference_signature},
        )
    )
    assert len(frame) == 1
    assert frame.iloc[0]["metric"] == "raw_schema_change::window_1"
    assert frame.iloc[0]["status"] == "WARNING"
