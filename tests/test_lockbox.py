import numpy as np
import pytest

from credit_risk.lockbox import (
    EXPECTED_LOCKBOX_CHUNKS,
    frozen_primary_scenario,
    frozen_risk_band_cutpoints,
    validate_lockbox_governance,
)


def _policy():
    return {
        "primary_policy": "balanced",
        "lockbox_chunks_reserved": [10, 11],
        "lockbox_evaluated": False,
        "scenarios": [
            {
                "name": "balanced",
                "performing_margin_rate": 0.12,
                "lgd": 0.45,
                "origination_cost_rate": 0.02,
                "risk_appetite_pd_cap": 0.12,
                "review_band_fraction": 0.20,
                "primary": True,
                "break_even_pd": 0.1754,
                "approve_pd_threshold": 0.096,
                "reject_pd_threshold": 0.12,
            }
        ],
    }


def _bands():
    return {"labels": ["A", "B", "C", "D", "E"], "cutpoints": [0.01, 0.02, 0.04, 0.08]}


def test_frozen_primary_scenario_reconstructs_predeclared_thresholds():
    s = frozen_primary_scenario(_policy())
    assert s.name == "balanced"
    assert np.isclose(s.approve_pd_threshold, 0.096)
    assert np.isclose(s.reject_pd_threshold, 0.12)


def test_frozen_risk_band_cutpoints_are_loaded_without_reestimation():
    cuts = frozen_risk_band_cutpoints(_bands())
    assert np.allclose(cuts, [0.01, 0.02, 0.04, 0.08])


def test_lockbox_governance_accepts_only_10_11_and_untouched_metadata():
    m4 = {"lockbox_chunks_reserved": [10, 11], "final_lockbox_evaluated": False}
    m5 = {"lockbox_chunks_reserved": [10, 11], "lockbox_evaluated": False, "primary_policy": "balanced"}
    validate_lockbox_governance(m4, m5, _policy(), _bands(), EXPECTED_LOCKBOX_CHUNKS)


def test_lockbox_governance_rejects_changed_population():
    m4 = {"lockbox_chunks_reserved": [10, 11], "final_lockbox_evaluated": False}
    m5 = {"lockbox_chunks_reserved": [10, 11], "lockbox_evaluated": False, "primary_policy": "balanced"}
    with pytest.raises(RuntimeError):
        validate_lockbox_governance(m4, m5, _policy(), _bands(), (9, 10))


def test_lockbox_governance_rejects_previously_evaluated_policy():
    m4 = {"lockbox_chunks_reserved": [10, 11], "final_lockbox_evaluated": False}
    m5 = {"lockbox_chunks_reserved": [10, 11], "lockbox_evaluated": True, "primary_policy": "balanced"}
    with pytest.raises(RuntimeError):
        validate_lockbox_governance(m4, m5, _policy(), _bands(), EXPECTED_LOCKBOX_CHUNKS)
