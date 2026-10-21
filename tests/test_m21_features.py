import numpy as np
import pandas as pd

from credit_risk.features import (
    aggregate_category_distributions,
    aggregate_extended_payment_dynamics,
    aggregate_legacy_credit_history,
)


def _toy_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2],
            "rn": [1, 2, 1],
            "pre_since_opened": [2, 4, 7],
            "enc_loans_credit_status": [1, 2, 1],
            "enc_loans_credit_type": [3, 3, 4],
            "enc_loans_account_cur": [1, 2, 1],
            "enc_loans_account_holder_type": [0, 0, 1],
            "is_zero_util": [1, 0, 1],
            "pclose_flag": [0, 1, 0],
            "enc_paym_0": [1, 2, 1],
            "enc_paym_1": [1, 3, 2],
            "enc_paym_2": [2, 3, 1],
            "enc_paym_3": [2, 2, 1],
        }
    )


def test_legacy_ablation_differs_only_by_latest_block():
    history = _toy_history()
    legacy = aggregate_legacy_credit_history(history, include_latest=True)
    no_latest = aggregate_legacy_credit_history(history, include_latest=False)

    latest_cols = [column for column in legacy.columns if column.endswith("__latest")]
    assert latest_cols
    assert set(no_latest.columns) == set(legacy.columns) - set(latest_cols)

    app1 = legacy.set_index("id").loc[1]
    assert app1["pre_since_opened__latest"] == 4


def test_category_distribution_is_fractional_and_not_numeric_mean():
    result = aggregate_category_distributions(_toy_history()).set_index("id")

    assert np.isclose(result.loc[1, "enc_loans_credit_status__frac__1"], 0.5)
    assert np.isclose(result.loc[1, "enc_loans_credit_status__frac__2"], 0.5)
    status_fraction_cols = [
        column for column in result.columns if column.startswith("enc_loans_credit_status__frac__")
    ]
    assert np.isclose(result.loc[1, status_fraction_cols].sum(), 1.0)
    assert not any(column.endswith("__mean") for column in result.columns)


def test_extended_payment_dynamics_are_reversal_invariant():
    history = _toy_history()
    forward = aggregate_extended_payment_dynamics(history).sort_values("id").reset_index(drop=True)

    reversed_history = history.copy()
    payment_cols = ["enc_paym_0", "enc_paym_1", "enc_paym_2", "enc_paym_3"]
    reversed_history[payment_cols] = history[payment_cols[::-1]].to_numpy()
    backward = aggregate_extended_payment_dynamics(reversed_history).sort_values("id").reset_index(drop=True)

    pd.testing.assert_frame_equal(forward, backward, check_exact=False, rtol=1e-6, atol=1e-6)
