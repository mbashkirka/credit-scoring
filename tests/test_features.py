import pandas as pd

from credit_risk.features import aggregate_credit_history


def test_m2_aggregation_uses_typed_features_and_no_latest():
    history = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "rn": [1, 2, 1],
            "pre_since_opened": [2, 4, 7],
            "enc_loans_credit_status": [1, 2, 1],
            "is_zero_util": [1, 0, 1],
            "pclose_flag": [0, 1, 0],
            "enc_paym_0": [1, 2, 1],
            "enc_paym_1": [1, 3, 2],
        }
    )

    result = aggregate_credit_history(history)

    assert len(result) == 2
    assert not any(column.endswith("__latest") for column in result.columns)
    assert "history__row_count" in result.columns
    assert "pre_since_opened__mean" in result.columns
    assert "enc_loans_credit_status__nunique" in result.columns
    assert "is_zero_util__mean" in result.columns
    assert "pclose_flag__sum" in result.columns
    assert "payment_state_nunique__mean" in result.columns
    assert "payment_transition_count__max" in result.columns
