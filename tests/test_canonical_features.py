import pandas as pd

from credit_risk.features import aggregate_canonical_m21_features


def test_canonical_m21_builder_has_one_row_per_application():
    history = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "rn": [1, 2, 1],
            "pre_since_opened": [1, 2, 3],
            "enc_loans_credit_status": [1, 2, 1],
            "enc_loans_credit_type": [1, 1, 2],
            "enc_loans_account_cur": [1, 1, 1],
            "enc_loans_account_holder_type": [1, 1, 1],
            "enc_paym_0": [0, 0, 1],
            "enc_paym_1": [0, 1, 1],
            "pclose_flag": [0, 1, 0],
        }
    )
    features = aggregate_canonical_m21_features(history)
    assert set(features["id"]) == {1, 2}
    assert not features["id"].duplicated().any()
    assert "enc_loans_credit_status__frac__1" in features.columns
    assert "payment_ext_transition_rate__mean" in features.columns
