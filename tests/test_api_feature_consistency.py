import numpy as np

from credit_risk.api.runtime import applications_to_history
from credit_risk.api.schemas import ApplicationRequest, CreditHistoryRecord
from credit_risk.features import aggregate_canonical_m21_features


def make_record(rn: int, payment_state: int, credit_type: int):
    payload = {name: 0 for name in CreditHistoryRecord.model_fields}
    payload.update(
        rn=rn,
        pre_util=rn,
        enc_loans_credit_type=credit_type,
        enc_loans_credit_status=1,
        enc_loans_account_holder_type=0,
        enc_loans_account_cur=0,
    )
    for i in range(25):
        payload[f"enc_paym_{i}"] = payment_state if i < 12 else payment_state + 1
    return CreditHistoryRecord(**payload)


def test_api_raw_frame_reuses_offline_canonical_feature_code():
    app = ApplicationRequest(
        application_id=123,
        credit_history=[make_record(1, 0, 2), make_record(2, 1, 5)],
    )
    raw = applications_to_history([app])
    features = aggregate_canonical_m21_features(raw)
    assert features["id"].tolist() == [123]
    assert features.shape[1] > 150
    assert np.isfinite(features["history__row_count"].iloc[0])
    assert features["history__row_count"].iloc[0] == 2
