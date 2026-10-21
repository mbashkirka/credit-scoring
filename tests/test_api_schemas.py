import pytest
from pydantic import ValidationError

from credit_risk.api.schemas import ApplicationRequest, BatchScoreRequest, CreditHistoryRecord


def valid_record(**updates):
    payload = {name: 0 for name in CreditHistoryRecord.model_fields}
    payload["rn"] = 1
    payload.update(updates)
    return payload


def test_raw_schema_rejects_target_and_unknown_fields():
    with pytest.raises(ValidationError):
        CreditHistoryRecord(**valid_record(flag=1))


def test_application_requires_history_and_unique_rn():
    with pytest.raises(ValidationError):
        ApplicationRequest(application_id=1, credit_history=[])
    rec = CreditHistoryRecord(**valid_record())
    with pytest.raises(ValidationError):
        ApplicationRequest(application_id=1, credit_history=[rec, rec])


def test_batch_rejects_duplicate_application_ids():
    app = ApplicationRequest(
        application_id=1,
        credit_history=[CreditHistoryRecord(**valid_record())],
    )
    with pytest.raises(ValidationError):
        BatchScoreRequest(applications=[app, app])
