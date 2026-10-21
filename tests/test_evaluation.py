import numpy as np
import pandas as pd
import pytest

from credit_risk.evaluation import align_to_frozen_feature_schema, assert_disjoint_ids


def test_align_to_frozen_schema_fills_structural_zero_and_drops_extra():
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "history__row_count": [2.0, 3.0],
            "enc_loans_credit_status__frac__1": [0.5, 0.2],
            "unexpected_new_category": [1.0, 1.0],
        }
    )
    schema = [
        "history__row_count",
        "enc_loans_credit_status__frac__1",
        "enc_loans_credit_status__frac__2",
        "some_nonstructural_feature",
    ]
    aligned, report = align_to_frozen_feature_schema(frame, schema)
    assert list(aligned.columns) == ["id", *schema]
    assert np.allclose(aligned["enc_loans_credit_status__frac__2"], 0.0)
    assert aligned["some_nonstructural_feature"].isna().all()
    assert "unexpected_new_category" in report.dropped_extra
    assert "enc_loans_credit_status__frac__2" in report.missing_structural_zero
    assert "some_nonstructural_feature" in report.missing_nonstructural


def test_assert_disjoint_ids_rejects_overlap():
    assert_disjoint_ids(np.array([1, 2]), np.array([3, 4]))
    with pytest.raises(RuntimeError, match="overlap"):
        assert_disjoint_ids(np.array([1, 2]), np.array([2, 3]))
