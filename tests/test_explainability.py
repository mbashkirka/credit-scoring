import numpy as np
import pandas as pd

from credit_risk.explainability import (
    choose_representative_applications,
    feature_semantics_table,
    global_feature_importance,
    global_group_importance,
    group_signed_contributions,
    local_reason_rows,
    semantic_feature,
)


def test_semantic_mapping_known_families():
    assert semantic_feature("pre_loans90__mean").group_code == "DELINQUENCY_PATTERN"
    assert semantic_feature("pre_util__max").group_code == "UTILIZATION_LIMIT"
    assert semantic_feature("enc_loans_credit_status__frac__3").group_code == "CREDIT_STATUS_MIX"
    assert semantic_feature("payment_transition_rate__mean").group_code == "REPAYMENT_PATTERN"
    assert semantic_feature("history__row_count").group_code == "HISTORY_BREADTH"


def test_feature_semantics_one_row_per_feature():
    names = ["pre_util__mean", "enc_paym_0__nunique", "history__row_count"]
    table = feature_semantics_table(names)
    assert table["feature"].tolist() == names
    assert table["feature"].is_unique
    assert table["group_code"].notna().all()


def test_global_importance_and_groups_sum_to_one():
    names = ["pre_util__mean", "pre_util__max", "pre_loans90__mean"]
    shap = np.array([[1.0, -1.0, 2.0], [3.0, -1.0, -2.0]])
    features = global_feature_importance(shap, names)
    groups = global_group_importance(features)
    assert np.isclose(features["importance_share"].sum(), 1.0)
    assert np.isclose(groups["importance_share"].sum(), 1.0)
    assert groups.loc[groups["group_code"] == "UTILIZATION_LIMIT", "n_features"].iloc[0] == 2


def test_group_signed_contributions_preserve_sum():
    names = ["pre_util__mean", "pre_util__max", "pre_loans90__mean"]
    shap = np.array([0.5, -0.2, 0.7])
    grouped = group_signed_contributions(shap, names)
    assert np.isclose(grouped["group_shap_log_odds"].sum(), shap.sum())


def test_local_reason_rows_are_group_deduplicated_and_directional():
    names = ["pre_util__mean", "pre_util__max", "pre_loans90__mean", "enc_paym_0__nunique"]
    values = np.array([1.0, 2.0, 3.0, 4.0])
    shap = np.array([0.4, 0.2, -0.5, 0.3])
    rows = local_reason_rows(
        application_id=1,
        calibrated_pd=0.1,
        decision="REVIEW",
        risk_band="E",
        feature_values=values,
        shap_row=shap,
        feature_names=names,
        top_n_each_direction=3,
    )
    increasing = rows[rows["direction"] == "risk_increasing"]
    reducing = rows[rows["direction"] == "risk_reducing"]
    assert increasing["reason_code"].is_unique
    assert reducing["reason_code"].is_unique
    assert (increasing["group_shap_log_odds"] > 0).all()
    assert (reducing["group_shap_log_odds"] < 0).all()


def test_representative_selection_is_deterministic_and_does_not_require_label():
    frame = pd.DataFrame(
        {
            "id": np.arange(30),
            "calibrated_pd": np.linspace(0.001, 0.2, 30),
            "decision_balanced": ["APPROVE"] * 10 + ["REVIEW"] * 10 + ["REJECT"] * 10,
            "source_chunk": [8] * 15 + [9] * 15,
        }
    )
    first = choose_representative_applications(frame, examples_per_decision=3)
    second = choose_representative_applications(frame, examples_per_decision=3)
    assert first["id"].tolist() == second["id"].tolist()
    assert len(first) == 9
    assert "flag" not in frame.columns
