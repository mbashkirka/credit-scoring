from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SemanticFeature:
    group_code: str
    group_name: str
    risk_increase_text: str
    risk_reduce_text: str
    caveat: str


_GENERIC_CAVEAT = (
    "Predictive association only. The underlying inputs are anonymized/encoded credit-history "
    "features; this text is an internal model-explanation label, not a causal claim or a "
    "compliance-approved adverse-action notice."
)


def semantic_feature(feature: str) -> SemanticFeature:
    """Map a technical M2.1-E feature into a stable semantic explanation family.

    The mapping intentionally stays at feature-family level because many Alfa-Bank inputs are
    anonymized/bin-encoded. It must not decode category IDs into meanings that are absent from
    the dataset documentation.
    """
    f = str(feature)

    if f == "history__row_count" or f.startswith("loan_categories__"):
        return SemanticFeature(
            "HISTORY_BREADTH",
            "Credit-history breadth and complexity",
            "Credit-history breadth/complexity pattern increases the model's estimated risk.",
            "Credit-history breadth/complexity pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if f.startswith("enc_paym_") or f.startswith("payment_") or f.startswith("payment"):
        return SemanticFeature(
            "REPAYMENT_PATTERN",
            "Repayment behavior pattern",
            "Repayment-history pattern increases the model's estimated risk.",
            "Repayment-history pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if "loans90" in f or "loans6090" in f or "loans3060" in f or "loans530" in f or "loans5" in f:
        return SemanticFeature(
            "DELINQUENCY_PATTERN",
            "Historical delinquency pattern",
            "Historical delinquency pattern increases the model's estimated risk.",
            "Historical delinquency pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if "total_overdue" in f or "max_overdue_sum" in f:
        return SemanticFeature(
            "OVERDUE_EXPOSURE",
            "Historical overdue exposure",
            "Historical overdue-exposure pattern increases the model's estimated risk.",
            "Historical overdue-exposure pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if (
        "pre_util" in f
        or "pre_over2limit" in f
        or "pre_maxover2limit" in f
        or "is_zero_util" in f
        or "is_zero_over2limit" in f
        or "is_zero_maxover2limit" in f
    ):
        return SemanticFeature(
            "UTILIZATION_LIMIT",
            "Utilization and limit behavior",
            "Utilization/limit-behavior pattern increases the model's estimated risk.",
            "Utilization/limit-behavior pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if (
        "credit_limit" in f
        or "next_pay_summ" in f
        or "outstanding" in f
        or "credit_cost_rate" in f
    ):
        return SemanticFeature(
            "EXPOSURE_BALANCE",
            "Credit exposure and balance pattern",
            "Credit exposure/balance pattern increases the model's estimated risk.",
            "Credit exposure/balance pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if (
        "since_opened" in f
        or "since_confirmed" in f
        or "pre_pterm" in f
        or "pre_fterm" in f
        or "till_pclose" in f
        or "till_fclose" in f
        or "pclose_flag" in f
        or "fclose_flag" in f
    ):
        return SemanticFeature(
            "AGE_TERM_CLOSURE",
            "Credit age, term, and closure pattern",
            "Credit age/term/closure pattern increases the model's estimated risk.",
            "Credit age/term/closure pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if f.startswith("enc_loans_credit_status"):
        return SemanticFeature(
            "CREDIT_STATUS_MIX",
            "Credit-status mix",
            "Credit-status mix pattern increases the model's estimated risk.",
            "Credit-status mix pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if f.startswith("enc_loans_credit_type"):
        return SemanticFeature(
            "CREDIT_PRODUCT_MIX",
            "Credit-product mix",
            "Credit-product mix pattern increases the model's estimated risk.",
            "Credit-product mix pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if f.startswith("enc_loans_account_holder_type"):
        return SemanticFeature(
            "ACCOUNT_HOLDER_MIX",
            "Account-holder-type mix",
            "Account-holder-type mix pattern increases the model's estimated risk.",
            "Account-holder-type mix pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    if f.startswith("enc_loans_account_cur"):
        return SemanticFeature(
            "ACCOUNT_CURRENCY_MIX",
            "Account-currency mix",
            "Account-currency mix pattern increases the model's estimated risk.",
            "Account-currency mix pattern decreases the model's estimated risk.",
            _GENERIC_CAVEAT,
        )

    return SemanticFeature(
        "OTHER_CREDIT_HISTORY",
        "Other credit-history pattern",
        "Credit-history pattern increases the model's estimated risk.",
        "Credit-history pattern decreases the model's estimated risk.",
        _GENERIC_CAVEAT,
    )


def feature_semantics_table(feature_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for feature in feature_names:
        semantic = semantic_feature(feature)
        rows.append(
            {
                "feature": feature,
                "group_code": semantic.group_code,
                "group_name": semantic.group_name,
                "risk_increase_text": semantic.risk_increase_text,
                "risk_reduce_text": semantic.risk_reduce_text,
                "caveat": semantic.caveat,
            }
        )
    return pd.DataFrame(rows)


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    values = np.asarray(shap_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("SHAP matrix shape does not match feature_names.")
    rows = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_signed_shap": np.mean(values, axis=0),
            "positive_contribution_share": np.mean(values > 0.0, axis=0),
        }
    )
    semantics = feature_semantics_table(feature_names)[["feature", "group_code", "group_name"]]
    rows = rows.merge(semantics, on="feature", how="left", validate="one_to_one")
    total = float(rows["mean_abs_shap"].sum())
    rows["importance_share"] = rows["mean_abs_shap"] / total if total > 0 else 0.0
    rows = rows.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1, dtype=int))
    return rows


def global_group_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    required = {"group_code", "group_name", "mean_abs_shap", "mean_signed_shap"}
    missing = required - set(feature_importance.columns)
    if missing:
        raise ValueError(f"feature_importance missing columns: {sorted(missing)}")
    grouped = (
        feature_importance.groupby(["group_code", "group_name"], as_index=False)
        .agg(
            n_features=("feature", "size"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            total_mean_signed_shap=("mean_signed_shap", "sum"),
        )
    )
    total = float(grouped["total_mean_abs_shap"].sum())
    grouped["importance_share"] = (
        grouped["total_mean_abs_shap"] / total if total > 0 else 0.0
    )
    grouped = grouped.sort_values(
        ["total_mean_abs_shap", "group_code"], ascending=[False, True]
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1, dtype=int))
    return grouped


def group_signed_contributions(
    shap_row: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    values = np.asarray(shap_row, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(feature_names):
        raise ValueError("shap_row shape does not match feature_names.")
    semantics = feature_semantics_table(feature_names)
    frame = semantics[["feature", "group_code", "group_name"]].copy()
    frame["shap_value"] = values
    grouped = (
        frame.groupby(["group_code", "group_name"], as_index=False)
        .agg(group_shap_log_odds=("shap_value", "sum"))
    )
    return grouped


def local_reason_rows(
    *,
    application_id: int,
    calibrated_pd: float,
    decision: str,
    risk_band: str,
    feature_values: np.ndarray,
    shap_row: np.ndarray,
    feature_names: list[str],
    top_n_each_direction: int = 3,
) -> pd.DataFrame:
    """Create de-duplicated semantic reason codes from signed SHAP contributions.

    Reason codes are selected at semantic-group level to avoid emitting several correlated
    technical features as separate human-facing explanations.
    """
    if top_n_each_direction <= 0:
        raise ValueError("top_n_each_direction must be positive.")
    values = np.asarray(feature_values, dtype=np.float64)
    shap = np.asarray(shap_row, dtype=np.float64)
    if values.ndim != 1 or shap.ndim != 1 or len(values) != len(feature_names) or len(shap) != len(feature_names):
        raise ValueError("feature_values/shap_row shapes must match feature_names.")

    semantics = feature_semantics_table(feature_names)
    tech = semantics[[
        "feature",
        "group_code",
        "group_name",
        "risk_increase_text",
        "risk_reduce_text",
        "caveat",
    ]].copy()
    tech["feature_value"] = values
    tech["shap_log_odds"] = shap

    group = (
        tech.groupby(["group_code", "group_name"], as_index=False)
        .agg(group_shap_log_odds=("shap_log_odds", "sum"))
    )

    rows: list[dict[str, object]] = []
    for direction, subset in (
        ("risk_increasing", group[group["group_shap_log_odds"] > 0].nlargest(top_n_each_direction, "group_shap_log_odds")),
        ("risk_reducing", group[group["group_shap_log_odds"] < 0].nsmallest(top_n_each_direction, "group_shap_log_odds")),
    ):
        for rank, g in enumerate(subset.itertuples(index=False), start=1):
            group_features = tech[tech["group_code"] == g.group_code].copy()
            if direction == "risk_increasing":
                representative = group_features.loc[group_features["shap_log_odds"].idxmax()]
                reason_text = str(representative["risk_increase_text"])
            else:
                representative = group_features.loc[group_features["shap_log_odds"].idxmin()]
                reason_text = str(representative["risk_reduce_text"])
            rows.append(
                {
                    "id": int(application_id),
                    "calibrated_pd": float(calibrated_pd),
                    "decision": str(decision),
                    "risk_band": str(risk_band),
                    "direction": direction,
                    "rank": rank,
                    "reason_code": str(g.group_code),
                    "reason_text": reason_text,
                    "group_shap_log_odds": float(g.group_shap_log_odds),
                    "representative_feature": str(representative["feature"]),
                    "representative_feature_value": float(representative["feature_value"]),
                    "representative_feature_shap_log_odds": float(representative["shap_log_odds"]),
                    "caveat": str(representative["caveat"]),
                }
            )
    return pd.DataFrame(rows)


def choose_representative_applications(
    frame: pd.DataFrame,
    *,
    decision_column: str = "decision_balanced",
    pd_column: str = "calibrated_pd",
    examples_per_decision: int = 3,
) -> pd.DataFrame:
    """Pick deterministic low/median/high-PD examples within each decision class.

    Outcome labels are intentionally not used for selection.
    """
    if examples_per_decision <= 0:
        raise ValueError("examples_per_decision must be positive.")
    required = {"id", decision_column, pd_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    quantiles = np.linspace(0.2, 0.8, examples_per_decision)
    selected: list[pd.Series] = []
    for decision in ("APPROVE", "REVIEW", "REJECT"):
        subset = frame[frame[decision_column] == decision].copy()
        if subset.empty:
            continue
        subset = subset.sort_values([pd_column, "id"]).reset_index(drop=True)
        used: set[int] = set()
        for q in quantiles:
            target = float(subset[pd_column].quantile(q))
            order = np.argsort(np.abs(subset[pd_column].to_numpy(dtype=float) - target), kind="stable")
            chosen_idx = next((int(i) for i in order if int(i) not in used), int(order[0]))
            used.add(chosen_idx)
            selected.append(subset.iloc[chosen_idx])
    if not selected:
        return frame.iloc[0:0].copy()
    result = pd.DataFrame(selected).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return result
