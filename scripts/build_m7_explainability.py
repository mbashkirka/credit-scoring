from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from credit_risk.decisioning import assign_decisions, assign_risk_bands
from credit_risk.explainability import (
    choose_representative_applications,
    feature_semantics_table,
    global_feature_importance,
    global_group_importance,
    group_signed_contributions,
    local_reason_rows,
)
from credit_risk.evaluation import align_to_frozen_feature_schema, load_frozen_feature_names
from credit_risk.lockbox import frozen_primary_scenario, frozen_risk_band_cutpoints


ARTIFACT_DIR = Path("artifacts")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
FEATURE_MANIFEST = ARTIFACT_DIR / "m3_features.csv"
M4_MODEL = ARTIFACT_DIR / "m4_catboost.cbm"
M4_IMPUTER = ARTIFACT_DIR / "m4_catboost_imputer.joblib"
M4_CALIBRATOR = ARTIFACT_DIR / "m4_calibrator.joblib"
M4_IMPORTANCE = ARTIFACT_DIR / "m4_catboost_feature_importance.csv"
M4_METADATA = ARTIFACT_DIR / "m4_metadata.json"
M5_DECISIONS = ARTIFACT_DIR / "m5_decisions.parquet"
M5_POLICY = ARTIFACT_DIR / "m5_policy_freeze.json"
M5_BANDS = ARTIFACT_DIR / "m5_risk_band_thresholds.json"
M6_MARKER = ARTIFACT_DIR / "m6_lockbox_opened.json"

DEVELOPMENT_CHUNKS = (8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M7 explainability for frozen Model V1. Uses CatBoost native TreeSHAP on a "
            "deterministic sample of M5 policy-development applications; no fitting or "
            "decision-policy changes are performed."
        )
    )
    parser.add_argument("--global-sample", type=int, default=10_000)
    parser.add_argument("--examples-per-decision", type=int, default=3)
    parser.add_argument("--top-reasons", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_inputs() -> tuple[list[str], dict, dict, dict]:
    required = [
        FEATURE_MANIFEST,
        M4_MODEL,
        M4_IMPUTER,
        M4_CALIBRATOR,
        M4_METADATA,
        M5_DECISIONS,
        M5_POLICY,
        M5_BANDS,
        M6_MARKER,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("M7 requires completed M4-M6 artifacts. Missing: " + ", ".join(missing))

    feature_names = load_frozen_feature_names(FEATURE_MANIFEST)
    if len(feature_names) != 212:
        raise RuntimeError(f"M7 expects 212 frozen Model V1 features; got {len(feature_names)}.")

    m4_meta = json.loads(M4_METADATA.read_text(encoding="utf-8"))
    policy = json.loads(M5_POLICY.read_text(encoding="utf-8"))
    bands = json.loads(M5_BANDS.read_text(encoding="utf-8"))
    marker = json.loads(M6_MARKER.read_text(encoding="utf-8"))
    if marker.get("lockbox_chunks") != [10, 11]:
        raise RuntimeError("M7 expects M6 lockbox evaluation to have opened chunks 10-11.")
    if m4_meta.get("selected_calibration_method") != "isotonic":
        raise RuntimeError("M7 expects frozen Model V1 isotonic calibration.")
    if policy.get("primary_policy") != "balanced":
        raise RuntimeError("M7 expects frozen Model V1 balanced policy.")
    return feature_names, m4_meta, policy, bands


def load_development_decisions(calibrator, scenario, cutpoints: np.ndarray) -> pd.DataFrame:
    frame = pd.read_parquet(M5_DECISIONS)
    required = {"id", "source_chunk", "raw_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"M5 decisions artifact is missing columns: {missing}")
    if set(frame["source_chunk"].astype(int).unique()) != set(DEVELOPMENT_CHUNKS):
        raise RuntimeError("M7 global explanations must use M5 policy-development chunks 8-9.")
    if frame["id"].duplicated().any():
        raise RuntimeError("Duplicate ids in M5 development decisions.")

    raw = frame["raw_score"].to_numpy(dtype=np.float64)
    pd_values = np.asarray(calibrator.predict(raw), dtype=np.float64)
    result = frame[["id", "source_chunk", "raw_score"]].copy()
    result["calibrated_pd"] = pd_values
    result["risk_band"] = assign_risk_bands(pd_values, cutpoints)
    result["decision_balanced"] = assign_decisions(pd_values, scenario)
    return result


def deterministic_global_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--global-sample must be positive.")
    n = min(n, len(frame))
    # Uniform application-level sample preserves portfolio-level SHAP importance.
    return frame.sample(n=n, random_state=seed, replace=False).sort_values("id").reset_index(drop=True)


def load_feature_rows(request: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    schema_rows: list[dict[str, object]] = []
    for chunk in sorted(request["source_chunk"].astype(int).unique()):
        cache = FEATURE_DIR / f"train_data_{chunk}.parquet"
        if not cache.exists():
            raise FileNotFoundError(f"M7 feature cache missing: {cache}")
        wanted = request.loc[request["source_chunk"] == chunk, "id"].astype(np.int64)
        raw = pd.read_parquet(cache)
        aligned, report = align_to_frozen_feature_schema(raw, feature_names)
        subset = aligned[aligned["id"].isin(set(wanted.tolist()))].copy()
        if len(subset) != len(wanted):
            missing_ids = sorted(set(wanted.tolist()) - set(subset["id"].astype(int).tolist()))
            raise RuntimeError(f"Feature rows missing for chunk {chunk}: {missing_ids[:10]}")
        rows.append(subset)
        schema_rows.append(
            {
                "source_chunk": chunk,
                "requested_rows": int(len(wanted)),
                "raw_feature_count": int(raw.shape[1] - 1),
                "frozen_feature_count": len(feature_names),
                "missing_structural_zero_count": len(report.missing_structural_zero),
                "missing_nonstructural_count": len(report.missing_nonstructural),
                "dropped_extra_count": len(report.dropped_extra),
                "dropped_extra_features": "|".join(report.dropped_extra),
            }
        )
        del raw, aligned, subset
        gc.collect()
    combined = pd.concat(rows, ignore_index=True)
    order = request[["id"]].copy()
    combined = order.merge(combined, on="id", how="left", validate="one_to_one")
    if combined[feature_names].isna().all(axis=1).any():
        raise RuntimeError("M7 failed to align one or more requested feature rows.")
    return combined, pd.DataFrame(schema_rows)


def compute_tree_shap(model, imputer, features: pd.DataFrame, feature_names: list[str]):
    matrix = imputer.transform(features[feature_names])
    pool = Pool(matrix, feature_names=feature_names)
    full = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=np.float64)
    if full.shape != (len(features), len(feature_names) + 1):
        raise RuntimeError(f"Unexpected CatBoost SHAP shape: {full.shape}")
    shap = full[:, :-1]
    base = full[:, -1]
    raw_margin = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=np.float64).reshape(-1)
    reconstructed = base + shap.sum(axis=1)
    max_error = float(np.max(np.abs(raw_margin - reconstructed))) if len(raw_margin) else 0.0
    if max_error > 1e-5:
        raise RuntimeError(f"TreeSHAP additivity check failed; max raw-margin error={max_error:.3e}")
    probability = np.asarray(model.predict_proba(pool)[:, 1], dtype=np.float64)
    return matrix, shap, base, raw_margin, probability, max_error


def merge_builtin_importance(global_features: pd.DataFrame) -> pd.DataFrame:
    if not M4_IMPORTANCE.exists():
        global_features["catboost_prediction_values_change"] = np.nan
        global_features["catboost_importance_rank"] = np.nan
        return global_features
    builtin = pd.read_csv(M4_IMPORTANCE)
    if not {"feature", "importance"}.issubset(builtin.columns):
        return global_features
    builtin = builtin[["feature", "importance"]].rename(
        columns={"importance": "catboost_prediction_values_change"}
    )
    builtin["catboost_importance_rank"] = builtin["catboost_prediction_values_change"].rank(
        method="min", ascending=False
    )
    return global_features.merge(builtin, on="feature", how="left", validate="one_to_one")


def save_global_plots(feature_importance: pd.DataFrame, group_importance: pd.DataFrame) -> None:
    top = feature_importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |TreeSHAP| (CatBoost raw log-odds)")
    ax.set_title("M7 global TreeSHAP — top 20 frozen Model V1 features")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m7_global_shap_top20.png", dpi=160)
    plt.close(fig)

    groups = group_importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(groups["group_name"], groups["importance_share"] * 100)
    ax.set_xlabel("Share of total mean |TreeSHAP|, %")
    ax.set_title("M7 semantic explanation groups")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m7_semantic_group_importance.png", dpi=160)
    plt.close(fig)


def save_local_plot(application_id: int, decision: str, groups: pd.DataFrame) -> str:
    data = groups.copy()
    data["abs"] = data["group_shap_log_odds"].abs()
    data = data.nlargest(8, "abs").sort_values("group_shap_log_odds")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(data["group_name"], data["group_shap_log_odds"])
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Signed TreeSHAP contribution (raw log-odds)")
    ax.set_title(f"M7 local explanation — id={application_id}, decision={decision}")
    fig.tight_layout()
    name = f"m7_local_example_{decision.lower()}.png"
    fig.savefig(ARTIFACT_DIR / name, dpi=160)
    plt.close(fig)
    return name


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    feature_names, m4_meta, policy_artifact, band_artifact = validate_inputs()
    model = CatBoostClassifier()
    model.load_model(M4_MODEL)
    imputer = joblib.load(M4_IMPUTER)
    calibrator = joblib.load(M4_CALIBRATOR)
    scenario = frozen_primary_scenario(policy_artifact)
    cutpoints = frozen_risk_band_cutpoints(band_artifact)

    development = load_development_decisions(calibrator, scenario, cutpoints)
    global_sample = deterministic_global_sample(development, args.global_sample, args.seed)
    local_examples = choose_representative_applications(
        development,
        examples_per_decision=args.examples_per_decision,
    )
    request = pd.concat(
        [
            global_sample[["id", "source_chunk"]],
            local_examples[["id", "source_chunk"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["id"]).reset_index(drop=True)

    print("=== M7 EXPLAINABILITY & INTERNAL REASON CODES ===")
    print("Frozen Model V1 changes: NONE")
    print(f"Global explanation population: M5 development chunks {list(DEVELOPMENT_CHUNKS)}")
    print(f"Uniform global SHAP sample: {len(global_sample):,} applications")
    print(f"Representative local examples: {len(local_examples):,}")
    print("SHAP domain: CatBoost raw log-odds (pre-isotonic); final calibrated PD shown separately")
    print("Reason codes: internal explanatory prototype; predictive association, not causal/legal claims")

    feature_rows, schema = load_feature_rows(request, feature_names)
    matrix, shap, base, raw_margin, raw_probability, additivity_error = compute_tree_shap(
        model, imputer, feature_rows, feature_names
    )
    index_by_id = {int(app_id): i for i, app_id in enumerate(feature_rows["id"].astype(int).tolist())}

    global_idx = np.asarray([index_by_id[int(x)] for x in global_sample["id"].astype(int)], dtype=int)
    global_shap = shap[global_idx]
    global_features = merge_builtin_importance(global_feature_importance(global_shap, feature_names))
    global_groups = global_group_importance(global_features)
    semantics = feature_semantics_table(feature_names)

    # Persist the sampled SHAP matrix as float32 for reproducible downstream diagnostics.
    shap_frame = pd.DataFrame(global_shap.astype(np.float32), columns=[f"shap__{f}" for f in feature_names])
    shap_meta = global_sample[["id", "source_chunk", "calibrated_pd", "risk_band", "decision_balanced"]].reset_index(drop=True)
    pd.concat([shap_meta, shap_frame], axis=1).to_parquet(
        ARTIFACT_DIR / "m7_global_shap_sample.parquet", index=False
    )

    reason_frames: list[pd.DataFrame] = []
    payloads: list[dict[str, object]] = []
    plotted: set[str] = set()
    for row in local_examples.itertuples(index=False):
        idx = index_by_id[int(row.id)]
        calibrated_pd = float(calibrator.predict(np.asarray([raw_probability[idx]], dtype=np.float64))[0])
        risk_band = str(assign_risk_bands(np.asarray([calibrated_pd]), cutpoints)[0])
        decision = str(assign_decisions(np.asarray([calibrated_pd]), scenario)[0])
        reasons = local_reason_rows(
            application_id=int(row.id),
            calibrated_pd=calibrated_pd,
            decision=decision,
            risk_band=risk_band,
            feature_values=np.asarray(matrix[idx], dtype=np.float64),
            shap_row=shap[idx],
            feature_names=feature_names,
            top_n_each_direction=args.top_reasons,
        )
        reason_frames.append(reasons)
        group_contrib = group_signed_contributions(shap[idx], feature_names)
        payloads.append(
            {
                "id": int(row.id),
                "source_chunk": int(row.source_chunk),
                "raw_catboost_probability": float(raw_probability[idx]),
                "raw_catboost_log_odds": float(raw_margin[idx]),
                "tree_shap_base_log_odds": float(base[idx]),
                "tree_shap_sum_log_odds": float(shap[idx].sum()),
                "calibrated_pd": calibrated_pd,
                "risk_band": risk_band,
                "decision": decision,
                "risk_increasing_reasons": reasons[reasons["direction"] == "risk_increasing"][[
                    "rank", "reason_code", "reason_text", "group_shap_log_odds",
                    "representative_feature", "representative_feature_value",
                    "representative_feature_shap_log_odds",
                ]].to_dict(orient="records"),
                "risk_reducing_reasons": reasons[reasons["direction"] == "risk_reducing"][[
                    "rank", "reason_code", "reason_text", "group_shap_log_odds",
                    "representative_feature", "representative_feature_value",
                    "representative_feature_shap_log_odds",
                ]].to_dict(orient="records"),
                "interpretation_note": (
                    "TreeSHAP values explain the CatBoost raw log-odds. Isotonic calibration is a "
                    "separate monotonic mapping to PD, so SHAP contributions are not additive in "
                    "calibrated-probability space. Explanations are associative, not causal."
                ),
            }
        )
        if decision not in plotted:
            save_local_plot(int(row.id), decision, group_contrib)
            plotted.add(decision)

    reason_codes = pd.concat(reason_frames, ignore_index=True) if reason_frames else pd.DataFrame()

    global_features.to_csv(ARTIFACT_DIR / "m7_global_shap_feature_importance.csv", index=False)
    global_groups.to_csv(ARTIFACT_DIR / "m7_global_shap_group_importance.csv", index=False)
    semantics.to_csv(ARTIFACT_DIR / "m7_feature_semantics.csv", index=False)
    reason_codes.to_csv(ARTIFACT_DIR / "m7_reason_codes.csv", index=False)
    schema.to_csv(ARTIFACT_DIR / "m7_schema_alignment.csv", index=False)
    with (ARTIFACT_DIR / "m7_local_explanations.jsonl").open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    save_global_plots(global_features, global_groups)

    metadata = {
        "stage": "M7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "V1_frozen_after_M6",
        "model_modified": False,
        "global_explanation_population": list(DEVELOPMENT_CHUNKS),
        "global_sample_n": int(len(global_sample)),
        "global_sample_method": "uniform_application_sample_without_outcome_based_selection",
        "random_seed": int(args.seed),
        "tree_shap_space": "CatBoost RawFormulaVal log-odds before isotonic calibration",
        "tree_shap_additivity_max_abs_error": additivity_error,
        "calibration_method": m4_meta["selected_calibration_method"],
        "reason_code_level": "semantic feature-family aggregation",
        "reason_codes_are_causal": False,
        "reason_codes_are_compliance_approved_adverse_action_notices": False,
        "lockbox_used_for_model_change": False,
        "lockbox_status": "historical_evaluation_after_M6",
    }
    (ARTIFACT_DIR / "m7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry = ARTIFACT_DIR / "experiments.csv"
    row = pd.DataFrame([{
        "timestamp_utc": metadata["created_at_utc"],
        "stage": "M7",
        "experiment": "frozen_v1_tree_shap_and_reason_codes",
        "feature_version": "e_typed_catdist_payment",
        "model": "m4_catboost_frozen_v1",
        "n_applications": int(len(global_sample)),
        "n_features": len(feature_names),
        "model_modified": False,
        "explanation_method": "catboost_native_tree_shap",
        "tree_shap_additivity_max_abs_error": additivity_error,
    }])
    if registry.exists():
        existing = pd.read_csv(registry)
        pd.concat([existing, row], ignore_index=True, sort=False).to_csv(registry, index=False)
    else:
        row.to_csv(registry, index=False)

    print("\n=== M7 GLOBAL TREE-SHAP TOP FEATURES ===")
    for item in global_features.head(15).itertuples(index=False):
        print(
            f"{int(item.rank):2d}. {item.feature:48s} | mean|SHAP|={item.mean_abs_shap:.6f} | "
            f"group={item.group_code}"
        )

    print("\n=== M7 GLOBAL SEMANTIC GROUPS ===")
    for item in global_groups.itertuples(index=False):
        print(
            f"{int(item.rank):2d}. {item.group_code:22s} | share={item.importance_share:.2%} | "
            f"features={int(item.n_features)}"
        )

    print("\n=== M7 LOCAL EXPLANATION EXAMPLES ===")
    for payload in payloads:
        print(
            f"id={payload['id']} | PD={payload['calibrated_pd']:.4%} | "
            f"band={payload['risk_band']} | decision={payload['decision']}"
        )
        for reason in payload["risk_increasing_reasons"]:
            print(
                f"  risk+ #{reason['rank']}: {reason['reason_code']} | "
                f"SHAP={reason['group_shap_log_odds']:+.4f}"
            )
        for reason in payload["risk_reducing_reasons"]:
            print(
                f"  risk- #{reason['rank']}: {reason['reason_code']} | "
                f"SHAP={reason['group_shap_log_odds']:+.4f}"
            )

    print("\n=== M7 EXPLANATION VALIDATION ===")
    print(f"TreeSHAP raw-margin additivity max abs error: {additivity_error:.3e}")
    print("SHAP explains CatBoost raw log-odds; calibrated PD is reported but not decomposed additively.")
    print("Reason codes are internal predictive explanations, not causal or legally validated notices.")

    print("\nSaved:")
    for path in [
        "artifacts/m7_global_shap_feature_importance.csv",
        "artifacts/m7_global_shap_group_importance.csv",
        "artifacts/m7_feature_semantics.csv",
        "artifacts/m7_global_shap_sample.parquet",
        "artifacts/m7_reason_codes.csv",
        "artifacts/m7_local_explanations.jsonl",
        "artifacts/m7_schema_alignment.csv",
        "artifacts/m7_global_shap_top20.png",
        "artifacts/m7_semantic_group_importance.png",
        "artifacts/m7_local_example_approve.png",
        "artifacts/m7_local_example_review.png",
        "artifacts/m7_local_example_reject.png",
        "artifacts/m7_metadata.json",
        "artifacts/experiments.csv",
    ]:
        if Path(path).exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
