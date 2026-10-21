from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.api.runtime import ArtifactPaths, ScoringRuntime
from credit_risk.calibration import calibration_metrics
from credit_risk.evaluation import align_to_frozen_feature_schema
from credit_risk.metrics import ranking_metrics
from credit_risk.monitoring import (
    MonitorAlert,
    MonitoringThresholds,
    NumericReferenceProfile,
    alerts_to_frame,
    compare_categorical_shares,
    compare_numeric_profile,
    feature_alerts,
    histogram_counts,
    increase_status,
    schema_alerts,
    schema_signature,
    threshold_status,
    worst_status,
    psi_from_counts,
    drop_status,
)

ARTIFACT_DIR = Path("artifacts")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
DEFAULT_CURRENT_CHUNKS = (10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Model-V1 monitoring. Default mode replays already-opened M6 chunks 10+11 "
            "as a historical monitoring case study; no model or policy changes are made."
        )
    )
    parser.add_argument("--current-chunks", nargs="+", type=int, default=list(DEFAULT_CURRENT_CHUNKS))
    parser.add_argument(
        "--current-predictions",
        type=Path,
        default=ARTIFACT_DIR / "m6_lockbox_predictions.parquet",
    )
    parser.add_argument("--historical-case-study", action="store_true", default=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds() -> MonitoringThresholds:
    payload = _load_json(ARTIFACT_DIR / "m10_alert_policy.json")
    return MonitoringThresholds(**payload["thresholds"])


def _load_feature_reference(runtime: ScoringRuntime) -> dict[str, NumericReferenceProfile]:
    payload = _load_json(ARTIFACT_DIR / "m10_reference_feature_profile.json")
    if payload.get("artifact_fingerprint") != runtime.artifact_fingerprint:
        raise RuntimeError("M10 feature reference fingerprint does not match loaded Model V1")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise RuntimeError("M10 reference feature profiles are invalid")
    if set(profiles) != set(runtime.feature_names):
        raise RuntimeError("M10 reference feature schema differs from frozen Model V1")
    return {name: NumericReferenceProfile.from_dict(profiles[name]) for name in runtime.feature_names}


def _current_feature_drift(
    chunks: list[int],
    runtime: ScoringRuntime,
    profiles: dict[str, NumericReferenceProfile],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accum = {}
    for feature, ref in profiles.items():
        accum[feature] = {
            "counts": np.zeros(len(ref.counts), dtype=np.int64),
            "missing": 0,
            "n": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "valid_n": 0,
        }
    schema_rows = []
    for chunk in chunks:
        path = FEATURE_DIR / f"train_data_{chunk}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Current feature cache missing: {path}. M6 should have created chunks 10-11."
            )
        raw = pd.read_parquet(path)
        aligned, report = align_to_frozen_feature_schema(raw, runtime.feature_names)
        raw_feature_names = sorted(set(raw.columns) - {"id"})
        schema_rows.append(
            {
                "population": f"chunk_{chunk}",
                "n": len(aligned),
                "raw_feature_count": raw.shape[1] - 1,
                "raw_schema_signature": schema_signature(raw_feature_names),
                "frozen_feature_count": len(runtime.feature_names),
                "missing_structural_zero_count": len(report.missing_structural_zero),
                "missing_structural_zero_features": "|".join(report.missing_structural_zero),
                "missing_nonstructural_count": len(report.missing_nonstructural),
                "missing_nonstructural_features": "|".join(report.missing_nonstructural),
                "dropped_extra_count": len(report.dropped_extra),
                "dropped_extra_features": "|".join(report.dropped_extra),
            }
        )
        for feature in runtime.feature_names:
            ref = profiles[feature]
            values = aligned[feature].to_numpy(dtype=np.float64, copy=False)
            counts, missing = histogram_counts(values, np.asarray(ref.edges, dtype=np.float64))
            state = accum[feature]
            state["counts"] += counts
            state["missing"] += missing
            state["n"] += len(values)
            valid = values[np.isfinite(values)]
            if valid.size:
                state["sum"] += float(valid.sum())
                state["sum_sq"] += float(np.square(valid).sum())
                state["valid_n"] += int(valid.size)
        print(
            f"Current chunk {chunk}: rows={len(aligned):,} | raw/frozen={raw.shape[1]-1}/"
            f"{len(runtime.feature_names)} | missing-nonstructural={len(report.missing_nonstructural)} | "
            f"dropped-extra={len(report.dropped_extra)}"
        )

    rows = []
    for feature in runtime.feature_names:
        ref = profiles[feature]
        state = accum[feature]
        vn = int(state["valid_n"])
        mean = state["sum"] / vn if vn else float("nan")
        variance = max(0.0, state["sum_sq"] / vn - mean**2) if vn else float("nan")
        std = float(np.sqrt(variance)) if vn else float("nan")
        ref_missing = ref.missing / ref.n if ref.n else float("nan")
        cur_missing = state["missing"] / state["n"] if state["n"] else float("nan")
        rows.append(
            {
                "feature": feature,
                "psi": psi_from_counts(
                    np.asarray(ref.counts),
                    ref.missing,
                    np.asarray(state["counts"]),
                    int(state["missing"]),
                ),
                "reference_mean": ref.mean,
                "current_mean": mean,
                "mean_delta": mean - ref.mean,
                "reference_std": ref.std,
                "current_std": std,
                "reference_missing_rate": ref_missing,
                "current_missing_rate": cur_missing,
                "missing_rate_delta": cur_missing - ref_missing,
                "reference_n": ref.n,
                "current_n": int(state["n"]),
            }
        )
    drift = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return drift, pd.DataFrame(schema_rows)


def _prediction_and_segment_drift(
    current: pd.DataFrame,
    reference: dict[str, object],
    thresholds: MonitoringThresholds,
) -> tuple[pd.DataFrame, pd.DataFrame, list[MonitorAlert]]:
    required = {"calibrated_pd", "risk_band", "decision"}
    missing = sorted(required - set(current.columns))
    if missing:
        raise RuntimeError(f"Current prediction file is missing columns: {missing}")
    pd_ref = NumericReferenceProfile.from_dict(reference["calibrated_pd"])
    pd_cmp = compare_numeric_profile(pd_ref, current["calibrated_pd"].to_numpy(dtype=np.float64))
    prediction_rows = [
        {
            "metric": "calibrated_pd_psi",
            "reference": 0.0,
            "current": float(pd_cmp["psi"]),
            "delta": float(pd_cmp["psi"]),
        },
        {
            "metric": "mean_pd",
            "reference": float(reference["mean_pd"]),
            "current": float(current["calibrated_pd"].mean()),
            "delta": float(current["calibrated_pd"].mean() - float(reference["mean_pd"])),
        },
    ]
    alerts = []
    psi = float(pd_cmp["psi"])
    psi_status = threshold_status(
        psi, thresholds.prediction_psi_warning, thresholds.prediction_psi_critical
    )
    if psi_status != "PASS":
        alerts.append(
            MonitorAlert(
                "prediction_drift",
                "calibrated_pd_psi",
                psi_status,
                psi,
                0.0,
                psi,
                thresholds.prediction_psi_warning,
                thresholds.prediction_psi_critical,
                "Calibrated-PD distribution PSI exceeded threshold.",
            )
        )
    mean_delta = float(prediction_rows[1]["delta"])
    mean_status = threshold_status(
        mean_delta,
        thresholds.mean_pd_abs_delta_warning,
        thresholds.mean_pd_abs_delta_critical,
    )
    if mean_status != "PASS":
        alerts.append(
            MonitorAlert(
                "prediction_drift",
                "mean_pd_delta",
                mean_status,
                float(prediction_rows[1]["current"]),
                float(prediction_rows[1]["reference"]),
                mean_delta,
                thresholds.mean_pd_abs_delta_warning,
                thresholds.mean_pd_abs_delta_critical,
                "Mean calibrated PD shifted beyond threshold.",
            )
        )

    band_cmp = compare_categorical_shares(reference["risk_band_shares"], current["risk_band"])
    band_cmp.insert(0, "segment_type", "risk_band")
    decision_cmp = compare_categorical_shares(
        reference["decision_shares"], current["decision"]
    )
    decision_cmp.insert(0, "segment_type", "decision")
    segment = pd.concat([band_cmp, decision_cmp], ignore_index=True)
    for row in segment.itertuples(index=False):
        status = threshold_status(
            float(row.share_delta),
            thresholds.segment_share_abs_delta_warning,
            thresholds.segment_share_abs_delta_critical,
        )
        if status != "PASS":
            alerts.append(
                MonitorAlert(
                    "segment_drift",
                    f"{row.segment_type}::{row.label}",
                    status,
                    float(row.current_share),
                    float(row.reference_share),
                    float(row.share_delta),
                    thresholds.segment_share_abs_delta_warning,
                    thresholds.segment_share_abs_delta_critical,
                    f"{row.segment_type} share shifted beyond threshold.",
                )
            )
    return pd.DataFrame(prediction_rows), segment, alerts


def _outcome_monitoring(
    current: pd.DataFrame,
    benchmark: dict[str, object],
    thresholds: MonitoringThresholds,
) -> tuple[pd.DataFrame, list[MonitorAlert]]:
    if "flag" not in current.columns:
        return pd.DataFrame(), []
    y = current["flag"].to_numpy(dtype=np.int8)
    pd_score = current["calibrated_pd"].to_numpy(dtype=np.float64)
    rank = ranking_metrics(y, pd_score)
    cal = calibration_metrics(y, pd_score, n_bins=10)
    current_metrics = {**rank, **cal}
    selected = benchmark["selected_metrics"]
    raw_reference = benchmark["raw_metrics"]
    raw_rank = None
    if "raw_score" in current.columns:
        raw_rank = ranking_metrics(y, current["raw_score"].to_numpy(dtype=np.float64))
    rows = []
    if raw_rank is not None:
        rows.append(
            {
                "metric": "raw_roc_auc",
                "reference": float(raw_reference["roc_auc"]),
                "current": float(raw_rank["roc_auc"]),
                "delta": float(raw_rank["roc_auc"]) - float(raw_reference["roc_auc"]),
            }
        )
    keys = [
        "roc_auc",
        "average_precision",
        "ks",
        "brier_score",
        "log_loss",
        "mean_predicted_pd",
        "observed_default_rate",
        "calibration_bias",
        "calibration_intercept",
        "calibration_slope",
    ]
    for key in keys:
        reference = float(selected[key]) if key in selected else float("nan")
        current_value = float(current_metrics[key])
        rows.append(
            {
                "metric": key,
                "reference": reference,
                "current": current_value,
                "delta": current_value - reference if np.isfinite(reference) else float("nan"),
            }
        )

    alerts = []
    if raw_rank is not None:
        ref_auc = float(raw_reference["roc_auc"])
        current_auc = float(raw_rank["roc_auc"])
        auc_metric = "raw_roc_auc_drop"
    else:
        ref_auc = float(selected["roc_auc"])
        current_auc = float(current_metrics["roc_auc"])
        auc_metric = "roc_auc_drop"
    auc_drop = ref_auc - current_auc
    status = drop_status(auc_drop, thresholds.auc_drop_warning, thresholds.auc_drop_critical)
    if status != "PASS":
        alerts.append(
            MonitorAlert(
                "delayed_outcome",
                auc_metric,
                status,
                current_auc,
                ref_auc,
                -auc_drop,
                thresholds.auc_drop_warning,
                thresholds.auc_drop_critical,
                "Base-model discrimination degraded versus the pre-deployment chunk-9 benchmark.",
            )
        )
    brier_increase = float(current_metrics["brier_score"]) - float(selected["brier_score"])
    status = increase_status(
        brier_increase, thresholds.brier_increase_warning, thresholds.brier_increase_critical
    )
    if status != "PASS":
        alerts.append(
            MonitorAlert(
                "delayed_outcome",
                "brier_increase",
                status,
                float(current_metrics["brier_score"]),
                float(selected["brier_score"]),
                brier_increase,
                thresholds.brier_increase_warning,
                thresholds.brier_increase_critical,
                "Brier score deteriorated versus the pre-deployment benchmark.",
            )
        )
    bias = float(current_metrics["calibration_bias"])
    status = threshold_status(
        bias,
        thresholds.calibration_bias_abs_warning,
        thresholds.calibration_bias_abs_critical,
    )
    if status != "PASS":
        alerts.append(
            MonitorAlert(
                "delayed_outcome",
                "calibration_bias",
                status,
                bias,
                0.0,
                bias,
                thresholds.calibration_bias_abs_warning,
                thresholds.calibration_bias_abs_critical,
                "Absolute portfolio calibration bias exceeded threshold.",
            )
        )
    slope_gap = float(current_metrics["calibration_slope"]) - 1.0
    status = threshold_status(
        slope_gap,
        thresholds.calibration_slope_abs_from_one_warning,
        thresholds.calibration_slope_abs_from_one_critical,
    )
    if status != "PASS":
        alerts.append(
            MonitorAlert(
                "delayed_outcome",
                "calibration_slope_gap",
                status,
                float(current_metrics["calibration_slope"]),
                1.0,
                slope_gap,
                thresholds.calibration_slope_abs_from_one_warning,
                thresholds.calibration_slope_abs_from_one_critical,
                "Calibration slope departed materially from the ideal value of 1.",
            )
        )
    return pd.DataFrame(rows), alerts


def _save_plots(
    feature_drift: pd.DataFrame,
    segments: pd.DataFrame,
    prediction_drift: pd.DataFrame,
) -> None:
    top = feature_drift.head(20).sort_values("psi")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["psi"])
    ax.set_xlabel("PSI")
    ax.set_title("M10 feature drift — top 20")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m10_feature_psi_top20.png", dpi=160)
    plt.close(fig)

    bands = segments[segments["segment_type"] == "risk_band"]
    x = np.arange(len(bands))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bands["reference_share"] * 100, width=width, label="Reference")
    ax.bar(x + width / 2, bands["current_share"] * 100, width=width, label="Current")
    ax.set_xticks(x, bands["label"])
    ax.set_ylabel("Population share, %")
    ax.set_title("M10 frozen risk-band population drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m10_risk_band_drift.png", dpi=160)
    plt.close(fig)

    decisions = segments[segments["segment_type"] == "decision"]
    x = np.arange(len(decisions))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - width / 2, decisions["reference_share"] * 100, width=width, label="Reference")
    ax.bar(x + width / 2, decisions["current_share"] * 100, width=width, label="Current")
    ax.set_xticks(x, decisions["label"])
    ax.set_ylabel("Applications, %")
    ax.set_title("M10 decision-rate drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m10_decision_drift.png", dpi=160)
    plt.close(fig)

    pd_row = prediction_drift[prediction_drift["metric"] == "mean_pd"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["Reference", "Current"], [pd_row["reference"] * 100, pd_row["current"] * 100])
    ax.set_ylabel("Mean calibrated PD, %")
    ax.set_title("M10 portfolio mean-PD monitoring")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m10_mean_pd_drift.png", dpi=160)
    plt.close(fig)


def _write_case_study(
    overall: str,
    feature_drift: pd.DataFrame,
    prediction_drift: pd.DataFrame,
    outcome: pd.DataFrame,
    alerts: pd.DataFrame,
) -> None:
    max_psi = float(feature_drift["psi"].max())
    pd_psi = float(
        prediction_drift.loc[prediction_drift["metric"] == "calibrated_pd_psi", "current"].iloc[0]
    )
    mean_pd_delta = float(
        prediction_drift.loc[prediction_drift["metric"] == "mean_pd", "delta"].iloc[0]
    )
    lines = [
        "# M10 Historical Monitoring Case Study — Model V1",
        "",
        "This report replays the already-opened M6 chunks 10–11 through monitoring rules ",
        "defined after M6. It is therefore retrospective evidence, not a prospective alerting claim.",
        "",
        f"**Overall status:** {overall}",
        "",
        f"- Maximum feature PSI: `{max_psi:.6f}`",
        f"- Calibrated-PD PSI: `{pd_psi:.6f}`",
        f"- Mean-PD shift: `{mean_pd_delta:+.4%}`",
    ]
    if not outcome.empty:
        auc_metric = "raw_roc_auc" if "raw_roc_auc" in set(outcome["metric"]) else "roc_auc"
        auc = outcome[outcome["metric"] == auc_metric].iloc[0]
        bias = outcome[outcome["metric"] == "calibration_bias"].iloc[0]
        slope = outcome[outcome["metric"] == "calibration_slope"].iloc[0]
        lines += [
            f"- Base-model AUC: `{auc['reference']:.6f}` reference → `{auc['current']:.6f}` current",
            f"- Current calibration bias: `{bias['current']:+.4%}`",
            f"- Current calibration slope: `{slope['current']:.4f}`",
        ]
    lines += ["", "## Alerts", ""]
    if alerts.empty:
        lines.append("No WARNING/CRITICAL alerts were triggered.")
    else:
        for row in alerts.itertuples(index=False):
            lines.append(f"- **{row.status}** `{row.category}/{row.metric}` — {row.message}")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- PSI and share shifts are distribution diagnostics, not proof of concept drift.",
        "- Delayed-outcome metrics require observed labels and should be evaluated only after outcomes mature.",
        "- M6 chunks 10–11 are historical evaluation data and must not be treated as a new untouched test set.",
        "- Alert thresholds are configurable project heuristics, not legal or regulatory limits.",
    ]
    (ARTIFACT_DIR / "m10_historical_case_study.md").write_text("\n".join(lines), encoding="utf-8")


def _append_experiment(summary: dict[str, object]) -> None:
    path = ARTIFACT_DIR / "experiments.csv"
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "M10",
                "experiment": "historical_monitoring_case_study",
                "model": "credit-risk-v1",
                "n_features": 212,
                "monitoring_status": summary["overall_status"],
                "max_feature_psi": summary["max_feature_psi"],
                "pd_psi": summary["pd_psi"],
                "historical_case_study": True,
            }
        ]
    )
    if path.exists():
        existing = pd.read_csv(path)
        pd.concat([existing, row], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        row.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    chunks = sorted(set(args.current_chunks))
    if chunks != list(DEFAULT_CURRENT_CHUNKS):
        raise ValueError(
            "This M10 historical case-study script expects already-opened M6 chunks 10+11. "
            "Use the reusable credit_risk.monitoring module for future production windows."
        )
    runtime = ScoringRuntime.load(ArtifactPaths(ARTIFACT_DIR))
    thresholds = _thresholds()
    reference_features = _load_feature_reference(runtime)
    prediction_reference = _load_json(ARTIFACT_DIR / "m10_reference_prediction_profile.json")
    schema_reference = _load_json(ARTIFACT_DIR / "m10_reference_schema_profile.json")
    if schema_reference.get("artifact_fingerprint") != runtime.artifact_fingerprint:
        raise RuntimeError("M10 schema reference fingerprint mismatch")
    observed_schema_signatures = {
        str(item["raw_schema_signature"])
        for item in schema_reference.get("observed_raw_schemas", [])
    }
    if not observed_schema_signatures:
        raise RuntimeError("M10 reference schema profile contains no observed raw schemas")
    benchmark = _load_json(ARTIFACT_DIR / "m10_reference_performance_benchmark.json")
    reference_meta = _load_json(ARTIFACT_DIR / "m10_reference_metadata.json")
    if reference_meta.get("artifact_fingerprint") != runtime.artifact_fingerprint:
        raise RuntimeError("M10 reference metadata fingerprint mismatch")

    current = pd.read_parquet(args.current_predictions)
    current = current[current["source_chunk"].astype(int).isin(chunks)].copy()
    if set(current["source_chunk"].astype(int).unique()) != set(chunks):
        raise RuntimeError("Current predictions do not contain exactly chunks 10+11")
    if current["id"].duplicated().any():
        raise RuntimeError("Duplicate application ids in current monitoring population")

    print("=== M10 PRODUCTION MONITORING — HISTORICAL CASE STUDY ===")
    print("Model V1 changes: NONE")
    print(f"artifact_fingerprint: {runtime.artifact_fingerprint}")
    print("reference: M5 chunks 8+9")
    print("current historical window: M6 chunks 10+11")
    print("Important: M10 alert thresholds were introduced after M6; this replay is retrospective.")

    feature_drift, schema = _current_feature_drift(chunks, runtime, reference_features)
    schema["reference_schema_match"] = schema["raw_schema_signature"].isin(
        observed_schema_signatures
    )
    prediction_drift, segments, prediction_alerts = _prediction_and_segment_drift(
        current, prediction_reference, thresholds
    )
    outcome, outcome_alerts = _outcome_monitoring(current, benchmark, thresholds)
    all_alerts = (
        schema_alerts(
            schema,
            thresholds,
            reference_raw_schema_signatures=observed_schema_signatures,
        )
        + feature_alerts(feature_drift, thresholds)
        + prediction_alerts
        + outcome_alerts
    )
    alert_frame = alerts_to_frame(all_alerts)
    overall = worst_status(alert_frame["status"].tolist() if not alert_frame.empty else [])

    feature_drift.to_csv(ARTIFACT_DIR / "m10_feature_drift.csv", index=False)
    schema.to_csv(ARTIFACT_DIR / "m10_schema_drift.csv", index=False)
    prediction_drift.to_csv(ARTIFACT_DIR / "m10_prediction_drift.csv", index=False)
    segments.to_csv(ARTIFACT_DIR / "m10_segment_drift.csv", index=False)
    outcome.to_csv(ARTIFACT_DIR / "m10_delayed_outcome_metrics.csv", index=False)
    alert_frame.to_csv(ARTIFACT_DIR / "m10_alerts.csv", index=False)
    _save_plots(feature_drift, segments, prediction_drift)

    max_feature_psi = float(feature_drift["psi"].max())
    pd_psi = float(
        prediction_drift.loc[prediction_drift["metric"] == "calibrated_pd_psi", "current"].iloc[0]
    )
    summary = {
        "stage": "M10",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "credit-risk-v1",
        "artifact_fingerprint": runtime.artifact_fingerprint,
        "monitoring_mode": "historical_case_study",
        "historical_current_population": "M6 chunks 10+11",
        "reference_population": "M5 chunks 8+9",
        "thresholds_defined_after_m6": True,
        "prospective_claim_for_m6": False,
        "n_current": int(len(current)),
        "overall_status": overall,
        "alert_count": int(len(alert_frame)),
        "warning_count": int((alert_frame["status"] == "WARNING").sum()) if not alert_frame.empty else 0,
        "critical_count": int((alert_frame["status"] == "CRITICAL").sum()) if not alert_frame.empty else 0,
        "max_feature_psi": max_feature_psi,
        "median_feature_psi": float(feature_drift["psi"].median()),
        "p90_feature_psi": float(feature_drift["psi"].quantile(0.90)),
        "pd_psi": pd_psi,
        "schema_baseline_mismatch_count": int((~schema["reference_schema_match"]).sum()),
        "model_changes": False,
        "policy_changes": False,
    }
    if not outcome.empty:
        for metric in ["raw_roc_auc", "roc_auc", "brier_score", "calibration_bias", "calibration_slope"]:
            if metric not in set(outcome["metric"]):
                continue
            row = outcome[outcome["metric"] == metric].iloc[0]
            summary[f"current_{metric}"] = float(row["current"])
            summary[f"reference_{metric}"] = float(row["reference"])
    (ARTIFACT_DIR / "m10_monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_case_study(overall, feature_drift, prediction_drift, outcome, alert_frame)
    _append_experiment(summary)

    print("\n=== M10 SCHEMA / DATA QUALITY ===")
    for row in schema.itertuples(index=False):
        print(
            f"{row.population}: raw/frozen={row.raw_feature_count}/{row.frozen_feature_count} | "
            f"missing_nonstructural={row.missing_nonstructural_count} | "
            f"dropped_extra={row.dropped_extra_count} | "
            f"reference_schema_match={row.reference_schema_match}"
        )
    print("\n=== M10 FEATURE DRIFT ===")
    print(
        f"median PSI={feature_drift['psi'].median():.6f} | "
        f"P90={feature_drift['psi'].quantile(0.90):.6f} | max={max_feature_psi:.6f}"
    )
    for row in feature_drift.head(10).itertuples(index=False):
        print(
            f"{row.feature:52s} PSI={row.psi:.6f} | "
            f"missingDelta={row.missing_rate_delta:+.3%}"
        )
    print("\n=== M10 PREDICTION / POLICY DRIFT ===")
    for row in prediction_drift.itertuples(index=False):
        if row.metric == "mean_pd":
            print(
                f"mean PD: {row.reference:.4%} -> {row.current:.4%} | delta={row.delta:+.4%}"
            )
        else:
            print(f"PD PSI: {row.current:.6f}")
    for row in segments.itertuples(index=False):
        print(
            f"{row.segment_type:9s} {row.label:8s}: {row.reference_share:.2%} -> "
            f"{row.current_share:.2%} | delta={row.share_delta:+.2%}"
        )
    if not outcome.empty:
        print("\n=== M10 DELAYED-OUTCOME MONITORING ===")
        for metric in ["raw_roc_auc", "roc_auc", "brier_score", "calibration_bias", "calibration_slope"]:
            if metric not in set(outcome["metric"]):
                continue
            row = outcome[outcome["metric"] == metric].iloc[0]
            print(
                f"{metric:20s} reference={row.reference:.6f} | current={row.current:.6f} | "
                f"delta={row.delta:+.6f}"
            )
    print("\n=== M10 ALERT SUMMARY ===")
    print(f"overall_status: {overall}")
    print(
        f"alerts: total={len(alert_frame)} | "
        f"warning={(alert_frame['status'] == 'WARNING').sum() if not alert_frame.empty else 0} | "
        f"critical={(alert_frame['status'] == 'CRITICAL').sum() if not alert_frame.empty else 0}"
    )
    if alert_frame.empty:
        print("No WARNING/CRITICAL alerts.")
    else:
        for row in alert_frame.itertuples(index=False):
            print(f"{row.status:8s} | {row.category:18s} | {row.metric} | {row.message}")
    print("\nSaved:")
    for path in [
        "artifacts/m10_feature_drift.csv",
        "artifacts/m10_schema_drift.csv",
        "artifacts/m10_prediction_drift.csv",
        "artifacts/m10_segment_drift.csv",
        "artifacts/m10_delayed_outcome_metrics.csv",
        "artifacts/m10_alerts.csv",
        "artifacts/m10_feature_psi_top20.png",
        "artifacts/m10_risk_band_drift.png",
        "artifacts/m10_decision_drift.png",
        "artifacts/m10_mean_pd_drift.png",
        "artifacts/m10_monitoring_summary.json",
        "artifacts/m10_historical_case_study.md",
        "artifacts/experiments.csv",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
