from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.decisioning import (
    DEFAULT_SCENARIOS,
    RISK_BAND_LABELS,
    assign_decisions,
    assign_risk_bands,
    derive_quantile_band_cutpoints,
    evaluate_policy,
    pd_to_score,
    risk_band_summary,
)

ARTIFACT_DIR = Path("artifacts")
M4_METADATA = ARTIFACT_DIR / "m4_metadata.json"
M4_PREDICTIONS = ARTIFACT_DIR / "m4_calibration_predictions.parquet"
M4_CALIBRATOR = ARTIFACT_DIR / "m4_calibrator.joblib"
LOCKBOX_CHUNKS = (10, 11)
POLICY_DEVELOPMENT_CHUNKS = (8, 9)
PRIMARY_POLICY = "balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M5 Credit Decision Engine. Builds frozen PD risk bands and three "
            "predeclared business-policy scenarios using only M4 chunks 8-9. "
            "Lockbox chunks 10-11 are never accessed."
        )
    )
    parser.add_argument("--risk-bands", type=int, default=5)
    parser.add_argument("--score-base", type=float, default=600.0)
    parser.add_argument("--score-base-odds", type=float, default=20.0)
    parser.add_argument("--score-pdo", type=float, default=50.0)
    return parser.parse_args()


def validate_m4_governance() -> dict[str, object]:
    required = [M4_METADATA, M4_PREDICTIONS, M4_CALIBRATOR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "M5 requires completed M4 artifacts. Missing: " + ", ".join(missing)
        )
    meta = json.loads(M4_METADATA.read_text(encoding="utf-8"))
    if meta.get("final_lockbox_evaluated") is not False:
        raise RuntimeError("Expected M4 metadata to mark the final lockbox as untouched.")
    if tuple(meta.get("lockbox_chunks_reserved", [])) != LOCKBOX_CHUNKS:
        raise RuntimeError("M4 lockbox governance does not match M5 expectations.")
    if set(meta.get("final_calibrator_refit_population", [])) != set(POLICY_DEVELOPMENT_CHUNKS):
        raise RuntimeError("Final M4 calibrator must be fitted on chunks 8+9 before M5.")
    return meta


def load_policy_development_frame(calibrator) -> pd.DataFrame:
    frame = pd.read_parquet(M4_PREDICTIONS)
    required_cols = {"id", "flag", "source_chunk", "raw_score"}
    missing = sorted(required_cols - set(frame.columns))
    if missing:
        raise RuntimeError(f"M4 prediction artifact is missing columns: {missing}")
    chunks = set(frame["source_chunk"].astype(int).unique().tolist())
    if chunks != set(POLICY_DEVELOPMENT_CHUNKS):
        raise RuntimeError(
            f"M5 policy development must use exactly chunks {POLICY_DEVELOPMENT_CHUNKS}; got {sorted(chunks)}."
        )
    if frame["id"].duplicated().any():
        raise RuntimeError("Duplicate application ids found in M4 policy-development predictions.")
    if set(frame["source_chunk"].astype(int).tolist()) & set(LOCKBOX_CHUNKS):
        raise RuntimeError("Lockbox rows unexpectedly appeared in M4 predictions.")

    raw = frame["raw_score"].to_numpy(dtype=np.float64)
    calibrated = np.asarray(calibrator.predict(raw), dtype=np.float64)
    if not np.all(np.isfinite(calibrated)) or np.any((calibrated < 0) | (calibrated > 1)):
        raise RuntimeError("Final M4 calibrator returned invalid probabilities.")
    result = frame[["id", "flag", "source_chunk", "raw_score"]].copy()
    result["calibrated_pd"] = calibrated.astype(np.float32)
    return result


def build_risk_band_outputs(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    pd_values = frame["calibrated_pd"].to_numpy(dtype=np.float64)
    cutpoints = derive_quantile_band_cutpoints(pd_values)
    frame["risk_band"] = assign_risk_bands(pd_values, cutpoints)

    summaries: list[pd.DataFrame] = []
    populations = [
        ("policy_dev_8_9", frame),
        ("chunk_8", frame[frame["source_chunk"] == 8]),
        ("chunk_9", frame[frame["source_chunk"] == 9]),
    ]
    for name, pop in populations:
        summary = risk_band_summary(
            pop["flag"].to_numpy(dtype=np.int8),
            pop["calibrated_pd"].to_numpy(dtype=np.float64),
            pop["risk_band"].to_numpy(dtype=object),
        )
        summary.insert(0, "population", name)
        summaries.append(summary)
    return cutpoints, pd.concat(summaries, ignore_index=True)


def build_policy_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    populations = [
        ("policy_dev_8_9", frame),
        ("chunk_8", frame[frame["source_chunk"] == 8]),
        ("chunk_9", frame[frame["source_chunk"] == 9]),
    ]
    for scenario in DEFAULT_SCENARIOS:
        decisions = assign_decisions(frame["calibrated_pd"].to_numpy(dtype=np.float64), scenario)
        frame[f"decision_{scenario.name}"] = decisions
        for name, pop in populations:
            pop_decisions = frame.loc[pop.index, f"decision_{scenario.name}"].to_numpy(dtype=object)
            metrics = evaluate_policy(
                pop["flag"].to_numpy(dtype=np.int8),
                pop["calibrated_pd"].to_numpy(dtype=np.float64),
                scenario,
                decisions=pop_decisions,
            )
            rows.append({"population": name, **metrics})
    return pd.DataFrame(rows)


def save_band_plot(summary: pd.DataFrame) -> None:
    data = summary[summary["population"] == "policy_dev_8_9"].copy()
    data["risk_band"] = pd.Categorical(
        data["risk_band"], categories=list(RISK_BAND_LABELS), ordered=True
    )
    data = data.sort_values("risk_band")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(data))
    ax.plot(x, data["mean_pd"] * 100, marker="o", label="Mean calibrated PD")
    ax.plot(x, data["observed_default_rate"] * 100, marker="o", label="Observed default rate")
    ax.set_xticks(x, data["risk_band"].astype(str))
    ax.set_xlabel("Frozen risk band (A = lowest risk)")
    ax.set_ylabel("Rate, %")
    ax.set_title("M5 risk bands — policy development population (chunks 8+9)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m5_risk_bands.png", dpi=160)
    plt.close(fig)


def save_policy_plot(summary: pd.DataFrame) -> None:
    data = summary[summary["population"] == "policy_dev_8_9"].copy()
    order = [s.name for s in DEFAULT_SCENARIOS]
    data["scenario"] = pd.Categorical(data["scenario"], categories=order, ordered=True)
    data = data.sort_values("scenario")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for column, label in [
        ("approval_rate", "Approve"),
        ("review_rate", "Review"),
        ("rejection_rate", "Reject"),
    ]:
        values = data[column].to_numpy(dtype=float) * 100
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set_xticks(x, data["scenario"].astype(str))
    ax.set_ylabel("Applications, %")
    ax.set_title("M5 predeclared decision-policy scenarios")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m5_policy_scenarios.png", dpi=160)
    plt.close(fig)


def append_experiment(policy_summary: pd.DataFrame) -> None:
    registry_path = ARTIFACT_DIR / "experiments.csv"
    primary = policy_summary[
        (policy_summary["population"] == "policy_dev_8_9")
        & (policy_summary["scenario"] == PRIMARY_POLICY)
    ].iloc[0]
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "M5",
                "experiment": "credit_decision_engine_policy_freeze",
                "feature_version": "e_typed_catdist_payment",
                "model": "m4_catboost_plus_isotonic_pd",
                "n_applications": int(primary["n"]),
                "n_features": 212,
                "calibration_method": "isotonic",
                "policy": PRIMARY_POLICY,
                "approval_rate": float(primary["approval_rate"]),
                "review_rate": float(primary["review_rate"]),
                "rejection_rate": float(primary["rejection_rate"]),
                "approved_mean_pd": float(primary["approved_mean_pd"]),
                "expected_profit_rate_per_applicant": float(
                    primary["expected_profit_rate_per_applicant"]
                ),
            }
        ]
    )
    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        pd.concat([existing, row], ignore_index=True, sort=False).to_csv(registry_path, index=False)
    else:
        row.to_csv(registry_path, index=False)


def print_risk_bands(summary: pd.DataFrame, cutpoints: np.ndarray) -> None:
    print("\n=== M5 FROZEN RISK BANDS ===")
    print("Band cutpoints (calibrated PD): " + ", ".join(f"{x:.4%}" for x in cutpoints))
    data = summary[summary["population"] == "policy_dev_8_9"]
    for row in data.itertuples(index=False):
        print(
            f"{row.risk_band}: share={row.population_share:.2%} | meanPD={row.mean_pd:.4%} | "
            f"observedDR={row.observed_default_rate:.4%} | badCapture={row.bad_capture_rate:.2%} | "
            f"riskMultiple={row.risk_multiple_vs_portfolio:.2f}x"
        )


def print_policy_summary(summary: pd.DataFrame) -> None:
    print("\n=== M5 BUSINESS SCENARIOS (DEVELOPMENT ONLY) ===")
    data = summary[summary["population"] == "policy_dev_8_9"]
    for row in data.itertuples(index=False):
        marker = " [PRIMARY]" if bool(row.primary) else ""
        print(f"\n{row.scenario}{marker}")
        print(
            f"  PD thresholds: approve <= {row.approve_pd_threshold:.2%}; "
            f"review <= {row.reject_pd_threshold:.2%}; reject above"
        )
        print(
            f"  decisions: approve={row.approval_rate:.2%} | review={row.review_rate:.2%} | "
            f"reject={row.rejection_rate:.2%}"
        )
        print(
            f"  approved portfolio: meanPD={row.approved_mean_pd:.4%} | "
            f"observedDR={row.approved_observed_default_rate:.4%}"
        )
        print(
            f"  expected: loss/approved={row.expected_loss_rate_per_approved:.4%} of EAD | "
            f"profit/approved={row.expected_profit_rate_per_approved:.4%} of EAD | "
            f"profit/applicant={row.expected_profit_rate_per_applicant:.4%} of EAD"
        )
        print(
            f"  default capture: reject={row.default_capture_reject:.2%} | "
            f"review+reject={row.default_capture_review_or_reject:.2%}"
        )
        print(
            f"  retrospective realized profit/applicant={row.realized_profit_rate_per_applicant_dev:.4%} "
            "of EAD (development diagnostic; not used to select policy)"
        )


def main() -> None:
    args = parse_args()
    if args.risk_bands != 5:
        raise ValueError("M5 governance currently freezes exactly five A-E risk bands.")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    m4_meta = validate_m4_governance()
    calibrator = joblib.load(M4_CALIBRATOR)
    frame = load_policy_development_frame(calibrator)

    print("=== M5 CREDIT DECISION ENGINE ===")
    print(f"PD model: frozen M4 CatBoost + {m4_meta['selected_calibration_method']} calibration")
    print(f"Policy-development chunks: {list(POLICY_DEVELOPMENT_CHUNKS)}")
    print(f"Applications: {len(frame):,}")
    print(f"Observed default rate: {frame['flag'].mean():.4%}")
    print(f"Reserved lockbox chunks (NOT ACCESSED): {list(LOCKBOX_CHUNKS)}")
    print("Business economics: normalized to EAD=1; illustrative, configurable assumptions")
    print(f"Primary policy predeclared before lockbox: {PRIMARY_POLICY}")

    frame["credit_score"] = pd_to_score(
        frame["calibrated_pd"].to_numpy(dtype=np.float64),
        base_score=args.score_base,
        base_good_to_bad_odds=args.score_base_odds,
        points_to_double_odds=args.score_pdo,
    ).astype(np.float32)

    cutpoints, band_summary = build_risk_band_outputs(frame)
    policy_summary = build_policy_outputs(frame)

    scenario_records = [scenario.to_record() for scenario in DEFAULT_SCENARIOS]
    band_thresholds = {
        "method": "development_equal-frequency_PD_cutpoints_with_tie_safe_distinct_boundaries",
        "development_chunks": list(POLICY_DEVELOPMENT_CHUNKS),
        "labels": list(RISK_BAND_LABELS),
        "cutpoints": [float(x) for x in cutpoints],
        "interpretation": "A is lowest calibrated PD; E is highest calibrated PD.",
    }
    policy_freeze = {
        "stage": "M5",
        "primary_policy": PRIMARY_POLICY,
        "primary_policy_selection_basis": (
            "predeclared medium-risk-appetite scenario; not optimized against observed defaults"
        ),
        "economic_unit": "normalized EAD=1",
        "economics_are_dataset_inferred": False,
        "review_outcome_modelled": False,
        "scenarios": scenario_records,
        "lockbox_chunks_reserved": list(LOCKBOX_CHUNKS),
        "lockbox_evaluated": False,
    }

    band_summary.to_csv(ARTIFACT_DIR / "m5_risk_band_summary.csv", index=False)
    policy_summary.to_csv(ARTIFACT_DIR / "m5_policy_summary.csv", index=False)
    frame.to_parquet(ARTIFACT_DIR / "m5_decisions.parquet", index=False)
    (ARTIFACT_DIR / "m5_risk_band_thresholds.json").write_text(
        json.dumps(band_thresholds, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "m5_policy_freeze.json").write_text(
        json.dumps(policy_freeze, indent=2), encoding="utf-8"
    )
    assumptions = pd.DataFrame(scenario_records)
    assumptions.to_csv(ARTIFACT_DIR / "m5_business_assumptions.csv", index=False)

    metadata = {
        "stage": "M5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pd_source": "m4_final_calibrator_refit_on_chunks_8_9",
        "selected_calibration_method": m4_meta["selected_calibration_method"],
        "policy_development_chunks": list(POLICY_DEVELOPMENT_CHUNKS),
        "n_policy_development": int(len(frame)),
        "policy_development_default_rate": float(frame["flag"].mean()),
        "risk_band_count": len(RISK_BAND_LABELS),
        "risk_band_cutpoints": [float(x) for x in cutpoints],
        "primary_policy": PRIMARY_POLICY,
        "policy_selection_used_observed_outcomes": False,
        "lockbox_chunks_reserved": list(LOCKBOX_CHUNKS),
        "lockbox_evaluated": False,
        "credit_score_mapping": {
            "base_score": args.score_base,
            "base_good_to_bad_odds": args.score_base_odds,
            "points_to_double_odds": args.score_pdo,
            "clip": [300.0, 850.0],
            "decision_use": False,
        },
    }
    (ARTIFACT_DIR / "m5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_band_plot(band_summary)
    save_policy_plot(policy_summary)
    append_experiment(policy_summary)

    print_risk_bands(band_summary, cutpoints)
    print_policy_summary(policy_summary)

    primary = policy_summary[
        (policy_summary["population"] == "policy_dev_8_9")
        & (policy_summary["scenario"] == PRIMARY_POLICY)
    ].iloc[0]
    print("\n=== M5 PRIMARY POLICY FREEZE ===")
    print(f"policy: {PRIMARY_POLICY}")
    print(f"approve PD threshold: {primary['approve_pd_threshold']:.4%}")
    print(f"reject PD threshold:  {primary['reject_pd_threshold']:.4%}")
    print(f"development approval rate: {primary['approval_rate']:.2%}")
    print(f"development review rate:   {primary['review_rate']:.2%}")
    print(f"development rejection rate:{primary['rejection_rate']:.2%}")
    print("Policy and A-E band thresholds are now frozen before lockbox evaluation.")
    print("Chunks 10-11 remain untouched.")

    print("\nSaved:")
    for path in [
        "artifacts/m5_risk_band_summary.csv",
        "artifacts/m5_risk_band_thresholds.json",
        "artifacts/m5_business_assumptions.csv",
        "artifacts/m5_policy_summary.csv",
        "artifacts/m5_policy_freeze.json",
        "artifacts/m5_decisions.parquet",
        "artifacts/m5_risk_bands.png",
        "artifacts/m5_policy_scenarios.png",
        "artifacts/m5_metadata.json",
        "artifacts/experiments.csv",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
