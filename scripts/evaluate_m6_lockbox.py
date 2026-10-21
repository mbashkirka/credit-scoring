from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from credit_risk.calibration import calibration_metrics, quantile_reliability_table
from credit_risk.data import find_train_history_files, load_target
from credit_risk.decisioning import (
    assign_decisions,
    assign_risk_bands,
    evaluate_policy,
    pd_to_score,
    risk_band_summary,
)
from credit_risk.evaluation import align_to_frozen_feature_schema, load_frozen_feature_names
from credit_risk.features import aggregate_canonical_m21_features
from credit_risk.lockbox import (
    EXPECTED_LOCKBOX_CHUNKS,
    frozen_primary_scenario,
    frozen_risk_band_cutpoints,
    validate_lockbox_governance,
)
from credit_risk.metrics import ranking_metrics
from credit_risk.statistics import stratified_bootstrap_probability_metrics

DATA_DIR = Path("data/raw")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
ARTIFACT_DIR = Path("artifacts")

M4_MODEL = ARTIFACT_DIR / "m4_catboost.cbm"
M4_IMPUTER = ARTIFACT_DIR / "m4_catboost_imputer.joblib"
M4_CALIBRATOR = ARTIFACT_DIR / "m4_calibrator.joblib"
M4_METADATA = ARTIFACT_DIR / "m4_metadata.json"
M4_PREDICTIONS = ARTIFACT_DIR / "m4_calibration_predictions.parquet"
FEATURE_MANIFEST = ARTIFACT_DIR / "m3_features.csv"
M5_METADATA = ARTIFACT_DIR / "m5_metadata.json"
M5_POLICY = ARTIFACT_DIR / "m5_policy_freeze.json"
M5_BANDS = ARTIFACT_DIR / "m5_risk_band_thresholds.json"
M6_OPEN_MARKER = ARTIFACT_DIR / "m6_lockbox_opened.json"

FROZEN_INPUTS = (
    M4_MODEL,
    M4_IMPUTER,
    M4_CALIBRATOR,
    M4_METADATA,
    FEATURE_MANIFEST,
    M5_METADATA,
    M5_POLICY,
    M5_BANDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M6 final lockbox evaluation. Opens chunks 10-11 using the frozen M4/M5 "
            "pipeline only. No fitting, tuning, recalibration or threshold changes are allowed."
        )
    )
    parser.add_argument("--lockbox-chunks", nargs="+", type=int, default=list(EXPECTED_LOCKBOX_CHUNKS))
    parser.add_argument("--prediction-batch-size", type=int, default=50_000)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_and_mark_lockbox_open(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    missing = [str(path) for path in FROZEN_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("M6 requires completed frozen M4/M5 artifacts: " + ", ".join(missing))
    chunks = tuple(args.lockbox_chunks)
    m4_meta = load_json(M4_METADATA)
    m5_meta = load_json(M5_METADATA)
    policy = load_json(M5_POLICY)
    bands = load_json(M5_BANDS)
    validate_lockbox_governance(m4_meta, m5_meta, policy, bands, chunks)
    if args.prediction_batch_size < 1:
        raise ValueError("--prediction-batch-size must be >=1.")
    if args.bootstrap_reps < 20:
        raise ValueError("--bootstrap-reps must be >=20.")
    if args.reliability_bins < 2:
        raise ValueError("--reliability-bins must be >=2.")

    fingerprints = {str(path): _sha256(path) for path in FROZEN_INPUTS}
    if M6_OPEN_MARKER.exists():
        marker = load_json(M6_OPEN_MARKER)
        if tuple(marker.get("lockbox_chunks", [])) != chunks:
            raise RuntimeError("Existing M6 marker was created for a different lockbox population.")
        if marker.get("frozen_input_sha256") != fingerprints:
            raise RuntimeError(
                "Frozen M4/M5 artifacts changed after lockbox was opened. "
                "M6 refuses to evaluate a modified pipeline on the historical lockbox."
            )
        print(
            "WARNING: M6 lockbox was already opened previously. Re-running the exact frozen "
            "pipeline is allowed for reproducibility, but chunks 10-11 are no longer untouched."
        )
    else:
        marker = {
            "stage": "M6",
            "first_opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "lockbox_chunks": list(chunks),
            "frozen_input_sha256": fingerprints,
            "fitting_allowed": False,
            "note": "Written before reading lockbox feature contents; freezes the evaluated pipeline.",
        }
        M6_OPEN_MARKER.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return m4_meta, m5_meta, policy, bands


def raw_chunk_paths(chunks: tuple[int, ...]) -> dict[int, Path]:
    available = {_chunk_index(path): path for path in find_train_history_files(DATA_DIR)}
    missing = [chunk for chunk in chunks if chunk not in available]
    if missing:
        raise FileNotFoundError(f"Raw lockbox chunks not found: {missing}")
    return {chunk: available[chunk] for chunk in chunks}


def ensure_feature_caches(chunks: tuple[int, ...], rebuild: bool) -> dict[int, Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = raw_chunk_paths(chunks)
    result: dict[int, Path] = {}
    for chunk in chunks:
        cache = FEATURE_DIR / f"train_data_{chunk}.parquet"
        result[chunk] = cache
        if cache.exists() and not rebuild:
            print(f"Using cached canonical lockbox features: {cache}")
            continue
        print(f"Reading {raw_paths[chunk]}")
        history = pd.read_parquet(raw_paths[chunk])
        print(f"Building frozen-compatible canonical M2.1-E features for lockbox chunk {chunk}...")
        features = aggregate_canonical_m21_features(history)
        features.to_parquet(cache, index=False)
        print(
            f"Cached lockbox chunk {chunk}: {len(features):,} applications x "
            f"{features.shape[1]-1:,} raw canonical features"
        )
        del history, features
        gc.collect()
    return result


def historical_development_ids() -> np.ndarray:
    """Read only id columns from already-used populations 0-9 for overlap protection."""
    arrays: list[np.ndarray] = []
    for chunk in range(10):
        path = FEATURE_DIR / f"train_data_{chunk}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Historical feature cache missing for chunk {chunk}; M6 cannot verify disjointness."
            )
        arrays.append(pd.read_parquet(path, columns=["id"])["id"].to_numpy(dtype=np.int64))
    ids = np.concatenate(arrays)
    if np.unique(ids).size != len(ids):
        raise RuntimeError("Historical development populations 0-9 contain overlapping ids.")
    return ids


def score_lockbox_chunk(
    cache_path: Path,
    chunk: int,
    feature_names: list[str],
    target_indexed: pd.DataFrame,
    model: CatBoostClassifier,
    imputer,
    calibrator,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_parquet(cache_path)
    aligned, report = align_to_frozen_feature_schema(raw, feature_names)
    ids = aligned["id"].to_numpy(dtype=np.int64)
    if pd.Index(ids).has_duplicates:
        raise RuntimeError(f"Duplicate ids inside lockbox chunk {chunk}.")
    try:
        y = target_indexed.loc[ids, "flag"].to_numpy(dtype=np.int8)
    except KeyError as exc:
        raise RuntimeError(f"Target rows missing for lockbox chunk {chunk}.") from exc
    raw_score = np.empty(len(aligned), dtype=np.float32)
    for start in range(0, len(aligned), batch_size):
        stop = min(len(aligned), start + batch_size)
        matrix = imputer.transform(aligned.iloc[start:stop][feature_names])
        raw_score[start:stop] = model.predict_proba(matrix)[:, 1].astype(np.float32)
        del matrix
    calibrated = np.asarray(calibrator.predict(raw_score.astype(np.float64)), dtype=np.float64)
    if not np.all(np.isfinite(calibrated)) or np.any((calibrated < 0) | (calibrated > 1)):
        raise RuntimeError("Frozen calibrator returned invalid lockbox probabilities.")
    frame = pd.DataFrame(
        {
            "id": ids,
            "flag": y,
            "source_chunk": chunk,
            "raw_score": raw_score,
            "calibrated_pd": calibrated.astype(np.float32),
        }
    )
    schema = {
        "source_chunk": chunk,
        "n_applications": int(len(frame)),
        "default_rate": float(y.mean()),
        "raw_feature_count": int(raw.shape[1] - 1),
        "frozen_feature_count": len(feature_names),
        "missing_structural_zero_count": len(report.missing_structural_zero),
        "missing_nonstructural_count": len(report.missing_nonstructural),
        "dropped_extra_count": len(report.dropped_extra),
        "dropped_extra_features": "|".join(report.dropped_extra),
    }
    del raw, aligned
    gc.collect()
    return frame, schema


def evaluate_population(name: str, frame: pd.DataFrame, reliability_bins: int) -> tuple[dict, dict, pd.DataFrame]:
    y = frame["flag"].to_numpy(dtype=np.int8)
    pd_score = frame["calibrated_pd"].to_numpy(dtype=np.float64)
    raw_score = frame["raw_score"].to_numpy(dtype=np.float64)
    ranking = ranking_metrics(y, pd_score)
    calibrated = calibration_metrics(y, pd_score, n_bins=reliability_bins)
    raw_calibration = calibration_metrics(y, raw_score, n_bins=reliability_bins)
    reliability = quantile_reliability_table(y, pd_score, n_bins=reliability_bins)
    reliability.insert(0, "population", name)
    return (
        {"population": name, **ranking},
        {"population": name, "method": "calibrated_isotonic", **calibrated},
        reliability,
    ), {"population": name, "method": "raw_catboost", **raw_calibration}, reliability


def plot_reliability(reliability: pd.DataFrame) -> None:
    overall = reliability[reliability["population"] == "lockbox_10_11"]
    max_value = max(
        0.05,
        float(overall["predicted_pd_mean"].max()),
        float(overall["observed_default_rate"].max()),
    ) * 1.15
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(
        overall["predicted_pd_mean"],
        overall["observed_default_rate"],
        marker="o",
        label="frozen M4 isotonic PD",
    )
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("M6 final lockbox calibration — chunks 10+11")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m6_lockbox_calibration.png", dpi=160)
    plt.close(fig)


def plot_risk_bands(summary: pd.DataFrame) -> None:
    data = summary[summary["population"] == "lockbox_10_11"].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(data))
    ax.plot(x, data["mean_pd"] * 100, marker="o", label="Mean frozen PD")
    ax.plot(x, data["observed_default_rate"] * 100, marker="o", label="Observed default rate")
    ax.set_xticks(x, data["risk_band"].astype(str))
    ax.set_xlabel("Frozen M5 risk band")
    ax.set_ylabel("Rate, %")
    ax.set_title("M6 frozen A-E risk bands on final lockbox")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m6_lockbox_risk_bands.png", dpi=160)
    plt.close(fig)


def append_experiment(ranking: pd.DataFrame, calibration: pd.DataFrame, policy: pd.DataFrame) -> None:
    registry = ARTIFACT_DIR / "experiments.csv"
    rank = ranking[ranking["population"] == "lockbox_10_11"].iloc[0]
    cal = calibration[
        (calibration["population"] == "lockbox_10_11")
        & (calibration["method"] == "calibrated_isotonic")
    ].iloc[0]
    pol = policy[policy["population"] == "lockbox_10_11"].iloc[0]
    row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "M6",
        "experiment": "final_lockbox_evaluation",
        "feature_version": "e_typed_catdist_payment",
        "model": "m4_catboost_plus_isotonic_plus_balanced_policy",
        "n_applications": int(pol["n"]),
        "n_features": 212,
        "roc_auc": float(rank["roc_auc"]),
        "average_precision": float(rank["average_precision"]),
        "brier_score": float(cal["brier_score"]),
        "log_loss": float(cal["log_loss"]),
        "approval_rate": float(pol["approval_rate"]),
        "review_rate": float(pol["review_rate"]),
        "rejection_rate": float(pol["rejection_rate"]),
        "expected_profit_rate_per_applicant": float(pol["expected_profit_rate_per_applicant"]),
        "realized_profit_rate_per_applicant": float(pol["realized_profit_rate_per_applicant"]),
    }])
    if registry.exists():
        existing = pd.read_csv(registry)
        pd.concat([existing, row], ignore_index=True, sort=False).to_csv(registry, index=False)
    else:
        row.to_csv(registry, index=False)


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    m4_meta, m5_meta, policy_artifact, band_artifact = validate_and_mark_lockbox_open(args)
    chunks = tuple(args.lockbox_chunks)
    scenario = frozen_primary_scenario(policy_artifact)
    cutpoints = frozen_risk_band_cutpoints(band_artifact)
    feature_names = load_frozen_feature_names(FEATURE_MANIFEST)
    if len(feature_names) != 212:
        raise RuntimeError(f"M6 expects 212 frozen features, got {len(feature_names)}.")

    print("=== M6 FINAL LOCKBOX EVALUATION ===")
    print(f"Lockbox chunks: {list(chunks)}")
    print("Fitting / tuning / recalibration / threshold adjustment: NONE")
    print(f"Frozen feature schema: {len(feature_names)} features")
    print(f"Frozen calibration: {m4_meta['selected_calibration_method']}")
    print(f"Frozen primary policy: {scenario.name}")
    print(
        f"Frozen policy thresholds: APPROVE <= {scenario.approve_pd_threshold:.4%}; "
        f"REVIEW <= {scenario.reject_pd_threshold:.4%}; REJECT above"
    )
    print("Frozen risk-band cutpoints: " + ", ".join(f"{x:.4%}" for x in cutpoints))

    caches = ensure_feature_caches(chunks, args.rebuild_features)
    historical_ids = historical_development_ids()
    target = load_target(DATA_DIR)
    target_indexed = target.set_index("id")

    model = CatBoostClassifier()
    model.load_model(M4_MODEL)
    imputer = joblib.load(M4_IMPUTER)
    calibrator = joblib.load(M4_CALIBRATOR)

    frames: list[pd.DataFrame] = []
    schema_rows: list[dict] = []
    for chunk in chunks:
        frame, schema = score_lockbox_chunk(
            caches[chunk], chunk, feature_names, target_indexed, model, imputer, calibrator,
            batch_size=args.prediction_batch_size,
        )
        overlap = np.intersect1d(historical_ids, frame["id"].to_numpy(dtype=np.int64))
        if overlap.size:
            raise RuntimeError(f"Lockbox chunk {chunk} overlaps historical development ids: {overlap.size:,}.")
        if frames:
            prior = np.concatenate([f["id"].to_numpy(dtype=np.int64) for f in frames])
            overlap_internal = np.intersect1d(prior, frame["id"].to_numpy(dtype=np.int64))
            if overlap_internal.size:
                raise RuntimeError(f"Lockbox chunks overlap by {overlap_internal.size:,} ids.")
        frame["risk_band"] = assign_risk_bands(frame["calibrated_pd"].to_numpy(), cutpoints)
        frame["decision"] = assign_decisions(frame["calibrated_pd"].to_numpy(), scenario)
        score_map = m5_meta.get("credit_score_mapping", {})
        frame["credit_score"] = pd_to_score(
            frame["calibrated_pd"].to_numpy(dtype=np.float64),
            base_score=float(score_map.get("base_score", 600.0)),
            base_good_to_bad_odds=float(score_map.get("base_good_to_bad_odds", 20.0)),
            points_to_double_odds=float(score_map.get("points_to_double_odds", 50.0)),
        ).astype(np.float32)
        frames.append(frame)
        schema_rows.append(schema)
        print(
            f"Lockbox chunk {chunk}: {len(frame):,} applications | default rate={frame['flag'].mean():.4%} | "
            f"raw/frozen features={schema['raw_feature_count']}/{schema['frozen_feature_count']} | "
            f"dropped-extra={schema['dropped_extra_count']}"
        )

    lockbox = pd.concat(frames, ignore_index=True)
    if lockbox["id"].duplicated().any():
        raise RuntimeError("Duplicate ids in pooled lockbox.")

    populations = [(f"chunk_{c}", lockbox[lockbox["source_chunk"] == c]) for c in chunks]
    populations.append(("lockbox_10_11", lockbox))
    ranking_rows, calibration_rows, reliability_frames = [], [], []
    band_frames, policy_rows = [], []
    for name, pop in populations:
        y = pop["flag"].to_numpy(dtype=np.int8)
        pd_score = pop["calibrated_pd"].to_numpy(dtype=np.float64)
        raw_score = pop["raw_score"].to_numpy(dtype=np.float64)
        ranking_rows.append({"population": name, **ranking_metrics(y, pd_score)})
        calibration_rows.append({"population": name, "method": "raw_catboost", **calibration_metrics(y, raw_score, n_bins=args.reliability_bins)})
        calibration_rows.append({"population": name, "method": "calibrated_isotonic", **calibration_metrics(y, pd_score, n_bins=args.reliability_bins)})
        rel = quantile_reliability_table(y, pd_score, n_bins=args.reliability_bins)
        rel.insert(0, "population", name)
        reliability_frames.append(rel)
        band = risk_band_summary(y, pd_score, pop["risk_band"].to_numpy(dtype=object))
        band.insert(0, "population", name)
        band_frames.append(band)
        decisions = pop["decision"].to_numpy(dtype=object)
        policy_rows.append({"population": name, **evaluate_policy(y, pd_score, scenario, decisions=decisions)})

    ranking_df = pd.DataFrame(ranking_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    band_df = pd.concat(band_frames, ignore_index=True)
    policy_df = pd.DataFrame(policy_rows)
    schema_df = pd.DataFrame(schema_rows)

    overall_rank = ranking_df[ranking_df["population"] == "lockbox_10_11"].iloc[0]
    overall_cal = calibration_df[(calibration_df["population"] == "lockbox_10_11") & (calibration_df["method"] == "calibrated_isotonic")].iloc[0]
    raw_cal = calibration_df[(calibration_df["population"] == "lockbox_10_11") & (calibration_df["method"] == "raw_catboost")].iloc[0]
    overall_policy = policy_df[policy_df["population"] == "lockbox_10_11"].iloc[0]

    print("\n=== M6 FINAL LOCKBOX DISCRIMINATION ===")
    for row in ranking_df.itertuples(index=False):
        print(
            f"{row.population:15s} AUC={row.roc_auc:.6f} | Gini={row.gini:.6f} | "
            f"AP={row.average_precision:.6f} | KS={row.ks:.6f} | "
            f"Lift@10={row.lift_top_10pct:.4f} | Capture@10={row.capture_rate_top_10pct:.4f}"
        )

    print("\n=== M6 FINAL LOCKBOX CALIBRATION ===")
    print(
        f"raw        Brier={raw_cal.brier_score:.6f} | LogLoss={raw_cal.log_loss:.6f} | "
        f"meanPD={raw_cal.mean_predicted_pd:.4%} | observed={raw_cal.observed_default_rate:.4%} | "
        f"intercept={raw_cal.calibration_intercept:+.4f} | slope={raw_cal.calibration_slope:.4f}"
    )
    print(
        f"isotonic   Brier={overall_cal.brier_score:.6f} | LogLoss={overall_cal.log_loss:.6f} | "
        f"meanPD={overall_cal.mean_predicted_pd:.4%} | observed={overall_cal.observed_default_rate:.4%} | "
        f"bias={overall_cal.calibration_bias:+.4%} | intercept={overall_cal.calibration_intercept:+.4f} | "
        f"slope={overall_cal.calibration_slope:.4f} | ECE={getattr(overall_cal, f'ece_quantile_{args.reliability_bins}'):.6f}"
    )

    print("\n=== M6 FINAL LOCKBOX FROZEN RISK BANDS ===")
    overall_bands = band_df[band_df["population"] == "lockbox_10_11"]
    for row in overall_bands.itertuples(index=False):
        print(
            f"{row.risk_band}: share={row.population_share:.2%} | meanPD={row.mean_pd:.4%} | "
            f"observedDR={row.observed_default_rate:.4%} | gap={row.calibration_gap:+.4%} | "
            f"badCapture={row.bad_capture_rate:.2%} | riskMultiple={row.risk_multiple_vs_portfolio:.2f}x"
        )

    print("\n=== M6 FINAL LOCKBOX BALANCED POLICY ===")
    print(
        f"decisions: approve={overall_policy.approval_rate:.2%} | review={overall_policy.review_rate:.2%} | "
        f"reject={overall_policy.rejection_rate:.2%}"
    )
    print(
        f"approved portfolio: meanPD={overall_policy.approved_mean_pd:.4%} | "
        f"observedDR={overall_policy.approved_observed_default_rate:.4%} | "
        f"calibrationGap={overall_policy.approved_calibration_gap:+.4%}"
    )
    print(
        f"loss/applicant: expected={overall_policy.expected_loss_rate_per_applicant:.4%} | "
        f"realized={overall_policy.realized_loss_rate_per_applicant:.4%} of EAD"
    )
    print(
        f"profit/applicant: expected={overall_policy.expected_profit_rate_per_applicant:.4%} | "
        f"realized={overall_policy.realized_profit_rate_per_applicant:.4%} of EAD | "
        f"forecastError={(overall_policy.expected_profit_rate_per_applicant-overall_policy.realized_profit_rate_per_applicant):+.4%}"
    )
    print(
        f"default capture: reject={overall_policy.default_capture_reject:.2%} | "
        f"review+reject={overall_policy.default_capture_review_or_reject:.2%}"
    )

    print(f"\nRunning final lockbox stratified bootstrap ({args.bootstrap_reps} replicates)...")
    ci_summaries, ci_draws = stratified_bootstrap_probability_metrics(
        lockbox["flag"].to_numpy(dtype=np.int8),
        lockbox["calibrated_pd"].to_numpy(dtype=np.float64),
        n_bootstrap=args.bootstrap_reps,
    )
    ci_df = pd.DataFrame([x.as_dict() for x in ci_summaries])
    print("\n=== M6 FINAL LOCKBOX 95% BOOTSTRAP CI ===")
    for row in ci_df.itertuples(index=False):
        print(f"{row.metric:18s} {row.point:.6f} | 95% CI [{row.ci_low:.6f}, {row.ci_high:.6f}]")

    lockbox.to_parquet(ARTIFACT_DIR / "m6_lockbox_predictions.parquet", index=False)
    ranking_df.to_csv(ARTIFACT_DIR / "m6_discrimination.csv", index=False)
    calibration_df.to_csv(ARTIFACT_DIR / "m6_calibration.csv", index=False)
    reliability_df.to_csv(ARTIFACT_DIR / "m6_reliability_table.csv", index=False)
    band_df.to_csv(ARTIFACT_DIR / "m6_risk_band_summary.csv", index=False)
    policy_df.to_csv(ARTIFACT_DIR / "m6_policy_summary.csv", index=False)
    schema_df.to_csv(ARTIFACT_DIR / "m6_schema_alignment.csv", index=False)
    ci_df.to_csv(ARTIFACT_DIR / "m6_bootstrap_ci.csv", index=False)
    np.save(ARTIFACT_DIR / "m6_bootstrap_draws.npy", ci_draws)
    plot_reliability(reliability_df)
    plot_risk_bands(band_df)

    metadata = {
        "stage": "M6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "final_one_shot_external_lockbox",
        "lockbox_chunks": list(chunks),
        "n_lockbox": int(len(lockbox)),
        "lockbox_default_rate": float(lockbox["flag"].mean()),
        "pipeline_frozen_before_lockbox": True,
        "fitting_on_lockbox": False,
        "tuning_on_lockbox": False,
        "recalibration_on_lockbox": False,
        "threshold_changes_on_lockbox": False,
        "feature_count": len(feature_names),
        "calibration_method": m4_meta["selected_calibration_method"],
        "primary_policy": scenario.name,
        "approve_pd_threshold": scenario.approve_pd_threshold,
        "reject_pd_threshold": scenario.reject_pd_threshold,
        "risk_band_cutpoints": [float(x) for x in cutpoints],
        "roc_auc": float(overall_rank.roc_auc),
        "average_precision": float(overall_rank.average_precision),
        "brier_score": float(overall_cal.brier_score),
        "log_loss": float(overall_cal.log_loss),
        "approval_rate": float(overall_policy.approval_rate),
        "review_rate": float(overall_policy.review_rate),
        "rejection_rate": float(overall_policy.rejection_rate),
        "expected_profit_rate_per_applicant": float(overall_policy.expected_profit_rate_per_applicant),
        "realized_profit_rate_per_applicant": float(overall_policy.realized_profit_rate_per_applicant),
        "lockbox_status_after_run": "OPENED_AND_EVALUATED",
        "future_use": "historical evaluation only; no longer an untouched test set",
    }
    (ARTIFACT_DIR / "m6_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_experiment(ranking_df, calibration_df, policy_df)

    print("\nLOCKBOX STATUS: OPENED_AND_EVALUATED")
    print("Chunks 10-11 must not be described as untouched in any future model-development stage.")
    print("\nSaved:")
    for path in [
        "artifacts/m6_lockbox_opened.json",
        "artifacts/m6_lockbox_predictions.parquet",
        "artifacts/m6_discrimination.csv",
        "artifacts/m6_calibration.csv",
        "artifacts/m6_reliability_table.csv",
        "artifacts/m6_risk_band_summary.csv",
        "artifacts/m6_policy_summary.csv",
        "artifacts/m6_schema_alignment.csv",
        "artifacts/m6_bootstrap_ci.csv",
        "artifacts/m6_bootstrap_draws.npy",
        "artifacts/m6_lockbox_calibration.png",
        "artifacts/m6_lockbox_risk_bands.png",
        "artifacts/m6_metadata.json",
        "artifacts/experiments.csv",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
