from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from credit_risk.data import find_train_history_files, load_target
from credit_risk.drift import (
    build_drift_accumulators,
    summarize_drift,
    update_drift_accumulators,
)
from credit_risk.evaluation import (
    align_to_frozen_feature_schema,
    assert_disjoint_ids,
    load_frozen_feature_names,
)
from credit_risk.features import aggregate_canonical_m21_features
from credit_risk.metrics import ranking_metrics
from credit_risk.statistics import paired_stratified_bootstrap_ranking_delta

DATA_DIR = Path("data/raw")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42
CANONICAL_VARIANT = "e_typed_catdist_payment"
DEV_CHUNK = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M3.2 external validation of frozen M3/M3.1 models on untouched raw "
            "history chunks. No fitting, early stopping, or hyperparameter selection "
            "is performed on holdout labels."
        )
    )
    parser.add_argument(
        "--holdout-chunks",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Raw train_data_<k>.pq chunks reserved for external validation.",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--prediction-batch-size", type=int, default=50_000)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--drift-bins", type=int, default=10)
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _required_artifacts() -> tuple[Path, ...]:
    return (
        ARTIFACT_DIR / "m3_features.csv",
        ARTIFACT_DIR / "m3_cv_assignments.parquet",
        ARTIFACT_DIR / "m3_logreg.joblib",
        ARTIFACT_DIR / "m3_catboost.cbm",
        ARTIFACT_DIR / "m3_catboost_imputer.joblib",
        ARTIFACT_DIR / "m3_metadata.json",
        ARTIFACT_DIR / "m31_catboost.cbm",
        ARTIFACT_DIR / "m31_catboost_imputer.joblib",
        ARTIFACT_DIR / "m31_metadata.json",
    )


def validate_stage_inputs(holdout_chunks: list[int]) -> None:
    if not holdout_chunks:
        raise ValueError("At least one holdout chunk is required.")
    if len(holdout_chunks) != len(set(holdout_chunks)):
        raise ValueError("--holdout-chunks contains duplicates.")
    if DEV_CHUNK in holdout_chunks:
        raise ValueError("Development chunk 0 cannot be part of the M3.2 holdout.")
    missing = [path for path in _required_artifacts() if not path.exists()]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "M3.2 requires the frozen M3 and M3.1 artifacts from chunk 0. Missing:\n"
            + rendered
        )


def raw_chunk_paths(chunks: list[int]) -> dict[int, Path]:
    available = {_chunk_index(path): path for path in find_train_history_files(DATA_DIR)}
    missing = [chunk for chunk in chunks if chunk not in available]
    if missing:
        raise FileNotFoundError(f"Raw train chunks not found: {missing}")
    return {chunk: available[chunk] for chunk in chunks}


def ensure_canonical_feature_caches(
    chunks: list[int],
    *,
    rebuild: bool,
) -> dict[int, Path]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = raw_chunk_paths(chunks)
    cache_paths: dict[int, Path] = {}
    for chunk in chunks:
        raw_path = raw_paths[chunk]
        cache_path = FEATURE_DIR / f"train_data_{chunk}.parquet"
        cache_paths[chunk] = cache_path
        if cache_path.exists() and not rebuild:
            print(f"Using cached canonical features: {cache_path}")
            continue
        print(f"Reading {raw_path}")
        history = pd.read_parquet(raw_path)
        print(f"Building canonical M2.1-E features for chunk {chunk}...")
        features = aggregate_canonical_m21_features(history)
        features.to_parquet(cache_path, index=False)
        print(
            f"Cached chunk {chunk}: {len(features):,} applications x "
            f"{features.shape[1] - 1:,} raw canonical features"
        )
        del history, features
        gc.collect()
    return cache_paths


def load_catboost(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def predict_in_batches(
    X: pd.DataFrame,
    *,
    logreg,
    m3_catboost: CatBoostClassifier,
    m3_imputer,
    m31_catboost: CatBoostClassifier,
    m31_imputer,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size < 1:
        raise ValueError("--prediction-batch-size must be >= 1")
    n = len(X)
    result = {
        "logistic_regression": np.empty(n, dtype="float32"),
        "catboost_m3_baseline": np.empty(n, dtype="float32"),
        "catboost_m31_tuned": np.empty(n, dtype="float32"),
    }
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        batch = X.iloc[start:stop]
        result["logistic_regression"][start:stop] = (
            logreg.predict_proba(batch)[:, 1].astype("float32")
        )
        m3_matrix = m3_imputer.transform(batch)
        result["catboost_m3_baseline"][start:stop] = (
            m3_catboost.predict_proba(m3_matrix)[:, 1].astype("float32")
        )
        del m3_matrix
        m31_matrix = m31_imputer.transform(batch)
        result["catboost_m31_tuned"][start:stop] = (
            m31_catboost.predict_proba(m31_matrix)[:, 1].astype("float32")
        )
        del m31_matrix
    return result


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    holdout_chunks = sorted(args.holdout_chunks)
    validate_stage_inputs(holdout_chunks)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = load_frozen_feature_names(ARTIFACT_DIR / "m3_features.csv")
    if len(feature_names) != 212:
        print(
            f"WARNING: frozen M3 schema has {len(feature_names)} features, not the "
            "historical 212. The file itself remains the source of truth."
        )

    # Development population is defined by the exact M3 CV assignment artifact,
    # not inferred from raw chunk naming.
    dev_assignments = pd.read_parquet(
        ARTIFACT_DIR / "m3_cv_assignments.parquet",
        columns=["id", "flag"],
    )
    if dev_assignments["id"].duplicated().any():
        raise RuntimeError("M3 development assignments contain duplicate ids.")

    m3_meta = load_json(ARTIFACT_DIR / "m3_metadata.json")
    m31_meta = load_json(ARTIFACT_DIR / "m31_metadata.json")
    m31_name = str(m31_meta.get("best_config", {}).get("name", "unknown"))

    print("=== M3.2 UNTOUCHED EXTERNAL HOLDOUT VALIDATION ===")
    print(f"Feature representation: {CANONICAL_VARIANT}")
    print(f"Frozen feature schema: {len(feature_names)} features")
    print(f"Development population: {len(dev_assignments):,} applications (M3 chunk 0)")
    print(f"Development default rate: {dev_assignments['flag'].mean():.4%}")
    print(f"Holdout chunks: {', '.join(map(str, holdout_chunks))}")
    print("Models: frozen M3 logistic, frozen M3 CatBoost, frozen M3.1 CatBoost")
    print(f"M3 CatBoost iterations: {m3_meta.get('catboost', {}).get('final_iterations', 'from model artifact')}")
    print(f"M3.1 selected config: {m31_name}; iterations={m31_meta.get('final_iterations', 'from model artifact')}")
    print("Holdout fitting / early stopping / tuning: NONE")

    cache_paths = ensure_canonical_feature_caches(
        holdout_chunks,
        rebuild=args.rebuild_features,
    )
    dev_cache = FEATURE_DIR / f"train_data_{DEV_CHUNK}.parquet"
    if not dev_cache.exists():
        raise FileNotFoundError(
            f"Development canonical cache is missing: {dev_cache}. "
            "Run M2.1/M3 first."
        )

    # Build an exact development drift reference using the frozen 212-column schema.
    dev_raw = pd.read_parquet(dev_cache)
    dev_aligned, dev_schema_report = align_to_frozen_feature_schema(dev_raw, feature_names)
    dev_ids = dev_aligned["id"].to_numpy()
    saved_dev_ids = dev_assignments["id"].to_numpy()
    if set(dev_ids.tolist()) != set(saved_dev_ids.tolist()):
        raise RuntimeError(
            "Development feature cache does not match the population used by M3."
        )
    drift_acc = build_drift_accumulators(
        dev_aligned[feature_names],
        feature_names,
        n_bins=args.drift_bins,
    )
    del dev_raw, dev_aligned
    gc.collect()

    logreg = joblib.load(ARTIFACT_DIR / "m3_logreg.joblib")
    m3_catboost = load_catboost(ARTIFACT_DIR / "m3_catboost.cbm")
    m3_imputer = joblib.load(ARTIFACT_DIR / "m3_catboost_imputer.joblib")
    m31_catboost = load_catboost(ARTIFACT_DIR / "m31_catboost.cbm")
    m31_imputer = joblib.load(ARTIFACT_DIR / "m31_catboost_imputer.joblib")

    target = load_target(DATA_DIR).set_index("id")
    all_prediction_frames: list[pd.DataFrame] = []
    chunk_metric_rows: list[dict[str, float | int | str]] = []
    population_rows: list[dict[str, float | int | str]] = [
        {
            "population": "development_m3",
            "source_chunk": DEV_CHUNK,
            "n_applications": int(len(dev_assignments)),
            "default_rate": float(dev_assignments["flag"].mean()),
        }
    ]
    schema_rows: list[dict[str, object]] = []
    all_holdout_ids: list[np.ndarray] = []

    for chunk in holdout_chunks:
        print(f"\n{'#' * 80}\nHOLDOUT CHUNK {chunk}")
        raw_features = pd.read_parquet(cache_paths[chunk])
        aligned, report = align_to_frozen_feature_schema(raw_features, feature_names)
        schema_rows.append(
            {
                "source_chunk": chunk,
                "n_raw_features": int(raw_features.shape[1] - 1),
                "n_frozen_features": len(feature_names),
                "missing_structural_zero_count": len(report.missing_structural_zero),
                "missing_nonstructural_count": len(report.missing_nonstructural),
                "dropped_extra_count": len(report.dropped_extra),
                "missing_structural_zero": "|".join(report.missing_structural_zero),
                "missing_nonstructural": "|".join(report.missing_nonstructural),
                "dropped_extra": "|".join(report.dropped_extra),
            }
        )
        ids = aligned["id"].to_numpy()
        all_holdout_ids.append(ids)
        assert_disjoint_ids(saved_dev_ids, ids)
        if pd.Index(ids).duplicated().any():
            raise RuntimeError(f"Duplicate application ids inside holdout chunk {chunk}.")
        missing_target = pd.Index(ids).difference(target.index)
        if len(missing_target):
            raise RuntimeError(
                f"Chunk {chunk} has {len(missing_target):,} applications without target labels."
            )
        y = target.loc[ids, "flag"].to_numpy(dtype="int8")
        population_rows.append(
            {
                "population": "external_holdout_chunk",
                "source_chunk": chunk,
                "n_applications": int(len(y)),
                "default_rate": float(y.mean()),
            }
        )

        update_drift_accumulators(drift_acc, aligned[feature_names])
        predictions = predict_in_batches(
            aligned[feature_names],
            logreg=logreg,
            m3_catboost=m3_catboost,
            m3_imputer=m3_imputer,
            m31_catboost=m31_catboost,
            m31_imputer=m31_imputer,
            batch_size=args.prediction_batch_size,
        )
        pred_frame = pd.DataFrame(
            {
                "id": ids,
                "flag": y,
                "source_chunk": chunk,
                **{f"score__{name}": score for name, score in predictions.items()},
            }
        )
        all_prediction_frames.append(pred_frame)

        print(
            f"Applications: {len(y):,} | default rate: {y.mean():.4%} | "
            f"raw/frozen features: {raw_features.shape[1]-1}/{len(feature_names)}"
        )
        for model_name, score in predictions.items():
            metrics = ranking_metrics(y, score)
            chunk_metric_rows.append(
                {"source_chunk": chunk, "model": model_name, **metrics}
            )
            print(
                f"{model_name:24s} AUC={metrics['roc_auc']:.6f} | "
                f"AP={metrics['average_precision']:.6f} | KS={metrics['ks']:.6f} | "
                f"Lift@10={metrics['lift_top_10pct']:.4f}"
            )

        del raw_features, aligned, predictions, pred_frame
        gc.collect()

    # Cross-chunk overlap is also forbidden.
    concatenated_ids = np.concatenate(all_holdout_ids)
    if len(np.unique(concatenated_ids)) != len(concatenated_ids):
        raise RuntimeError("Application id overlap detected across M3.2 holdout chunks.")
    assert_disjoint_ids(saved_dev_ids, concatenated_ids)

    predictions_all = pd.concat(all_prediction_frames, ignore_index=True)
    y_all = predictions_all["flag"].to_numpy(dtype="int8")

    model_cols = {
        "logistic_regression": "score__logistic_regression",
        "catboost_m3_baseline": "score__catboost_m3_baseline",
        "catboost_m31_tuned": "score__catboost_m31_tuned",
    }
    summary_rows: list[dict[str, float | int | str]] = []
    print("\n=== M3.2 EXTERNAL HOLDOUT MODEL COMPARISON ===")
    for model_name, column in model_cols.items():
        metrics = ranking_metrics(y_all, predictions_all[column].to_numpy())
        summary_rows.append(
            {
                "model": model_name,
                "n_applications": len(y_all),
                "n_features": len(feature_names),
                "default_rate": float(y_all.mean()),
                **metrics,
            }
        )
        print(
            f"{model_name:24s} AUC={metrics['roc_auc']:.6f} | "
            f"Gini={metrics['gini']:.6f} | AP={metrics['average_precision']:.6f} | "
            f"KS={metrics['ks']:.6f} | Lift@10={metrics['lift_top_10pct']:.4f} | "
            f"Capture@10={metrics['capture_rate_top_10pct']:.4f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    print(f"\n=== M3.2 PAIRED BOOTSTRAP ({args.bootstrap_reps} replicates) ===")
    bootstrap_rows: list[dict[str, object]] = []
    bootstrap_raw: list[pd.DataFrame] = []

    def compare(label: str, reference_col: str, candidate_col: str) -> None:
        auc, ap, raw = paired_stratified_bootstrap_ranking_delta(
            y_all,
            predictions_all[reference_col].to_numpy(),
            predictions_all[candidate_col].to_numpy(),
            n_bootstrap=args.bootstrap_reps,
            confidence=args.bootstrap_confidence,
            random_state=RANDOM_STATE,
        )
        for item in (auc, ap):
            bootstrap_rows.append({"comparison": label, **item.as_dict()})
        bootstrap_raw.append(
            pd.DataFrame(
                {
                    "comparison": label,
                    "bootstrap_rep": np.arange(1, len(raw) + 1),
                    "delta_roc_auc": raw[:, 0],
                    "delta_average_precision": raw[:, 1],
                }
            )
        )
        pct = int(args.bootstrap_confidence * 100)
        print(f"\n{label}")
        print(
            f"  ΔAUC={auc.point_delta:+.6f} | {pct}% CI "
            f"[{auc.ci_low:+.6f}, {auc.ci_high:+.6f}] | P(Δ>0)={auc.win_probability:.4f}"
        )
        print(
            f"  ΔAP ={ap.point_delta:+.6f} | {pct}% CI "
            f"[{ap.ci_low:+.6f}, {ap.ci_high:+.6f}] | P(Δ>0)={ap.win_probability:.4f}"
        )

    compare(
        "m3_catboost_minus_logistic",
        "score__logistic_regression",
        "score__catboost_m3_baseline",
    )
    compare(
        "m31_tuned_minus_m3_catboost",
        "score__catboost_m3_baseline",
        "score__catboost_m31_tuned",
    )
    compare(
        "m31_tuned_minus_logistic",
        "score__logistic_regression",
        "score__catboost_m31_tuned",
    )

    drift = summarize_drift(drift_acc)
    population = pd.DataFrame(population_rows)
    combined_holdout_row = pd.DataFrame(
        [
            {
                "population": "external_holdout_combined",
                "source_chunk": "|".join(map(str, holdout_chunks)),
                "n_applications": int(len(y_all)),
                "default_rate": float(y_all.mean()),
            }
        ]
    )
    population = pd.concat([population, combined_holdout_row], ignore_index=True)

    dev_rate = float(dev_assignments["flag"].mean())
    holdout_rate = float(y_all.mean())
    print("\n=== M3.2 POPULATION SHIFT ===")
    print(f"Development default rate: {dev_rate:.4%}")
    print(f"Holdout default rate:     {holdout_rate:.4%}")
    print(f"Default-rate delta:       {holdout_rate - dev_rate:+.4%}")
    print(f"Median feature PSI:       {drift['psi'].median():.6f}")
    print(f"90th pct feature PSI:     {drift['psi'].quantile(0.90):.6f}")
    print(f"Max feature PSI:          {drift['psi'].max():.6f}")
    print("\nTop 10 features by PSI:")
    print(
        drift[["feature", "psi", "missing_rate_delta", "standardized_mean_diff"]]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    summary.insert(0, "timestamp_utc", timestamp)
    summary.insert(1, "stage", "M3.2")
    summary.to_csv(ARTIFACT_DIR / "m32_external_validation.csv", index=False)
    pd.DataFrame(chunk_metric_rows).to_csv(
        ARTIFACT_DIR / "m32_chunk_metrics.csv", index=False
    )
    predictions_all.to_parquet(
        ARTIFACT_DIR / "m32_holdout_predictions.parquet", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        ARTIFACT_DIR / "m32_bootstrap_summary.csv", index=False
    )
    pd.concat(bootstrap_raw, ignore_index=True).to_parquet(
        ARTIFACT_DIR / "m32_bootstrap_deltas.parquet", index=False
    )
    drift.to_csv(ARTIFACT_DIR / "m32_feature_drift.csv", index=False)
    population.to_csv(ARTIFACT_DIR / "m32_population_summary.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(
        ARTIFACT_DIR / "m32_schema_alignment.csv", index=False
    )

    metadata = {
        "timestamp_utc": timestamp,
        "stage": "M3.2",
        "experiment": "untouched_external_holdout_validation",
        "evaluation_note": (
            "Primary M3.2 metrics are external-holdout estimates from frozen models "
            "trained on the M3 development population. No holdout fitting, early "
            "stopping, feature expansion, or hyperparameter selection is performed."
        ),
        "feature_variant": CANONICAL_VARIANT,
        "frozen_feature_count": len(feature_names),
        "development_chunk": DEV_CHUNK,
        "development_n": int(len(dev_assignments)),
        "development_default_rate": dev_rate,
        "holdout_chunks": holdout_chunks,
        "holdout_n": int(len(y_all)),
        "holdout_default_rate": holdout_rate,
        "default_rate_delta": holdout_rate - dev_rate,
        "m3_metadata": m3_meta,
        "m31_selected_config": m31_meta.get("best_config"),
        "m31_final_iterations": m31_meta.get("final_iterations"),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_confidence": args.bootstrap_confidence,
        "drift_bins": args.drift_bins,
        "drift_median_psi": float(drift["psi"].median()),
        "drift_p90_psi": float(drift["psi"].quantile(0.90)),
        "drift_max_psi": float(drift["psi"].max()),
        "development_schema_report": {
            "missing_structural_zero": list(dev_schema_report.missing_structural_zero),
            "missing_nonstructural": list(dev_schema_report.missing_nonstructural),
            "dropped_extra": list(dev_schema_report.dropped_extra),
        },
    }
    (ARTIFACT_DIR / "m32_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    registry_path = ARTIFACT_DIR / "experiments.csv"
    registry_rows = summary.copy()
    if registry_path.exists():
        registry = pd.concat(
            [pd.read_csv(registry_path), registry_rows],
            ignore_index=True,
            sort=False,
        )
    else:
        registry = registry_rows
    registry.to_csv(registry_path, index=False)

    print("\nSaved:")
    for path in (
        ARTIFACT_DIR / "m32_external_validation.csv",
        ARTIFACT_DIR / "m32_chunk_metrics.csv",
        ARTIFACT_DIR / "m32_holdout_predictions.parquet",
        ARTIFACT_DIR / "m32_bootstrap_summary.csv",
        ARTIFACT_DIR / "m32_bootstrap_deltas.parquet",
        ARTIFACT_DIR / "m32_feature_drift.csv",
        ARTIFACT_DIR / "m32_population_summary.csv",
        ARTIFACT_DIR / "m32_schema_alignment.csv",
        ARTIFACT_DIR / "m32_metadata.json",
        ARTIFACT_DIR / "experiments.csv",
    ):
        print("-", path)


if __name__ == "__main__":
    main()
