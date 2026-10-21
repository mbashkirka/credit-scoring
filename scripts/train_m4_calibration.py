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

from credit_risk.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_metrics,
    quantile_reliability_table,
)
from credit_risk.data import find_train_history_files, load_target
from credit_risk.evaluation import align_to_frozen_feature_schema, load_frozen_feature_names
from credit_risk.features import aggregate_canonical_m21_features
from credit_risk.modeling import FrozenMedianImputer, balanced_binary_class_weights

DATA_DIR = Path("data/raw")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
M4_DIR = Path("data/processed/m4")
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42
CANONICAL_VARIANT = "e_typed_catdist_payment"

DEFAULT_TRAIN_CHUNKS = tuple(range(0, 8))
DEFAULT_CALIBRATION_CHUNK = 8
DEFAULT_VALIDATION_CHUNK = 9
LOCKBOX_CHUNKS = (10, 11)

M4_DEPTH = 6
M4_LEARNING_RATE = 0.05
M4_L2 = 5.0
M4_ITERATIONS = 314


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M4 Probability-of-Default calibration. Fits the frozen M3 CatBoost "
            "configuration on chunks 0-7, fits calibration on chunk 8, validates "
            "calibration choice on chunk 9, and never reads lockbox chunks 10-11."
        )
    )
    parser.add_argument("--train-chunks", nargs="+", type=int, default=list(DEFAULT_TRAIN_CHUNKS))
    parser.add_argument("--calibration-chunk", type=int, default=DEFAULT_CALIBRATION_CHUNK)
    parser.add_argument("--validation-chunk", type=int, default=DEFAULT_VALIDATION_CHUNK)
    parser.add_argument("--prediction-batch-size", type=int, default=50_000)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--rebuild-training-matrix", action="store_true")
    parser.add_argument("--refit-model", action="store_true")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--used-ram-limit",
        type=str,
        default=None,
        help="Optional CatBoost CPU RAM hint, e.g. 8gb or 12gb.",
    )
    parser.add_argument("--reliability-bins", type=int, default=10)
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _schema_fingerprint(feature_names: list[str]) -> str:
    payload = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_governance(args: argparse.Namespace) -> tuple[list[int], int, int]:
    train_chunks = sorted(set(args.train_chunks))
    calibration_chunk = int(args.calibration_chunk)
    validation_chunk = int(args.validation_chunk)
    if train_chunks != list(DEFAULT_TRAIN_CHUNKS):
        raise ValueError(
            f"M4 governance requires training chunks {list(DEFAULT_TRAIN_CHUNKS)}; got {train_chunks}."
        )
    if calibration_chunk != DEFAULT_CALIBRATION_CHUNK:
        raise ValueError(f"M4 governance requires calibration chunk {DEFAULT_CALIBRATION_CHUNK}.")
    if validation_chunk != DEFAULT_VALIDATION_CHUNK:
        raise ValueError(f"M4 governance requires validation chunk {DEFAULT_VALIDATION_CHUNK}.")
    used = set(train_chunks) | {calibration_chunk, validation_chunk}
    if used & set(LOCKBOX_CHUNKS):
        raise RuntimeError("M4 must never access lockbox chunks 10-11.")
    if len(used) != len(train_chunks) + 2:
        raise ValueError("Train/calibration/validation populations must be disjoint.")
    if args.prediction_batch_size < 1:
        raise ValueError("--prediction-batch-size must be >= 1.")
    return train_chunks, calibration_chunk, validation_chunk


def raw_chunk_paths(chunks: list[int]) -> dict[int, Path]:
    available = {_chunk_index(path): path for path in find_train_history_files(DATA_DIR)}
    missing = [chunk for chunk in chunks if chunk not in available]
    if missing:
        raise FileNotFoundError(f"Raw train chunks not found: {missing}")
    # Deliberately request only the authorized chunks. Lockbox files are not read.
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
        cache_path = FEATURE_DIR / f"train_data_{chunk}.parquet"
        cache_paths[chunk] = cache_path
        if cache_path.exists() and not rebuild:
            print(f"Using cached canonical features: {cache_path}")
            continue
        raw_path = raw_paths[chunk]
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


def _row_count(path: Path) -> int:
    # Reading one id column keeps memory bounded and avoids loading the 212-feature cache.
    return int(len(pd.read_parquet(path, columns=["id"])))


def _training_matrix_paths() -> dict[str, Path]:
    return {
        "X": M4_DIR / "train_X_imputed_float32.npy",
        "y": M4_DIR / "train_y_int8.npy",
        "ids": M4_DIR / "train_ids_int64.npy",
        "meta": M4_DIR / "train_matrix_metadata.json",
        "imputer": ARTIFACT_DIR / "m4_catboost_imputer.joblib",
    }


def build_or_load_training_matrix(
    cache_paths: dict[int, Path],
    train_chunks: list[int],
    feature_names: list[str],
    target: pd.DataFrame,
    *,
    rebuild: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FrozenMedianImputer]:
    M4_DIR.mkdir(parents=True, exist_ok=True)
    paths = _training_matrix_paths()
    fingerprint = _schema_fingerprint(feature_names)
    expected_meta = {
        "train_chunks": train_chunks,
        "feature_count": len(feature_names),
        "schema_sha256": fingerprint,
    }

    if not rebuild and all(paths[k].exists() for k in ("X", "y", "ids", "meta", "imputer")):
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in expected_meta.items()) and meta.get("complete") is True:
            print(f"Using cached M4 training matrix: {paths['X']}")
            return (
                np.load(paths["X"], mmap_mode="r"),
                np.load(paths["y"], mmap_mode="r"),
                np.load(paths["ids"], mmap_mode="r"),
                joblib.load(paths["imputer"]),
            )

    total_rows = sum(_row_count(cache_paths[chunk]) for chunk in train_chunks)
    print(
        f"Materializing disk-backed M4 training matrix: {total_rows:,} x "
        f"{len(feature_names):,} float32"
    )
    X_mm = np.lib.format.open_memmap(
        paths["X"], mode="w+", dtype=np.float32, shape=(total_rows, len(feature_names))
    )
    y_mm = np.lib.format.open_memmap(paths["y"], mode="w+", dtype=np.int8, shape=(total_rows,))
    id_mm = np.lib.format.open_memmap(paths["ids"], mode="w+", dtype=np.int64, shape=(total_rows,))

    target_indexed = target.set_index("id")
    offset = 0
    for chunk in train_chunks:
        raw = pd.read_parquet(cache_paths[chunk])
        aligned, report = align_to_frozen_feature_schema(raw, feature_names)
        n = len(aligned)
        ids = aligned["id"].to_numpy(dtype=np.int64)
        if pd.Index(ids).has_duplicates:
            raise RuntimeError(f"Duplicate ids inside training chunk {chunk}.")
        try:
            y_chunk = target_indexed.loc[ids, "flag"].to_numpy(dtype=np.int8)
        except KeyError as exc:
            raise RuntimeError(f"Target rows are missing for training chunk {chunk}.") from exc
        X_mm[offset : offset + n] = aligned[feature_names].to_numpy(dtype=np.float32, copy=False)
        y_mm[offset : offset + n] = y_chunk
        id_mm[offset : offset + n] = ids
        print(
            f"  chunk {chunk}: rows={n:,}, structural-zero-missing={len(report.missing_structural_zero)}, "
            f"nonstructural-missing={len(report.missing_nonstructural)}, dropped-extra={len(report.dropped_extra)}"
        )
        offset += n
        del raw, aligned, ids, y_chunk
        gc.collect()

    if offset != total_rows:
        raise RuntimeError("Training matrix row-count mismatch.")
    X_mm.flush()
    y_mm.flush()
    id_mm.flush()

    ids_view = np.asarray(id_mm)
    if np.unique(ids_view).size != total_rows:
        raise RuntimeError("Training chunks contain overlapping application ids.")

    print("Computing exact per-feature medians from disk-backed matrix...")
    medians = np.empty(len(feature_names), dtype=np.float32)
    for j, feature in enumerate(feature_names):
        column = np.asarray(X_mm[:, j])
        valid = column[np.isfinite(column)]
        medians[j] = np.float32(np.median(valid)) if valid.size else np.float32(0.0)
        if (j + 1) % 25 == 0 or j + 1 == len(feature_names):
            print(f"  medians: {j + 1}/{len(feature_names)}")

    print("Imputing training matrix in place...")
    for j in range(len(feature_names)):
        column = X_mm[:, j]
        mask = ~np.isfinite(column)
        if np.any(mask):
            column[mask] = medians[j]
        if (j + 1) % 25 == 0 or j + 1 == len(feature_names):
            print(f"  imputed: {j + 1}/{len(feature_names)}")
    X_mm.flush()

    imputer = FrozenMedianImputer(medians)
    joblib.dump(imputer, paths["imputer"])
    meta = {
        **expected_meta,
        "complete": True,
        "n_rows": total_rows,
        "default_rate": float(np.asarray(y_mm).mean()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return (
        np.load(paths["X"], mmap_mode="r"),
        np.load(paths["y"], mmap_mode="r"),
        np.load(paths["ids"], mmap_mode="r"),
        imputer,
    )


def make_m4_catboost(class_weights: list[float], args: argparse.Namespace) -> CatBoostClassifier:
    kwargs = dict(
        iterations=M4_ITERATIONS,
        depth=M4_DEPTH,
        learning_rate=M4_LEARNING_RATE,
        l2_leaf_reg=M4_L2,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=args.thread_count,
        verbose=50,
    )
    if args.used_ram_limit:
        kwargs["used_ram_limit"] = args.used_ram_limit
    return CatBoostClassifier(**kwargs)


def model_meta_matches(path: Path, train_chunks: list[int], feature_names: list[str]) -> bool:
    if not path.exists():
        return False
    meta = json.loads(path.read_text(encoding="utf-8"))
    return (
        meta.get("train_chunks") == train_chunks
        and meta.get("schema_sha256") == _schema_fingerprint(feature_names)
        and meta.get("iterations") == M4_ITERATIONS
        and meta.get("depth") == M4_DEPTH
        and meta.get("learning_rate") == M4_LEARNING_RATE
        and meta.get("l2_leaf_reg") == M4_L2
    )


def fit_or_load_m4_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_chunks: list[int],
    feature_names: list[str],
    args: argparse.Namespace,
) -> CatBoostClassifier:
    model_path = ARTIFACT_DIR / "m4_catboost.cbm"
    meta_path = ARTIFACT_DIR / "m4_model_metadata.json"
    if not args.refit_model and model_path.exists() and model_meta_matches(meta_path, train_chunks, feature_names):
        print(f"Using frozen fitted M4 model: {model_path}")
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model

    weights = balanced_binary_class_weights(np.asarray(y_train, dtype=np.int8))
    print("\n=== M4 BASE MODEL FIT ===")
    print(f"Training applications: {len(y_train):,}")
    print(f"Training default rate: {np.asarray(y_train).mean():.4%}")
    print(
        f"Frozen CatBoost: depth={M4_DEPTH}, lr={M4_LEARNING_RATE}, "
        f"l2={M4_L2}, iterations={M4_ITERATIONS}"
    )
    print(
        f"Balanced class weights: negative={weights.negative:.6f}, "
        f"positive={weights.positive:.6f}"
    )
    model = make_m4_catboost(weights.as_catboost_list(), args)
    # No eval_set / early stopping: complexity was frozen before these populations.
    model.fit(X_train, np.asarray(y_train, dtype=np.int8))
    model.save_model(model_path)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACT_DIR / "m4_catboost_feature_importance.csv", index=False)
    meta = {
        "stage": "M4",
        "train_chunks": train_chunks,
        "n_train": int(len(y_train)),
        "train_default_rate": float(np.asarray(y_train).mean()),
        "feature_count": len(feature_names),
        "schema_sha256": _schema_fingerprint(feature_names),
        "iterations": M4_ITERATIONS,
        "depth": M4_DEPTH,
        "learning_rate": M4_LEARNING_RATE,
        "l2_leaf_reg": M4_L2,
        "class_weight": "balanced",
        "early_stopping": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model


def score_chunk(
    cache_path: Path,
    chunk: int,
    feature_names: list[str],
    target_indexed: pd.DataFrame,
    model: CatBoostClassifier,
    imputer: FrozenMedianImputer,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_parquet(cache_path)
    aligned, report = align_to_frozen_feature_schema(raw, feature_names)
    ids = aligned["id"].to_numpy(dtype=np.int64)
    y = target_indexed.loc[ids, "flag"].to_numpy(dtype=np.int8)
    raw_score = np.empty(len(aligned), dtype=np.float32)
    for start in range(0, len(aligned), batch_size):
        stop = min(len(aligned), start + batch_size)
        matrix = imputer.transform(aligned.iloc[start:stop][feature_names])
        raw_score[start:stop] = model.predict_proba(matrix)[:, 1].astype(np.float32)
        del matrix
    frame = pd.DataFrame(
        {
            "id": ids,
            "flag": y,
            "source_chunk": chunk,
            "raw_score": raw_score,
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
    }
    del raw, aligned
    gc.collect()
    return frame, schema


def fit_calibrators(raw_score: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {
        "raw": IdentityCalibrator().fit(raw_score, y),
        "platt": PlattCalibrator().fit(raw_score, y),
        "isotonic": IsotonicCalibrator().fit(raw_score, y),
    }


def evaluate_calibrators(
    population_name: str,
    frame: pd.DataFrame,
    calibrators: dict[str, object],
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | str]] = []
    reliability_frames: list[pd.DataFrame] = []
    y = frame["flag"].to_numpy(dtype=np.int8)
    raw = frame["raw_score"].to_numpy(dtype=np.float64)
    for method, calibrator in calibrators.items():
        pd_score = calibrator.predict(raw)
        metrics = calibration_metrics(y, pd_score, n_bins=n_bins)
        metric_rows.append({"population": population_name, "method": method, **metrics})
        table = quantile_reliability_table(y, pd_score, n_bins=n_bins)
        table.insert(0, "method", method)
        table.insert(0, "population", population_name)
        reliability_frames.append(table)
        frame[f"pd_{method}"] = pd_score.astype(np.float32)
    return pd.DataFrame(metric_rows), pd.concat(reliability_frames, ignore_index=True)


def print_calibration_table(title: str, metrics: pd.DataFrame, n_bins: int) -> None:
    print(f"\n=== {title} ===")
    ece_col = f"ece_quantile_{n_bins}"
    for row in metrics.itertuples(index=False):
        print(
            f"{row.method:9s} Brier={row.brier_score:.6f} | LogLoss={row.log_loss:.6f} | "
            f"meanPD={row.mean_predicted_pd:.4%} | observed={row.observed_default_rate:.4%} | "
            f"ECE={getattr(row, ece_col):.6f} | intercept={row.calibration_intercept:+.4f} | "
            f"slope={row.calibration_slope:.4f} | AUC={row.roc_auc:.6f}"
        )


def select_calibration_method(validation_metrics: pd.DataFrame) -> str:
    ranked = validation_metrics.sort_values(
        ["brier_score", "log_loss", "method"], ascending=[True, True, True]
    )
    return str(ranked.iloc[0]["method"])


def refit_selected_calibrator(
    method: str,
    combined: pd.DataFrame,
):
    cls = {
        "raw": IdentityCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
    }[method]
    return cls().fit(
        combined["raw_score"].to_numpy(dtype=np.float64),
        combined["flag"].to_numpy(dtype=np.int8),
    )


def save_reliability_plot(validation_reliability: pd.DataFrame, selected: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    max_value = float(
        max(
            validation_reliability["predicted_pd_mean"].max(),
            validation_reliability["observed_default_rate"].max(),
            0.05,
        )
    )
    limit = min(1.0, max_value * 1.15)
    ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1, label="perfect calibration")
    for method in ["raw", "platt", "isotonic"]:
        data = validation_reliability[validation_reliability["method"] == method]
        ax.plot(
            data["predicted_pd_mean"],
            data["observed_default_rate"],
            marker="o",
            label=f"{method}{' (selected)' if method == selected else ''}",
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("M4 calibration validation — chunk 9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "m4_calibration_curve_validation.png", dpi=160)
    plt.close(fig)


def append_experiment(selected: str, validation_metrics: pd.DataFrame, n_train: int) -> None:
    registry_path = ARTIFACT_DIR / "experiments.csv"
    best = validation_metrics[validation_metrics["method"] == selected].iloc[0]
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "M4",
                "experiment": "pd_calibration_validation",
                "feature_version": CANONICAL_VARIANT,
                "model": "catboost_m3_baseline_scaled",
                "n_applications": int(n_train),
                "n_features": 212,
                "calibration_method": selected,
                "brier_score": float(best["brier_score"]),
                "log_loss": float(best["log_loss"]),
                "roc_auc": float(best["roc_auc"]),
            }
        ]
    )
    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        pd.concat([existing, row], ignore_index=True, sort=False).to_csv(registry_path, index=False)
    else:
        row.to_csv(registry_path, index=False)


def main() -> None:
    args = parse_args()
    train_chunks, calibration_chunk, validation_chunk = validate_governance(args)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    M4_DIR.mkdir(parents=True, exist_ok=True)

    feature_manifest = ARTIFACT_DIR / "m3_features.csv"
    feature_names = load_frozen_feature_names(feature_manifest)
    target = load_target(DATA_DIR)

    authorized_chunks = [*train_chunks, calibration_chunk, validation_chunk]
    print("=== M4 PROBABILITY-OF-DEFAULT CALIBRATION ===")
    print(f"Feature representation: {CANONICAL_VARIANT}")
    print(f"Frozen feature schema: {len(feature_names)} features")
    print(f"Base-model training chunks: {train_chunks}")
    print(f"Calibration-fit chunk: {calibration_chunk}")
    print(f"Calibration-validation chunk: {validation_chunk}")
    print(f"Reserved lockbox chunks (NOT ACCESSED): {list(LOCKBOX_CHUNKS)}")
    print(
        f"Frozen CatBoost config: depth={M4_DEPTH}, learning_rate={M4_LEARNING_RATE}, "
        f"l2_leaf_reg={M4_L2}, iterations={M4_ITERATIONS}; early stopping=NONE"
    )

    caches = ensure_canonical_feature_caches(authorized_chunks, rebuild=args.rebuild_features)
    X_train, y_train, train_ids, imputer = build_or_load_training_matrix(
        caches,
        train_chunks,
        feature_names,
        target,
        rebuild=args.rebuild_training_matrix,
    )
    model = fit_or_load_m4_model(X_train, y_train, train_chunks, feature_names, args)

    target_indexed = target.set_index("id")
    calibration_frame, calibration_schema = score_chunk(
        caches[calibration_chunk],
        calibration_chunk,
        feature_names,
        target_indexed,
        model,
        imputer,
        batch_size=args.prediction_batch_size,
    )
    validation_frame, validation_schema = score_chunk(
        caches[validation_chunk],
        validation_chunk,
        feature_names,
        target_indexed,
        model,
        imputer,
        batch_size=args.prediction_batch_size,
    )

    train_ids_array = np.asarray(train_ids)
    if np.intersect1d(train_ids_array, calibration_frame["id"].to_numpy()).size:
        raise RuntimeError("Training/calibration id overlap detected.")
    if np.intersect1d(train_ids_array, validation_frame["id"].to_numpy()).size:
        raise RuntimeError("Training/validation id overlap detected.")
    if np.intersect1d(
        calibration_frame["id"].to_numpy(), validation_frame["id"].to_numpy()
    ).size:
        raise RuntimeError("Calibration/validation id overlap detected.")

    print("\n=== M4 POPULATIONS ===")
    print(f"Training:    {len(y_train):,} applications | default rate={np.asarray(y_train).mean():.4%}")
    print(
        f"Calibration: {len(calibration_frame):,} applications | "
        f"default rate={calibration_frame['flag'].mean():.4%}"
    )
    print(
        f"Validation:  {len(validation_frame):,} applications | "
        f"default rate={validation_frame['flag'].mean():.4%}"
    )

    fit8_calibrators = fit_calibrators(
        calibration_frame["raw_score"].to_numpy(),
        calibration_frame["flag"].to_numpy(),
    )
    joblib.dump(fit8_calibrators["platt"], ARTIFACT_DIR / "m4_platt_calibrator_fit8.joblib")
    joblib.dump(fit8_calibrators["isotonic"], ARTIFACT_DIR / "m4_isotonic_calibrator_fit8.joblib")

    fit_metrics, fit_reliability = evaluate_calibrators(
        "calibration_fit_chunk8", calibration_frame, fit8_calibrators, args.reliability_bins
    )
    validation_metrics, validation_reliability = evaluate_calibrators(
        "calibration_validation_chunk9", validation_frame, fit8_calibrators, args.reliability_bins
    )
    print_calibration_table("M4 CALIBRATION FIT DIAGNOSTIC (CHUNK 8)", fit_metrics, args.reliability_bins)
    print_calibration_table("M4 CALIBRATION VALIDATION (CHUNK 9)", validation_metrics, args.reliability_bins)

    selected = select_calibration_method(validation_metrics)
    print(f"\nSelected calibration method by chunk-9 Brier score: {selected}")
    print("Chunk 9 is calibration-selection validation, not the final lockbox estimate.")

    combined = pd.concat([calibration_frame, validation_frame], ignore_index=True)
    final_calibrator = refit_selected_calibrator(selected, combined)
    joblib.dump(final_calibrator, ARTIFACT_DIR / "m4_calibrator.joblib")

    all_metrics = pd.concat([fit_metrics, validation_metrics], ignore_index=True)
    all_reliability = pd.concat([fit_reliability, validation_reliability], ignore_index=True)
    all_predictions = pd.concat([calibration_frame, validation_frame], ignore_index=True)
    all_metrics.to_csv(ARTIFACT_DIR / "m4_calibration_metrics.csv", index=False)
    all_reliability.to_csv(ARTIFACT_DIR / "m4_reliability_table.csv", index=False)
    all_predictions.to_parquet(ARTIFACT_DIR / "m4_calibration_predictions.parquet", index=False)
    pd.DataFrame([calibration_schema, validation_schema]).to_csv(
        ARTIFACT_DIR / "m4_schema_alignment.csv", index=False
    )
    save_reliability_plot(validation_reliability, selected)

    selected_validation = validation_metrics[validation_metrics["method"] == selected].iloc[0]
    raw_validation = validation_metrics[validation_metrics["method"] == "raw"].iloc[0]
    metadata = {
        "stage": "M4",
        "feature_version": CANONICAL_VARIANT,
        "feature_count": len(feature_names),
        "train_chunks": train_chunks,
        "calibration_chunk": calibration_chunk,
        "validation_chunk": validation_chunk,
        "lockbox_chunks_reserved": list(LOCKBOX_CHUNKS),
        "n_train": int(len(y_train)),
        "train_default_rate": float(np.asarray(y_train).mean()),
        "calibration_default_rate": float(calibration_frame["flag"].mean()),
        "validation_default_rate": float(validation_frame["flag"].mean()),
        "selected_calibration_method": selected,
        "selection_metric": "validation_brier_score",
        "validation_brier_selected": float(selected_validation["brier_score"]),
        "validation_brier_raw": float(raw_validation["brier_score"]),
        "validation_logloss_selected": float(selected_validation["log_loss"]),
        "validation_mean_pd_selected": float(selected_validation["mean_predicted_pd"]),
        "validation_observed_default_rate": float(selected_validation["observed_default_rate"]),
        "final_calibrator_refit_population": [calibration_chunk, validation_chunk],
        "final_lockbox_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACT_DIR / "m4_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_experiment(selected, validation_metrics, len(y_train))

    print("\n=== M4 SELECTED CALIBRATION ===")
    print(f"method: {selected}")
    print(f"validation Brier: {selected_validation['brier_score']:.6f} (raw={raw_validation['brier_score']:.6f})")
    print(f"validation LogLoss: {selected_validation['log_loss']:.6f} (raw={raw_validation['log_loss']:.6f})")
    print(
        f"validation mean PD: {selected_validation['mean_predicted_pd']:.4%} | "
        f"observed default rate: {selected_validation['observed_default_rate']:.4%}"
    )
    print(
        f"calibration intercept/slope: {selected_validation['calibration_intercept']:+.4f} / "
        f"{selected_validation['calibration_slope']:.4f}"
    )
    print("Selected calibrator refitted on chunks 8+9. Chunks 10-11 remain untouched.")

    print("\nSaved:")
    for path in [
        "artifacts/m4_catboost.cbm",
        "artifacts/m4_catboost_imputer.joblib",
        "artifacts/m4_catboost_feature_importance.csv",
        "artifacts/m4_model_metadata.json",
        "artifacts/m4_platt_calibrator_fit8.joblib",
        "artifacts/m4_isotonic_calibrator_fit8.joblib",
        "artifacts/m4_calibrator.joblib",
        "artifacts/m4_calibration_metrics.csv",
        "artifacts/m4_reliability_table.csv",
        "artifacts/m4_calibration_predictions.parquet",
        "artifacts/m4_calibration_curve_validation.png",
        "artifacts/m4_schema_alignment.csv",
        "artifacts/m4_metadata.json",
        "artifacts/experiments.csv",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
