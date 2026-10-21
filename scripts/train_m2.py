from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import iter_train_history, load_target
from credit_risk.features import aggregate_credit_history
from credit_risk.metrics import ranking_metrics

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/app_features_m2")
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Number of raw train parquet chunks. Use 1 for the first M2 run.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "'balanced' matches M1 ranking-oriented training; 'none' is useful "
            "later when studying probability calibration."
        ),
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore cached application-level parquet features and rebuild them.",
    )
    return parser.parse_args()


def build_or_load_features(max_files: int, rebuild: bool) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app_frames: list[pd.DataFrame] = []

    for raw_path, history in iter_train_history(DATA_DIR, max_files=max_files):
        cache_path = PROCESSED_DIR / f"{raw_path.stem}__m2.parquet"
        if cache_path.exists() and not rebuild:
            print(f"Loading cached features {cache_path}")
            app_features = pd.read_parquet(cache_path)
        else:
            print(f"Aggregating {raw_path.name} -> application level")
            app_features = aggregate_credit_history(history)
            app_features.to_parquet(cache_path, index=False)
            print(
                f"Cached {len(app_features):,} applications x "
                f"{app_features.shape[1] - 1:,} features"
            )

        app_frames.append(app_features)
        del history, app_features
        gc.collect()

    features = pd.concat(app_frames, ignore_index=True, sort=False)
    feature_cols = [column for column in features.columns if column != "id"]
    features[feature_cols] = features[feature_cols].fillna(np.nan).astype("float32")

    duplicates = int(features["id"].duplicated().sum())
    if duplicates:
        raise RuntimeError(
            f"Found {duplicates:,} duplicate application ids across parquet chunks. "
            "The chunk-wise aggregation assumption is invalid and must be revised."
        )
    return features


def make_model(feature_names: list[str], class_weight: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None if class_weight == "none" else "balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    target = load_target(DATA_DIR)
    features = build_or_load_features(args.max_files, args.rebuild_features)
    dataset = target.merge(features, on="id", how="inner", validate="one_to_one")
    if dataset.empty:
        raise RuntimeError("Target/history merge produced zero rows.")

    y = dataset["flag"].astype("int8").to_numpy()
    ids = dataset["id"].to_numpy()
    X = dataset.drop(columns=["id", "flag"])
    feature_names = X.columns.tolist()

    print(f"\nApplications used: {len(dataset):,}")
    print(f"M2 features: {X.shape[1]:,}")
    print(f"Default rate: {y.mean():.4%}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Class weight: {args.class_weight}")

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof = np.zeros(len(X), dtype="float32")
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n=== FOLD {fold}/{args.cv_folds} ===")
        model = make_model(feature_names, args.class_weight)
        model.fit(X.iloc[train_idx], y[train_idx])
        valid_prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = valid_prob.astype("float32")

        metrics = ranking_metrics(y[valid_idx], valid_prob)
        fold_row: dict[str, float | int] = {"fold": fold, **metrics}
        fold_rows.append(fold_row)
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")

    fold_metrics = pd.DataFrame(fold_rows)
    summary: dict[str, float | int | str] = {
        "feature_version": "m2_typed_aggregation_v1",
        "max_files": args.max_files,
        "cv_folds": args.cv_folds,
        "class_weight": args.class_weight,
        "n_applications": int(len(dataset)),
        "n_features": int(X.shape[1]),
        "default_rate": float(y.mean()),
    }
    for column in fold_metrics.columns:
        if column == "fold":
            continue
        summary[f"cv_{column}_mean"] = float(fold_metrics[column].mean())
        summary[f"cv_{column}_std"] = float(fold_metrics[column].std(ddof=1))

    print("\n=== CV SUMMARY ===")
    for name, value in summary.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    final_model = make_model(feature_names, args.class_weight)
    print("\nFitting final M2 model on all selected applications...")
    final_model.fit(X, y)

    joblib.dump(final_model, ARTIFACT_DIR / "logreg_m2.joblib")
    fold_metrics.to_csv(ARTIFACT_DIR / "m2_cv_folds.csv", index=False)
    pd.DataFrame({"id": ids, "flag": y, "oof_probability": oof}).to_parquet(
        ARTIFACT_DIR / "m2_oof_predictions.parquet",
        index=False,
    )
    (ARTIFACT_DIR / "m2_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.Series(feature_names, name="feature").to_csv(
        ARTIFACT_DIR / "m2_features.csv", index=False
    )

    registry_path = ARTIFACT_DIR / "experiments.csv"
    registry_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    new_row = pd.DataFrame([registry_row])
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        registry = pd.concat([registry, new_row], ignore_index=True)
    else:
        registry = new_row
    registry.to_csv(registry_path, index=False)

    print("\nSaved:")
    for path in (
        ARTIFACT_DIR / "logreg_m2.joblib",
        ARTIFACT_DIR / "m2_metrics.json",
        ARTIFACT_DIR / "m2_cv_folds.csv",
        ARTIFACT_DIR / "m2_oof_predictions.parquet",
        ARTIFACT_DIR / "m2_features.csv",
        ARTIFACT_DIR / "experiments.csv",
    ):
        print("-", path)


if __name__ == "__main__":
    main()
