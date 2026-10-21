from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import load_target, load_train_history
from credit_risk.features import aggregate_legacy_credit_history

DATA_DIR = Path("data/raw")
ARTIFACT_DIR = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Number of train parquet chunks to load. Start with 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    target = load_target(DATA_DIR)
    history = load_train_history(DATA_DIR, max_files=args.max_files)

    print("Aggregating history...")
    features = aggregate_legacy_credit_history(history, include_latest=True)

    dataset = target.merge(features, on="id", how="inner")
    if dataset.empty:
        raise RuntimeError("Target/history merge produced zero rows.")

    y = dataset["flag"].astype(int)
    X = dataset.drop(columns=["id", "flag"])

    print(f"Applications used: {len(dataset):,}")
    print(f"Features: {X.shape[1]:,}")
    print(f"Default rate: {y.mean():.4%}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X.columns.tolist()

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
                numeric_features,
            )
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training logistic-regression baseline...")
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "max_files": args.max_files,
        "n_applications": int(len(dataset)),
        "n_features": int(X.shape[1]),
        "default_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_valid, valid_prob)),
        "average_precision": float(average_precision_score(y_valid, valid_prob)),
    }

    print("\n=== VALIDATION METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    joblib.dump(model, ARTIFACT_DIR / "logreg_baseline.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.Series(X.columns, name="feature").to_csv(
        ARTIFACT_DIR / "features.csv",
        index=False,
    )

    print("\nSaved:")
    print("-", ARTIFACT_DIR / "logreg_baseline.joblib")
    print("-", ARTIFACT_DIR / "metrics.json")
    print("-", ARTIFACT_DIR / "features.csv")


if __name__ == "__main__":
    main()
