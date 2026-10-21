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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from credit_risk.data import load_target
from credit_risk.metrics import ranking_metrics
from credit_risk.modeling import balanced_binary_class_weights, make_logistic_pipeline

DATA_DIR = Path("data/raw")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42
CANONICAL_VARIANT = "e_typed_catdist_payment"

STRUCTURAL_ZERO_PREFIXES = (
    "enc_loans_account_holder_type__frac__",
    "enc_loans_credit_status__frac__",
    "enc_loans_credit_type__frac__",
    "enc_loans_account_cur__frac__",
    "payment_overall_state__frac__",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M3 controlled model-family benchmark: Logistic Regression vs CatBoost "
            "on the canonical M2.1 feature representation."
        )
    )
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument("--catboost-iterations", type=int, default=1200)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--thread-count",
        type=int,
        default=-1,
        help="CatBoost CPU threads; -1 lets CatBoost choose.",
    )
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def load_canonical_features(max_files: int) -> pd.DataFrame:
    paths = sorted(FEATURE_DIR.glob("train_data_*.parquet"), key=_chunk_index)
    paths = paths[:max_files]
    if len(paths) < max_files:
        raise FileNotFoundError(
            f"Requested {max_files} feature chunk(s), but found {len(paths)} under {FEATURE_DIR}. "
            "Run scripts/train_m21_ablation.py with the same --max-files first."
        )

    frames = [pd.read_parquet(path) for path in paths]
    features = pd.concat(frames, ignore_index=True, sort=False)

    zero_cols = [
        column
        for column in features.columns
        if column.startswith(STRUCTURAL_ZERO_PREFIXES)
    ]
    if zero_cols:
        features[zero_cols] = features[zero_cols].fillna(0.0)

    duplicates = int(features["id"].duplicated().sum())
    if duplicates:
        raise RuntimeError(f"Found {duplicates:,} duplicate application ids.")
    return features


def build_population(max_files: int) -> tuple[pd.DataFrame, list[str]]:
    target = load_target(DATA_DIR)
    features = load_canonical_features(max_files)
    dataset = (
        target.merge(features, on="id", how="inner", validate="one_to_one")
        .sort_values("id")
        .reset_index(drop=True)
    )
    if dataset.empty:
        raise RuntimeError("Target/features merge produced zero rows.")
    feature_names = [column for column in dataset.columns if column not in {"id", "flag"}]
    return dataset, feature_names


def deterministic_folds(y: np.ndarray, n_splits: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    result = np.full(len(y), -1, dtype="int8")
    dummy = np.zeros(len(y), dtype="int8")
    for fold, (_, valid_idx) in enumerate(cv.split(dummy, y), start=1):
        result[valid_idx] = fold
    if np.any(result < 0):
        raise RuntimeError("Some rows were not assigned to a fold.")
    return result


def get_frozen_folds(population: pd.DataFrame, cv_folds: int) -> tuple[np.ndarray, str]:
    """Reuse the M2.1 assignments when they exactly match the M3 population."""
    m21_path = ARTIFACT_DIR / "m21_cv_assignments.parquet"
    if m21_path.exists():
        saved = pd.read_parquet(m21_path)
        required = {"id", "flag", "cv_fold"}
        if required.issubset(saved.columns):
            saved = saved[["id", "flag", "cv_fold"]].sort_values("id").reset_index(drop=True)
            current = population[["id", "flag"]].sort_values("id").reset_index(drop=True)
            same_population = (
                len(saved) == len(current)
                and saved["id"].equals(current["id"])
                and saved["flag"].equals(current["flag"])
            )
            same_fold_count = saved["cv_fold"].nunique() == cv_folds
            if same_population and same_fold_count:
                return saved["cv_fold"].to_numpy(dtype="int8"), "m21_frozen"

    y = population["flag"].to_numpy(dtype="int8")
    return deterministic_folds(y, cv_folds), "m3_deterministic_fallback"


def make_catboost(
    class_weights: list[float] | None,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    thread_count: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=thread_count,
    )


def evaluate_logistic(
    X: pd.DataFrame,
    y: np.ndarray,
    fold_id: np.ndarray,
    class_weight_mode: str,
) -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    rows: list[dict[str, float | int | str]] = []
    oof = np.zeros(len(X), dtype="float32")

    print("\n" + "=" * 72)
    print("MODEL: logistic_regression")

    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)
        weights = (
            balanced_binary_class_weights(y[train_idx])
            if class_weight_mode == "balanced"
            else None
        )
        model = make_logistic_pipeline(weights)
        model.fit(X.iloc[train_idx], y[train_idx])
        prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = prob.astype("float32")
        metrics = ranking_metrics(y[valid_idx], prob)
        rows.append({
            "model": "logistic_regression",
            "fold": int(fold),
            "trees": np.nan,
            **metrics,
        })
        print(
            f"fold {fold}: AUC={metrics['roc_auc']:.6f} | "
            f"Gini={metrics['gini']:.6f} | AP={metrics['average_precision']:.6f} | "
            f"KS={metrics['ks']:.6f} | Lift@10={metrics['lift_top_10pct']:.4f}"
        )
        del model
        gc.collect()

    return rows, oof


def evaluate_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    fold_id: np.ndarray,
    class_weight_mode: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, list[int]]:
    rows: list[dict[str, float | int | str]] = []
    oof = np.zeros(len(X), dtype="float32")
    tree_counts: list[int] = []

    print("\n" + "=" * 72)
    print("MODEL: catboost")

    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)

        # Same missing-value treatment as Logistic Regression for a clean model-family comparison.
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_valid = imputer.transform(X.iloc[valid_idx])

        weights = (
            balanced_binary_class_weights(y[train_idx]).as_catboost_list()
            if class_weight_mode == "balanced"
            else None
        )
        model = make_catboost(
            weights,
            iterations=args.catboost_iterations,
            depth=args.catboost_depth,
            learning_rate=args.catboost_learning_rate,
            l2_leaf_reg=args.catboost_l2_leaf_reg,
            thread_count=args.thread_count,
        )
        model.fit(
            X_train,
            y[train_idx],
            eval_set=(X_valid, y[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=False,
        )
        prob = model.predict_proba(X_valid)[:, 1]
        oof[valid_idx] = prob.astype("float32")
        tree_count = int(model.tree_count_)
        tree_counts.append(tree_count)
        metrics = ranking_metrics(y[valid_idx], prob)
        rows.append({
            "model": "catboost",
            "fold": int(fold),
            "trees": tree_count,
            **metrics,
        })
        print(
            f"fold {fold}: AUC={metrics['roc_auc']:.6f} | "
            f"Gini={metrics['gini']:.6f} | AP={metrics['average_precision']:.6f} | "
            f"KS={metrics['ks']:.6f} | Lift@10={metrics['lift_top_10pct']:.4f} | "
            f"trees={tree_count}"
        )

        del model, imputer, X_train, X_valid
        gc.collect()

    return rows, oof, tree_counts


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in {"model", "fold", "trees"}
    ]
    rows: list[dict[str, float | int | str]] = []
    for model_name, group in fold_metrics.groupby("model", sort=False):
        row: dict[str, float | int | str] = {"model": model_name}
        for metric in metric_columns:
            row[f"cv_{metric}_mean"] = float(group[metric].mean())
            row[f"cv_{metric}_std"] = float(group[metric].std(ddof=1))
        valid_trees = group["trees"].dropna()
        row["median_best_trees"] = (
            int(np.median(valid_trees.to_numpy())) if len(valid_trees) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def fit_final_models(
    X: pd.DataFrame,
    y: np.ndarray,
    class_weight_mode: str,
    catboost_trees: int,
    args: argparse.Namespace,
    feature_names: list[str],
) -> None:
    final_weights = (
        balanced_binary_class_weights(y) if class_weight_mode == "balanced" else None
    )

    logreg = make_logistic_pipeline(final_weights)
    logreg.fit(X, y)
    joblib.dump(logreg, ARTIFACT_DIR / "m3_logreg.joblib")

    # CatBoost final model uses the median best tree count discovered only inside CV.
    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X)
    catboost = make_catboost(
        None if final_weights is None else final_weights.as_catboost_list(),
        iterations=catboost_trees,
        depth=args.catboost_depth,
        learning_rate=args.catboost_learning_rate,
        l2_leaf_reg=args.catboost_l2_leaf_reg,
        thread_count=args.thread_count,
    )
    catboost.fit(X_all, y, verbose=False)
    catboost.save_model(ARTIFACT_DIR / "m3_catboost.cbm")
    joblib.dump(imputer, ARTIFACT_DIR / "m3_catboost_imputer.joblib")

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": catboost.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACT_DIR / "m3_catboost_feature_importance.csv", index=False)


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, feature_names = build_population(args.max_files)
    y = dataset["flag"].to_numpy(dtype="int8")
    X = dataset[feature_names]
    fold_id, fold_source = get_frozen_folds(dataset, args.cv_folds)

    print("=== M3 CONTROLLED MODEL BENCHMARK ===")
    print(f"Feature representation: {CANONICAL_VARIANT}")
    print(f"Applications: {len(dataset):,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Default rate: {y.mean():.4%}")
    print(f"Shared CV folds: {args.cv_folds} ({fold_source})")
    print(f"Class weight: {args.class_weight}")
    print(
        "CatBoost config: "
        f"max_iter={args.catboost_iterations}, depth={args.catboost_depth}, "
        f"learning_rate={args.catboost_learning_rate}, "
        f"l2_leaf_reg={args.catboost_l2_leaf_reg}, "
        f"early_stopping={args.early_stopping_rounds}"
    )

    log_rows, log_oof = evaluate_logistic(X, y, fold_id, args.class_weight)
    cat_rows, cat_oof, tree_counts = evaluate_catboost(
        X, y, fold_id, args.class_weight, args
    )

    fold_metrics = pd.DataFrame([*log_rows, *cat_rows])
    summary = summarize(fold_metrics)

    print("\n=== M3 MODEL COMPARISON (sorted by ROC-AUC) ===")
    display = [
        "model",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_gini_mean",
        "cv_average_precision_mean",
        "cv_ks_mean",
        "cv_lift_top_10pct_mean",
        "cv_capture_rate_top_10pct_mean",
        "median_best_trees",
    ]
    print(summary[display].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    by_model = summary.set_index("model")
    auc_delta = float(
        by_model.loc["catboost", "cv_roc_auc_mean"]
        - by_model.loc["logistic_regression", "cv_roc_auc_mean"]
    )
    ap_delta = float(
        by_model.loc["catboost", "cv_average_precision_mean"]
        - by_model.loc["logistic_regression", "cv_average_precision_mean"]
    )
    lift_delta = float(
        by_model.loc["catboost", "cv_lift_top_10pct_mean"]
        - by_model.loc["logistic_regression", "cv_lift_top_10pct_mean"]
    )
    print("\n=== CATBOOST - LOGISTIC DELTA ===")
    print(f"ROC-AUC: {auc_delta:+.6f}")
    print(f"Average Precision: {ap_delta:+.6f}")
    print(f"Lift@10: {lift_delta:+.6f}")

    median_trees = max(1, int(np.median(tree_counts)))
    print(f"\nFitting final models; CatBoost iterations={median_trees} (CV median)...")
    fit_final_models(
        X,
        y,
        args.class_weight,
        median_trees,
        args,
        feature_names,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    summary.insert(0, "feature_variant", CANONICAL_VARIANT)
    summary.insert(0, "n_features", len(feature_names))
    summary.insert(0, "n_applications", len(dataset))
    summary.insert(0, "max_files", args.max_files)
    summary.to_csv(ARTIFACT_DIR / "m3_model_comparison.csv", index=False)
    fold_metrics.to_csv(ARTIFACT_DIR / "m3_folds.csv", index=False)

    oof = dataset[["id", "flag"]].copy()
    oof["cv_fold"] = fold_id
    oof["oof__logistic_regression"] = log_oof
    oof["oof__catboost"] = cat_oof
    oof.to_parquet(ARTIFACT_DIR / "m3_oof_predictions.parquet", index=False)

    dataset[["id", "flag"]].assign(cv_fold=fold_id).to_parquet(
        ARTIFACT_DIR / "m3_cv_assignments.parquet", index=False
    )
    pd.DataFrame({"feature": feature_names}).to_csv(
        ARTIFACT_DIR / "m3_features.csv", index=False
    )

    metadata = {
        "timestamp_utc": timestamp,
        "stage": "M3",
        "experiment": "controlled_model_family_benchmark",
        "feature_variant": CANONICAL_VARIANT,
        "max_files": args.max_files,
        "cv_folds": args.cv_folds,
        "fold_source": fold_source,
        "random_state": RANDOM_STATE,
        "class_weight": args.class_weight,
        "n_applications": int(len(dataset)),
        "n_features": int(len(feature_names)),
        "default_rate": float(y.mean()),
        "catboost": {
            "max_iterations": args.catboost_iterations,
            "depth": args.catboost_depth,
            "learning_rate": args.catboost_learning_rate,
            "l2_leaf_reg": args.catboost_l2_leaf_reg,
            "early_stopping_rounds": args.early_stopping_rounds,
            "fold_tree_counts": tree_counts,
            "final_iterations": median_trees,
        },
        "delta_catboost_minus_logistic": {
            "roc_auc": auc_delta,
            "average_precision": ap_delta,
            "lift_top_10pct": lift_delta,
        },
    }
    (ARTIFACT_DIR / "m3_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    registry_path = ARTIFACT_DIR / "experiments.csv"
    registry_rows = summary.copy()
    registry_rows.insert(0, "timestamp_utc", timestamp)
    registry_rows.insert(1, "stage", "M3")
    if registry_path.exists():
        old = pd.read_csv(registry_path)
        registry = pd.concat([old, registry_rows], ignore_index=True, sort=False)
    else:
        registry = registry_rows
    registry.to_csv(registry_path, index=False)

    print("\nSaved:")
    for path in (
        ARTIFACT_DIR / "m3_model_comparison.csv",
        ARTIFACT_DIR / "m3_folds.csv",
        ARTIFACT_DIR / "m3_oof_predictions.parquet",
        ARTIFACT_DIR / "m3_cv_assignments.parquet",
        ARTIFACT_DIR / "m3_features.csv",
        ARTIFACT_DIR / "m3_logreg.joblib",
        ARTIFACT_DIR / "m3_catboost.cbm",
        ARTIFACT_DIR / "m3_catboost_imputer.joblib",
        ARTIFACT_DIR / "m3_catboost_feature_importance.csv",
        ARTIFACT_DIR / "m3_metadata.json",
        ARTIFACT_DIR / "experiments.csv",
    ):
        print("-", path)


if __name__ == "__main__":
    main()
