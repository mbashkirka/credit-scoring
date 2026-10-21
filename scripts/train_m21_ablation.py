from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import iter_train_history, load_target
from credit_risk.features import (
    aggregate_category_distributions,
    aggregate_credit_history,
    aggregate_extended_payment_dynamics,
    aggregate_legacy_credit_history,
    join_feature_blocks,
)
from credit_risk.metrics import ranking_metrics

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/app_features_m21")
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42

VARIANTS = (
    "a_legacy_m1",
    "b_legacy_no_latest",
    "c_typed_m2",
    "d_typed_catdist",
    "e_typed_catdist_payment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2.1 controlled feature-ablation experiment."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Number of raw train parquet chunks. Keep 1 for the first M2.1 run.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified CV folds shared by every feature variant.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Ignore all M2.1 feature caches and rebuild from raw parquet.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Optional subset of ablation variants to evaluate.",
    )
    return parser.parse_args()


def _cache_path(variant: str, raw_stem: str) -> Path:
    return PROCESSED_DIR / variant / f"{raw_stem}.parquet"


def _save_feature_block(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(
        f"  cached {path.parent.name}: {len(frame):,} applications x "
        f"{frame.shape[1] - 1:,} features"
    )


def ensure_feature_caches(max_files: int, rebuild: bool) -> None:
    """Build all nested M2.1 feature sets once per raw chunk.

    The feature variants are nested so we can reuse blocks instead of doing five
    independent expensive raw aggregations.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for raw_path, history in iter_train_history(DATA_DIR, max_files=max_files):
        paths = {variant: _cache_path(variant, raw_path.stem) for variant in VARIANTS}
        if not rebuild and all(path.exists() for path in paths.values()):
            print(f"All M2.1 caches exist for {raw_path.name}; skipping aggregation")
            del history
            gc.collect()
            continue

        print(f"\nBuilding M2.1 feature variants for {raw_path.name}")

        # A/B: exactly reproduce M1 and remove only the controversial latest-by-rn block.
        if rebuild or not paths["a_legacy_m1"].exists() or not paths["b_legacy_no_latest"].exists():
            legacy = aggregate_legacy_credit_history(history, include_latest=True)
            _save_feature_block(legacy, paths["a_legacy_m1"])
            no_latest_cols = [column for column in legacy.columns if not column.endswith("__latest")]
            _save_feature_block(legacy[no_latest_cols].copy(), paths["b_legacy_no_latest"])
            del legacy
            gc.collect()

        # C: current typed M2 representation.
        if rebuild or not paths["c_typed_m2"].exists():
            typed = aggregate_credit_history(history)
            _save_feature_block(typed, paths["c_typed_m2"])
            del typed
            gc.collect()

        # D: M2 + proper distributions for the four encoded loan categories.
        if rebuild or not paths["d_typed_catdist"].exists():
            typed = pd.read_parquet(paths["c_typed_m2"])
            catdist = aggregate_category_distributions(history)
            typed_catdist = join_feature_blocks(typed, catdist)
            _save_feature_block(typed_catdist, paths["d_typed_catdist"])
            del typed, catdist, typed_catdist
            gc.collect()

        # E: D + category-code-safe, direction-agnostic payment dynamics.
        if rebuild or not paths["e_typed_catdist_payment"].exists():
            typed_catdist = pd.read_parquet(paths["d_typed_catdist"])
            payment = aggregate_extended_payment_dynamics(history)
            full = join_feature_blocks(typed_catdist, payment)
            _save_feature_block(full, paths["e_typed_catdist_payment"])
            del typed_catdist, payment, full
            gc.collect()

        del history
        gc.collect()


def load_variant_features(variant: str, max_files: int) -> pd.DataFrame:
    paths = sorted((PROCESSED_DIR / variant).glob("train_data_*.parquet"))
    if max_files is not None:
        # Lexicographic filenames are not guaranteed to match numeric order;
        # parse the final suffix so this mirrors the raw-data iterator semantics.
        paths = sorted(
            paths,
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )[:max_files]
    if not paths:
        raise FileNotFoundError(f"No cached feature files found for {variant}.")

    frames = [pd.read_parquet(path) for path in paths]
    features = pd.concat(frames, ignore_index=True, sort=False)

    # Structural absence of a category level in one raw chunk means zero
    # frequency for applications from that chunk, not an unknown measurement.
    structural_zero_prefixes = (
        "enc_loans_account_holder_type__frac__",
        "enc_loans_credit_status__frac__",
        "enc_loans_credit_type__frac__",
        "enc_loans_account_cur__frac__",
        "payment_overall_state__frac__",
    )
    zero_cols = [
        column
        for column in features.columns
        if column.startswith(structural_zero_prefixes)
    ]
    if zero_cols:
        features[zero_cols] = features[zero_cols].fillna(0.0)

    duplicates = int(features["id"].duplicated().sum())
    if duplicates:
        raise RuntimeError(
            f"Found {duplicates:,} duplicate application ids across parquet chunks."
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


def build_master_population(target: pd.DataFrame, max_files: int) -> pd.DataFrame:
    reference = load_variant_features("c_typed_m2", max_files)[["id"]]
    population = (
        target.merge(reference, on="id", how="inner", validate="one_to_one")
        .sort_values("id")
        .reset_index(drop=True)
    )
    if population.empty:
        raise RuntimeError("Target/history merge produced zero rows.")
    return population


def shared_folds(y: np.ndarray, n_splits: int) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_id = np.full(len(y), -1, dtype="int8")
    dummy = np.zeros(len(y), dtype="int8")
    for fold, (_, valid_idx) in enumerate(cv.split(dummy, y), start=1):
        fold_id[valid_idx] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("Some rows were not assigned to a CV fold.")
    return fold_id


def evaluate_variant(
    variant: str,
    target_population: pd.DataFrame,
    fold_id: np.ndarray,
    max_files: int,
    class_weight: str,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, list[str]]:
    features = load_variant_features(variant, max_files)
    dataset = target_population.merge(features, on="id", how="left", validate="one_to_one")
    if len(dataset) != len(target_population):
        raise RuntimeError(f"Population mismatch for {variant}.")

    y = dataset["flag"].to_numpy(dtype="int8")
    X = dataset.drop(columns=["id", "flag"])
    feature_names = X.columns.tolist()
    oof = np.zeros(len(X), dtype="float32")
    rows: list[dict[str, float | int | str]] = []

    print(f"\n{'=' * 72}")
    print(f"VARIANT: {variant}")
    print(f"features: {len(feature_names):,}")

    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)
        model = make_model(feature_names, class_weight)
        model.fit(X.iloc[train_idx], y[train_idx])
        valid_prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = valid_prob.astype("float32")
        metrics = ranking_metrics(y[valid_idx], valid_prob)
        row: dict[str, float | int | str] = {
            "variant": variant,
            "fold": int(fold),
            "n_features": len(feature_names),
            **metrics,
        }
        rows.append(row)
        print(
            f"fold {fold}: AUC={metrics['roc_auc']:.6f} | "
            f"Gini={metrics['gini']:.6f} | AP={metrics['average_precision']:.6f} | "
            f"KS={metrics['ks']:.6f} | Lift@10={metrics['lift_top_10pct']:.4f}"
        )

    del features, dataset, X
    gc.collect()
    return rows, oof, feature_names


def summarize(fold_metrics: pd.DataFrame, population: pd.DataFrame, max_files: int, cv_folds: int, class_weight: str) -> pd.DataFrame:
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in {"variant", "fold", "n_features"}
    ]
    summary_rows = []
    for variant, group in fold_metrics.groupby("variant", sort=False):
        row: dict[str, float | int | str] = {
            "variant": variant,
            "max_files": max_files,
            "cv_folds": cv_folds,
            "class_weight": class_weight,
            "n_applications": len(population),
            "n_features": int(group["n_features"].iloc[0]),
            "default_rate": float(population["flag"].mean()),
        }
        for metric in metric_columns:
            row[f"cv_{metric}_mean"] = float(group[metric].mean())
            row[f"cv_{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return summary.sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_feature_caches(args.max_files, args.rebuild_features)
    target = load_target(DATA_DIR)
    population = build_master_population(target, args.max_files)
    y = population["flag"].to_numpy(dtype="int8")
    fold_id = shared_folds(y, args.cv_folds)

    print("\n=== M2.1 CONTROLLED ABLATION ===")
    print(f"Applications: {len(population):,}")
    print(f"Default rate: {y.mean():.4%}")
    print(f"Shared CV folds: {args.cv_folds}")
    print(f"Class weight: {args.class_weight}")
    print("Variants:", ", ".join(args.variants))

    all_rows: list[dict[str, float | int | str]] = []
    oof_frame = population[["id", "flag"]].copy()
    feature_manifest_rows: list[dict[str, str]] = []

    for variant in args.variants:
        rows, oof, features = evaluate_variant(
            variant,
            population,
            fold_id,
            args.max_files,
            args.class_weight,
        )
        all_rows.extend(rows)
        oof_frame[f"oof__{variant}"] = oof
        feature_manifest_rows.extend(
            {"variant": variant, "feature": feature} for feature in features
        )

    fold_metrics = pd.DataFrame(all_rows)
    summary = summarize(
        fold_metrics,
        population,
        args.max_files,
        args.cv_folds,
        args.class_weight,
    )

    print("\n=== M2.1 ABLATION SUMMARY (sorted by ROC-AUC) ===")
    display_cols = [
        "variant",
        "n_features",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_gini_mean",
        "cv_average_precision_mean",
        "cv_ks_mean",
        "cv_lift_top_10pct_mean",
        "cv_capture_rate_top_10pct_mean",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    best = summary.iloc[0]
    print(
        f"\nBest M2.1 variant: {best['variant']} "
        f"(AUC={best['cv_roc_auc_mean']:.6f} ± {best['cv_roc_auc_std']:.6f})"
    )

    fold_metrics.to_csv(ARTIFACT_DIR / "m21_ablation_folds.csv", index=False)
    summary.to_csv(ARTIFACT_DIR / "m21_ablation_summary.csv", index=False)
    oof_frame.to_parquet(ARTIFACT_DIR / "m21_ablation_oof.parquet", index=False)
    pd.DataFrame(feature_manifest_rows).to_csv(
        ARTIFACT_DIR / "m21_feature_manifest.csv", index=False
    )
    population.assign(cv_fold=fold_id).to_parquet(
        ARTIFACT_DIR / "m21_cv_assignments.parquet", index=False
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "M2.1",
        "experiment": "controlled_feature_ablation",
        "max_files": args.max_files,
        "cv_folds": args.cv_folds,
        "random_state": RANDOM_STATE,
        "class_weight": args.class_weight,
        "n_applications": int(len(population)),
        "default_rate": float(y.mean()),
        "best_variant": str(best["variant"]),
        "best_cv_roc_auc_mean": float(best["cv_roc_auc_mean"]),
        "best_cv_roc_auc_std": float(best["cv_roc_auc_std"]),
    }
    (ARTIFACT_DIR / "m21_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    registry_path = ARTIFACT_DIR / "experiments.csv"
    registry_rows = summary.copy()
    registry_rows.insert(0, "timestamp_utc", metadata["timestamp_utc"])
    registry_rows.insert(1, "stage", "M2.1")
    if registry_path.exists():
        old = pd.read_csv(registry_path)
        registry = pd.concat([old, registry_rows], ignore_index=True, sort=False)
    else:
        registry = registry_rows
    registry.to_csv(registry_path, index=False)

    print("\nSaved:")
    for path in (
        ARTIFACT_DIR / "m21_ablation_summary.csv",
        ARTIFACT_DIR / "m21_ablation_folds.csv",
        ARTIFACT_DIR / "m21_ablation_oof.parquet",
        ARTIFACT_DIR / "m21_cv_assignments.parquet",
        ARTIFACT_DIR / "m21_feature_manifest.csv",
        ARTIFACT_DIR / "m21_metadata.json",
        ARTIFACT_DIR / "experiments.csv",
    ):
        print("-", path)


if __name__ == "__main__":
    main()
