from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

from credit_risk.data import load_target
from credit_risk.metrics import ranking_metrics
from credit_risk.modeling import balanced_binary_class_weights
from credit_risk.statistics import paired_stratified_bootstrap_ranking_delta

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


@dataclass(frozen=True)
class CatBoostConfig:
    name: str
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.0
    random_strength: float | None = None
    bootstrap_type: str | None = None
    bagging_temperature: float | None = None

    def model_kwargs(self) -> dict[str, object]:
        result: dict[str, object] = {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
        }
        if self.random_strength is not None:
            result["random_strength"] = self.random_strength
        if self.bootstrap_type is not None:
            result["bootstrap_type"] = self.bootstrap_type
        if self.bagging_temperature is not None:
            result["bagging_temperature"] = self.bagging_temperature
        return result


def tuning_configs() -> list[CatBoostConfig]:
    """Small, interpretable one-factor-at-a-time tuning set around the M3 baseline."""
    return [
        CatBoostConfig("baseline_d6_lr005_l2_5"),
        CatBoostConfig("shallower_d5", depth=5),
        CatBoostConfig("deeper_d7", depth=7),
        CatBoostConfig("deeper_d8", depth=8),
        CatBoostConfig("slower_lr003", learning_rate=0.03),
        CatBoostConfig("faster_lr008", learning_rate=0.08),
        CatBoostConfig("stronger_l2_10", l2_leaf_reg=10.0),
        CatBoostConfig("lower_random_strength_05", random_strength=0.5),
        CatBoostConfig("higher_random_strength_2", random_strength=2.0),
        CatBoostConfig(
            "bayesian_bagging_temp_1",
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M3.1 controlled CatBoost tuning on frozen M3 folds plus paired "
            "stratified bootstrap of OOF model deltas."
        )
    )
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--catboost-iterations", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional prefix of the tuning set, useful only for smoke tests.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    return parser.parse_args()


def _chunk_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def load_population(max_files: int) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(FEATURE_DIR.glob("train_data_*.parquet"), key=_chunk_index)[:max_files]
    if len(paths) < max_files:
        raise FileNotFoundError(
            f"Requested {max_files} feature chunk(s), but found {len(paths)} under {FEATURE_DIR}. "
            "Run scripts/train_m21_ablation.py with the same --max-files first."
        )
    features = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)
    zero_cols = [c for c in features.columns if c.startswith(STRUCTURAL_ZERO_PREFIXES)]
    if zero_cols:
        features[zero_cols] = features[zero_cols].fillna(0.0)
    if features["id"].duplicated().any():
        raise RuntimeError("Duplicate application ids in canonical feature cache.")

    target = load_target(DATA_DIR)
    dataset = (
        target.merge(features, on="id", how="inner", validate="one_to_one")
        .sort_values("id")
        .reset_index(drop=True)
    )
    feature_names = [c for c in dataset.columns if c not in {"id", "flag"}]
    return dataset, feature_names


def load_frozen_folds(dataset: pd.DataFrame, cv_folds: int) -> np.ndarray:
    for path in (ARTIFACT_DIR / "m3_cv_assignments.parquet", ARTIFACT_DIR / "m21_cv_assignments.parquet"):
        if not path.exists():
            continue
        saved = pd.read_parquet(path)[["id", "flag", "cv_fold"]].sort_values("id").reset_index(drop=True)
        current = dataset[["id", "flag"]].sort_values("id").reset_index(drop=True)
        same = (
            len(saved) == len(current)
            and saved["id"].equals(current["id"])
            and saved["flag"].equals(current["flag"])
            and saved["cv_fold"].nunique() == cv_folds
        )
        if same:
            return saved["cv_fold"].to_numpy(dtype="int8")
    raise RuntimeError(
        "Frozen M3/M2.1 fold assignments do not match this population. "
        "Run M3 first with the same --max-files/--cv-folds; M3.1 intentionally refuses to create new folds."
    )


def make_catboost(
    config: CatBoostConfig,
    *,
    iterations: int,
    class_weights: list[float] | None,
    thread_count: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=thread_count,
        **config.model_kwargs(),
    )


def summarize(folds: pd.DataFrame, config_lookup: dict[str, CatBoostConfig]) -> pd.DataFrame:
    metric_cols = [
        "roc_auc",
        "gini",
        "average_precision",
        "ks",
        "capture_rate_top_10pct",
        "lift_top_10pct",
        "precision_top_10pct",
    ]
    rows: list[dict[str, object]] = []
    for name, group in folds.groupby("config", sort=False):
        cfg = config_lookup[name]
        row: dict[str, object] = {"config": name, **asdict(cfg)}
        row.pop("name", None)
        for metric in metric_cols:
            row[f"cv_{metric}_mean"] = float(group[metric].mean())
            row[f"cv_{metric}_std"] = float(group[metric].std(ddof=1))
        row["median_best_trees"] = int(np.median(group["trees"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def load_m3_oof(dataset: pd.DataFrame, fold_id: np.ndarray) -> pd.DataFrame:
    path = ARTIFACT_DIR / "m3_oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("artifacts/m3_oof_predictions.parquet is required. Run M3 first.")
    oof = pd.read_parquet(path).sort_values("id").reset_index(drop=True)
    current = dataset[["id", "flag"]].sort_values("id").reset_index(drop=True)
    required = {"id", "flag", "cv_fold", "oof__logistic_regression", "oof__catboost"}
    if not required.issubset(oof.columns):
        raise RuntimeError(f"M3 OOF file misses columns: {sorted(required - set(oof.columns))}")
    if not (
        len(oof) == len(current)
        and oof["id"].equals(current["id"])
        and oof["flag"].equals(current["flag"])
        and np.array_equal(oof["cv_fold"].to_numpy(), fold_id)
    ):
        raise RuntimeError("M3 OOF population/folds do not match M3.1 population.")
    return oof


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configs = tuning_configs()
    if args.max_configs is not None:
        if args.max_configs < 1:
            raise ValueError("--max-configs must be >= 1")
        configs = configs[:args.max_configs]

    dataset, feature_names = load_population(args.max_files)
    y = dataset["flag"].to_numpy(dtype="int8")
    fold_id = load_frozen_folds(dataset, args.cv_folds)
    X = dataset[feature_names]

    # One global imputation is NOT used for model evaluation because fitting a median
    # on all rows would leak validation-distribution information. We process one fold
    # at a time: fit its imputer once, reuse those matrices across all configurations,
    # then release them before moving to the next fold.
    print("=== M3.1 CONTROLLED CATBOOST TUNING ===")
    print(f"Feature representation: {CANONICAL_VARIANT}")
    print(f"Applications: {len(dataset):,}")
    print(f"Features: {len(feature_names):,}")
    print(f"Default rate: {y.mean():.4%}")
    print(f"Frozen CV folds: {args.cv_folds}")
    print(f"Configs: {len(configs)}")
    print(f"Class weight: {args.class_weight}")
    print(f"Max iterations: {args.catboost_iterations}; early stopping: {args.early_stopping_rounds}")

    all_rows: list[dict[str, object]] = []
    config_oof = {config.name: np.zeros(len(y), dtype="float32") for config in configs}
    config_trees = {config.name: [] for config in configs}

    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        valid_idx = np.flatnonzero(fold_id == fold)
        imputer = SimpleImputer(strategy="median")
        # Keep the same float precision as M3 so the baseline config is directly reproducible.
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_valid = imputer.transform(X.iloc[valid_idx])
        weights = (
            balanced_binary_class_weights(y[train_idx]).as_catboost_list()
            if args.class_weight == "balanced"
            else None
        )

        print("\n" + "#" * 80)
        print(f"FOLD {fold}/{args.cv_folds}")
        for config_index, config in enumerate(configs, start=1):
            print(f"[{config_index}/{len(configs)}] {config.name}: ", end="", flush=True)
            model = make_catboost(
                config,
                iterations=args.catboost_iterations,
                class_weights=weights,
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
            config_oof[config.name][valid_idx] = prob.astype("float32")
            tree_count = int(model.tree_count_)
            config_trees[config.name].append(tree_count)
            metrics = ranking_metrics(y[valid_idx], prob)
            all_rows.append({
                "config": config.name,
                "fold": int(fold),
                "trees": tree_count,
                **config.model_kwargs(),
                **metrics,
            })
            print(
                f"AUC={metrics['roc_auc']:.6f} | AP={metrics['average_precision']:.6f} | "
                f"KS={metrics['ks']:.6f} | Lift@10={metrics['lift_top_10pct']:.4f} | trees={tree_count}"
            )
            del model
            gc.collect()

        del imputer, X_train, X_valid
        gc.collect()

    fold_metrics = pd.DataFrame(all_rows)
    config_lookup = {c.name: c for c in configs}
    summary = summarize(fold_metrics, config_lookup)
    best_name = str(summary.iloc[0]["config"])
    best_cfg = config_lookup[best_name]
    baseline_name = configs[0].name

    print("\n=== M3.1 TUNING SUMMARY (sorted by ROC-AUC) ===")
    display_cols = [
        "config",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "bootstrap_type",
        "bagging_temperature",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_average_precision_mean",
        "cv_ks_mean",
        "cv_lift_top_10pct_mean",
        "median_best_trees",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    best = summary.set_index("config").loc[best_name]
    baseline = summary.set_index("config").loc[baseline_name]
    print("\n=== BEST TUNED - M3 BASELINE-CONFIG DELTA ===")
    print(f"Best config: {best_name}")
    print(f"ROC-AUC: {float(best['cv_roc_auc_mean'] - baseline['cv_roc_auc_mean']):+.6f}")
    print(f"Average Precision: {float(best['cv_average_precision_mean'] - baseline['cv_average_precision_mean']):+.6f}")
    print(f"Lift@10: {float(best['cv_lift_top_10pct_mean'] - baseline['cv_lift_top_10pct_mean']):+.6f}")

    m3_oof = load_m3_oof(dataset, fold_id)
    print(f"\nRunning paired stratified bootstrap ({args.bootstrap_reps} replicates)...")

    comparisons: list[dict[str, object]] = []
    bootstrap_raw: list[pd.DataFrame] = []

    def run_bootstrap(label: str, reference: np.ndarray, candidate: np.ndarray) -> None:
        auc_summary, ap_summary, raw = paired_stratified_bootstrap_ranking_delta(
            y,
            reference,
            candidate,
            n_bootstrap=args.bootstrap_reps,
            confidence=args.bootstrap_confidence,
            random_state=RANDOM_STATE,
        )
        for item in (auc_summary, ap_summary):
            row = {"comparison": label, **item.as_dict()}
            comparisons.append(row)
        bootstrap_raw.append(pd.DataFrame({
            "comparison": label,
            "bootstrap_rep": np.arange(1, len(raw) + 1),
            "delta_roc_auc": raw[:, 0],
            "delta_average_precision": raw[:, 1],
        }))
        print(f"\n{label}")
        print(
            f"  ΔAUC={auc_summary.point_delta:+.6f} | "
            f"{int(args.bootstrap_confidence*100)}% CI [{auc_summary.ci_low:+.6f}, {auc_summary.ci_high:+.6f}] | "
            f"P(Δ>0)={auc_summary.win_probability:.4f}"
        )
        print(
            f"  ΔAP ={ap_summary.point_delta:+.6f} | "
            f"{int(args.bootstrap_confidence*100)}% CI [{ap_summary.ci_low:+.6f}, {ap_summary.ci_high:+.6f}] | "
            f"P(Δ>0)={ap_summary.win_probability:.4f}"
        )

    run_bootstrap(
        "m3_catboost_minus_logistic",
        m3_oof["oof__logistic_regression"].to_numpy(),
        m3_oof["oof__catboost"].to_numpy(),
    )
    run_bootstrap(
        "m31_best_minus_m3_catboost",
        m3_oof["oof__catboost"].to_numpy(),
        config_oof[best_name],
    )
    run_bootstrap(
        "m31_best_minus_logistic",
        m3_oof["oof__logistic_regression"].to_numpy(),
        config_oof[best_name],
    )

    # Fit the selected model on all selected applications. Iterations come only
    # from fold-internal early stopping and are therefore not chosen on full-data labels.
    final_trees = max(1, int(np.median(config_trees[best_name])))
    final_imputer = SimpleImputer(strategy="median")
    X_all = final_imputer.fit_transform(X)
    final_weights = (
        balanced_binary_class_weights(y).as_catboost_list()
        if args.class_weight == "balanced"
        else None
    )
    final_model = make_catboost(
        best_cfg,
        iterations=final_trees,
        class_weights=final_weights,
        thread_count=args.thread_count,
    )
    print(f"\nFitting selected CatBoost on all applications; iterations={final_trees}...")
    final_model.fit(X_all, y, verbose=False)
    final_model.save_model(ARTIFACT_DIR / "m31_catboost.cbm")
    joblib.dump(final_imputer, ARTIFACT_DIR / "m31_catboost_imputer.joblib")

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": final_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACT_DIR / "m31_catboost_feature_importance.csv", index=False)

    timestamp = datetime.now(timezone.utc).isoformat()
    summary.insert(0, "n_features", len(feature_names))
    summary.insert(0, "n_applications", len(dataset))
    summary.insert(0, "max_files", args.max_files)
    summary.to_csv(ARTIFACT_DIR / "m31_tuning_summary.csv", index=False)
    fold_metrics.to_csv(ARTIFACT_DIR / "m31_tuning_folds.csv", index=False)

    oof_out = dataset[["id", "flag"]].copy()
    oof_out["cv_fold"] = fold_id
    oof_out["oof__m3_logistic"] = m3_oof["oof__logistic_regression"].to_numpy(dtype="float32")
    oof_out["oof__m3_catboost"] = m3_oof["oof__catboost"].to_numpy(dtype="float32")
    for name, scores in config_oof.items():
        oof_out[f"oof__{name}"] = scores
    oof_out.to_parquet(ARTIFACT_DIR / "m31_oof_predictions.parquet", index=False)

    bootstrap_summary = pd.DataFrame(comparisons)
    bootstrap_summary.to_csv(ARTIFACT_DIR / "m31_bootstrap_summary.csv", index=False)
    pd.concat(bootstrap_raw, ignore_index=True).to_parquet(
        ARTIFACT_DIR / "m31_bootstrap_deltas.parquet", index=False
    )

    metadata = {
        "timestamp_utc": timestamp,
        "stage": "M3.1",
        "experiment": "controlled_catboost_tuning_and_paired_bootstrap",
        "selection_note": (
            "M3.1 CV metrics are model-selection estimates on reused frozen folds; "
            "they are not an untouched final generalization estimate."
        ),
        "feature_variant": CANONICAL_VARIANT,
        "max_files": args.max_files,
        "cv_folds": args.cv_folds,
        "random_state": RANDOM_STATE,
        "class_weight": args.class_weight,
        "n_applications": int(len(dataset)),
        "n_features": int(len(feature_names)),
        "default_rate": float(y.mean()),
        "max_iterations": args.catboost_iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
        "configs": [asdict(c) for c in configs],
        "best_config": asdict(best_cfg),
        "best_fold_tree_counts": config_trees[best_name],
        "final_iterations": final_trees,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_confidence": args.bootstrap_confidence,
    }
    (ARTIFACT_DIR / "m31_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    registry_path = ARTIFACT_DIR / "experiments.csv"
    registry_rows = summary.copy()
    registry_rows.insert(0, "timestamp_utc", timestamp)
    registry_rows.insert(1, "stage", "M3.1")
    if registry_path.exists():
        registry = pd.concat([pd.read_csv(registry_path), registry_rows], ignore_index=True, sort=False)
    else:
        registry = registry_rows
    registry.to_csv(registry_path, index=False)

    print("\nSaved:")
    for path in (
        ARTIFACT_DIR / "m31_tuning_summary.csv",
        ARTIFACT_DIR / "m31_tuning_folds.csv",
        ARTIFACT_DIR / "m31_oof_predictions.parquet",
        ARTIFACT_DIR / "m31_bootstrap_summary.csv",
        ARTIFACT_DIR / "m31_bootstrap_deltas.parquet",
        ARTIFACT_DIR / "m31_catboost.cbm",
        ARTIFACT_DIR / "m31_catboost_imputer.joblib",
        ARTIFACT_DIR / "m31_catboost_feature_importance.csv",
        ARTIFACT_DIR / "m31_metadata.json",
        ARTIFACT_DIR / "experiments.csv",
    ):
        print("-", path)


if __name__ == "__main__":
    main()
