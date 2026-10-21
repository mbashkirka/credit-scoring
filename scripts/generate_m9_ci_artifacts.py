from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

from credit_risk.modeling import FrozenMedianImputer


REQUIRED_NAMES = (
    "m4_catboost.cbm",
    "m4_catboost_imputer.joblib",
    "m4_calibrator.joblib",
    "m3_features.csv",
    "m4_metadata.json",
    "m5_policy_freeze.json",
    "m5_risk_band_thresholds.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tiny synthetic Model V1-compatible artifacts for CI container-contract tests. "
            "These artifacts are NEVER accuracy benchmarks or production model files."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def feature_names() -> list[str]:
    meaningful = [
        "history__row_count",
        "payment_overall_state__frac__0",
        "pre_util__min",
        "is_zero_util__mean",
        "pre_loans_outstanding__std",
        "is_zero_loans3060__mean",
        "enc_loans_credit_type__frac__0",
        "enc_loans_credit_status__frac__0",
        "payment_transition_count__mean",
        "pre_since_opened__min",
    ]
    fillers = [f"ci_fixture_feature__{i:03d}" for i in range(212 - len(meaningful))]
    return meaningful + fillers


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in REQUIRED_NAMES if (args.output_dir / name).exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing serving artifacts. Use --force only in an ephemeral CI workspace: "
            + ", ".join(map(str, existing))
        )

    names = feature_names()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(512, len(names))).astype(np.float32)
    signal = 0.9 * X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2] + 0.3 * X[:, 5]
    probability = 1.0 / (1.0 + np.exp(-signal))
    y = (rng.random(len(X)) < probability).astype(np.int8)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Synthetic CI target unexpectedly lost a class")

    model = CatBoostClassifier(
        iterations=12,
        depth=3,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    train_pool = Pool(X, label=y, feature_names=names)
    model.fit(train_pool)
    model.save_model(str(args.output_dir / "m4_catboost.cbm"))

    joblib.dump(FrozenMedianImputer(np.zeros(len(names), dtype=np.float32)), args.output_dir / "m4_catboost_imputer.joblib")

    raw = model.predict_proba(X)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, y)
    joblib.dump(calibrator, args.output_dir / "m4_calibrator.joblib")

    pd.DataFrame({"feature": names}).to_csv(args.output_dir / "m3_features.csv", index=False)
    (args.output_dir / "m4_metadata.json").write_text(
        json.dumps(
            {
                "stage": "M9_CI_SYNTHETIC_FIXTURE",
                "selected_calibration_method": "isotonic",
                "not_for_model_quality_evaluation": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "m5_risk_band_thresholds.json").write_text(
        json.dumps(
            {
                "method": "M9_CI_SYNTHETIC_FIXTURE",
                "labels": ["A", "B", "C", "D", "E"],
                "cutpoints": [0.20, 0.40, 0.60, 0.80],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "m5_policy_freeze.json").write_text(
        json.dumps(
            {
                "stage": "M9_CI_SYNTHETIC_FIXTURE",
                "primary_policy": "balanced",
                "scenarios": [
                    {
                        "name": "balanced",
                        "performing_margin_rate": 0.12,
                        "lgd": 0.45,
                        "origination_cost_rate": 0.02,
                        "risk_appetite_pd_cap": 0.12,
                        "review_band_fraction": 0.20,
                        "primary": True,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== M9 CI SYNTHETIC ARTIFACT FIXTURE ===")
    print(f"output_dir: {args.output_dir}")
    print("feature_count: 212")
    print("purpose: container/API contract testing only")
    print("production_model: False")


if __name__ == "__main__":
    main()
