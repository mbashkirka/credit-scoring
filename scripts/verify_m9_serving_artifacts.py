from __future__ import annotations

import argparse
from pathlib import Path

from credit_risk.api.runtime import ArtifactPaths, ScoringRuntime


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
    parser = argparse.ArgumentParser(description="Verify frozen Model V1 serving artifacts for M9.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--load-runtime",
        action="store_true",
        help="Load CatBoost/imputer/calibrator and validate the complete runtime contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = []
    empty = []
    for name in REQUIRED_NAMES:
        path = args.artifact_dir / name
        if not path.exists():
            missing.append(str(path))
        elif path.stat().st_size == 0:
            empty.append(str(path))
    if missing or empty:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if empty:
            details.append("empty=" + ", ".join(empty))
        raise FileNotFoundError("Invalid M9 serving artifact set: " + "; ".join(details))

    print("=== M9 SERVING ARTIFACT CHECK ===")
    print(f"artifact_dir: {args.artifact_dir}")
    print(f"files: {len(REQUIRED_NAMES)}")
    print("all required files present and non-empty: True")

    if args.load_runtime:
        runtime = ScoringRuntime.load(ArtifactPaths(root=args.artifact_dir))
        info = runtime.model_info()
        print(f"model_version: {info['model_version']}")
        print(f"features: {info['n_features']}")
        print(f"calibration: {info['calibration_method']}")
        print(f"policy: {info['primary_policy']}")
        print(f"artifact_fingerprint: {info['artifact_fingerprint']}")
        print("runtime contract: PASS")


if __name__ == "__main__":
    main()
