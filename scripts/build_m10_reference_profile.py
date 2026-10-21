from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.api.runtime import ArtifactPaths, ScoringRuntime
from credit_risk.evaluation import align_to_frozen_feature_schema
from credit_risk.monitoring import (
    MonitoringThresholds,
    NumericReferenceProfile,
    build_numeric_reference_profile,
    categorical_shares,
    histogram_counts,
    schema_signature,
)

ARTIFACT_DIR = Path("artifacts")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
REFERENCE_CHUNKS = (8, 9)
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build M10 Model-V1 monitoring reference profiles from the frozen M5 "
            "development population (chunks 8+9). No model fitting or policy changes."
        )
    )
    parser.add_argument("--reference-chunks", nargs="+", type=int, default=list(REFERENCE_CHUNKS))
    parser.add_argument("--feature-bins", type=int, default=10)
    parser.add_argument("--pd-bins", type=int, default=10)
    parser.add_argument("--quantile-sample-per-chunk", type=int, default=25_000)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_cache(chunk: int) -> Path:
    return FEATURE_DIR / f"train_data_{chunk}.parquet"


def _load_reference_predictions() -> pd.DataFrame:
    path = ARTIFACT_DIR / "m5_decisions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"M10 requires {path}")
    frame = pd.read_parquet(path)
    required = {
        "id",
        "source_chunk",
        "calibrated_pd",
        "risk_band",
        "decision_balanced",
        "credit_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"M5 decisions artifact is missing columns: {missing}")
    if set(frame["source_chunk"].astype(int).unique()) != set(REFERENCE_CHUNKS):
        raise RuntimeError("M10 reference predictions must come exactly from M5 chunks 8+9")
    return frame


def _load_predeployment_performance_benchmark() -> dict[str, object]:
    path = ARTIFACT_DIR / "m4_calibration_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"M10 requires {path}")
    metrics = pd.read_csv(path)
    validation = metrics[metrics["population"] == "calibration_validation_chunk9"].copy()
    if validation.empty:
        raise RuntimeError("M4 calibration-validation benchmark for chunk 9 is missing")
    m4_meta = json.loads((ARTIFACT_DIR / "m4_metadata.json").read_text(encoding="utf-8"))
    selected = str(m4_meta.get("selected_calibration_method"))
    selected_row = validation[validation["method"] == selected]
    raw_row = validation[validation["method"] == "raw"]
    if len(selected_row) != 1 or len(raw_row) != 1:
        raise RuntimeError("Unable to resolve M4 raw/selected validation rows")
    return {
        "population": "calibration_validation_chunk9",
        "selected_method": selected,
        "selected_metrics": selected_row.iloc[0].to_dict(),
        "raw_metrics": raw_row.iloc[0].to_dict(),
        "interpretation": (
            "Pre-deployment outcome benchmark. The selected calibrator here was fitted on chunk 8 "
            "and evaluated on chunk 9; the final production calibrator was subsequently refitted on 8+9."
        ),
    }


def _sample_reference_features(
    chunks: list[int],
    feature_names: list[str],
    per_chunk: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    samples = []
    schema_observations: list[dict[str, object]] = []
    for chunk in chunks:
        path = _feature_cache(chunk)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference feature cache missing: {path}. Run M4/M5 pipeline first."
            )
        raw = pd.read_parquet(path)
        aligned, report = align_to_frozen_feature_schema(raw, feature_names)
        if report.missing_nonstructural:
            raise RuntimeError(
                f"Reference chunk {chunk} is missing non-structural frozen features: "
                f"{list(report.missing_nonstructural)[:5]}"
            )
        raw_feature_names = sorted(set(raw.columns) - {"id"})
        schema_observations.append(
            {
                "chunk": int(chunk),
                "raw_feature_count": len(raw_feature_names),
                "raw_schema_signature": schema_signature(raw_feature_names),
                "raw_feature_names": raw_feature_names,
                "missing_structural_zero_features": list(report.missing_structural_zero),
                "missing_nonstructural_features": list(report.missing_nonstructural),
                "dropped_extra_features": list(report.dropped_extra),
            }
        )
        n = min(per_chunk, len(aligned))
        sample = aligned[feature_names].sample(n=n, random_state=RANDOM_STATE + chunk)
        samples.append(sample)
        print(
            f"Reference chunk {chunk}: rows={len(aligned):,} | sample={n:,} | "
            f"raw/frozen={raw.shape[1]-1}/{len(feature_names)} | "
            f"dropped-extra={len(report.dropped_extra)}"
        )
    return pd.concat(samples, ignore_index=True), schema_observations


def _build_feature_profiles(
    chunks: list[int],
    feature_names: list[str],
    *,
    n_bins: int,
    sample_per_chunk: int,
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    print("Building deterministic quantile-edge sample...")
    sample, schema_observations = _sample_reference_features(
        chunks, feature_names, sample_per_chunk
    )
    edges_by_feature = {
        feature: np.asarray(
            build_numeric_reference_profile(sample[feature].to_numpy(), n_bins=n_bins).edges,
            dtype=np.float64,
        )
        for feature in feature_names
    }
    del sample

    counts = {
        feature: np.zeros(len(edges_by_feature[feature]) - 1, dtype=np.int64)
        for feature in feature_names
    }
    missing = {feature: 0 for feature in feature_names}
    total_n = {feature: 0 for feature in feature_names}
    sums = {feature: 0.0 for feature in feature_names}
    sums_sq = {feature: 0.0 for feature in feature_names}
    valid_n = {feature: 0 for feature in feature_names}

    print("Accumulating exact reference counts/statistics in a bounded-memory second pass...")
    for chunk in chunks:
        raw = pd.read_parquet(_feature_cache(chunk))
        aligned, _ = align_to_frozen_feature_schema(raw, feature_names)
        for feature in feature_names:
            values = aligned[feature].to_numpy(dtype=np.float64, copy=False)
            c, m = histogram_counts(values, edges_by_feature[feature])
            counts[feature] += c
            missing[feature] += m
            total_n[feature] += len(values)
            valid = values[np.isfinite(values)]
            if valid.size:
                sums[feature] += float(valid.sum())
                sums_sq[feature] += float(np.square(valid).sum())
                valid_n[feature] += int(valid.size)
        print(f"  accumulated chunk {chunk}: {len(aligned):,} applications")

    profiles: dict[str, dict[str, object]] = {}
    summary_rows = []
    for feature in feature_names:
        vn = valid_n[feature]
        mean = sums[feature] / vn if vn else float("nan")
        variance = max(0.0, sums_sq[feature] / vn - mean**2) if vn else float("nan")
        std = float(np.sqrt(variance)) if vn else float("nan")
        profile = NumericReferenceProfile(
            edges=tuple(float(x) for x in edges_by_feature[feature]),
            counts=tuple(int(x) for x in counts[feature]),
            missing=int(missing[feature]),
            n=int(total_n[feature]),
            mean=float(mean),
            std=std,
        )
        profiles[feature] = profile.to_dict()
        summary_rows.append(
            {
                "feature": feature,
                "n": profile.n,
                "mean": profile.mean,
                "std": profile.std,
                "missing_rate": profile.missing / profile.n if profile.n else float("nan"),
                "n_bins": len(profile.counts),
            }
        )
    return profiles, pd.DataFrame(summary_rows), schema_observations


def main() -> None:
    args = parse_args()
    chunks = sorted(set(args.reference_chunks))
    if chunks != list(REFERENCE_CHUNKS):
        raise ValueError(f"M10 reference governance requires chunks {list(REFERENCE_CHUNKS)}")
    if args.feature_bins < 2 or args.pd_bins < 2:
        raise ValueError("feature-bins and pd-bins must be >= 2")
    if args.quantile_sample_per_chunk < 1000:
        raise ValueError("quantile-sample-per-chunk must be >= 1000")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = ScoringRuntime.load(ArtifactPaths(ARTIFACT_DIR))
    predictions = _load_reference_predictions()
    print("=== M10 MONITORING REFERENCE PROFILE ===")
    print("model_version: credit-risk-v1")
    print(f"artifact_fingerprint: {runtime.artifact_fingerprint}")
    print(f"reference chunks: {chunks}")
    print(f"applications: {len(predictions):,}")
    print(f"features: {len(runtime.feature_names)}")

    feature_profiles, feature_summary, schema_observations = _build_feature_profiles(
        chunks,
        runtime.feature_names,
        n_bins=args.feature_bins,
        sample_per_chunk=args.quantile_sample_per_chunk,
    )
    pd_profile = build_numeric_reference_profile(
        predictions["calibrated_pd"].to_numpy(dtype=np.float64), n_bins=args.pd_bins
    )
    prediction_profile = {
        "calibrated_pd": pd_profile.to_dict(),
        "mean_pd": float(predictions["calibrated_pd"].mean()),
        "risk_band_shares": categorical_shares(
            predictions["risk_band"], runtime.band_labels
        ),
        "decision_shares": categorical_shares(
            predictions["decision_balanced"], ["APPROVE", "REVIEW", "REJECT"]
        ),
        "mean_credit_score": float(predictions["credit_score"].mean()),
    }
    unique_schemas: dict[str, dict[str, object]] = {}
    for observation in schema_observations:
        signature = str(observation["raw_schema_signature"])
        if signature not in unique_schemas:
            unique_schemas[signature] = {
                "raw_schema_signature": signature,
                "raw_feature_count": int(observation["raw_feature_count"]),
                "raw_feature_names": observation["raw_feature_names"],
                "missing_structural_zero_features": observation[
                    "missing_structural_zero_features"
                ],
                "missing_nonstructural_features": observation["missing_nonstructural_features"],
                "dropped_extra_features": observation["dropped_extra_features"],
                "observed_chunks": [],
            }
        unique_schemas[signature]["observed_chunks"].append(int(observation["chunk"]))
    schema_profile = {
        "model_version": "credit-risk-v1",
        "artifact_fingerprint": runtime.artifact_fingerprint,
        "reference_chunks": chunks,
        "comparison_rule": (
            "A current raw schema is baseline-compatible when its exact unordered feature-name "
            "set matches any schema observed in the M10 reference population. Extra features "
            "relative to the frozen 212-feature model are not drift when already present in "
            "that reference schema."
        ),
        "observed_raw_schemas": list(unique_schemas.values()),
    }

    performance = _load_predeployment_performance_benchmark()
    thresholds = MonitoringThresholds()

    feature_payload = {
        "model_version": "credit-risk-v1",
        "artifact_fingerprint": runtime.artifact_fingerprint,
        "reference_chunks": chunks,
        "feature_count": len(runtime.feature_names),
        "feature_bins": args.feature_bins,
        "quantile_edge_estimation": {
            "method": "deterministic_sample_then_exact_full_population_counts",
            "sample_per_chunk": args.quantile_sample_per_chunk,
            "random_state": RANDOM_STATE,
        },
        "profiles": feature_profiles,
    }
    (ARTIFACT_DIR / "m10_reference_feature_profile.json").write_text(
        json.dumps(feature_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    feature_summary.to_csv(ARTIFACT_DIR / "m10_reference_feature_summary.csv", index=False)
    (ARTIFACT_DIR / "m10_reference_schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, allow_nan=False), encoding="utf-8"
    )
    (ARTIFACT_DIR / "m10_reference_prediction_profile.json").write_text(
        json.dumps(prediction_profile, indent=2, allow_nan=False), encoding="utf-8"
    )
    (ARTIFACT_DIR / "m10_reference_performance_benchmark.json").write_text(
        json.dumps(performance, indent=2, allow_nan=False), encoding="utf-8"
    )
    alert_policy = {
        "stage": "M10",
        "thresholds": thresholds.to_dict(),
        "status_levels": ["PASS", "WARNING", "CRITICAL"],
        "threshold_provenance": (
            "Project-level operational heuristics introduced after M6; not regulatory limits and "
            "not claimed to have been predeclared before the final lockbox."
        ),
        "historical_m6_use": "retrospective case study only",
        "future_use": "prospective monitoring rules once this M10 policy is frozen",
    }
    (ARTIFACT_DIR / "m10_alert_policy.json").write_text(
        json.dumps(alert_policy, indent=2, allow_nan=False), encoding="utf-8"
    )
    metadata = {
        "stage": "M10_REFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "credit-risk-v1",
        "artifact_fingerprint": runtime.artifact_fingerprint,
        "reference_population": "M5 policy-development chunks 8+9",
        "reference_chunks": chunks,
        "n_reference": int(len(predictions)),
        "feature_count": len(runtime.feature_names),
        "reference_raw_schema_signatures": sorted(unique_schemas),
        "reference_raw_schema_count": len(unique_schemas),
        "model_changes": False,
        "calibration_changes": False,
        "policy_changes": False,
        "thresholds_defined_after_m6": True,
        "source_sha256": {
            "m5_decisions.parquet": _sha256(ARTIFACT_DIR / "m5_decisions.parquet"),
            "m4_calibration_metrics.csv": _sha256(ARTIFACT_DIR / "m4_calibration_metrics.csv"),
        },
    }
    (ARTIFACT_DIR / "m10_reference_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("\n=== M10 REFERENCE PREDICTION MIX ===")
    print(f"mean PD: {prediction_profile['mean_pd']:.4%}")
    print("risk bands: " + " | ".join(
        f"{k}={v:.2%}" for k, v in prediction_profile["risk_band_shares"].items()
    ))
    print("decisions: " + " | ".join(
        f"{k}={v:.2%}" for k, v in prediction_profile["decision_shares"].items()
    ))
    selected = performance["selected_metrics"]
    print("\n=== M10 PRE-DEPLOYMENT OUTCOME BENCHMARK ===")
    print(
        f"chunk 9 selected={performance['selected_method']} | "
        f"AUC={float(selected['roc_auc']):.6f} | Brier={float(selected['brier_score']):.6f} | "
        f"bias={float(selected['calibration_bias']):+.4%} | "
        f"slope={float(selected['calibration_slope']):.4f}"
    )
    print("\nReference profile frozen. M6 remains a historical case study, not prospective monitoring.")
    print("Saved:")
    for path in [
        "artifacts/m10_reference_feature_profile.json",
        "artifacts/m10_reference_feature_summary.csv",
        "artifacts/m10_reference_schema_profile.json",
        "artifacts/m10_reference_prediction_profile.json",
        "artifacts/m10_reference_performance_benchmark.json",
        "artifacts/m10_alert_policy.json",
        "artifacts/m10_reference_metadata.json",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
