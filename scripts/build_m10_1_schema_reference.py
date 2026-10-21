from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_risk.api.runtime import ArtifactPaths, ScoringRuntime
from credit_risk.evaluation import align_to_frozen_feature_schema
from credit_risk.monitoring import schema_signature

ARTIFACT_DIR = Path("artifacts")
FEATURE_DIR = Path("data/processed/app_features_m21/e_typed_catdist_payment")
REFERENCE_CHUNKS = (8, 9)


def main() -> None:
    runtime = ScoringRuntime.load(ArtifactPaths(ARTIFACT_DIR))
    observed: dict[str, dict[str, object]] = {}

    print("=== M10.1 BASELINE-AWARE SCHEMA REFERENCE ===")
    print(f"artifact_fingerprint: {runtime.artifact_fingerprint}")
    print(f"reference chunks: {list(REFERENCE_CHUNKS)}")

    for chunk in REFERENCE_CHUNKS:
        path = FEATURE_DIR / f"train_data_{chunk}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Reference feature cache missing: {path}")
        raw = pd.read_parquet(path)
        _, report = align_to_frozen_feature_schema(raw, runtime.feature_names)
        if report.missing_nonstructural:
            raise RuntimeError(
                f"Reference chunk {chunk} is missing non-structural frozen features: "
                f"{list(report.missing_nonstructural)[:5]}"
            )

        raw_feature_names = sorted(set(raw.columns) - {"id"})
        signature = schema_signature(raw_feature_names)
        if signature not in observed:
            observed[signature] = {
                "raw_schema_signature": signature,
                "raw_feature_count": len(raw_feature_names),
                "raw_feature_names": raw_feature_names,
                "missing_structural_zero_features": list(report.missing_structural_zero),
                "missing_nonstructural_features": list(report.missing_nonstructural),
                "dropped_extra_features": list(report.dropped_extra),
                "observed_chunks": [],
            }
        observed[signature]["observed_chunks"].append(int(chunk))
        print(
            f"chunk_{chunk}: raw/frozen={len(raw_feature_names)}/{len(runtime.feature_names)} | "
            f"dropped_extra={len(report.dropped_extra)} | signature={signature[:12]}..."
        )

    payload = {
        "stage": "M10.1_SCHEMA_REFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "credit-risk-v1",
        "artifact_fingerprint": runtime.artifact_fingerprint,
        "reference_chunks": list(REFERENCE_CHUNKS),
        "logic_revision": "baseline_aware_raw_schema_v1",
        "introduced_after_m6": True,
        "comparison_rule": (
            "A current raw schema is baseline-compatible when its exact unordered feature-name "
            "set matches any schema observed in the M10 reference population. Extra features "
            "relative to the frozen 212-feature model are not drift when already present in "
            "that reference schema."
        ),
        "observed_raw_schemas": list(observed.values()),
    }
    out = ARTIFACT_DIR / "m10_reference_schema_profile.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"observed unique raw schemas: {len(observed)}")
    print(f"Saved: {out}")
    print("M10/M6 remains a retrospective historical case study; no threshold was changed.")


if __name__ == "__main__":
    main()
