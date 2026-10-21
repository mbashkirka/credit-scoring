from __future__ import annotations

from dataclasses import fields

import numpy as np

from credit_risk.decisioning import BusinessScenario, RISK_BAND_LABELS


EXPECTED_LOCKBOX_CHUNKS = (10, 11)
EXPECTED_POLICY_NAME = "balanced"


def frozen_primary_scenario(policy_freeze: dict[str, object]) -> BusinessScenario:
    """Reconstruct the predeclared M5 primary scenario without recomputing thresholds."""
    name = str(policy_freeze.get("primary_policy", ""))
    if name != EXPECTED_POLICY_NAME:
        raise RuntimeError(
            f"M6 expects frozen primary policy '{EXPECTED_POLICY_NAME}', got '{name}'."
        )
    records = policy_freeze.get("scenarios")
    if not isinstance(records, list):
        raise RuntimeError("Frozen policy artifact does not contain scenario records.")
    matches = [row for row in records if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError("Frozen primary scenario must appear exactly once.")
    allowed = {f.name for f in fields(BusinessScenario)}
    kwargs = {k: v for k, v in matches[0].items() if k in allowed}
    scenario = BusinessScenario(**kwargs)
    if not scenario.primary:
        raise RuntimeError("Frozen primary scenario is not marked primary.")
    return scenario


def frozen_risk_band_cutpoints(
    threshold_artifact: dict[str, object],
) -> np.ndarray:
    labels = tuple(threshold_artifact.get("labels", []))
    if labels != RISK_BAND_LABELS:
        raise RuntimeError(
            f"M6 expects frozen risk-band labels {RISK_BAND_LABELS}, got {labels}."
        )
    cuts = np.asarray(threshold_artifact.get("cutpoints", []), dtype=np.float64)
    if cuts.shape != (len(RISK_BAND_LABELS) - 1,):
        raise RuntimeError("Frozen risk-band artifact must contain exactly four cutpoints.")
    if not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise RuntimeError("Frozen risk-band cutpoints must be finite and strictly increasing.")
    return cuts


def validate_lockbox_governance(
    m4_meta: dict[str, object],
    m5_meta: dict[str, object],
    policy_freeze: dict[str, object],
    threshold_artifact: dict[str, object],
    lockbox_chunks: tuple[int, ...] = EXPECTED_LOCKBOX_CHUNKS,
) -> None:
    """Fail closed if any pre-lockbox governance artifact is inconsistent."""
    if tuple(lockbox_chunks) != EXPECTED_LOCKBOX_CHUNKS:
        raise RuntimeError(
            f"M6 is authorized only for lockbox chunks {EXPECTED_LOCKBOX_CHUNKS}."
        )
    for name, meta in (("M4", m4_meta), ("M5", m5_meta)):
        reserved = tuple(meta.get("lockbox_chunks_reserved", []))
        if reserved != EXPECTED_LOCKBOX_CHUNKS:
            raise RuntimeError(f"{name} reserved lockbox chunks do not match M6 governance.")
        if meta.get("lockbox_evaluated") is not False and name == "M5":
            raise RuntimeError("M5 metadata does not mark lockbox as untouched.")
    if m4_meta.get("final_lockbox_evaluated") is not False:
        raise RuntimeError("M4 metadata does not mark final lockbox as untouched.")
    if tuple(policy_freeze.get("lockbox_chunks_reserved", [])) != EXPECTED_LOCKBOX_CHUNKS:
        raise RuntimeError("Frozen policy lockbox reservation does not match M6.")
    if policy_freeze.get("lockbox_evaluated") is not False:
        raise RuntimeError("Frozen policy artifact does not mark lockbox as untouched.")
    if m5_meta.get("primary_policy") != EXPECTED_POLICY_NAME:
        raise RuntimeError("M5 metadata primary policy differs from the frozen M6 policy.")
    frozen_primary_scenario(policy_freeze)
    frozen_risk_band_cutpoints(threshold_artifact)
