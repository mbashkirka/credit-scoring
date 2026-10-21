from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STRUCTURAL_ZERO_PREFIXES = (
    "enc_loans_account_holder_type__frac__",
    "enc_loans_credit_status__frac__",
    "enc_loans_credit_type__frac__",
    "enc_loans_account_cur__frac__",
    "payment_overall_state__frac__",
)


@dataclass(frozen=True)
class SchemaAlignmentReport:
    missing_structural_zero: tuple[str, ...]
    missing_nonstructural: tuple[str, ...]
    dropped_extra: tuple[str, ...]


def align_to_frozen_feature_schema(
    features: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, SchemaAlignmentReport]:
    """Align application features to a development-time feature schema.

    Category-distribution features absent from a raw chunk are structural zeros,
    not missing measurements. Other absent features remain NaN so the frozen
    model's development-fitted imputer handles them. Features not present in the
    development schema are dropped: M3.2 must not expand its representation
    after looking at the holdout population.
    """
    if "id" not in features.columns:
        raise ValueError("Feature frame must contain 'id'.")
    if features["id"].duplicated().any():
        raise ValueError("Feature frame must contain one row per application id.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Frozen feature schema contains duplicate names.")

    source_cols = set(features.columns) - {"id"}
    frozen_cols = set(feature_names)
    missing = frozen_cols - source_cols
    missing_structural = sorted(
        col for col in missing if col.startswith(STRUCTURAL_ZERO_PREFIXES)
    )
    missing_nonstructural = sorted(missing - set(missing_structural))
    dropped_extra = sorted(source_cols - frozen_cols)

    aligned = features.reindex(columns=["id", *feature_names]).copy()
    structural_cols = [
        col for col in feature_names if col.startswith(STRUCTURAL_ZERO_PREFIXES)
    ]
    if structural_cols:
        aligned[structural_cols] = aligned[structural_cols].fillna(0.0)

    return aligned, SchemaAlignmentReport(
        missing_structural_zero=tuple(missing_structural),
        missing_nonstructural=tuple(missing_nonstructural),
        dropped_extra=tuple(dropped_extra),
    )


def load_frozen_feature_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen feature manifest not found: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["feature"]:
        raise ValueError(f"Expected one 'feature' column in {path}.")
    names = frame["feature"].astype(str).tolist()
    if not names or len(names) != len(set(names)):
        raise ValueError("Frozen feature manifest is empty or contains duplicates.")
    return names


def assert_disjoint_ids(development_ids: np.ndarray, holdout_ids: np.ndarray) -> None:
    development_ids = np.asarray(development_ids)
    holdout_ids = np.asarray(holdout_ids)
    overlap = np.intersect1d(development_ids, holdout_ids, assume_unique=False)
    if overlap.size:
        preview = ", ".join(map(str, overlap[:10]))
        raise RuntimeError(
            f"Development/holdout id overlap detected: {overlap.size:,} ids. "
            f"Examples: {preview}"
        )
