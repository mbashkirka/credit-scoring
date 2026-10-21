from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.exceptions import InconsistentVersionWarning

from credit_risk.decisioning import BusinessScenario, assign_decisions, assign_risk_bands, pd_to_score
from credit_risk.evaluation import align_to_frozen_feature_schema, load_frozen_feature_names
from credit_risk.explainability import local_reason_rows
from credit_risk.features import aggregate_canonical_m21_features

from .schemas import ApplicationRequest, ReasonCodeResponse, ScoreResponse

MODEL_VERSION = "credit-risk-v1"
API_VERSION = "v1"
EXPLANATION_DOMAIN = "catboost_raw_log_odds_pre_isotonic"


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path = field(
        default_factory=lambda: Path(os.environ.get("CREDIT_RISK_ARTIFACT_DIR", "artifacts"))
    )

    @property
    def model(self) -> Path:
        return self.root / "m4_catboost.cbm"

    @property
    def imputer(self) -> Path:
        return self.root / "m4_catboost_imputer.joblib"

    @property
    def calibrator(self) -> Path:
        return self.root / "m4_calibrator.joblib"

    @property
    def feature_manifest(self) -> Path:
        return self.root / "m3_features.csv"

    @property
    def m4_metadata(self) -> Path:
        return self.root / "m4_metadata.json"

    @property
    def policy(self) -> Path:
        return self.root / "m5_policy_freeze.json"

    @property
    def bands(self) -> Path:
        return self.root / "m5_risk_band_thresholds.json"

    def frozen_inputs(self) -> tuple[Path, ...]:
        return (
            self.model,
            self.imputer,
            self.calibrator,
            self.feature_manifest,
            self.m4_metadata,
            self.policy,
            self.bands,
        )


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _primary_scenario(policy: dict[str, object]) -> BusinessScenario:
    primary_name = str(policy.get("primary_policy", ""))
    scenarios = policy.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError("m5_policy_freeze.json does not contain scenario records")
    matches = [row for row in scenarios if isinstance(row, dict) and row.get("name") == primary_name]
    if len(matches) != 1:
        raise RuntimeError("Unable to resolve exactly one frozen primary policy scenario")
    row = matches[0]
    return BusinessScenario(
        name=str(row["name"]),
        performing_margin_rate=float(row["performing_margin_rate"]),
        lgd=float(row["lgd"]),
        origination_cost_rate=float(row["origination_cost_rate"]),
        risk_appetite_pd_cap=float(row["risk_appetite_pd_cap"]),
        review_band_fraction=float(row["review_band_fraction"]),
        primary=bool(row.get("primary", True)),
    )


def applications_to_history(applications: list[ApplicationRequest]) -> pd.DataFrame:
    """Convert validated API requests to the exact raw-history frame used offline."""
    rows: list[dict[str, object]] = []
    for application in applications:
        for record in application.credit_history:
            row = {"id": int(application.application_id), **record.model_dump()}
            rows.append(row)
    if not rows:
        raise ValueError("At least one credit-history row is required")
    frame = pd.DataFrame(rows)
    # Keep the source dtypes numeric while preserving explicit nulls as NaN.
    for column in frame.columns:
        if column != "id":
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


class ScoringRuntime:
    """Immutable, process-local runtime for frozen Model V1 artifacts."""

    def __init__(
        self,
        *,
        model: CatBoostClassifier,
        imputer,
        calibrator,
        feature_names: list[str],
        band_labels: list[str],
        band_cutpoints: np.ndarray,
        primary_scenario: BusinessScenario,
        calibration_method: str,
        artifact_fingerprint: str,
    ) -> None:
        self.model = model
        self.imputer = imputer
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.band_labels = list(band_labels)
        self.band_cutpoints = np.asarray(band_cutpoints, dtype=np.float64)
        self.primary_scenario = primary_scenario
        self.calibration_method = str(calibration_method)
        self.artifact_fingerprint = str(artifact_fingerprint)

    @classmethod
    def load(cls, paths: ArtifactPaths | None = None) -> "ScoringRuntime":
        paths = paths or ArtifactPaths()
        missing = [str(path) for path in paths.frozen_inputs() if not path.exists()]
        if missing:
            raise FileNotFoundError("Frozen serving artifacts missing: " + ", ".join(missing))

        feature_names = load_frozen_feature_names(paths.feature_manifest)
        if len(feature_names) != 212:
            raise RuntimeError(f"Expected frozen 212-feature Model V1 schema; got {len(feature_names)}")

        m4_meta = _load_json(paths.m4_metadata)
        policy = _load_json(paths.policy)
        bands = _load_json(paths.bands)
        calibration_method = str(m4_meta.get("selected_calibration_method", ""))
        if calibration_method != "isotonic":
            raise RuntimeError(f"Model V1 expects isotonic calibration; got {calibration_method!r}")

        labels = [str(x) for x in bands.get("labels", [])]
        cutpoints = np.asarray(bands.get("cutpoints", []), dtype=np.float64)
        if labels != ["A", "B", "C", "D", "E"] or cutpoints.shape != (4,):
            raise RuntimeError("Frozen M5 risk-band artifact is incompatible with Model V1")

        model = CatBoostClassifier()
        model.load_model(str(paths.model))
        # Pickle/joblib estimators are not supported across scikit-learn versions.
        # Treat a version mismatch as a readiness failure rather than a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            imputer = joblib.load(paths.imputer)
            calibrator = joblib.load(paths.calibrator)

        return cls(
            model=model,
            imputer=imputer,
            calibrator=calibrator,
            feature_names=feature_names,
            band_labels=labels,
            band_cutpoints=cutpoints,
            primary_scenario=_primary_scenario(policy),
            calibration_method=calibration_method,
            artifact_fingerprint=_artifact_fingerprint(paths.frozen_inputs()),
        )

    def model_info(self) -> dict[str, object]:
        return {
            "model_version": MODEL_VERSION,
            "api_version": API_VERSION,
            "n_features": len(self.feature_names),
            "calibration_method": self.calibration_method,
            "primary_policy": self.primary_scenario.name,
            "approve_pd_threshold": float(self.primary_scenario.approve_pd_threshold),
            "reject_pd_threshold": float(self.primary_scenario.reject_pd_threshold),
            "risk_band_labels": list(self.band_labels),
            "risk_band_cutpoints": self.band_cutpoints.astype(float).tolist(),
            "explanation_domain": EXPLANATION_DOMAIN,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    def _frozen_matrix(self, applications: list[ApplicationRequest]) -> tuple[pd.DataFrame, np.ndarray]:
        history = applications_to_history(applications)
        features = aggregate_canonical_m21_features(history)
        aligned, _ = align_to_frozen_feature_schema(features, self.feature_names)
        requested_ids = [int(application.application_id) for application in applications]
        indexed = aligned.set_index("id", drop=False)
        missing_ids = [app_id for app_id in requested_ids if app_id not in indexed.index]
        if missing_ids:
            raise RuntimeError(f"Feature pipeline lost application ids: {missing_ids[:5]}")
        aligned = indexed.loc[requested_ids].reset_index(drop=True)
        matrix = np.asarray(self.imputer.transform(aligned[self.feature_names]), dtype=np.float32)
        if matrix.shape != (len(applications), len(self.feature_names)):
            raise RuntimeError("Frozen serving matrix has unexpected shape")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("Frozen imputer returned non-finite serving features")
        return aligned, matrix

    def score_many(
        self,
        applications: list[ApplicationRequest],
        *,
        include_explanations: bool,
        top_reasons_each_direction: int = 3,
    ) -> list[ScoreResponse]:
        if not applications:
            raise ValueError("applications must not be empty")
        ids = [int(item.application_id) for item in applications]
        if len(ids) != len(set(ids)):
            raise ValueError("applications contains duplicate application_id values")

        _, matrix = self._frozen_matrix(applications)
        raw_probability = np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float64)
        calibrated = np.asarray(self.calibrator.predict(raw_probability), dtype=np.float64)
        if not np.all(np.isfinite(calibrated)) or np.any((calibrated < 0.0) | (calibrated > 1.0)):
            raise RuntimeError("Frozen calibrator returned invalid probability values")

        bands = assign_risk_bands(calibrated, self.band_cutpoints, labels=tuple(self.band_labels))
        decisions = assign_decisions(calibrated, self.primary_scenario)
        scores = np.rint(pd_to_score(calibrated)).astype(int)

        shap_values: np.ndarray | None = None
        if include_explanations:
            pool = Pool(matrix, feature_names=self.feature_names)
            shap_full = np.asarray(self.model.get_feature_importance(pool, type="ShapValues"), dtype=np.float64)
            shap_values = shap_full[:, :-1]
            base = shap_full[:, -1]
            raw_margin = np.asarray(self.model.predict(matrix, prediction_type="RawFormulaVal"), dtype=np.float64)
            reconstructed = base + shap_values.sum(axis=1)
            error = float(np.max(np.abs(reconstructed - raw_margin)))
            if error > 1e-8:
                raise RuntimeError(f"TreeSHAP additivity check failed in serving: max error={error:.3e}")

        outputs: list[ScoreResponse] = []
        for i, app_id in enumerate(ids):
            increasing: list[ReasonCodeResponse] = []
            reducing: list[ReasonCodeResponse] = []
            if shap_values is not None:
                reason_frame = local_reason_rows(
                    application_id=app_id,
                    calibrated_pd=float(calibrated[i]),
                    decision=str(decisions[i]),
                    risk_band=str(bands[i]),
                    feature_values=matrix[i],
                    shap_row=shap_values[i],
                    feature_names=self.feature_names,
                    top_n_each_direction=top_reasons_each_direction,
                )
                for row in reason_frame.to_dict(orient="records"):
                    item = ReasonCodeResponse(
                        code=str(row["reason_code"]),
                        description=str(row["reason_text"]),
                        direction=str(row["direction"]),
                        contribution_log_odds=float(row["group_shap_log_odds"]),
                        representative_feature=str(row["representative_feature"]),
                        representative_feature_value=(
                            None
                            if pd.isna(row["representative_feature_value"])
                            else float(row["representative_feature_value"])
                        ),
                        representative_feature_contribution_log_odds=float(
                            row["representative_feature_shap_log_odds"]
                        ),
                        caveat=str(row["caveat"]),
                    )
                    if item.direction == "risk_increasing":
                        increasing.append(item)
                    else:
                        reducing.append(item)

            outputs.append(
                ScoreResponse(
                    model_version=MODEL_VERSION,
                    artifact_fingerprint=self.artifact_fingerprint,
                    application_id=app_id,
                    probability_of_default=float(calibrated[i]),
                    raw_model_probability=float(raw_probability[i]),
                    credit_score=int(scores[i]),
                    risk_band=str(bands[i]),
                    decision=str(decisions[i]),
                    explanation_domain=EXPLANATION_DOMAIN,
                    risk_increasing_reasons=increasing,
                    risk_reducing_reasons=reducing,
                )
            )
        return outputs
