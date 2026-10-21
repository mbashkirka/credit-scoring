from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CreditHistoryRecord(BaseModel):
    """One raw credit-history row from the Alfa-style source schema.

    Fields are required but nullable. This distinguishes an explicitly missing source
    measurement from an accidentally omitted API field. Target/outcome fields are not
    part of the serving contract and extra fields are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    rn: int | None
    pre_since_opened: int | None
    pre_since_confirmed: int | None
    pre_pterm: int | None
    pre_fterm: int | None
    pre_till_pclose: int | None
    pre_till_fclose: int | None
    pre_loans_credit_limit: int | None
    pre_loans_next_pay_summ: int | None
    pre_loans_outstanding: int | None
    pre_loans_total_overdue: int | None
    pre_loans_max_overdue_sum: int | None
    pre_loans_credit_cost_rate: int | None
    pre_loans5: int | None
    pre_loans530: int | None
    pre_loans3060: int | None
    pre_loans6090: int | None
    pre_loans90: int | None
    is_zero_loans5: int | None
    is_zero_loans530: int | None
    is_zero_loans3060: int | None
    is_zero_loans6090: int | None
    is_zero_loans90: int | None
    pre_util: int | None
    pre_over2limit: int | None
    pre_maxover2limit: int | None
    is_zero_util: int | None
    is_zero_over2limit: int | None
    is_zero_maxover2limit: int | None
    enc_paym_0: int | None
    enc_paym_1: int | None
    enc_paym_2: int | None
    enc_paym_3: int | None
    enc_paym_4: int | None
    enc_paym_5: int | None
    enc_paym_6: int | None
    enc_paym_7: int | None
    enc_paym_8: int | None
    enc_paym_9: int | None
    enc_paym_10: int | None
    enc_paym_11: int | None
    enc_paym_12: int | None
    enc_paym_13: int | None
    enc_paym_14: int | None
    enc_paym_15: int | None
    enc_paym_16: int | None
    enc_paym_17: int | None
    enc_paym_18: int | None
    enc_paym_19: int | None
    enc_paym_20: int | None
    enc_paym_21: int | None
    enc_paym_22: int | None
    enc_paym_23: int | None
    enc_paym_24: int | None
    enc_loans_account_holder_type: int | None
    enc_loans_credit_status: int | None
    enc_loans_credit_type: int | None
    enc_loans_account_cur: int | None
    pclose_flag: int | None
    fclose_flag: int | None


class ApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: int
    credit_history: list[CreditHistoryRecord] = Field(min_length=1, max_length=2048)

    @model_validator(mode="after")
    def validate_record_numbers(self) -> "ApplicationRequest":
        rn = [row.rn for row in self.credit_history if row.rn is not None]
        if len(rn) != len(set(rn)):
            raise ValueError("credit_history contains duplicate non-null rn values")
        return self


class ScoreRequest(ApplicationRequest):
    include_explanations: bool = True


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[ApplicationRequest] = Field(min_length=1, max_length=128)
    include_explanations: bool = False

    @model_validator(mode="after")
    def validate_application_ids(self) -> "BatchScoreRequest":
        ids = [item.application_id for item in self.applications]
        if len(ids) != len(set(ids)):
            raise ValueError("applications contains duplicate application_id values")
        return self


class ReasonCodeResponse(BaseModel):
    code: str
    description: str
    direction: Literal["risk_increasing", "risk_reducing"]
    contribution_log_odds: float
    representative_feature: str
    representative_feature_value: float | None
    representative_feature_contribution_log_odds: float
    caveat: str


class ScoreResponse(BaseModel):
    model_version: str
    artifact_fingerprint: str
    application_id: int
    probability_of_default: float
    raw_model_probability: float
    credit_score: int
    risk_band: Literal["A", "B", "C", "D", "E"]
    decision: Literal["APPROVE", "REVIEW", "REJECT"]
    explanation_domain: str
    risk_increasing_reasons: list[ReasonCodeResponse]
    risk_reducing_reasons: list[ReasonCodeResponse]
    request_id: str | None = None
    processing_ms: float | None = None


class BatchScoreResponse(BaseModel):
    model_version: str
    artifact_fingerprint: str
    request_id: str
    batch_size: int
    processing_ms: float
    items: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    model_version: str
    artifact_fingerprint: str


class ModelInfoResponse(BaseModel):
    model_version: str
    api_version: str
    n_features: int
    calibration_method: str
    primary_policy: str
    approve_pd_threshold: float
    reject_pd_threshold: float
    risk_band_labels: list[str]
    risk_band_cutpoints: list[float]
    explanation_domain: str
    artifact_fingerprint: str
