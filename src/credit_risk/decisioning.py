from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


_EPS = 1e-8
RISK_BAND_LABELS = ("A", "B", "C", "D", "E")
DECISION_LABELS = ("APPROVE", "REVIEW", "REJECT")


@dataclass(frozen=True)
class BusinessScenario:
    """Transparent normalized economics for a decision-policy sensitivity scenario.

    All monetary terms are expressed as fractions of a normalized Exposure at Default
    (EAD=1). They are illustrative portfolio assumptions, not values inferred from the
    anonymized Kaggle data.
    """

    name: str
    performing_margin_rate: float
    lgd: float
    origination_cost_rate: float
    risk_appetite_pd_cap: float
    review_band_fraction: float
    primary: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scenario name must be non-empty.")
        for field_name in (
            "performing_margin_rate",
            "lgd",
            "origination_cost_rate",
            "risk_appetite_pd_cap",
            "review_band_fraction",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if not 0.0 < self.performing_margin_rate < 1.0:
            raise ValueError("performing_margin_rate must be in (0, 1).")
        if not 0.0 < self.lgd <= 1.0:
            raise ValueError("lgd must be in (0, 1].")
        if not 0.0 <= self.origination_cost_rate < self.performing_margin_rate:
            raise ValueError("origination_cost_rate must be >=0 and lower than margin.")
        if not 0.0 < self.risk_appetite_pd_cap < 1.0:
            raise ValueError("risk_appetite_pd_cap must be in (0, 1).")
        if not 0.0 <= self.review_band_fraction < 1.0:
            raise ValueError("review_band_fraction must be in [0, 1).")

    @property
    def break_even_pd(self) -> float:
        """PD where expected normalized profit from approval equals zero."""
        return (self.performing_margin_rate - self.origination_cost_rate) / (
            self.performing_margin_rate + self.lgd
        )

    @property
    def reject_pd_threshold(self) -> float:
        """Hard automated-risk ceiling: the tighter of economics and risk appetite."""
        return min(self.break_even_pd, self.risk_appetite_pd_cap)

    @property
    def approve_pd_threshold(self) -> float:
        """Lower boundary of a review zone immediately below the risk ceiling."""
        return self.reject_pd_threshold * (1.0 - self.review_band_fraction)

    def to_record(self) -> dict[str, float | str | bool]:
        return {
            **asdict(self),
            "break_even_pd": self.break_even_pd,
            "approve_pd_threshold": self.approve_pd_threshold,
            "reject_pd_threshold": self.reject_pd_threshold,
        }


DEFAULT_SCENARIOS: tuple[BusinessScenario, ...] = (
    BusinessScenario(
        name="growth",
        performing_margin_rate=0.14,
        lgd=0.40,
        origination_cost_rate=0.015,
        risk_appetite_pd_cap=0.20,
        review_band_fraction=0.15,
        primary=False,
    ),
    BusinessScenario(
        name="balanced",
        performing_margin_rate=0.12,
        lgd=0.45,
        origination_cost_rate=0.020,
        risk_appetite_pd_cap=0.12,
        review_band_fraction=0.20,
        primary=True,
    ),
    BusinessScenario(
        name="conservative",
        performing_margin_rate=0.10,
        lgd=0.55,
        origination_cost_rate=0.025,
        risk_appetite_pd_cap=0.08,
        review_band_fraction=0.25,
        primary=False,
    ),
)


def validate_probability(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probability must be one-dimensional.")
    if not np.all(np.isfinite(p)):
        raise ValueError("probability contains non-finite values.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must lie in [0, 1].")
    return np.clip(p, _EPS, 1.0 - _EPS)


def derive_quantile_band_cutpoints(
    probability: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80),
) -> np.ndarray:
    """Derive deployable PD cutpoints approximating equal-frequency development bands.

    Isotonic calibration can create probability ties. Cutpoints are therefore placed
    strictly between adjacent *distinct* PD values, choosing cumulative-mass boundaries
    closest to the requested quantiles while preserving increasing thresholds.
    """
    p = validate_probability(probability)
    if len(quantiles) != 4:
        raise ValueError("Exactly four quantiles are required for five risk bands.")
    q = np.asarray(quantiles, dtype=np.float64)
    if np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q) <= 0):
        raise ValueError("quantiles must be strictly increasing and inside (0, 1).")

    unique, counts = np.unique(p, return_counts=True)
    if unique.size < len(q) + 1:
        raise ValueError("Not enough distinct PD values to construct five frozen bands.")

    cumulative_mass = np.cumsum(counts)[:-1] / len(p)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    chosen: list[int] = []
    previous = -1
    total_candidates = len(midpoints)
    for i, target in enumerate(q):
        remaining_boundaries = len(q) - i - 1
        start = previous + 1
        stop_exclusive = total_candidates - remaining_boundaries
        candidate_idx = np.arange(start, stop_exclusive)
        if candidate_idx.size == 0:
            raise RuntimeError("Unable to construct strictly increasing PD band cutpoints.")
        best = int(candidate_idx[np.argmin(np.abs(cumulative_mass[candidate_idx] - target))])
        chosen.append(best)
        previous = best
    cutpoints = midpoints[np.asarray(chosen, dtype=int)]
    if np.any(np.diff(cutpoints) <= 0):
        raise RuntimeError("Derived risk-band cutpoints are not strictly increasing.")
    return cutpoints.astype(np.float64)


def assign_risk_bands(
    probability: np.ndarray,
    cutpoints: np.ndarray,
    *,
    labels: tuple[str, ...] = RISK_BAND_LABELS,
) -> np.ndarray:
    p = validate_probability(probability)
    cuts = np.asarray(cutpoints, dtype=np.float64)
    if cuts.ndim != 1 or len(cuts) != len(labels) - 1:
        raise ValueError("cutpoints must contain one fewer value than labels.")
    if not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("cutpoints must be finite and strictly increasing.")
    idx = np.digitize(p, cuts, right=True)
    return np.asarray(labels, dtype=object)[idx]


def risk_band_summary(
    y_true: np.ndarray,
    probability: np.ndarray,
    bands: np.ndarray,
    *,
    labels: tuple[str, ...] = RISK_BAND_LABELS,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=np.int8)
    p = validate_probability(probability)
    b = np.asarray(bands, dtype=object)
    if not (len(y) == len(p) == len(b)):
        raise ValueError("y_true, probability and bands lengths differ.")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("y_true must be binary.")
    total_bad = int(y.sum())
    portfolio_dr = float(y.mean())
    rows: list[dict[str, float | int | str]] = []
    for label in labels:
        mask = b == label
        if not np.any(mask):
            continue
        bads = int(y[mask].sum())
        observed = float(y[mask].mean())
        predicted = float(p[mask].mean())
        rows.append(
            {
                "risk_band": label,
                "n": int(mask.sum()),
                "population_share": float(mask.mean()),
                "pd_min": float(p[mask].min()),
                "pd_max": float(p[mask].max()),
                "mean_pd": predicted,
                "median_pd": float(np.median(p[mask])),
                "observed_default_rate": observed,
                "calibration_gap": predicted - observed,
                "bad_count": bads,
                "bad_capture_rate": float(bads / total_bad) if total_bad else float("nan"),
                "risk_multiple_vs_portfolio": (
                    float(observed / portfolio_dr) if portfolio_dr > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_profit_rate(probability: np.ndarray, scenario: BusinessScenario) -> np.ndarray:
    """Expected profit as a fraction of normalized EAD for an approved application."""
    p = validate_probability(probability)
    return (
        (1.0 - p) * scenario.performing_margin_rate
        - p * scenario.lgd
        - scenario.origination_cost_rate
    )


def expected_loss_rate(probability: np.ndarray, scenario: BusinessScenario) -> np.ndarray:
    p = validate_probability(probability)
    return p * scenario.lgd


def realized_profit_rate(y_true: np.ndarray, scenario: BusinessScenario) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.int8)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("y_true must be binary.")
    return (
        (1.0 - y) * scenario.performing_margin_rate
        - y * scenario.lgd
        - scenario.origination_cost_rate
    ).astype(np.float64)


def assign_decisions(probability: np.ndarray, scenario: BusinessScenario) -> np.ndarray:
    p = validate_probability(probability)
    decision = np.full(len(p), "REJECT", dtype=object)
    decision[p <= scenario.reject_pd_threshold] = "REVIEW"
    decision[p <= scenario.approve_pd_threshold] = "APPROVE"
    return decision


def evaluate_policy(
    y_true: np.ndarray,
    probability: np.ndarray,
    scenario: BusinessScenario,
    decisions: np.ndarray | None = None,
) -> dict[str, float | int | str | bool]:
    y = np.asarray(y_true, dtype=np.int8)
    p = validate_probability(probability)
    if len(y) != len(p):
        raise ValueError("y_true and probability lengths differ.")
    d = assign_decisions(p, scenario) if decisions is None else np.asarray(decisions, dtype=object)
    if len(d) != len(p):
        raise ValueError("decisions length differs from probability.")
    unknown = set(np.unique(d).tolist()) - set(DECISION_LABELS)
    if unknown:
        raise ValueError(f"Unknown decision labels: {sorted(unknown)}")

    approve = d == "APPROVE"
    review = d == "REVIEW"
    reject = d == "REJECT"
    bad_total = int(y.sum())

    exp_profit = expected_profit_rate(p, scenario)
    exp_loss = expected_loss_rate(p, scenario)
    realized_profit = realized_profit_rate(y, scenario)
    realized_loss = y.astype(np.float64) * scenario.lgd

    def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if np.any(mask) else float("nan")

    approved_bad_count = int(y[approve].sum()) if np.any(approve) else 0
    rejected_bad_count = int(y[reject].sum()) if np.any(reject) else 0
    nonapproved_bad_count = int(y[~approve].sum()) if np.any(~approve) else 0

    return {
        "scenario": scenario.name,
        "primary": scenario.primary,
        "n": int(len(y)),
        "portfolio_default_rate": float(y.mean()),
        "break_even_pd": scenario.break_even_pd,
        "approve_pd_threshold": scenario.approve_pd_threshold,
        "reject_pd_threshold": scenario.reject_pd_threshold,
        "approval_rate": float(approve.mean()),
        "review_rate": float(review.mean()),
        "rejection_rate": float(reject.mean()),
        "approved_mean_pd": _safe_mean(p, approve),
        "approved_observed_default_rate": _safe_mean(y.astype(np.float64), approve),
        "approved_bad_count": approved_bad_count,
        "approved_calibration_gap": (
            _safe_mean(p, approve) - _safe_mean(y.astype(np.float64), approve)
        ),
        "expected_loss_rate_per_approved": _safe_mean(exp_loss, approve),
        "expected_loss_rate_per_applicant": float(np.where(approve, exp_loss, 0.0).mean()),
        "realized_loss_rate_per_approved": _safe_mean(realized_loss, approve),
        "realized_loss_rate_per_applicant": float(np.where(approve, realized_loss, 0.0).mean()),
        "expected_profit_rate_per_approved": _safe_mean(exp_profit, approve),
        "expected_profit_rate_per_applicant": float(np.where(approve, exp_profit, 0.0).mean()),
        "realized_profit_rate_per_approved": _safe_mean(realized_profit, approve),
        "realized_profit_rate_per_applicant": float(np.where(approve, realized_profit, 0.0).mean()),
        # Backward-compatible M5 names. They are diagnostics on M5 development data;
        # M6 uses the generic aliases above for genuine lockbox outcomes.
        "realized_profit_rate_per_approved_dev": _safe_mean(realized_profit, approve),
        "realized_profit_rate_per_applicant_dev": float(
            np.where(approve, realized_profit, 0.0).mean()
        ),
        "default_capture_reject": (
            float(rejected_bad_count / bad_total) if bad_total else float("nan")
        ),
        "default_capture_review_or_reject": (
            float(nonapproved_bad_count / bad_total) if bad_total else float("nan")
        ),
        "default_share_approved": (
            float(approved_bad_count / bad_total) if bad_total else float("nan")
        ),
    }


def pd_to_score(
    probability: np.ndarray,
    *,
    base_score: float = 600.0,
    base_good_to_bad_odds: float = 20.0,
    points_to_double_odds: float = 50.0,
    min_score: float = 300.0,
    max_score: float = 850.0,
) -> np.ndarray:
    """Illustrative scorecard-style monotone mapping from PD to a credit score.

    At ``base_score``, good:bad odds equal ``base_good_to_bad_odds``. Every
    ``points_to_double_odds`` points doubles the good:bad odds. This mapping is
    presentation-only and does not alter PD or decisions.
    """
    p = validate_probability(probability)
    if base_good_to_bad_odds <= 0 or points_to_double_odds <= 0:
        raise ValueError("Odds and PDO must be positive.")
    good_bad_odds = (1.0 - p) / p
    factor = points_to_double_odds / np.log(2.0)
    score = base_score + factor * np.log(good_bad_odds / base_good_to_bad_odds)
    return np.clip(score, min_score, max_score)
