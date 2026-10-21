# Governance, Assumptions, and Limitations

## 1. Experimental governance

The project progressively freezes components rather than re-optimizing everything after each holdout result.

- **Feature freeze:** M2.1 selects the 212-feature canonical representation.
- **Base-model freeze:** M3.2 confirms the nonlinear CatBoost gain externally and rejects an unconvincing small tuning gain.
- **Calibration protocol:** chunk 8 fits candidate calibrators; chunk 9 selects the method by predeclared Brier rule.
- **Policy freeze:** M5 freezes risk bands and the `balanced` decision thresholds before M6.
- **Lockbox:** M6 opens chunks 10–11 once and records a fingerprint of the frozen upstream artifacts.
- **Monitoring:** M10 thresholds are introduced after M6 and explicitly labeled retrospective for the M6 replay.

## 2. Lockbox status

Chunks 10–11 are **OPENED_AND_EVALUATED**. They can be used to explain the historical final evaluation and monitoring replay, but not to support future claims of unbiased model selection or tuning.

## 3. No timestamp assumption

The repository does not establish that parquet chunk number or `rn` is a real chronological variable. Therefore:

- no `latest-by-rn` feature is used in Model V1;
- chunk-to-chunk score changes are not called temporal drift without timestamp evidence;
- concept drift is not claimed from AUC degradation alone; the project does not claim temporal drift from chunk ordering.

## 4. Probability semantics

Balanced class weights distort raw probability levels. The base CatBoost `predict_proba` is therefore treated as a ranking-oriented score until a separate calibration layer is applied.

The selected isotonic mapping is monotone but piecewise constant; it may introduce ties and therefore cause very small differences between raw-score and calibrated-PD AUC.

## 5. Economic assumptions

M5 demonstrates a decision engine using normalized `EAD=1` and configurable performing margin, LGD, and origination-cost assumptions. The anonymous dataset does not contain enough information to estimate real LGD, EAD, pricing, or lifetime profitability.

No absolute profit result should be interpreted as an estimate of a bank's actual economics.

## 6. Explainability boundary

TreeSHAP provides a predictive decomposition of the CatBoost raw score. It is not causal. The project does not convert SHAP values into additive calibrated-PD percentage-point contributions.

Semantic reason groups are internal engineering/model-governance prototypes. Regulatory adverse-action notices would require institution-specific legal/compliance validation and stable human-readable feature definitions.

## 7. Fairness and protected attributes

The anonymized dataset does not provide a defensible basis for comprehensive protected-class fairness analysis. The absence of a fairness result is not evidence of fairness.

A real deployment would require jurisdiction-specific fair-lending review, proxy analysis, subgroup performance/calibration checks where legally permissible, and governance of protected data.

## 8. External validity

The project uses one Kaggle source. Chunks 1–3 provide an external holdout relative to the initial development chunk, but they are not a different bank, geography, underwriting policy, or macroeconomic regime.

## 9. Production gaps intentionally not fabricated

The project does not claim to have:

- real online A/B experiment results;
- production traffic/latency SLO history;
- real bank profit uplift;
- legal approval for automated lending;
- causal explanations;
- evidence of temporal drift without timestamps.

These omissions are deliberate rather than filled with simulated claims.
