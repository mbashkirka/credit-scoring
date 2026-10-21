# Model Card — Credit Risk Engine Model V1

## Model details

- **Model name:** Credit Risk Engine Model V1
- **Repository release:** M11 / package `0.11.0`
- **Task:** binary default-risk ranking and calibrated probability-of-default estimation on anonymized credit-history data.
- **Base model:** CatBoost classifier.
- **Frozen configuration:** depth 6, learning rate 0.05, `l2_leaf_reg=5`, 314 trees.
- **Feature representation:** 212 application-level features from `e_typed_catdist_payment`.
- **Calibration:** isotonic regression.
- **Primary policy:** `balanced`; approve PD ≤ 9.6%, review 9.6–12%, reject > 12%.
- **Serving fingerprint observed in M8–M10 runs:** `14a17e7ac80680624a0587f9cf009b0d4e0f61b98eec300fe5297087c496f301`.

The fingerprint identifies the seven frozen serving artifacts; M11 documentation changes do not alter Model V1.

## Intended use

This model is intended as an **educational portfolio demonstration** of end-to-end credit-risk ML engineering: feature construction, validation, probability calibration, decisioning, explainability, serving, containerization, and monitoring.

It is **not** intended for real lending decisions without institution-specific data governance, legal/compliance review, model-risk validation, economic parameterization, fairness testing, and production controls. In short, this portfolio model is not intended for real lending decisions as-is.

## Data

The project uses the Kaggle `alfa-bank-pd-credit-history` competition dataset:

- target table: 3,000,000 applications;
- observed overall positive/default rate: ~3.55%;
- 12 credit-history parquet chunks;
- fields are anonymized integer-coded variables.

The project does not infer unsupported real-world meanings for anonymized coded columns.

### Data-role governance

- chunk 0: initial development/model comparison;
- chunks 1–3: external validation of frozen development candidates;
- chunks 0–7: final base-model training;
- chunk 8: calibration fit;
- chunk 9: calibration-method validation;
- chunks 8+9: final selected calibrator refit, risk-band/policy development, monitoring reference;
- chunks 10+11: M6 final lockbox, now `OPENED_AND_EVALUATED`.

Chunk order is not treated as a verified timeline. No temporal-drift claim is made from chunk number alone.

## Feature engineering

The canonical representation contains 212 features and emphasizes category-code-safe summaries:

- `pre_*`: compact numeric distribution statistics;
- binary flags: rates/counts and related summaries;
- encoded categorical loan fields: per-application category distributions/diversity;
- payment state sequences: equality/change/transition/composition statistics rather than numeric distance between arbitrary codes;
- no `latest` feature based on maximum `rn`.

The representation was selected by controlled ablation. A larger legacy 414-feature block produced slightly higher Logistic CV AUC (0.744054 vs 0.743519), but the 212-feature representation was chosen for stability, compactness, and safer encoded-category semantics.

## Training and selection

Model-family comparison used the same frozen feature representation, rows, folds, median imputation, and balanced class-weighting policy.

M3 CatBoost achieved CV ROC-AUC `0.755291 ± 0.002703`, compared with Logistic `0.743519 ± 0.002169`. On external chunks 1–3, frozen M3 CatBoost achieved ROC-AUC `0.748592` vs Logistic `0.734594`; paired bootstrap delta was `+0.013998` with 95% CI `[+0.012506, +0.015203]`.

Controlled M3.1 tuning produced only a small external delta over the baseline CatBoost (`+0.000436`, 95% CI `[-0.000073, +0.000908]`). Therefore the simpler 314-tree M3 baseline configuration was retained for Model V1.

## Probability calibration

The base model uses balanced class weights, so its raw `predict_proba` is not treated as PD. M4 fits Platt and isotonic calibration on chunk 8, selects by predeclared Brier-score rule on chunk 9, then refits the selected isotonic calibrator on chunks 8+9.

Chunk-9 validation for isotonic calibration:

- ROC-AUC: 0.735132;
- Brier: 0.031197;
- log loss: 0.134081;
- mean PD: 3.7227%;
- observed default rate: 3.3136%;
- calibration bias: +0.4091 percentage points;
- calibration slope: 0.9932.

## Final lockbox evaluation

M6 evaluates the fully frozen pipeline once on chunks 10+11 (500,000 applications):

- ROC-AUC: **0.716828**, bootstrap 95% CI **[0.713127, 0.720598]**;
- Average Precision: **0.077197**, 95% CI **[0.075467, 0.079246]**;
- KS: **0.317859**;
- Brier: **0.031068**, 95% CI **[0.031028, 0.031105]**;
- log loss: **0.134804**, 95% CI **[0.134414, 0.135185]**;
- mean calibrated PD: **3.5848%**;
- observed default rate: **3.2814%**;
- calibration bias: **+0.3034 percentage points**;
- calibration slope: **0.9177**.

The lower final AUC relative to M3.2 is reported rather than hidden. Both lockbox chunks showed similar degradation, so it was not attributable to a single anomalous chunk. Small marginal PSI does not prove concept drift, and the repository does not make that claim.

## Decision layer

Risk bands A–E were frozen on chunks 8+9. The final lockbox maintained monotonic observed default rates from A through E; band E represented about 22% of applications and captured about 49% of defaults.

Frozen `balanced` policy on the lockbox:

- APPROVE: 94.65%;
- REVIEW: 2.05%;
- REJECT: 3.30%;
- approved mean PD: 3.0948%;
- approved observed default rate: 2.8604%;
- reject default capture: 12.26%;
- review + reject default capture: 17.49%.

Business expected-loss/profit calculations use normalized `EAD=1` and illustrative margins/LGD/costs. They are methodology demonstrations, not real bank economics.

## Explainability

TreeSHAP explanations are predictive and not causal.

Native CatBoost TreeSHAP is computed in `RawFormulaVal` / raw log-odds space. Additivity was verified numerically (maximum absolute error around `5.3e-15` in the M7 run).

Largest semantic groups by global mean absolute SHAP importance included repayment pattern, exposure/balance, utilization/limit, credit-product mix, and delinquency pattern; the top five groups contributed roughly 88% of global importance in the sampled development population.

**Interpretation constraint:** a negative SHAP value for an overdue-related feature does not mean overdue behavior is causally beneficial. It only means that the application's encoded feature value moved this model's raw score below its model baseline.

## Serving

The M8 API reuses the offline feature implementation. A training-serving consistency test on 10 applications found:

- maximum calibrated-PD difference: `2.841e-09`;
- risk bands: identical;
- decisions: identical;
- rounded credit scores: identical.

M9 packages the API in a non-root Docker image with pinned runtime dependencies and an end-to-end CI container contract.

## Monitoring

M10 reference population: chunks 8+9. Historical M6 replay (chunks 10+11):

- feature PSI median: 0.001790;
- feature PSI P90: 0.004964;
- maximum PSI: 0.122707 (`pre_since_confirmed__max`);
- calibrated-PD PSI: 0.000739;
- mean PD: 3.5430% → 3.5848%;
- raw ROC-AUC: 0.735320 → 0.717098 (warning);
- overall status: WARNING, 2 warnings, 0 critical.

M10 thresholds were defined **after** M6. Therefore this is a retrospective case study, not evidence that the alert policy was prospectively predeclared for the lockbox.

## Limitations and risk considerations

1. **Anonymized semantics.** Coded variables cannot support institution-grade policy interpretation or legal reasoning.
2. **No verified timestamps.** Chunk/rn order is not proof of time; temporal validation is therefore unavailable.
3. **Single competition dataset.** External validation is across held-out chunks from the same source, not a different institution or economic regime.
4. **Class-weighted base fit.** Calibration is essential; raw weighted-model probabilities are not PD.
5. **Calibration transport.** Lockbox slope fell to ~0.918 even though overall Brier remained strong.
6. **Fairness.** Protected-class fairness cannot be adequately evaluated from the available anonymized data and is not claimed.
7. **Decision economics.** LGD/EAD/pricing are illustrative assumptions.
8. **Reason codes.** Internal predictive explanations are not regulatory adverse-action notices.
9. **Monitoring causality.** Co-occurring feature drift and AUC degradation do not establish causal mechanism.
10. **No real online experiment.** Static Kaggle data cannot provide genuine A/B production evidence.

## Ethical / compliance note

A real credit-decision system would require approved feature inventories, data lineage, permissible-purpose review, jurisdiction-specific fair-lending controls, adverse-action validation, independent model-risk validation, audit logging, change management, and ongoing performance/fairness monitoring. Those are outside the evidence available in this dataset.
