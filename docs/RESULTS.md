# Consolidated Results

This file collects the main empirical results without requiring readers to reconstruct them from stage artifacts.

## 1. Feature representation

| Variant / stage | Features | Evaluation | ROC-AUC | Notes |
|---|---:|---|---:|---|
| M1 legacy | 414 | single holdout | 0.738626 | generic arithmetic aggregation; reference only |
| M2 typed | 167 | 3-fold CV | 0.722140 ± 0.001428 | safer semantics but underfit representation |
| M2.1 legacy M1 | 414 | same 3-fold CV | 0.744054 | includes questionable latest-by-rn block |
| M2.1 canonical E | 212 | same 3-fold CV | **0.743519 ± 0.002169** | selected representation |

M2.1 category-distribution and payment-dynamics additions recovered most of the predictive gap between the compact typed representation and the legacy feature set.

## 2. Model family and tuning

| Model | Development ROC-AUC | External chunks 1–3 ROC-AUC | Interpretation |
|---|---:|---:|---|
| Logistic Regression | 0.743519 ± 0.002169 | 0.734594 | linear baseline |
| M3 CatBoost baseline | 0.755291 ± 0.002703 | **0.748592** | production candidate |
| M3.1 tuned CatBoost | 0.756779 ± 0.002575 | 0.749028 | small tuning gain did not convincingly generalize |

External paired bootstrap:

- M3 CatBoost − Logistic: `+0.013998` AUC, 95% CI `[+0.012506, +0.015203]`.
- Tuned − M3 CatBoost: `+0.000436` AUC, 95% CI `[-0.000073, +0.000908]`.

The production-oriented choice remains the simpler baseline CatBoost (314 vs 474 median trees for the tuned candidate).

## 3. Calibration

### Chunk 9 method validation

| Mapping | Brier | Log loss | Mean PD | Observed DR | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| raw weighted score | 0.196911 | 0.574826 | 40.5641% | 3.3136% | 0.735320 |
| Platt | 0.031208 | **0.134056** | 3.7273% | 3.3136% | 0.735320 |
| Isotonic | **0.031197** | 0.134081 | 3.7227% | 3.3136% | 0.735132 |

Isotonic was selected by the predeclared Brier rule. Platt and isotonic were practically very close; no substantive superiority is claimed.

## 4. Final lockbox

Population: chunks 10+11, 500,000 applications.

| Metric | Result | 95% bootstrap CI |
|---|---:|---:|
| ROC-AUC | **0.716828** | [0.713127, 0.720598] |
| Average Precision | **0.077197** | [0.075467, 0.079246] |
| Brier | **0.031068** | [0.031028, 0.031105] |
| Log loss | **0.134804** | [0.134414, 0.135185] |
| KS | 0.317859 | — |
| Mean calibrated PD | 3.5848% | — |
| Observed default rate | 3.2814% | — |
| Calibration bias | +0.3034 pp | — |
| Calibration slope | 0.9177 | — |

The discrimination gap versus M3.2 is material and is retained in the public project narrative.

## 5. Risk bands on final lockbox

| Band | Share | Mean PD | Observed DR | Default capture |
|---|---:|---:|---:|---:|
| A | 20.11% | 0.6136% | 0.6732% | 4.13% |
| B | 19.23% | 1.4902% | 1.5390% | 9.02% |
| C | 18.60% | 2.6148% | 2.5035% | 14.19% |
| D | 20.01% | 4.3406% | 3.8834% | 23.69% |
| E | 22.04% | 8.2551% | 7.2908% | 48.98% |

Observed default rate remains monotonic across A→E.

## 6. Balanced decision policy

| Metric | Development chunks 8+9 | Final lockbox 10+11 |
|---|---:|---:|
| APPROVE | 94.68% | 94.65% |
| REVIEW | 2.04% | 2.05% |
| REJECT | 3.27% | 3.30% |
| Approved observed DR | 3.0540% | 2.8604% |
| Reject default capture | 12.48% | 12.26% |
| Review+reject default capture | 18.39% | 17.49% |

Economics are normalized/illustrative, so portfolio-rate stability is more meaningful here than absolute profit claims.

## 7. Monitoring historical replay

| Signal | Reference 8+9 / chunk 9 | Historical current 10+11 | Status |
|---|---:|---:|---|
| median feature PSI | — | 0.001790 | stable overall |
| max feature PSI | — | 0.122707 | warning: localized |
| PD PSI | — | 0.000739 | stable |
| mean PD | 3.5430% | 3.5848% | stable |
| APPROVE | 94.68% | 94.65% | stable |
| raw ROC-AUC | 0.735320 | 0.717098 | **WARNING** |
| calibration slope | 0.9932 | 0.9177 | below configured warning threshold |

Final M10.1 overall status: **WARNING — 2 warnings, 0 critical**.
