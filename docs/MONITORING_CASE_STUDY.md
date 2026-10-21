# Monitoring Case Study — Stable Portfolio Mix, Weaker Discrimination

## Context

M10/M10.1 adds production-style monitoring to the already frozen Model V1. The reference population is chunks 8+9; the historical current window is the already-opened M6 lockbox (chunks 10+11).

The alert thresholds were introduced **after** M6 had been evaluated. This page therefore documents a **retrospective replay**, not a claim that the rules were prospectively predeclared before the lockbox.

## What label-free monitoring saw

### Schema

Reference chunks 8 and 9 each produced 214 raw canonical columns, aligned to the frozen 212-feature Model V1 schema by dropping the same two extra category-derived columns. Chunks 10 and 11 produced the same raw feature-name signature.

M10.1 therefore reports:

- `reference_schema_match=True` for chunk 10;
- `reference_schema_match=True` for chunk 11;
- no schema-drift alert.

This is preferable to warning merely because `raw/frozen=214/212`: behavior already present in the baseline is not drift.

### Feature distributions

- median PSI: **0.001790**;
- P90 PSI: **0.004964**;
- maximum PSI: **0.122707** for `pre_since_confirmed__max`.

Only the maximum crossed the configured warning threshold (0.10). Related members of the same feature family also moved (`pre_since_confirmed__mean` PSI 0.0635; `__min` PSI 0.0218), suggesting a localized family-level shift rather than widespread marginal covariate drift.

The project does **not** infer the real-world meaning of this anonymized feature. This does not establish that the localized shift caused downstream performance degradation.

### Prediction / policy stability

- PD PSI: **0.000739**;
- mean PD: **3.5430% → 3.5848%** (+0.0418 pp);
- band A share: 21.01% → 20.11%;
- band E share: 21.75% → 22.04%;
- APPROVE: **94.68% → 94.65%**;
- REVIEW: **2.04% → 2.05%**;
- REJECT: **3.27% → 3.30%**.

From an operational, label-free perspective, the portfolio looked broadly stable.

## What delayed labels revealed

Once outcomes are available:

| Metric | Reference | Current | Delta |
|---|---:|---:|---:|
| raw ROC-AUC | 0.735320 | **0.717098** | **-0.018223** |
| calibrated ROC-AUC | 0.735132 | 0.716828 | -0.018304 |
| Brier | 0.031197 | 0.031068 | -0.000130 |
| calibration bias | +0.4091 pp | +0.3034 pp | -0.1057 pp |
| calibration slope | 0.9932 | 0.9177 | -0.0755 |

The configured AUC warning threshold is a drop of 0.015; the raw-score decline therefore triggered `raw_roc_auc_drop`.

Brier did not worsen, partly illustrating why no single metric is sufficient: probability error, ranking quality, prevalence, and calibration can move differently.

## Final replay status

```text
overall_status: WARNING
alerts: 2 warning / 0 critical

1. feature_drift   psi::pre_since_confirmed__max
2. delayed_outcome raw_roc_auc_drop
```

## Production lesson

**Stable feature PSI, PD distribution, and decision rates do not guarantee stable ranking performance.** A credit-risk monitoring program needs both:

1. immediate label-free operational indicators; and
2. delayed-outcome surveillance when labels mature.

The replay also demonstrates an important restraint: the co-occurrence of one localized feature shift and lower AUC is not evidence of causation or proof of concept drift.
