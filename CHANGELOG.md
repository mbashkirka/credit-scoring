# Changelog

## 0.11.0 — M11 Portfolio Release
- Fixed the M11 source-release validator to distinguish local ignored ML state from files actually publishable/tracked by Git.

- Replaced experiment-log-style root README with a concise portfolio narrative.
- Added architecture, model card, consolidated results, governance, reproducibility, monitoring case study, and demo/interview documentation.
- Added repository-level release validator and CI release-contract checks.
- Added explicit guards against calling chunks 10–11 untouched after M6.
- No Model V1, calibration, risk-band, decision-policy, serving-artifact, or M10 alert-threshold changes.

## 0.10.1 — M10.1

- Made schema drift baseline-aware by freezing observed raw reference schemas on chunks 8+9.
- Removed false-positive alerts for the two baseline extra category-derived columns.

## 0.10.0 — M10

- Added production monitoring for schema/data quality, feature PSI, PD/band/decision drift, and delayed-outcome performance/calibration.

## 0.9.1 — M9.1

- Matched serving scikit-learn runtime to the serialized isotonic calibrator and added a hard compatibility guard.

## 0.9.0 — M9

- Added Docker packaging, pinned serving dependencies, non-root runtime, container healthcheck, and GitHub Actions container contract.

## 0.8.0 — M8

- Added FastAPI serving with shared offline/online feature engineering and artifact fingerprinting.

Earlier milestones M1–M7 are documented in `docs/RESULTS.md` and repository history.
