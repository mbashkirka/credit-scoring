# Reproducibility Guide

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
python scripts/check_m9_environment.py
make static
make test
make release-check
```

Production direct pins are in `requirements/runtime.txt`; research/test dependencies are in `requirements/dev.txt`.

## 2. Data layout

```text
data/raw/data_for_competition/
├── train_target.csv
├── train_data_0.pq
├── train_data_1.pq
├── ...
└── train_data_11.pq
```

Raw data and generated model artifacts are ignored by Git.

## 3. Stage order

The project is deliberately staged. Do not run later selection stages and then reinterpret earlier holdouts as untouched.

### M1 — baseline

```bash
python scripts/inspect_data.py
python scripts/train_baseline.py --max-files 1
```

### M2 / M2.1 — feature engineering and ablation

```bash
python scripts/train_m2.py --max-files 1 --cv-folds 3
python scripts/train_m21_ablation.py --max-files 1 --cv-folds 3
```

### M3 — model family

```bash
python scripts/train_m3_models.py --max-files 1 --cv-folds 3
```

### M3.1 — controlled tuning

```bash
python scripts/train_m31_tuning.py --max-files 1 --cv-folds 3
```

### M3.2 — external validation

```bash
python scripts/evaluate_m32_holdout.py --holdout-chunks 1 2 3
```

### M4 — final base fit + calibration

```bash
python scripts/train_m4_calibration.py
```

### M5 — decision engine

```bash
python scripts/build_m5_decision_engine.py
```

### M6 — final one-shot lockbox

```bash
python scripts/evaluate_m6_lockbox.py
```

After this command, chunks 10–11 are no longer untouched.

### M7 — explanations

```bash
python scripts/build_m7_explainability.py
```

### M8 — serving consistency

```bash
python scripts/validate_m8_serving_consistency.py --source-chunk 9 --n-applications 10
make api
```

### M9 — packaging

```bash
make verify-artifacts
make docker-up
python scripts/m9_container_smoke_test.py --base-url http://127.0.0.1:8000
docker compose exec credit-risk-api id -u
make docker-down
```

### M10 / M10.1 — monitoring

```bash
python scripts/build_m10_reference_profile.py
python scripts/build_m10_1_schema_reference.py
python scripts/run_m10_monitoring.py
```

### M11 — source release validation

```bash
python scripts/validate_m11_release.py
```

## 4. Serving artifact contract

A production Docker build requires:

```text
artifacts/m4_catboost.cbm
artifacts/m4_catboost_imputer.joblib
artifacts/m4_calibrator.joblib
artifacts/m3_features.csv
artifacts/m4_metadata.json
artifacts/m5_policy_freeze.json
artifacts/m5_risk_band_thresholds.json
```

Verify before serving:

```bash
python scripts/verify_m9_serving_artifacts.py --load-runtime
```

The expected Model V1 fingerprint from the documented run is:

```text
14a17e7ac80680624a0587f9cf009b0d4e0f61b98eec300fe5297087c496f301
```

If the fingerprint changes, treat it as a different serving bundle until investigated.

## 5. CI

GitHub Actions runs:

- Python 3.11 and 3.12 compile/Ruff/pytest/release validation;
- ephemeral synthetic Model-V1-compatible artifact generation;
- actual Docker build;
- container startup;
- `/health`, `/ready`, `/v1/model-info`, `/v1/score` smoke;
- Docker health-state check;
- non-root UID check.

Synthetic CI artifacts validate packaging contracts only; they are not accuracy benchmarks.


## Local data and artifacts

`data/` and `artifacts/` are intentionally allowed to remain in a developer's working tree. They contain downloaded Kaggle data, derived feature caches, fitted model objects, predictions, and experiment outputs needed for local reproducibility. The public-release contract is therefore **Git-based**, not a requirement to delete local state: `.gitignore` excludes all payloads under these roots while permitting `.gitkeep` placeholders. If files were tracked before the ignore policy was added, remove them from the Git index without deleting the local copies (for example, `git rm -r --cached data artifacts`).

