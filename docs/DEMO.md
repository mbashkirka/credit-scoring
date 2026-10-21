# Demo and Interview Walkthrough

## 30-second summary

> I built an end-to-end credit-risk system on 3 million anonymized applications. I started with a reproducible Logistic baseline, ran controlled feature ablation, demonstrated a statistically supported CatBoost gain on a frozen external holdout, calibrated the class-weighted model into PD, froze risk bands and a decision policy before a one-shot final lockbox, then packaged the exact same feature/model path behind FastAPI and Docker. The most interesting result is that final monitoring showed almost no PD or decision drift while delayed labels revealed an AUC drop, which is why I implemented both label-free and delayed-outcome monitoring.

## 90-second technical walkthrough

1. **Feature semantics:** encoded categorical fields are not naively averaged; the canonical 212-feature set uses category distributions and payment transition/composition features.
2. **Controlled comparison:** Logistic and CatBoost use the same rows, folds, imputation, and class weights.
3. **Model-selection restraint:** a tuned CatBoost improved CV slightly but not convincingly on external validation, so the simpler baseline configuration was retained.
4. **PD calibration:** because balanced class weights make raw probabilities unsuitable as PD, Platt and isotonic mappings were compared on a dedicated calibration-validation split.
5. **Policy freeze:** risk bands and APPROVE/REVIEW/REJECT thresholds were frozen before the final lockbox.
6. **Final result:** lockbox AUC fell to ~0.717 from ~0.749 on the earlier external holdout; the project reports the gap rather than tuning it away.
7. **Serving:** the API takes raw credit history and calls the same aggregation code used offline; consistency tests match PD to ~`1e-9`.
8. **Monitoring:** PD PSI was only `0.000739` and approval mix was essentially unchanged, but delayed raw AUC dropped by `0.0182` — a concrete reason to monitor labels, not only PSI.

## Live demo

### Start API

```bash
make verify-artifacts
make api
```

### Inspect model contract

```bash
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/model-info
```

### Score the checked-in example

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/score \
  -H 'content-type: application/json' \
  --data-binary @examples/m8_score_request.json
```

### Docker demo

```bash
make docker-up
python scripts/m9_container_smoke_test.py --base-url http://127.0.0.1:8000
docker compose exec credit-risk-api id -u
make docker-down
```

Expected runtime UID: `10001`.

## Questions this project is ready to answer

### Why CatBoost?

Not because it is fashionable: the frozen external comparison showed an AUC gain of ~0.014 over Logistic with a paired-bootstrap CI fully above zero.

### Why not use the tuned model?

The tuned candidate improved development CV by ~0.0015 AUC, but its external delta over baseline was only +0.00044 with a CI crossing zero and it required ~51% more trees.

### Why calibrate?

Balanced class weights made the raw model output average around 40% on a population with ~3–4% default. Calibration reduced Brier/log-loss dramatically and restored PD-level interpretability.

### Did the model drift over time?

The dataset does not establish timestamps. The project can say performance differs across held-out chunks and that a monitoring replay detects degradation; it cannot honestly call the chunk sequence temporal drift.

### Are SHAP reason codes causal?

No. They explain the model's raw log-odds prediction relative to its baseline. They are internal predictive reason codes, not causal explanations or approved adverse-action notices.

### Why is the final AUC lower?

That is exactly why the lockbox exists. The earlier external score was not guaranteed to replicate on the final reserved population. The project preserves the lower result and uses it to motivate monitoring rather than tuning on the lockbox.
