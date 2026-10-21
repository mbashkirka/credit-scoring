from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from credit_risk.api.app import create_app
from credit_risk.api.runtime import ScoringRuntime
from credit_risk.api.schemas import ApplicationRequest, CreditHistoryRecord

RAW_DIR = Path("data/raw/data_for_competition/train_data")
OFFLINE = Path("artifacts/m5_decisions.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate M8 raw-history serving path against frozen offline M5 outputs."
    )
    parser.add_argument("--source-chunk", type=int, default=9, choices=[8, 9])
    parser.add_argument("--n-applications", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_applications < 1 or args.n_applications > 100:
        raise ValueError("--n-applications must be in [1, 100]")
    if not OFFLINE.exists():
        raise FileNotFoundError(OFFLINE)
    raw_path = RAW_DIR / f"train_data_{args.source_chunk}.pq"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    offline = pd.read_parquet(OFFLINE)
    offline = offline[offline["source_chunk"] == args.source_chunk].sort_values("id")
    chosen = offline.head(args.n_applications).copy()
    ids = chosen["id"].astype(int).tolist()

    try:
        raw = pd.read_parquet(raw_path, filters=[("id", "in", ids)])
    except Exception:
        raw = pd.read_parquet(raw_path)
        raw = raw[raw["id"].isin(ids)].copy()

    requests: list[ApplicationRequest] = []
    record_columns = list(CreditHistoryRecord.model_fields)
    for app_id in ids:
        rows = raw[raw["id"] == app_id].sort_values("rn")
        records = []
        for row in rows[record_columns].itertuples(index=False, name=None):
            payload = {
                col: (None if pd.isna(value) else int(value))
                for col, value in zip(record_columns, row, strict=True)
            }
            records.append(CreditHistoryRecord(**payload))
        requests.append(ApplicationRequest(application_id=app_id, credit_history=records))

    runtime = ScoringRuntime.load()
    app = create_app(runtime_override=runtime)
    online_rows: list[dict[str, object]] = []
    with TestClient(app) as client:
        for request in requests:
            payload = {
                **request.model_dump(mode="json"),
                "include_explanations": False,
            }
            response = client.post("/v1/score", json=payload)
            if response.status_code != 200:
                raise RuntimeError(
                    f"M8 HTTP scoring failed for id={request.application_id}: "
                    f"{response.status_code} {response.text}"
                )
            body = response.json()
            online_rows.append(
                {
                    "id": int(body["application_id"]),
                    "online_pd": float(body["probability_of_default"]),
                    "online_band": str(body["risk_band"]),
                    "online_decision": str(body["decision"]),
                    "online_score": int(body["credit_score"]),
                }
            )
    online_frame = pd.DataFrame(online_rows)
    compare = chosen[["id", "calibrated_pd", "risk_band", "decision_balanced", "credit_score"]].merge(
        online_frame, on="id", validate="one_to_one"
    )
    pd_error = np.max(np.abs(compare["calibrated_pd"] - compare["online_pd"]))
    bands_match = bool((compare["risk_band"] == compare["online_band"]).all())
    decisions_match = bool((compare["decision_balanced"] == compare["online_decision"]).all())
    # M5 stores float presentation scores; API intentionally rounds to integer score points.
    scores_match = bool((np.rint(compare["credit_score"]).astype(int) == compare["online_score"]).all())

    print("=== M8 TRAINING-SERVING CONSISTENCY ===")
    print(f"applications: {len(compare)}")
    print(f"max |PD offline-online|: {pd_error:.3e}")
    print(f"risk bands identical: {bands_match}")
    print(f"decisions identical:  {decisions_match}")
    print(f"credit scores identical after rounding: {scores_match}")
    if pd_error > 1e-7 or not bands_match or not decisions_match or not scores_match:
        raise RuntimeError("M8 serving consistency validation FAILED")
    print("SERVING CONSISTENCY: PASS")


if __name__ == "__main__":
    main()
