from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


VALID_BANDS = {"A", "B", "C", "D", "E"}
VALID_DECISIONS = {"APPROVE", "REVIEW", "REJECT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9 live-container HTTP smoke/integration test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", type=Path, default=Path("examples/m8_score_request.json"))
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser.parse_args()


def request_json(method: str, url: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP {response.status} from {url}")
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            ready = request_json("GET", f"{base_url}/ready", timeout=2.0)
            if ready.get("status") == "ready":
                return ready
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Container did not become ready within {timeout_seconds:.0f}s: {last_error}")


def main() -> None:
    args = parse_args()
    if not args.request.exists():
        raise FileNotFoundError(args.request)
    payload = json.loads(args.request.read_text(encoding="utf-8"))

    ready = wait_until_ready(args.base_url.rstrip("/"), args.timeout_seconds)
    health = request_json("GET", f"{args.base_url.rstrip('/')}/health")
    info = request_json("GET", f"{args.base_url.rstrip('/')}/v1/model-info")
    score = request_json("POST", f"{args.base_url.rstrip('/')}/v1/score", payload)

    fingerprint = str(ready["artifact_fingerprint"])
    if health != {"status": "ok"}:
        raise RuntimeError(f"Unexpected /health payload: {health}")
    if info.get("artifact_fingerprint") != fingerprint or score.get("artifact_fingerprint") != fingerprint:
        raise RuntimeError("Artifact fingerprint differs across readiness/model-info/score endpoints")
    if info.get("model_version") != "credit-risk-v1" or score.get("model_version") != "credit-risk-v1":
        raise RuntimeError("Unexpected model version")
    if int(info.get("n_features", -1)) != 212:
        raise RuntimeError("Serving image does not expose the frozen 212-feature contract")

    pd_value = float(score["probability_of_default"])
    if not 0.0 <= pd_value <= 1.0:
        raise RuntimeError("Scored PD lies outside [0, 1]")
    if score.get("risk_band") not in VALID_BANDS:
        raise RuntimeError("Invalid risk band")
    if score.get("decision") not in VALID_DECISIONS:
        raise RuntimeError("Invalid credit decision")
    if not 300 <= int(score["credit_score"]) <= 850:
        raise RuntimeError("Credit score lies outside presentation range")
    if score.get("explanation_domain") != "catboost_raw_log_odds_pre_isotonic":
        raise RuntimeError("Unexpected explanation domain")
    if not isinstance(score.get("risk_increasing_reasons"), list) or not isinstance(
        score.get("risk_reducing_reasons"), list
    ):
        raise RuntimeError("Reason-code response schema is invalid")

    print("=== M9 CONTAINER SMOKE TEST ===")
    print("health: PASS")
    print("readiness: PASS")
    print("model-info: PASS")
    print("score endpoint: PASS")
    print(f"model_version: {score['model_version']}")
    print(f"artifact_fingerprint: {fingerprint}")
    print(f"PD: {pd_value:.6f}")
    print(f"band/decision: {score['risk_band']} / {score['decision']}")
    print("CONTAINER CONTRACT: PASS")


if __name__ == "__main__":
    main()
