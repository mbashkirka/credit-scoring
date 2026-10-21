from fastapi.testclient import TestClient

from credit_risk.api.app import create_app
from credit_risk.api.runtime import API_VERSION, EXPLANATION_DOMAIN, MODEL_VERSION
from credit_risk.api.schemas import CreditHistoryRecord, ScoreResponse


class FakeRuntime:
    artifact_fingerprint = "abc123"

    def model_info(self):
        return {
            "model_version": MODEL_VERSION,
            "api_version": API_VERSION,
            "n_features": 212,
            "calibration_method": "isotonic",
            "primary_policy": "balanced",
            "approve_pd_threshold": 0.096,
            "reject_pd_threshold": 0.12,
            "risk_band_labels": ["A", "B", "C", "D", "E"],
            "risk_band_cutpoints": [0.01, 0.02, 0.03, 0.06],
            "explanation_domain": EXPLANATION_DOMAIN,
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    def score_many(self, applications, *, include_explanations):
        return [
            ScoreResponse(
                model_version=MODEL_VERSION,
                artifact_fingerprint=self.artifact_fingerprint,
                application_id=a.application_id,
                probability_of_default=0.1,
                raw_model_probability=0.4,
                credit_score=550,
                risk_band="E",
                decision="REVIEW",
                explanation_domain=EXPLANATION_DOMAIN,
                risk_increasing_reasons=[],
                risk_reducing_reasons=[],
            )
            for a in applications
        ]


def valid_record():
    payload = {name: 0 for name in CreditHistoryRecord.model_fields}
    payload["rn"] = 1
    return payload


def test_health_ready_model_info_and_request_id():
    app = create_app(runtime_override=FakeRuntime())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready", headers={"x-request-id": "test-id"})
        assert ready.status_code == 200
        assert ready.headers["x-request-id"] == "test-id"
        info = client.get("/v1/model-info")
        assert info.json()["n_features"] == 212


def test_single_and_batch_scoring_contract():
    app = create_app(runtime_override=FakeRuntime())
    payload = {"application_id": 7, "credit_history": [valid_record()]}
    with TestClient(app) as client:
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["application_id"] == 7
        assert body["decision"] == "REVIEW"
        batch = client.post(
            "/v1/score/batch",
            json={"applications": [payload], "include_explanations": False},
        )
        assert batch.status_code == 200
        assert batch.json()["batch_size"] == 1


def test_validation_error_is_structured_and_target_is_forbidden():
    app = create_app(runtime_override=FakeRuntime())
    record = valid_record()
    record["flag"] = 1
    with TestClient(app) as client:
        response = client.post(
            "/v1/score",
            json={"application_id": 7, "credit_history": [record]},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["request_id"]


def test_readiness_fails_cleanly_when_artifacts_cannot_load():
    def broken_loader():
        raise FileNotFoundError("missing model")

    app = create_app(runtime_loader=broken_loader)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "MODEL_NOT_READY"
