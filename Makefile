.PHONY: install-dev test static api verify-artifacts docker-build docker-up docker-down docker-smoke ci-local m10-reference m10-schema-reference m10-monitor release-check

install-dev:
	python -m pip install -r requirements/dev.txt
	python -m pip install --no-deps -e .
	python -m pip check

test:
	pytest -q

static:
	python -m compileall -q src scripts tests
	ruff check src scripts tests

api:
	uvicorn credit_risk.api.app:app --host 0.0.0.0 --port 8000

verify-artifacts:
	python scripts/verify_m9_serving_artifacts.py --load-runtime

docker-build: verify-artifacts
	docker build -t credit-risk-engine:v1 .

docker-up: verify-artifacts
	docker compose up --build -d

docker-down:
	docker compose down

docker-smoke:
	python scripts/m9_container_smoke_test.py --base-url http://127.0.0.1:8000

ci-local: static test release-check


m10-reference:
	python scripts/build_m10_reference_profile.py

m10-schema-reference:
	python scripts/build_m10_1_schema_reference.py

m10-monitor:
	python scripts/run_m10_monitoring.py

release-check:
	python scripts/validate_m11_release.py
