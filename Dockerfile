# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CREDIT_RISK_ARTIFACT_DIR=/app/artifacts

WORKDIR /app

# Runtime packages are pinned independently from the research environment.
COPY requirements/runtime.txt /tmp/runtime.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/runtime.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps .

# Fail the build if any frozen serving artifact is absent from the build context.
RUN mkdir -p /app/artifacts
COPY artifacts/m4_catboost.cbm /app/artifacts/m4_catboost.cbm
COPY artifacts/m4_catboost_imputer.joblib /app/artifacts/m4_catboost_imputer.joblib
COPY artifacts/m4_calibrator.joblib /app/artifacts/m4_calibrator.joblib
COPY artifacts/m3_features.csv /app/artifacts/m3_features.csv
COPY artifacts/m4_metadata.json /app/artifacts/m4_metadata.json
COPY artifacts/m5_policy_freeze.json /app/artifacts/m5_policy_freeze.json
COPY artifacts/m5_risk_band_thresholds.json /app/artifacts/m5_risk_band_thresholds.json

RUN groupadd --system --gid 10001 creditrisk \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin creditrisk \
    && chown -R creditrisk:creditrisk /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).read()" || exit 1

STOPSIGNAL SIGTERM
CMD ["uvicorn", "credit_risk.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
