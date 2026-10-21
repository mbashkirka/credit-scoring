from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .runtime import MODEL_VERSION, ScoringRuntime
from .schemas import (
    ApplicationRequest,
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    ScoreRequest,
    ScoreResponse,
)


def create_app(
    *,
    runtime_override: ScoringRuntime | object | None = None,
    runtime_loader: Callable[[], ScoringRuntime] | None = None,
) -> FastAPI:
    loader = runtime_loader or ScoringRuntime.load

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = None
        app.state.load_error = None
        try:
            app.state.runtime = runtime_override if runtime_override is not None else loader()
        except Exception as exc:  # keep liveness endpoint up while readiness fails
            app.state.load_error = f"{type(exc).__name__}: {exc}"
        yield
        app.state.runtime = None

    app = FastAPI(
        title="Credit Risk Engine",
        version="1.0.0",
        description=(
            "Frozen Model V1 inference API. Returns calibrated PD, risk band, "
            "Balanced-policy decision and optional internal TreeSHAP reason codes."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request payload failed schema validation.",
                    "request_id": getattr(request.state, "request_id", None),
                    "details": exc.errors(),
                }
            },
        )

    def get_runtime(request: Request):
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_NOT_READY",
                    "message": getattr(request.app.state, "load_error", None)
                    or "Frozen scoring runtime is not loaded.",
                },
            )
        return runtime

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["operations"])
    def ready(runtime=Depends(get_runtime)) -> ReadyResponse:
        return ReadyResponse(
            model_version=MODEL_VERSION,
            artifact_fingerprint=runtime.artifact_fingerprint,
        )

    @app.get("/v1/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(runtime=Depends(get_runtime)) -> ModelInfoResponse:
        return ModelInfoResponse(**runtime.model_info())

    @app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
    def score(
        payload: ScoreRequest,
        request: Request,
        runtime=Depends(get_runtime),
    ) -> ScoreResponse:
        application = ApplicationRequest.model_validate(
            payload.model_dump(exclude={"include_explanations"})
        )
        started = time.perf_counter()
        try:
            result = runtime.score_many(
                [application], include_explanations=payload.include_explanations
            )[0]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "SCORING_INPUT_ERROR", "message": str(exc)}) from exc
        return result.model_copy(
            update={
                "request_id": request.state.request_id,
                "processing_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    @app.post("/v1/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
    def score_batch(
        payload: BatchScoreRequest,
        request: Request,
        runtime=Depends(get_runtime),
    ) -> BatchScoreResponse:
        started = time.perf_counter()
        try:
            items = runtime.score_many(
                payload.applications,
                include_explanations=payload.include_explanations,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "SCORING_INPUT_ERROR", "message": str(exc)}) from exc
        processing_ms = (time.perf_counter() - started) * 1000.0
        return BatchScoreResponse(
            model_version=MODEL_VERSION,
            artifact_fingerprint=runtime.artifact_fingerprint,
            request_id=request.state.request_id,
            batch_size=len(items),
            processing_ms=processing_ms,
            items=items,
        )

    return app


app = create_app()
