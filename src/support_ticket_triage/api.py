"""FastAPI application with strict schemas and privacy-preserving logging."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.staticfiles import StaticFiles

from support_ticket_triage.inference import (
    InferenceEngine,
    InputValidationError,
    ReviewReason,
)

LOGGER = logging.getLogger("support_ticket_triage.api")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    text: Annotated[str, Field(min_length=1, max_length=1000)]


class CandidateResponse(BaseModel):
    intent: str
    display_name: str
    routing_group: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_intent: str
    display_name: str
    routing_group: str
    calibrated_confidence: float
    top_three: list[CandidateResponse]
    review_required: bool
    review_reasons: list[ReviewReason]
    model_name: str
    model_version: str
    review_policy_version: str


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Create an application that loads one frozen model at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = InferenceEngine.from_environment()
        yield

    app = FastAPI(
        title="Support Ticket Triage ML",
        version="0.1.0",
        lifespan=lifespan,
    )
    demo_root = Path(__file__).resolve().parents[2] / "demo"
    if demo_root.exists():
        app.mount("/demo", StaticFiles(directory=demo_root, html=True), name="demo")

    @app.middleware("http")
    async def metadata_only_access_log(request: Request, call_next):
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        LOGGER.info(
            "request_complete request_id=%s method=%s path=%s status=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        loaded = getattr(request.app.state, "engine", None)
        return {"status": "ready" if loaded is not None else "not_ready"}

    @app.get("/metadata")
    async def metadata(request: Request) -> dict[str, Any]:
        return request.app.state.engine.metadata()

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        try:
            result = request.app.state.engine.predict(payload.text)
        except InputValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        LOGGER.info(
            "prediction_complete model=%s version=%s review_required=%s reasons=%s",
            result.model_name,
            result.model_version,
            result.review_required,
            ",".join(result.review_reasons) if result.review_reasons else "none",
        )
        return result.as_dict()

    return app


app = create_app()
