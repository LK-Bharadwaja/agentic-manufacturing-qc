"""FastAPI service exposing surface roughness prediction over HTTP."""

import io
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api import service
from api.schemas import (
    ErrorResponse,
    FeaturesRequest,
    HealthResponse,
    ModelsStatusResponse,
    PredictionResponse,
)
from modules.config import ALLOWED_UPLOAD_SUFFIXES, MAX_UPLOAD_BYTES, RATE_LIMIT
from modules.exceptions import InvalidInputError, SurfaceRoughnessError
from modules.feature_extraction import (
    FEATURE_NAMES,
    available_channels,
    extract_features_from_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models...")
    service.warm_up()
    yield


app = FastAPI(
    title="Surface Roughness Predictor API",
    version=VERSION,
    description=(
        "Predicts machined-part surface roughness **Ra (µm)** from 3-axis vibration "
        "signals using three approaches in parallel: Multiple Linear Regression, "
        "a 81-rule Mamdani fuzzy system, and a Conv1D neural network."
    ),
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="rate_limit_exceeded", detail=str(exc.detail)
        ).model_dump(),
    )


@app.exception_handler(InvalidInputError)
async def _invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="invalid_input", detail=str(exc)).model_dump(),
    )


@app.exception_handler(SurfaceRoughnessError)
async def _domain_error_handler(request: Request, exc: SurfaceRoughnessError):
    logger.exception("Domain error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="prediction_failed", detail=str(exc)).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok", version=VERSION)


@app.get("/models/status", response_model=ModelsStatusResponse, tags=["ops"])
async def models_status():
    statuses = service.model_statuses()
    return ModelsStatusResponse(
        models=statuses, all_ready=all(s["ready"] for s in statuses)
    )


def _read_upload(filename: str, raw: bytes) -> pd.DataFrame:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise InvalidInputError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Expected one of: {sorted(ALLOWED_UPLOAD_SUFFIXES)}"
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidInputError(
            f"File is {len(raw) / 1e6:.1f} MB, exceeding the "
            f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB limit."
        )
    try:
        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(raw))
        return pd.read_excel(io.BytesIO(raw))
    except Exception as exc:
        raise InvalidInputError(f"Could not parse '{filename}': {exc}") from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
@limiter.limit(RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)):  # noqa: B008 — FastAPI idiom
    """Upload a vibration signal file and get Ra predictions from all three models."""
    raw = await file.read()
    df = _read_upload(file.filename or "upload", raw)
    features_df = extract_features_from_df(df)
    result = service.run_all(features_df, filename=file.filename)
    result["channels_found"] = available_channels(df)
    return PredictionResponse(**result)


@app.post(
    "/predict/features",
    response_model=PredictionResponse,
    tags=["prediction"],
    responses={400: {"model": ErrorResponse}},
)
@limiter.limit(RATE_LIMIT)
async def predict_from_features(request: Request, payload: FeaturesRequest):
    """Predict directly from a pre-computed set of the 36 features."""
    missing = [name for name in FEATURE_NAMES if name not in payload.features]
    if missing:
        raise InvalidInputError(
            f"Missing {len(missing)} of {len(FEATURE_NAMES)} features: {missing[:5]}..."
        )
    features_df = pd.DataFrame([payload.features], columns=FEATURE_NAMES)
    result = service.run_all(features_df, filename=None)
    result["channels_found"] = []
    return PredictionResponse(**result)
