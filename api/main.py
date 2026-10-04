"""FastAPI service exposing surface roughness prediction over HTTP."""

import concurrent.futures
import faulthandler
import io
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from agent.qc_agent import run_qc_pipeline
from api import service
from api.schemas import (
    AgentPredictResponse,
    ErrorResponse,
    FeaturesRequest,
    HealthResponse,
    ModelsStatusResponse,
    PredictionResponse,
    RagAskRequest,
    RagAskResponse,
)
from modules.config import ALLOWED_UPLOAD_SUFFIXES, MAX_UPLOAD_BYTES, RATE_LIMIT
from modules.exceptions import InvalidInputError, SurfaceRoughnessError
from modules.feature_extraction import (
    FEATURE_NAMES,
    available_channels,
    extract_features_from_df,
)
from rag.qa_engine import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"

limiter = Limiter(key_func=get_remote_address)

# The agent occasionally freezes after a Gemini call returns — a stall
# downstream of the network call itself, so a per-call LLM timeout doesn't
# catch it (see agent/qc_agent.py's _LLM_TIMEOUT_SECONDS). This wall-clock
# deadline bounds the whole /agent/predict request regardless of cause. The
# stuck worker thread can't be killed (Python has no safe thread-kill), so
# it's left running in the background; it's a plain ThreadPoolExecutor, not
# the anyio pool other endpoints use via run_in_threadpool.
_AGENT_TIMEOUT_SECONDS = 90.0
_agent_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="qc-agent"
)


def _dump_agent_hang(progress: dict) -> Path:
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    path = logs_dir / f"agent_hang_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Agent did not complete within {_AGENT_TIMEOUT_SECONDS:.0f}s.\n")
        f.write(f"Last completed tool call: {progress.get('last_tool_call')!r}\n\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    return path


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


@app.post(
    "/agent/predict",
    response_model=AgentPredictResponse,
    tags=["agent"],
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
@limiter.limit(RATE_LIMIT)
async def agent_predict(request: Request, file: UploadFile = File(...)):  # noqa: B008 — FastAPI idiom
    """Upload a vibration signal file and get the QC agent's full reasoning trace.

    Same input shape as /predict — a .xlsx/.xls/.csv file — but runs the
    LangGraph tool-calling agent (agent/qc_agent.py) instead of calling all
    three models directly, returning its tool-by-tool trace and final report.

    Runs on a dedicated thread pool (not the anyio pool run_in_threadpool
    uses elsewhere) so a stuck run can't starve other requests, and is
    bounded by _AGENT_TIMEOUT_SECONDS regardless of cause: the agent's tools
    call back into this same server's /predict/features over HTTP, and the
    agent has occasionally been observed to hang after a Gemini call
    returns — downstream of the network call itself, so the LLM client's
    own timeout doesn't cover it.
    """
    raw = await file.read()
    df = _read_upload(file.filename or "upload", raw)
    features_df = extract_features_from_df(df)
    features = features_df.iloc[0].to_dict()

    progress: dict = {}
    future = _agent_executor.submit(run_qc_pipeline, features, progress)
    try:
        result = await run_in_threadpool(future.result, _AGENT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        hang_path = _dump_agent_hang(progress)
        logger.error(
            "Agent QC pipeline exceeded %.0fs deadline; last tool call=%s; trace written to %s",
            _AGENT_TIMEOUT_SECONDS, progress.get("last_tool_call"), hang_path,
        )
        return JSONResponse(
            status_code=504,
            content=ErrorResponse(
                error="agent_timeout",
                detail=(
                    "The QC agent did not complete in time — this is a known "
                    "intermittent issue with the underlying AI service. Please try again."
                ),
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Agent QC pipeline failed")
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="agent_unavailable", detail=str(exc)).model_dump(),
        )
    result["filename"] = file.filename
    return AgentPredictResponse(**result)


@app.post(
    "/rag/ask",
    response_model=RagAskResponse,
    tags=["agent"],
    responses={502: {"model": ErrorResponse}},
)
@limiter.limit(RATE_LIMIT)
async def rag_ask(request: Request, payload: RagAskRequest):
    """Answer a question about the project's methodology/documentation via RAG."""
    try:
        result = await run_in_threadpool(answer_question, payload.question)
    except Exception as exc:
        logger.exception("RAG query failed")
        return JSONResponse(
            status_code=502,
            content=ErrorResponse(error="rag_unavailable", detail=str(exc)).model_dump(),
        )
    return RagAskResponse(**result)
