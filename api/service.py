"""Runs all three models over an extracted feature row and records the result."""

import json
import logging
import time
from datetime import UTC, datetime

import pandas as pd

import modules.cnn_model as cnn_module
import modules.fuzzy_logic as fuzzy_module
import modules.mlr as mlr_module
from modules.config import PREDICTION_LOG_FILE
from modules.exceptions import ModelNotAvailableError

logger = logging.getLogger(__name__)

_MODELS = (
    ("MLR", mlr_module),
    ("Fuzzy Logic", fuzzy_module),
    ("CNN", cnn_module),
)


def model_statuses() -> list[dict]:
    statuses = []
    for name, module in _MODELS:
        ready = module.is_available()
        detail = None
        if not ready:
            detail = (
                "TensorFlow not installed or model artifacts missing"
                if name == "CNN"
                else "Model artifacts missing — run save_models.py"
            )
        elif name == "Fuzzy Logic":
            detail = "Trained membership bounds loaded"
        statuses.append({"model": name, "ready": ready, "detail": detail})
    return statuses


def warm_up() -> None:
    """Load every available model once at startup so requests don't pay the cost."""
    loaders = {
        "MLR": mlr_module._load,
        "Fuzzy Logic": fuzzy_module._load_bounds,
        "CNN": cnn_module._load,
    }
    for name, module in _MODELS:
        if not module.is_available():
            logger.warning("Skipping warm-up for %s — not available", name)
            continue
        try:
            loaders[name]()
            logger.info("Warmed up %s", name)
        except ModelNotAvailableError as exc:
            logger.warning("Warm-up failed for %s: %s", name, exc)


def run_all(features_df: pd.DataFrame, filename: str | None = None) -> dict:
    """Predict Ra with every model, returning a response-shaped dict."""
    started = time.perf_counter()
    predictions = []
    values = []
    category = None

    for name, module in _MODELS:
        try:
            ra = float(module.predict(features_df))
            values.append(ra)
            predictions.append({"model": name, "ra": ra, "available": True, "detail": None})
            if name == "Fuzzy Logic":
                category = fuzzy_module.categorize(ra)
        except ModelNotAvailableError as exc:
            predictions.append(
                {"model": name, "ra": None, "available": False, "detail": str(exc)}
            )
        except Exception:
            logger.exception("%s prediction failed", name)
            predictions.append(
                {
                    "model": name,
                    "ra": None,
                    "available": False,
                    "detail": "Prediction failed — see server logs",
                }
            )

    latency_ms = (time.perf_counter() - started) * 1000
    result = {
        "filename": filename,
        "predictions": predictions,
        "spread_um": (max(values) - min(values)) if len(values) > 1 else None,
        "category": category,
        "features": {k: float(v) for k, v in features_df.iloc[0].items()},
        "latency_ms": round(latency_ms, 2),
    }

    _log_prediction(result)
    return result


def _log_prediction(result: dict) -> None:
    """Append one line of prediction telemetry. Never fails the request."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "filename": result["filename"],
        "predictions": {p["model"]: p["ra"] for p in result["predictions"]},
        "category": result["category"],
        "latency_ms": result["latency_ms"],
    }
    try:
        PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Could not write prediction log: %s", exc)
