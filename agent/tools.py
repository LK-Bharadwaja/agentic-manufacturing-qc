"""LangChain @tool definitions for the Surface Roughness Predictor agent.

Each tool wraps one capability the LLM can invoke. The docstrings are the
primary signal Claude uses to decide which tool to call and when, so they
describe inputs, outputs, and the exact situation that warrants the call.

Prediction tools POST to the FastAPI /predict/features endpoint (requires
the server to be running at API_BASE_URL = http://localhost:8000 by default).
query_rag and get_signal_noise_level run in-process, no server needed.
"""

import requests
from langchain_core.tools import tool

from modules.config import API_BASE_URL, TRAINING_RA_MAX, TRAINING_RA_MIN
from rag.qa_engine import answer_question

_FEATURES_URL = f"{API_BASE_URL}/predict/features"
_TIMEOUT = 30  # seconds


def _post_features(features: dict) -> dict:
    """POST features to the API and return the parsed JSON. Raises on HTTP error."""
    try:
        resp = requests.post(
            _FEATURES_URL,
            json={"features": features},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach the prediction API at {API_BASE_URL}. "
            "Start the server with: uvicorn api.main:app --port 8000"
        ) from exc


def _extract(data: dict, model_name: str) -> str:
    """Pull one model's result out of a PredictionResponse dict."""
    for pred in data["predictions"]:
        if pred["model"] == model_name:
            if not pred["available"]:
                return f"{model_name} model unavailable: {pred['detail']}"
            return f"{model_name} predicted Ra = {pred['ra']:.4f} µm"
    return f"Error: '{model_name}' entry not found in API response."


# ---------------------------------------------------------------------------
# Prediction tools
# ---------------------------------------------------------------------------

@tool
def predict_mlr(features: dict) -> str:
    """Get the Multiple Linear Regression (MLR) model's surface roughness prediction.

    The MLR model is a 4-feature linear regression (using Channel 2 RMS, PEAK,
    KURTOSIS, and PSD). It is the simplest and fastest model but has the lowest
    accuracy (R²=0.46). Call this tool when:
      - The user asks specifically about the MLR model's output.
      - You need a baseline or interpretable linear estimate alongside the other models.
      - You are running a side-by-side comparison of all three models.

    The API runs all three models in one call but this tool extracts only the MLR
    result. If you need results from multiple models, call each tool separately.

    Args:
        features: Dict of all 36 vibration feature values keyed by their exact
                  name (e.g. "Channel 1 RMS", "Channel 2 KURTOSIS"). All 36
                  names from FEATURE_NAMES must be present.

    Returns:
        A string of the form "MLR predicted Ra = X.XXXX µm", or an error message
        if the model is unavailable or the API is unreachable.
    """
    try:
        data = _post_features(features)
        return _extract(data, "MLR")
    except Exception as exc:
        return f"Error: {exc}"


@tool
def predict_fuzzy(features: dict) -> str:
    """Get the 81-rule Mamdani fuzzy logic system's surface roughness prediction.

    The fuzzy model uses 4 Channel-2 features (RMS, PEAK, KURTOSIS, PSD) with
    triangular membership functions and centroid defuzzification. It also produces
    a surface quality category: 'Smooth' (Ra < 2.175), 'Average' (2.175–2.55),
    or 'Rough' (Ra > 2.55). Accuracy is 90.25% (R²=0.18). Call this tool when:
      - The user wants a qualitative, rules-based assessment of surface quality.
      - The user asks why 81 rules or how the fuzzy system works (prefer query_rag
        for the explanation, but call this for the actual number).
      - You are comparing all three models and need the fuzzy result.

    Args:
        features: Dict of all 36 vibration feature values keyed by their exact
                  name. The fuzzy model internally uses only 4 Channel-2 features,
                  but the API expects the full 36.

    Returns:
        A string with the predicted Ra in µm and the surface category, e.g.
        "Fuzzy Logic predicted Ra = X.XXXX µm" plus category from the API response,
        or an error message if the model or API is unavailable.
    """
    try:
        data = _post_features(features)
        base = _extract(data, "Fuzzy Logic")
        category = data.get("category")
        if category and "Error" not in base and "unavailable" not in base:
            return f"{base} | surface category: {category}"
        return base
    except Exception as exc:
        return f"Error: {exc}"


@tool
def predict_cnn(features: dict) -> str:
    """Get the Conv1D Convolutional Neural Network (CNN) model's surface roughness prediction.

    The CNN is a 1D convolutional network (Conv1D(32) → Dropout(0.2) → Dense(16) →
    Dense(1)) trained on all 36 vibration features, MinMax-scaled and reshaped to
    (36, 1) spatial input. Training-set fit is R²=0.895 (96.77%) on the 27-part
    training set — this is a training-set fit, not a held-out test score. Call this
    tool when:
      - The user asks specifically about the neural network or CNN result.
      - The user wants the CNN's architecture or training-fit numbers.
      - You are comparing all three models and need the CNN result.

    Args:
        features: Dict of all 36 vibration feature values keyed by their exact
                  name. The CNN applies a trained MinMaxScaler to all 36 features
                  before inference.

    Returns:
        A string of the form "CNN predicted Ra = X.XXXX µm", or an error message
        if TensorFlow is unavailable, model artifacts are missing, or the API
        is unreachable.
    """
    try:
        data = _post_features(features)
        return _extract(data, "CNN")
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Extrapolation sanity check
# ---------------------------------------------------------------------------

@tool
def check_training_range(ra: float) -> str:
    """Check whether a predicted Ra value falls inside the models' training data range.

    All three models were fit on 27 training parts whose ground-truth Ra values
    range from 1.931 to 2.812 µm (see Y_TRAIN in save_models.py). None of the
    models have ever seen a genuine example outside that range, so a prediction
    outside it is an extrapolation, not an interpolation — a model's R² or
    accuracy figures describe how well it fits the range it was trained on and
    say nothing about how it behaves outside it. Call this tool on every numeric
    Ra prediction you get from predict_mlr, predict_fuzzy, or predict_cnn before
    deciding which one to trust — especially before treating a higher-R² model's
    result as the correct answer.

    Args:
        ra: A predicted Ra value in µm, as returned by one of the prediction tools.

    Returns:
        A string stating whether the value falls within the known training
        range. If it does not, the message flags it as a potential
        extrapolation error that should be downweighted rather than trusted by
        default — particularly if other models agree closely with each other
        and fall inside the range.
    """
    try:
        ra = float(ra)
    except (TypeError, ValueError):
        return f"Error: {ra!r} is not a numeric Ra value."

    if TRAINING_RA_MIN <= ra <= TRAINING_RA_MAX:
        return (
            f"Ra = {ra:.4f} µm is within the known training range "
            f"({TRAINING_RA_MIN:.3f}-{TRAINING_RA_MAX:.3f} µm). No extrapolation concern."
        )

    if ra < TRAINING_RA_MIN:
        delta, direction = TRAINING_RA_MIN - ra, "below"
    else:
        delta, direction = ra - TRAINING_RA_MAX, "above"

    return (
        f"WARNING: Ra = {ra:.4f} µm is {delta:.4f} µm {direction} the known "
        f"training range ({TRAINING_RA_MIN:.3f}-{TRAINING_RA_MAX:.3f} µm). "
        "This is an extrapolation beyond any part the models were fit on. "
        "Flag it as a potential extrapolation error and downweight it rather "
        "than trusting it by default — especially if other models agree "
        "closely with each other and fall inside the known range."
    )


# ---------------------------------------------------------------------------
# RAG tool
# ---------------------------------------------------------------------------

@tool
def query_rag(question: str) -> str:
    """Search the project report documents to answer a question about methodology.

    Retrieves the top-3 most relevant document chunks from the embedded project
    reports (Final Report, Individual Report, Stage 1&2 Report, Metrics Clarification)
    using cosine similarity, then asks Claude to answer using only those chunks.
    If the question is not related to this project's documentation, returns a
    "not found" message without calling the LLM (distance threshold = 0.35).

    Call this tool when the user asks about:
      - Why design decisions were made (e.g. why 81 fuzzy rules, why 4 features).
      - What the report says about accuracy, R², or methodology.
      - Background theory: CNC machining, surface roughness, Ra units.
      - The authoritative R² or accuracy figures for any model.
      - How features were extracted or which features the models use.

    Do NOT call this tool for numeric predictions — use predict_mlr, predict_fuzzy,
    or predict_cnn for that. query_rag answers "why/how/what does the report say",
    not "what is the predicted Ra for these features".

    Args:
        question: A natural-language question about the project methodology,
                  design decisions, or background theory.

    Returns:
        A dict-like string with keys: answer (str), sources (list of filenames),
        grounded (bool). If grounded is False, the question was out of scope.
    """
    try:
        result = answer_question(question)
        if not result["grounded"]:
            return (
                "Not grounded: the question appears to be outside the scope of "
                "the project documentation. " + result["answer"]
            )
        sources = ", ".join(result["sources"]) if result["sources"] else "unknown"
        return f"Answer: {result['answer']}\nSources: {sources}"
    except Exception as exc:
        return f"Error querying RAG: {exc}"


# ---------------------------------------------------------------------------
# Signal diagnostics tool
# ---------------------------------------------------------------------------

@tool
def get_signal_noise_level(features: dict) -> str:
    """Assess the vibration signal noise level from a pre-extracted feature set.

    No noise-level function exists in modules/ — this computes it fresh from
    two dimensionless, scale-independent indicators already present in the features:

    Crest Factor (PEAK / RMS) per channel — measures impulsivity:
      < 3   → clean, smooth signal (steady-state cutting)
      3–5   → moderate noise (typical CNC vibration)
      > 5   → high impulsive noise (tool chatter, interrupted cut, sensor issue)

    Kurtosis (Fisher, excess over Gaussian = 0) per channel — measures tail weight:
      near 0  → Gaussian background noise, normal operation
      2–5     → mild impulsive events
      > 5     → strong impulsive events, possible defect or tool wear

    Call this tool when:
      - The user asks about signal quality, noise, or reliability of the features.
      - A prediction looks unexpectedly high or low and you want to diagnose the input.
      - The user asks whether the sensor data is clean enough to trust.

    Args:
        features: Dict containing at least the six keys "Channel N CREST FACTOR"
                  and "Channel N KURTOSIS" for N in {1, 2, 3}. Extra keys are ignored.

    Returns:
        A per-channel breakdown of Crest Factor and Kurtosis plus a composite
        noise rating: Low / Moderate / High / Very High.
    """
    try:
        lines = []
        cf_vals = []
        kurt_vals = []

        for ch in range(1, 4):
            cf_key = f"Channel {ch} CREST FACTOR"
            kurt_key = f"Channel {ch} KURTOSIS"
            cf = features.get(cf_key)
            kurt = features.get(kurt_key)

            if cf is None or kurt is None:
                lines.append(f"Channel {ch}: missing features ({cf_key} or {kurt_key})")
                continue

            cf = float(cf)
            kurt = float(kurt)
            cf_vals.append(cf)
            kurt_vals.append(kurt)

            if cf < 3:
                cf_label = "low noise"
            elif cf < 5:
                cf_label = "moderate noise"
            else:
                cf_label = "high impulsive noise"

            if kurt < 2:
                kurt_label = "Gaussian (normal)"
            elif kurt < 5:
                kurt_label = "mild impulsive events"
            else:
                kurt_label = "strong impulsive events"

            lines.append(
                f"Channel {ch}: Crest Factor={cf:.2f} ({cf_label}), "
                f"Kurtosis={kurt:.2f} ({kurt_label})"
            )

        if not cf_vals:
            return "Error: no Crest Factor or Kurtosis features found in the dict."

        avg_cf = sum(cf_vals) / len(cf_vals)
        avg_kurt = sum(kurt_vals) / len(kurt_vals)

        if avg_cf < 3 and avg_kurt < 2:
            rating = "Low — signal is clean, predictions are reliable"
        elif avg_cf < 5 and avg_kurt < 5:
            rating = "Moderate — typical CNC signal, predictions should be valid"
        elif avg_cf < 7 or avg_kurt < 8:
            rating = "High — notable impulsive noise, treat predictions with caution"
        else:
            rating = "Very High — strong noise or sensor anomaly, verify the input"

        lines.append(f"\nComposite noise rating: {rating}")
        lines.append(f"(avg Crest Factor={avg_cf:.2f}, avg Kurtosis={avg_kurt:.2f})")

        return "\n".join(lines)

    except Exception as exc:
        return f"Error computing noise level: {exc}"
