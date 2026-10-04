"""Central configuration. Values come from env vars with prototype defaults."""

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = _PACKAGE_DIR.parent


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).resolve() if raw else default


# ── Signal processing ─────────────────────────────────────────────────────────
SAMPLING_RATE = _env_int("SRP_SAMPLING_RATE", 50_000)
WINDOW_SIZE = SAMPLING_RATE // 50
OVERLAP = WINDOW_SIZE // 2
STFT_WINDOW = WINDOW_SIZE // 2

CHANNELS = ["Channel1 [g]", "Channel2 [g]", "Channel3 [g]"]

# ── Model artifacts ───────────────────────────────────────────────────────────
MODELS_DIR = _env_path("SRP_MODELS_DIR", PROJECT_DIR / "models")

MLR_MODEL_FILE = MODELS_DIR / "mlr_model.pkl"
MLR_FEATURES_FILE = MODELS_DIR / "mlr_features.pkl"
FUZZY_BOUNDS_FILE = MODELS_DIR / "fuzzy_bounds.pkl"
CNN_MODEL_FILE = MODELS_DIR / "cnn_model.keras"
CNN_SCALER_FILE = MODELS_DIR / "cnn_scaler.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# ── Fuzzy inference ───────────────────────────────────────────────────────────
FUZZY_FEATURES = [
    "Channel 2 RMS",
    "Channel 2 PEAK",
    "Channel 2 KURTOSIS",
    "Channel 2 PSD",
]
FUZZY_OUTPUT_MAP = {"Smooth": 1.9970, "Average": 2.3511, "Rough": 2.7480}
FUZZY_SMOOTH_MAX = 2.175
FUZZY_AVERAGE_MAX = 2.55

# ── Training data range ────────────────────────────────────────────────────────
# Ground-truth Ra range across the 27 training parts (min/max of Y_TRAIN in
# save_models.py). None of the three models have seen a genuine example outside
# this range, so a prediction outside it is an extrapolation, not an
# interpolation, regardless of that model's training-set R².
TRAINING_RA_MIN = 1.931
TRAINING_RA_MAX = 2.812

# ── API ───────────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = _env_int("SRP_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
ALLOWED_UPLOAD_SUFFIXES = {".csv", ".xlsx", ".xls"}
PREDICTION_LOG_FILE = _env_path(
    "SRP_PREDICTION_LOG", PROJECT_DIR / "logs" / "predictions.jsonl"
)
RATE_LIMIT = os.getenv("SRP_RATE_LIMIT", "30/minute")
API_BASE_URL = os.getenv("SRP_API_URL", "http://localhost:8000")
