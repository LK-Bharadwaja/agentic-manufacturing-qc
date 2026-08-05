import logging
import pickle

import pandas as pd

from modules.config import CNN_MODEL_FILE, CNN_SCALER_FILE
from modules.exceptions import InvalidInputError, ModelNotAvailableError
from modules.feature_extraction import FEATURE_NAMES

logger = logging.getLogger(__name__)

_cache: dict = {}


def _tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    if _cache:
        return _cache['model'], _cache['scaler']

    if not _tensorflow_available():
        raise ModelNotAvailableError(
            'CNN', 'TensorFlow is not installed — pip install tensorflow'
        )

    for path, label in ((CNN_MODEL_FILE, 'model'), (CNN_SCALER_FILE, 'scaler')):
        if not path.exists():
            raise ModelNotAvailableError(
                'CNN',
                f"{label} not found at {path} — run 'python save_models.py' to train it",
            )

    import tensorflow as tf
    model = tf.keras.models.load_model(CNN_MODEL_FILE)

    with open(CNN_SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    logger.info("Loaded CNN model from %s", CNN_MODEL_FILE)
    _cache['model'] = model
    _cache['scaler'] = scaler
    return model, scaler


def is_available() -> bool:
    """Return True if the CNN model files exist and TensorFlow is installed."""
    return (
        _tensorflow_available()
        and CNN_MODEL_FILE.exists()
        and CNN_SCALER_FILE.exists()
    )


def predict(features_df: pd.DataFrame) -> float:
    """
    Predict surface roughness using the CNN model.
    MinMaxScaler is applied automatically before inference.
    Returns Ra in µm.
    """
    model, scaler = _load()

    missing = [c for c in FEATURE_NAMES if c not in features_df.columns]
    if missing:
        raise InvalidInputError(
            f"Input is missing {len(missing)} required feature column(s): {missing}"
        )

    X = features_df[FEATURE_NAMES].values          # (1, 36)
    X_scaled = scaler.transform(X)                  # normalize with training scaler
    X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # (1, 36, 1)

    pred = model.predict(X_seq, verbose=0).flatten()[0]
    return float(pred)
