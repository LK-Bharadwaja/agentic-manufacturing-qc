import os
import pickle
import numpy as np
import pandas as pd

from modules.feature_extraction import FEATURE_NAMES

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_FILE = os.path.join(_MODELS_DIR, 'cnn_model.keras')
_SCALER_FILE = os.path.join(_MODELS_DIR, 'cnn_scaler.pkl')

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
        raise ImportError(
            "TensorFlow is not installed. Install it with:\n  pip install tensorflow"
        )

    model_path = os.path.abspath(_MODEL_FILE)
    scaler_path = os.path.abspath(_SCALER_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at:\n  {model_path}\n\n"
            "Run  python save_models.py  from the project directory to train and save it."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"CNN scaler not found at:\n  {scaler_path}\n\n"
            "Run  python save_models.py  to regenerate it."
        )

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    _cache['model'] = model
    _cache['scaler'] = scaler
    return model, scaler


def is_available() -> bool:
    """Return True if the CNN model files exist and TensorFlow is installed."""
    return (
        _tensorflow_available()
        and os.path.exists(os.path.abspath(_MODEL_FILE))
        and os.path.exists(os.path.abspath(_SCALER_FILE))
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
        raise ValueError(
            f"Input DataFrame is missing {len(missing)} required column(s): {missing}"
        )

    X = features_df[FEATURE_NAMES].values          # (1, 36)
    X_scaled = scaler.transform(X)                  # normalize with training scaler
    X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # (1, 36, 1)

    pred = model.predict(X_seq, verbose=0).flatten()[0]
    return float(pred)
