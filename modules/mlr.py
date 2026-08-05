import logging
import pickle

import pandas as pd

from modules.config import MLR_FEATURES_FILE, MLR_MODEL_FILE
from modules.exceptions import InvalidInputError, ModelNotAvailableError

logger = logging.getLogger(__name__)

# Cached after first load
_cache: dict = {}


def is_available() -> bool:
    """Return True if the MLR artifacts exist on disk."""
    return MLR_MODEL_FILE.exists() and MLR_FEATURES_FILE.exists()


def _load():
    if _cache:
        return _cache['model'], _cache['features']

    for path, label in ((MLR_MODEL_FILE, 'model'), (MLR_FEATURES_FILE, 'feature list')):
        if not path.exists():
            raise ModelNotAvailableError(
                'MLR',
                f"{label} not found at {path} — run 'python save_models.py' to train it",
            )

    with open(MLR_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(MLR_FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)

    logger.info("Loaded MLR model with %d features", len(features))
    _cache['model'] = model
    _cache['features'] = features
    return model, features


def predict(features_df: pd.DataFrame) -> float:
    """Return MLR surface roughness prediction (Ra, µm)."""
    model, best_features = _load()

    missing = [c for c in best_features if c not in features_df.columns]
    if missing:
        raise InvalidInputError(
            f"Input is missing {len(missing)} required feature column(s): {missing}"
        )

    X = features_df[best_features].values.reshape(1, -1)
    return float(model.predict(X)[0])


def get_feature_names() -> list:
    """Return the list of features used by the saved MLR model."""
    _, features = _load()
    return list(features)


def get_equation() -> str:
    """Return the regression equation string."""
    model, features = _load()
    terms = " + ".join(
        f"({c:.4f} × {f})" for c, f in zip(model.coef_, features)
    )
    return f"Ra = {terms} + {model.intercept_:.4f}"
