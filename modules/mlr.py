import os
import pickle
import numpy as np
import pandas as pd

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_FILE = os.path.join(_MODELS_DIR, 'mlr_model.pkl')
_FEATS_FILE = os.path.join(_MODELS_DIR, 'mlr_features.pkl')

# Cached after first load
_cache: dict = {}


def _load():
    if _cache:
        return _cache['model'], _cache['features']

    model_path = os.path.abspath(_MODEL_FILE)
    feats_path = os.path.abspath(_FEATS_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MLR model not found at:\n  {model_path}\n\n"
            "Run  python save_models.py  from the project directory to train and save it."
        )
    if not os.path.exists(feats_path):
        raise FileNotFoundError(
            f"MLR feature list not found at:\n  {feats_path}\n\n"
            "Run  python save_models.py  to regenerate it."
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feats_path, 'rb') as f:
        features = pickle.load(f)

    _cache['model'] = model
    _cache['features'] = features
    return model, features


def predict(features_df: pd.DataFrame) -> float:
    """Return MLR surface roughness prediction (Ra, µm)."""
    model, best_features = _load()

    missing = [c for c in best_features if c not in features_df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing {len(missing)} required column(s): {missing}"
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
