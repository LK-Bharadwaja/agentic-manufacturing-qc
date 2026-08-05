import itertools
import os
import pickle
import warnings
import numpy as np
import pandas as pd

FUZZY_FEATURES = [
    'Channel 2 RMS',
    'Channel 2 PEAK',
    'Channel 2 KURTOSIS',
    'Channel 2 PSD',
]
LEVELS = ['Low', 'Medium', 'High']
OUTPUT_MAP = {'Smooth': 1.9970, 'Average': 2.3511, 'Rough': 2.7480}

_BOUNDS_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'fuzzy_bounds.pkl')
_bounds_cache: dict = {}


def _load_bounds() -> dict:
    if _bounds_cache:
        return _bounds_cache
    path = os.path.abspath(_BOUNDS_FILE)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        _bounds_cache.update(data)
    return _bounds_cache


def _triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function. Returns membership degree in [0, 1]."""
    if c <= a:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def _mf_params(bounds: dict, feat: str, val: float) -> dict:
    """Return (a, b, c) for Low/Medium/High given feature bounds."""
    if feat in bounds:
        b = bounds[feat]
        return {
            'Low':    (b['min'],  b['min'],  b['mean']),
            'Medium': (b['q1'],   b['q2'],   b['q3']),
            'High':   (b['mean'], b['max'],  b['max']),
        }
    # Fallback: symmetric around the observed value — outputs 'Average'
    warnings.warn(
        f"No training bounds found for '{feat}'. "
        "Using symmetric fallback MFs — run save_models.py for accurate predictions.",
        UserWarning,
        stacklevel=4,
    )
    return {
        'Low':    (val * 0.50, val * 0.50, val),
        'Medium': (val * 0.75, val,        val * 1.25),
        'High':   (val,        val * 1.50, val * 1.50),
    }


def _output_label(combo: tuple) -> str:
    counts = {lv: combo.count(lv) for lv in LEVELS}
    if counts['High'] >= 3:
        return 'Rough'
    if counts['Low'] >= 3:
        return 'Smooth'
    return 'Average'


# Pre-build all 81 rules (3^4 combinations)
_RULES = [
    {'inputs': combo, 'output': _output_label(combo), 'value': OUTPUT_MAP[_output_label(combo)]}
    for combo in itertools.product(LEVELS, repeat=4)
]


def predict(features_df: pd.DataFrame) -> float:
    """
    Predict surface roughness using the 81-rule Mamdani fuzzy system.
    Uses centroid defuzzification.
    Returns Ra estimate in µm.
    """
    missing = [c for c in FUZZY_FEATURES if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing columns for fuzzy prediction: {missing}")

    bounds = _load_bounds()
    row = features_df.iloc[0]

    # Build membership function params for each feature
    mf = {
        feat: _mf_params(bounds, feat, float(row[feat]))
        for feat in FUZZY_FEATURES
    }

    # Evaluate all 81 rules (min t-norm), accumulate weighted sum
    num = 0.0
    den = 0.0
    for rule in _RULES:
        strength = min(
            _triangular_mf(float(row[FUZZY_FEATURES[i]]), *mf[FUZZY_FEATURES[i]][rule['inputs'][i]])
            for i in range(4)
        )
        if strength > 0:
            num += strength * rule['value']
            den += strength

    if den < 1e-12:
        # No rules fired — return centroid of all outputs (Average)
        return OUTPUT_MAP['Average']

    result = num / den
    # Clamp to physical range
    result = float(np.clip(result, OUTPUT_MAP['Smooth'], OUTPUT_MAP['Rough']))
    return result


def get_rule_count() -> int:
    return len(_RULES)
