import itertools
import logging
import pickle

import numpy as np
import pandas as pd

from modules.config import (
    FUZZY_AVERAGE_MAX,
    FUZZY_BOUNDS_FILE,
    FUZZY_FEATURES,
    FUZZY_OUTPUT_MAP,
    FUZZY_SMOOTH_MAX,
)
from modules.exceptions import InvalidInputError

logger = logging.getLogger(__name__)

OUTPUT_MAP = FUZZY_OUTPUT_MAP
LEVELS = ['Low', 'Medium', 'High']

_bounds_cache: dict = {}


def is_available() -> bool:
    """Return True if trained membership bounds exist (fallback MFs work without them)."""
    return FUZZY_BOUNDS_FILE.exists()


def categorize(ra: float) -> str:
    """Map a predicted Ra to its surface quality category."""
    if ra < FUZZY_SMOOTH_MAX:
        return 'Smooth'
    if ra < FUZZY_AVERAGE_MAX:
        return 'Average'
    return 'Rough'


def _load_bounds() -> dict:
    if _bounds_cache:
        return _bounds_cache
    if FUZZY_BOUNDS_FILE.exists():
        with open(FUZZY_BOUNDS_FILE, 'rb') as f:
            _bounds_cache.update(pickle.load(f))
        logger.info("Loaded fuzzy bounds for %d features", len(_bounds_cache))
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
    logger.warning(
        "No training bounds for '%s'; using symmetric fallback MFs. "
        "Run save_models.py for accurate predictions.",
        feat,
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
        raise InvalidInputError(f"Missing columns for fuzzy prediction: {missing}")

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
