"""
save_models.py
--------------
Extracts vibration features from parts 1-27 (training set),
selects the best 4-feature MLR combination, trains CNN (all 36 features),
computes fuzzy membership bounds, and saves everything to models/.

Run from the project directory:
    python save_models.py
"""

import json
import os
import pickle
import sys
import warnings
import zipfile
from datetime import UTC, datetime
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Resolve paths relative to this script
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from modules.feature_extraction import (  # noqa: E402 — needs PROJECT_DIR on sys.path
    FEATURE_NAMES,
    extract_features_from_file,
)
from modules.fuzzy_logic import FUZZY_FEATURES  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
PARTS_ZIP = os.path.join(PROJECT_DIR, 'data', 'parts.zip')
PARTS_DIR = os.path.join(PROJECT_DIR, 'data', 'parts')

# Ground-truth Ra values for parts 1-27 (training)
Y_TRAIN = [
    2.024, 1.983, 1.945, 2.384, 2.439, 2.467, 2.752, 2.812, 2.795, 2.185,
    2.024, 1.931, 2.405, 2.408, 2.481, 2.781, 2.748, 2.745, 2.085, 2.002,
    2.023, 2.441, 2.478, 2.440, 2.776, 2.693, 2.757,
]

# Feature groups — one representative feature per group is selected by R²
FEATURE_GROUPS = [
    "RMS", "PEAK", "CREST FACTOR", "KURTOSIS",
    "PSD", "Dominant Frequency", "Total Energy", "Spectral Entropy",
    "STFT Centroid", "STFT Bandwidth", "STFT Entropy", "STFT Kurtosis",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _banner(msg: str):
    print(f"\n{'-' * 60}")
    print(f"  {msg}")
    print('-' * 60)


def _custom_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(1 - np.abs(y_true - y_pred) / np.abs(y_true))) * 100


# ── Step 1: Locate / extract part files ─────────────────────────────────────

def find_parts_dir() -> str:
    """Return directory where part xlsx files live, extracting zip if needed."""
    # Check data/parts/ first (normal location after restructure)
    if os.path.exists(os.path.join(PARTS_DIR, 'part 1.xlsx')):
        return PARTS_DIR
    # Fallback: files extracted directly to project root (legacy layout)
    if os.path.exists(os.path.join(PROJECT_DIR, 'part 1.xlsx')):
        return PROJECT_DIR
    # Extract from zip
    if not os.path.exists(PARTS_ZIP):
        raise FileNotFoundError(
            f"parts.zip not found at {PARTS_ZIP} and no extracted part files found.\n"
            "Place parts.zip in the project directory and re-run."
        )
    print(f"Extracting {PARTS_ZIP} -> {PARTS_DIR}  (this may take a moment)...")
    os.makedirs(PARTS_DIR, exist_ok=True)
    with zipfile.ZipFile(PARTS_ZIP, 'r') as z:
        z.extractall(PARTS_DIR)
    print("Extraction complete.")
    return PARTS_DIR


# ── Step 2: Feature extraction ───────────────────────────────────────────────

def build_training_features(parts_dir: str) -> pd.DataFrame:
    _banner("Step 2 — Extracting features from 27 training parts")
    rows = []
    for i in range(1, 28):
        path = os.path.join(parts_dir, f'part {i}.xlsx')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training file not found: {path}\n"
                f"Expected part files 1-27 in {parts_dir}"
            )
        print(f"  Part {i:2d}/27 ...", end='\r', flush=True)
        feat_df = extract_features_from_file(path)
        rows.append(feat_df)
    print(f"  Done — {len(rows)} parts processed.           ")
    return pd.concat(rows, ignore_index=True)


# ── Step 3: MLR — feature selection + training ───────────────────────────────

def train_mlr(features_df: pd.DataFrame, y_train: list) -> list:
    _banner("Step 3 — MLR: feature selection and training")

    X_all = features_df[FEATURE_NAMES]
    Y = pd.Series(y_train, dtype=float)

    # R² per feature
    r2_map = {}
    for col in FEATURE_NAMES:
        try:
            m = sm.OLS(Y, sm.add_constant(X_all[[col]])).fit()
            r2_map[col] = m.rsquared
        except Exception:
            r2_map[col] = 0.0

    # Best feature per group
    top12 = []
    for group in FEATURE_GROUPS:
        group_cols = [c for c in FEATURE_NAMES if group in c]
        if group_cols:
            best = max(group_cols, key=lambda c: r2_map.get(c, 0.0))
            if best not in top12:
                top12.append(best)

    print("  Top-12 selected features:")
    for f in top12:
        print(f"    {f}  (R²={r2_map.get(f, 0):.4f})")

    # Exhaustive 4-feature combination search
    print(f"\n  Searching all C({len(top12)},4) = {len(list(combinations(top12, 4)))} combinations...")
    best_combo, best_acc, best_r2, best_model = None, 0.0, 0.0, None

    for combo in combinations(top12, 4):
        X = features_df[list(combo)].values
        m = LinearRegression().fit(X, Y)
        yp = m.predict(X)
        acc = _custom_accuracy(Y, yp)
        r2 = r2_score(Y, yp)
        if acc > best_acc:
            best_acc, best_r2, best_combo, best_model = acc, r2, combo, m

    print(f"\n  Best 4-feature combination: {best_combo}")
    print(f"  Custom accuracy: {best_acc:.2f}%  |  R²: {best_r2:.4f}")

    # Build equation string
    terms = "  +  ".join(
        f"({c:.4f} × {f})" for c, f in zip(best_model.coef_, best_combo, strict=True)
    )
    print(f"  Equation: Ra = {terms} + {best_model.intercept_:.4f}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'mlr_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODELS_DIR, 'mlr_features.pkl'), 'wb') as f:
        pickle.dump(list(best_combo), f)
    print("  Saved: models/mlr_model.pkl  +  models/mlr_features.pkl")
    return {
        'accuracy_pct': round(best_acc, 2),
        'r2': round(best_r2, 4),
        'features_used': len(best_combo),
        'notes': f"Best 4-feature combination of {len(top12)} group representatives",
    }


# ── Step 4: Fuzzy logic bounds ───────────────────────────────────────────────

def save_fuzzy_bounds(features_df: pd.DataFrame):
    _banner("Step 4 — Fuzzy logic: computing membership function bounds")
    bounds = {}
    for feat in FUZZY_FEATURES:
        col = features_df[feat]
        bounds[feat] = {
            'min':  float(col.min()),
            'q1':   float(col.quantile(0.25)),
            'q2':   float(col.quantile(0.50)),
            'q3':   float(col.quantile(0.75)),
            'mean': float(col.mean()),
            'max':  float(col.max()),
        }
        b = bounds[feat]
        print(f"  {feat}: min={b['min']:.5f}  mean={b['mean']:.5f}  max={b['max']:.5f}")

    with open(os.path.join(MODELS_DIR, 'fuzzy_bounds.pkl'), 'wb') as f:
        pickle.dump(bounds, f)
    print("  Saved: models/fuzzy_bounds.pkl")


def evaluate_fuzzy(features_df: pd.DataFrame, y_train: list) -> dict:
    """Score the fuzzy system on the training set using the bounds just written."""
    import modules.fuzzy_logic as fuzzy_module

    fuzzy_module._bounds_cache.clear()
    preds = [
        fuzzy_module.predict(features_df.iloc[[i]]) for i in range(len(features_df))
    ]
    acc = _custom_accuracy(y_train, preds)
    r2 = r2_score(y_train, preds)
    print(f"  Fuzzy custom accuracy: {acc:.2f}%  |  R²: {r2:.4f}")
    return {
        'accuracy_pct': round(acc, 2),
        'r2': round(r2, 4),
        'features_used': len(FUZZY_FEATURES),
        'notes': '81-rule Mamdani system, centroid defuzzification',
    }


# ── Step 5: CNN training ──────────────────────────────────────────────────────

def train_cnn(features_df: pd.DataFrame, y_train: list):
    _banner("Step 5 — CNN: training Conv1D model (300 epochs)")
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("  WARNING: TensorFlow not installed — skipping CNN training.")
        print("  Install with:  pip install tensorflow")
        return None

    X = features_df[FEATURE_NAMES].values.astype(float)
    Y = np.array(y_train, dtype=float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu',
               input_shape=(X_seq.shape[1], 1)),
        Dropout(0.2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    class _ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:3d}/300  loss={logs['loss']:.6f}")

    model.fit(X_seq, Y, epochs=300, batch_size=8,
              verbose=0, callbacks=[_ProgressCallback()])

    y_pred = model.predict(X_seq, verbose=0).flatten()
    acc = _custom_accuracy(Y, y_pred)
    r2 = r2_score(Y, y_pred)
    print(f"\n  CNN custom accuracy: {acc:.2f}%  |  R²: {r2:.4f}")

    model.save(os.path.join(MODELS_DIR, 'cnn_model.keras'))
    with open(os.path.join(MODELS_DIR, 'cnn_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("  Saved: models/cnn_model.keras  +  models/cnn_scaler.pkl")
    return {
        'accuracy_pct': round(acc, 2),
        'r2': round(r2, 4),
        'features_used': len(FEATURE_NAMES),
        'notes': 'Conv1D(32) → Dropout(0.2) → Dense(16) → Dense(1), 300 epochs',
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    _banner("Capstone - save_models.py")
    print(f"  Project dir : {PROJECT_DIR}")
    print(f"  Models dir  : {MODELS_DIR}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    parts_dir = find_parts_dir()
    _banner("Step 1 — Part files")
    print(f"  Using parts from: {parts_dir}")

    features_df = build_training_features(parts_dir)
    y_train = Y_TRAIN[:len(features_df)]

    metrics = {'MLR': train_mlr(features_df, y_train)}
    save_fuzzy_bounds(features_df)
    metrics['Fuzzy Logic'] = evaluate_fuzzy(features_df, y_train)
    cnn_metrics = train_cnn(features_df, y_train)
    if cnn_metrics:
        metrics['CNN'] = cnn_metrics

    _banner("Step 6 — Writing models/metrics.json")
    payload = {
        'trained_at': datetime.now(UTC).isoformat(),
        'training_parts': len(features_df),
        'models': metrics,
    }
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print("  Saved: models/metrics.json")

    _banner("All models saved successfully!")
    print(f"  Output directory: {MODELS_DIR}")
    saved = os.listdir(MODELS_DIR)
    for f in sorted(saved):
        fpath = os.path.join(MODELS_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
