"""
app.py — Streamlit UI for Surface Roughness Predictor
Run:  streamlit run app.py
"""

import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Ensure project root is on the import path
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from modules.feature_extraction import extract_features_from_df, FEATURE_NAMES, CHANNELS
import modules.mlr as mlr_module
import modules.fuzzy_logic as fuzzy_module
import modules.cnn_model as cnn_module

# ── Model file locations ──────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(_PROJECT_DIR, 'models')

_MODEL_FILES = {
    'MLR': [
        os.path.join(_MODELS_DIR, 'mlr_model.pkl'),
        os.path.join(_MODELS_DIR, 'mlr_features.pkl'),
    ],
    'Fuzzy Logic': [
        os.path.join(_MODELS_DIR, 'fuzzy_bounds.pkl'),
    ],
    'CNN': [
        os.path.join(_MODELS_DIR, 'cnn_model.keras'),
        os.path.join(_MODELS_DIR, 'cnn_scaler.pkl'),
    ],
}


def _model_ready(name: str) -> bool:
    return all(os.path.exists(p) for p in _MODEL_FILES[name])


def _tensorflow_installed() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surface Roughness Predictor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Surface Roughness Predictor")
st.markdown(
    "Predict machined-part surface roughness **Ra (µm)** from 3-axis vibration signals "
    "using Multiple Linear Regression, Fuzzy Logic, and CNN."
)

# ── Sidebar: model status ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Status")
    any_missing = False
    for name in ['MLR', 'Fuzzy Logic', 'CNN']:
        ready = _model_ready(name)
        if name == 'CNN' and ready and not _tensorflow_installed():
            st.warning(f"⚠️ {name} — TensorFlow not installed")
            any_missing = True
        elif ready:
            st.success(f"✅ {name}")
        else:
            st.error(f"❌ {name} — files missing")
            any_missing = True

    st.divider()
    st.caption("Missing models? Run from project directory:")
    st.code("python save_models.py", language="bash")

    st.divider()
    st.header("Expected file format")
    st.markdown(
        "Upload an `.xlsx` or `.csv` file with columns:\n"
        "- `Channel1 [g]`\n"
        "- `Channel2 [g]`\n"
        "- `Channel3 [g]`"
    )

if any_missing:
    st.warning(
        "⚠️ **One or more model files are missing.**  \n"
        "Predictions for missing models will be unavailable.  \n"
        "Run `python save_models.py` from the project directory to train all models."
    )

st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
st.header("1. Upload Vibration Data")
uploaded = st.file_uploader(
    "Choose a part file",
    type=['xlsx', 'xls', 'csv'],
    help="File must contain at least one of: Channel1 [g], Channel2 [g], Channel3 [g]",
)

if uploaded is None:
    st.info("📂 Upload a vibration signal file to begin.")
    st.stop()

# ── Load and validate ─────────────────────────────────────────────────────────
try:
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"❌ Could not read '{uploaded.name}': {e}")
    st.stop()

st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")

# Channel availability
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("File preview")
    st.dataframe(df.head(6), use_container_width=True)

with col_b:
    st.subheader("Channel status")
    found_channels = []
    for ch in CHANNELS:
        if ch in df.columns:
            n = df[ch].dropna().shape[0]
            st.success(f"✅ {ch} — {n:,} samples")
            found_channels.append(ch)
        else:
            st.warning(f"⚠️ {ch} — not found (will use zeros)")

    if not found_channels:
        st.error(
            "No expected channel columns found. "
            "Columns must be named  Channel1 [g], Channel2 [g], Channel3 [g]."
        )
        st.stop()

# ── Feature extraction ────────────────────────────────────────────────────────
st.divider()
st.header("2. Feature Extraction")

with st.spinner("Extracting 36 vibration features (time / frequency / STFT)..."):
    try:
        features_df = extract_features_from_df(df)
        st.success("✅ 36 features extracted successfully")
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

with st.expander("View all 36 extracted features"):
    display = features_df.T.rename(columns={0: 'Value'})
    display['Value'] = display['Value'].map('{:.6f}'.format)
    st.dataframe(display, use_container_width=True)

# ── Predictions ───────────────────────────────────────────────────────────────
st.divider()
st.header("3. Predictions (Ra, µm)")

col1, col2, col3 = st.columns(3)

# — MLR —
with col1:
    st.subheader("📐 Multiple Linear Regression")
    if not _model_ready('MLR'):
        st.error("Model not available — run save_models.py")
    else:
        try:
            pred_mlr = mlr_module.predict(features_df)
            st.metric("Predicted Ra", f"{pred_mlr:.4f} µm")
            with st.expander("Equation"):
                st.code(mlr_module.get_equation())
            with st.expander("Features used"):
                for f in mlr_module.get_feature_names():
                    st.markdown(f"- `{f}`")
        except Exception as e:
            st.error(f"MLR prediction failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# — Fuzzy Logic —
with col2:
    st.subheader("🔀 Fuzzy Logic (81 rules)")
    fuzzy_bounds_ready = _model_ready('Fuzzy Logic')
    if not fuzzy_bounds_ready:
        st.warning("Bounds file missing — using fallback MFs (less accurate)")
    try:
        pred_fuzzy = fuzzy_module.predict(features_df)
        st.metric("Predicted Ra", f"{pred_fuzzy:.4f} µm")
        st.caption(
            f"Inputs: Ch2 RMS / PEAK / KURTOSIS / PSD  →  "
            f"{fuzzy_module.get_rule_count()} rules  →  centroid defuzz"
        )
        # Show output category
        if pred_fuzzy < 2.175:
            st.info("Category: **Smooth** (< 2.175 µm)")
        elif pred_fuzzy < 2.55:
            st.info("Category: **Average** (2.175 – 2.55 µm)")
        else:
            st.info("Category: **Rough** (> 2.55 µm)")
    except Exception as e:
        st.error(f"Fuzzy prediction failed: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

# — CNN —
with col3:
    st.subheader("🧠 CNN (Conv1D)")
    if not _tensorflow_installed():
        st.error("TensorFlow not installed — run:  pip install tensorflow")
    elif not _model_ready('CNN'):
        st.error("Model not available — run save_models.py")
    else:
        try:
            pred_cnn = cnn_module.predict(features_df)
            st.metric("Predicted Ra", f"{pred_cnn:.4f} µm")
            st.caption("36 features → MinMaxScaler → Conv1D(32) → Dense(16) → Dense(1)")
        except Exception as e:
            st.error(f"CNN prediction failed: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

# ── Summary ───────────────────────────────────────────────────────────────────
st.divider()
st.header("4. Model Comparison")

summary_data = {}
if _model_ready('MLR'):
    try:
        summary_data['MLR'] = round(mlr_module.predict(features_df), 4)
    except Exception:
        pass

try:
    summary_data['Fuzzy Logic'] = round(fuzzy_module.predict(features_df), 4)
except Exception:
    pass

if _tensorflow_installed() and _model_ready('CNN'):
    try:
        summary_data['CNN'] = round(cnn_module.predict(features_df), 4)
    except Exception:
        pass

if summary_data:
    summary_df = pd.DataFrame(
        list(summary_data.items()), columns=['Model', 'Predicted Ra (µm)']
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if len(summary_data) > 1:
        vals = list(summary_data.values())
        spread = max(vals) - min(vals)
        st.caption(f"Prediction spread across models: **{spread:.4f} µm**")
