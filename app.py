"""Streamlit front end. Talks to the prediction API over HTTP.

Run:  streamlit run app.py   (with the API reachable at SRP_API_URL)
"""

import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("SRP_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("SRP_REQUEST_TIMEOUT", "120"))

st.set_page_config(
    page_title="Surface Roughness Predictor",
    page_icon="⚙️",
    layout="wide",
)


def _get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


st.title("⚙️ Surface Roughness Predictor")
st.markdown(
    "Predict machined-part surface roughness **Ra (µm)** from 3-axis vibration signals "
    "using Multiple Linear Regression, Fuzzy Logic, and a Conv1D CNN."
)

# ── Sidebar: backend + model status ───────────────────────────────────────────
status = _get("/models/status")

with st.sidebar:
    st.header("Backend")
    if status is None:
        st.error("❌ API unreachable")
        st.caption(f"Tried `{API_URL}`")
    else:
        st.success("✅ API connected")
        st.caption(f"`{API_URL}`")

    st.divider()
    st.header("Model Status")
    if status is None:
        st.info("Unavailable until the API responds.")
    else:
        for model in status["models"]:
            if model["ready"]:
                st.success(f"✅ {model['model']}")
            else:
                st.error(f"❌ {model['model']}")
                if model["detail"]:
                    st.caption(model["detail"])

    st.divider()
    st.header("Expected file format")
    st.markdown(
        "Upload an `.xlsx` or `.csv` file containing:\n"
        "- `Channel1 [g]`\n"
        "- `Channel2 [g]`\n"
        "- `Channel3 [g]`"
    )
    st.caption(f"API docs: {API_URL}/docs")

if status is None:
    st.error(
        f"Cannot reach the prediction API at `{API_URL}`.  \n"
        "Start the full stack with `docker compose up`, or run the API directly with "
        "`uvicorn api.main:app --port 8000`."
    )
    st.stop()

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
st.header("1. Upload Vibration Data")
uploaded = st.file_uploader(
    "Choose a part file",
    type=["xlsx", "xls", "csv"],
    help="File must contain at least one of: Channel1 [g], Channel2 [g], Channel3 [g]",
)

if uploaded is None:
    st.info("📂 Upload a vibration signal file to begin.")
    st.stop()

# ── Predict ───────────────────────────────────────────────────────────────────
with st.spinner("Extracting 36 features and running all three models..."):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (uploaded.name, uploaded.getvalue())},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Request to the API failed: {exc}")
        st.stop()

if response.status_code != 200:
    try:
        detail = response.json().get("detail", response.text)
    except json.JSONDecodeError:
        detail = response.text
    st.error(f"❌ {detail}")
    st.stop()

result = response.json()
st.success(
    f"✅ Analysed **{uploaded.name}** in {result['latency_ms']:.0f} ms — "
    f"{len(result['channels_found'])} of 3 channels found"
)

# ── Features ──────────────────────────────────────────────────────────────────
st.divider()
st.header("2. Extracted Features")
with st.expander(f"View all {len(result['features'])} features"):
    features_df = pd.DataFrame(
        sorted(result["features"].items()), columns=["Feature", "Value"]
    )
    features_df["Value"] = features_df["Value"].map("{:.6f}".format)
    st.dataframe(features_df, use_container_width=True, hide_index=True)

# ── Predictions ───────────────────────────────────────────────────────────────
st.divider()
st.header("3. Predictions (Ra, µm)")

icons = {"MLR": "📐", "Fuzzy Logic": "🔀", "CNN": "🧠"}
for column, prediction in zip(st.columns(3), result["predictions"], strict=True):
    with column:
        name = prediction["model"]
        st.subheader(f"{icons.get(name, '•')} {name}")
        if prediction["available"]:
            st.metric("Predicted Ra", f"{prediction['ra']:.4f} µm")
            if name == "Fuzzy Logic" and result["category"]:
                st.info(f"Category: **{result['category']}**")
        else:
            st.error(prediction["detail"] or "Unavailable")

# ── Comparison ────────────────────────────────────────────────────────────────
st.divider()
st.header("4. Model Comparison")

ran = [p for p in result["predictions"] if p["available"]]
if ran:
    st.dataframe(
        pd.DataFrame(
            [{"Model": p["model"], "Predicted Ra (µm)": round(p["ra"], 4)} for p in ran]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if result["spread_um"] is not None:
        st.caption(f"Prediction spread across models: **{result['spread_um']:.4f} µm**")

# ── Training performance ──────────────────────────────────────────────────────
metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    st.subheader("Training-set performance")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy (%)": m["accuracy_pct"],
                    "R²": m["r2"],
                    "Features": m["features_used"],
                    "Architecture": m["notes"],
                }
                for name, m in metrics["models"].items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Trained on {metrics['training_parts']} parts. "
        "These are training-set fits, not held-out scores."
    )
