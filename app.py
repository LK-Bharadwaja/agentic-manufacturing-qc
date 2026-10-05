"""Streamlit front end. Talks to the prediction API over HTTP.

Run:  streamlit run app.py   (with the API reachable at SRP_API_URL)
"""

import html
import json
import os

import pandas as pd
import requests
import streamlit as st

from modules.config import TRAINING_RA_MAX, TRAINING_RA_MIN

API_URL = os.getenv("SRP_API_URL", "http://localhost:8000").rstrip("/")
# Browser-facing address for links; API_URL may be an internal Docker hostname.
PUBLIC_API_URL = os.getenv("SRP_PUBLIC_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("SRP_REQUEST_TIMEOUT", "120"))
RA_TOLERANCE_UM = 2.3

st.set_page_config(
    page_title="Surface Roughness Predictor",
    page_icon="⚙️",
    layout="wide",
)

THEME_CSS = """
<style>
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
  --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --danger:#f85149; --mono-accent:#79c0ff;
  --mono:'JetBrains Mono','Roboto Mono',ui-monospace,SFMono-Regular,Consolas,'Courier New',monospace;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background:var(--bg) !important; color:var(--text);
}
[data-testid="stHeader"] { height:2.2rem; border-bottom:1px solid var(--border); }
.block-container { padding:1.2rem 1.6rem 1.5rem !important; max-width:100% !important; }
[data-testid="stVerticalBlock"] { gap:0.55rem !important; }
hr { margin:0.6rem 0 !important; border-color:var(--border) !important; }
p, li, label, .stMarkdown { color:var(--text); }
[data-testid="stCaptionContainer"], small, .stCaption { color:var(--muted) !important; }
h1 { font-size:1.45rem !important; letter-spacing:0.08em; text-transform:uppercase; padding:0 0 .3rem !important; }
h2 { font-size:0.95rem !important; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted) !important;
     border-bottom:1px solid var(--border); padding:0.2rem 0 0.3rem !important; margin-top:0.4rem; }
h3 { font-size:0.82rem !important; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted) !important;
     padding:0.2rem 0 !important; }
code, pre, kbd, [data-testid="stCode"] * { font-family:var(--mono) !important; }
code { color:var(--mono-accent) !important; background:var(--panel) !important; border:1px solid var(--border);
       border-radius:3px; padding:0 .3em; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"], [data-testid="stTable"] {
  font-family:var(--mono) !important; }

[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] h2 { border:none; }

[data-testid="stAlert"] { background:var(--panel) !important; border:1px solid var(--border);
  border-left:3px solid var(--accent); border-radius:4px; padding:.4rem .7rem; color:var(--text); }

.stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { letter-spacing:0.08em; text-transform:uppercase; font-size:.78rem;
  color:var(--muted); padding:.4rem 1rem; background:transparent; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent) !important; }

[data-testid="stFileUploaderDropzone"], .stTextInput input {
  background:var(--panel) !important; border:1px solid var(--border) !important; color:var(--text) !important;
  border-radius:4px; }
.stTextInput input:focus { border-color:var(--accent) !important; box-shadow:none !important; }
.stButton > button { background:var(--panel); color:var(--accent); border:1px solid var(--accent);
  border-radius:4px; letter-spacing:.1em; text-transform:uppercase; font-size:.78rem; }
.stButton > button:hover { background:var(--accent); color:var(--bg); border-color:var(--accent); }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--border) !important; border-radius:4px; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:4px; }

.card-row { display:flex; gap:.6rem; flex-wrap:wrap; }
.card { flex:1 1 200px; background:var(--panel); border:1px solid var(--border); border-radius:4px;
  padding:.55rem .8rem; }
.card.warn { border-color:var(--warn); background:rgba(210,153,34,.10); }
.card.danger { border-color:var(--danger); background:rgba(248,81,73,.10); }
.card .label { font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.card .value { font-family:var(--mono); font-size:1.55rem; color:var(--mono-accent); line-height:1.3; }
.card .value small { font-size:.8rem; color:var(--muted); }
.card.warn .value { color:var(--warn); } .card.danger .value { color:var(--danger); }
.card .sub { font-size:.74rem; color:var(--muted); letter-spacing:.04em; min-height:1.1em; }
.card .flag { font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; margin-top:.2rem; }
.card.warn .flag { color:var(--warn); } .card.danger .flag { color:var(--danger); }
.mono { font-family:var(--mono); color:var(--mono-accent); }

.timeline { border-left:1px solid var(--border); margin-left:.5rem; padding-left:.9rem; }
.tl-row { position:relative; background:var(--panel); border:1px solid var(--border); border-radius:4px;
  padding:.4rem .7rem; margin-bottom:.45rem; }
.tl-row::before { content:""; position:absolute; left:-1.32rem; top:.75rem; width:.55rem; height:.55rem;
  border-radius:50%; background:var(--accent); box-shadow:0 0 0 3px var(--bg); }
.tl-row.warn { border-color:var(--warn); } .tl-row.warn::before { background:var(--warn); }
.tl-row.danger { border-color:var(--danger); } .tl-row.danger::before { background:var(--danger); }
.tl-head { display:flex; gap:.6rem; align-items:baseline; }
.tl-idx { font-family:var(--mono); color:var(--muted); font-size:.75rem; }
.tl-name { font-family:var(--mono); color:var(--mono-accent); font-size:.88rem; }
.tl-row.warn .tl-name { color:var(--warn); } .tl-row.danger .tl-name { color:var(--danger); }
.tl-tag { margin-left:auto; font-size:.65rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.tl-reason { color:var(--muted); font-size:.78rem; margin:.2rem 0; }
.tl-row details { margin-top:.2rem; }
.tl-row summary { cursor:pointer; color:var(--muted); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; }
.tl-row pre { font-family:var(--mono); font-size:.74rem; background:var(--bg); border:1px solid var(--border);
  border-radius:3px; padding:.4rem .6rem; margin:.3rem 0; white-space:pre-wrap; word-break:break-word;
  color:var(--text); }

.answer { background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:4px; padding:.7rem .9rem; }
.answer.ungrounded { border-color:var(--warn); background:rgba(210,153,34,.10); }
.answer .badge { font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; font-family:var(--mono);
  color:var(--accent); margin-bottom:.3rem; }
.answer.ungrounded .badge { color:var(--warn); }
.pill { display:inline-block; font-family:var(--mono); font-size:.72rem; color:var(--mono-accent);
  background:var(--panel); border:1px solid var(--border); border-radius:999px; padding:.1rem .6rem;
  margin:.15rem .3rem 0 0; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def _esc(value) -> str:
    return html.escape(str(value))


def _ra_state(ra: float) -> tuple[str, str]:
    """Return (css class, flag text) for an Ra value."""
    if ra < TRAINING_RA_MIN or ra > TRAINING_RA_MAX:
        return "danger", "Extrapolation: outside training range"
    if ra > RA_TOLERANCE_UM:
        return "warn", "Exceeds tolerance"
    return "", ""


def _card(label: str, value: str, sub: str = "", state: str = "", flag: str = "") -> str:
    flag_html = f'<div class="flag">{_esc(flag)}</div>' if flag else ""
    return (
        f'<div class="card {state}"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{value}</div><div class="sub">{_esc(sub)}</div>{flag_html}</div>'
    )


def _tool_state(call: dict) -> tuple[str, str]:
    out = str(call["output"])
    if out.startswith("Error") or "unavailable" in out:
        return "danger", "failed"
    if call["tool"] == "check_training_range" and out.startswith("WARNING"):
        return "warn", "range warning"
    return "", "ok"


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
    st.caption(f"API docs: {PUBLIC_API_URL}/docs")

if status is None:
    st.error(
        f"Cannot reach the prediction API at `{API_URL}`.  \n"
        "Start the full stack with `docker compose up`, or run the API directly with "
        "`uvicorn api.main:app --port 8000`."
    )
    st.stop()

st.divider()

def _post_file(path: str, uploaded_file):
    return requests.post(
        f"{API_URL}{path}",
        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
        timeout=REQUEST_TIMEOUT,
    )


def _error_detail(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except json.JSONDecodeError:
        return response.text


def render_predictions_tab():
    # ── Upload ────────────────────────────────────────────────────────────────
    st.header("1. Upload Vibration Data")
    uploaded = st.file_uploader(
        "Choose a part file",
        type=["xlsx", "xls", "csv"],
        help="File must contain at least one of: Channel1 [g], Channel2 [g], Channel3 [g]",
        key="predict_upload",
    )

    if uploaded is None:
        st.info("📂 Upload a vibration signal file to begin.")
        return

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Extracting 36 features and running all three models..."):
        try:
            response = _post_file("/predict", uploaded)
        except requests.RequestException as exc:
            st.error(f"Request to the API failed: {exc}")
            return

    if response.status_code != 200:
        st.error(f"❌ {_error_detail(response)}")
        return

    result = response.json()
    st.success(
        f"✅ Analysed **{uploaded.name}** in {result['latency_ms']:.0f} ms — "
        f"{len(result['channels_found'])} of 3 channels found"
    )

    # ── Features ──────────────────────────────────────────────────────────────
    st.divider()
    st.header("2. Extracted Features")
    with st.expander(f"View all {len(result['features'])} features"):
        features_df = pd.DataFrame(
            sorted(result["features"].items()), columns=["Feature", "Value"]
        )
        features_df["Value"] = features_df["Value"].map("{:.6f}".format)
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    st.divider()
    st.header("3. Predictions (Ra, µm)")

    cards = []
    for prediction in result["predictions"]:
        name = prediction["model"]
        if prediction["available"]:
            state, flag = _ra_state(prediction["ra"])
            category = result["category"] if name == "Fuzzy Logic" and result["category"] else ""
            cards.append(
                _card(
                    name,
                    f"{prediction['ra']:.4f} <small>µm</small>",
                    f"Category: {category}" if category else "",
                    state,
                    flag,
                )
            )
        else:
            cards.append(_card(name, "—", prediction["detail"] or "Unavailable", "danger", "Unavailable"))
    st.markdown(f'<div class="card-row">{"".join(cards)}</div>', unsafe_allow_html=True)

    # ── Comparison ──────────────────────────────────────────────────────────────
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

    # ── Training performance ──────────────────────────────────────────────────
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


def render_agent_tab():
    st.header("Agentic QC Pipeline")
    st.markdown(
        "Runs the LangGraph tool-calling QC agent instead of calling all three models "
        "directly. The agent decides for itself which models to call, checks signal "
        "noise, validates each prediction against the training range, and consults the "
        "project documentation (RAG) when predictions are borderline or disagree."
    )

    uploaded = st.file_uploader(
        "Choose a part file",
        type=["xlsx", "xls", "csv"],
        help="File must contain at least one of: Channel1 [g], Channel2 [g], Channel3 [g]",
        key="agent_upload",
    )

    if uploaded is None:
        st.info("📂 Upload a vibration signal file to run the QC agent.")
        return

    with st.spinner("Running the QC agent (multiple LLM + tool round-trips — this can take a minute)..."):
        try:
            response = _post_file("/agent/predict", uploaded)
        except requests.RequestException as exc:
            st.error(f"Request to the API failed: {exc}")
            return

    if response.status_code != 200:
        st.error(f"❌ {_error_detail(response)}")
        return

    result = response.json()

    st.divider()
    st.subheader("Final Ra Estimate")
    ra = result["ra"]
    if ra is None:
        st.error("The agent did not produce a final Ra estimate — see the report below for why.")
    else:
        state, flag = _ra_state(ra)
        sub = f"Tolerance {RA_TOLERANCE_UM} µm" if state != "warn" else (
            f"Exceeds the {RA_TOLERANCE_UM} µm smooth/average tolerance"
        )
        st.markdown(
            f'<div class="card-row">{_card("Final Ra", f"{ra:.4f} <small>µm</small>", sub, state, flag if state == "danger" else "")}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Models consulted")
    st.write(", ".join(result["models_used"]) if result["models_used"] else "None")

    if result["noise_level"]:
        st.subheader("Signal noise assessment")
        st.text(result["noise_level"])

    st.subheader("Documentation (RAG) consulted?")
    rag_calls = [c for c in result["tool_calls"] if c["tool"] == "query_rag"]
    if result["rag_consulted"]:
        st.info(f"✅ Yes — queried the project documentation {len(rag_calls)} time(s).")
        first_reason = rag_calls[0]["reasoning"] if rag_calls and rag_calls[0]["reasoning"] else None
        if first_reason:
            st.caption(f"Agent's stated reason: {first_reason}")
    else:
        st.caption(
            "❌ No — the agent judged the predictions consistent and in-range enough "
            "not to need the docs."
        )

    st.subheader(f"Tool-call trace ({len(result['tool_calls'])} calls)")
    rows = []
    for i, call in enumerate(result["tool_calls"], 1):
        state, tag = _tool_state(call)
        reason = (
            f'<div class="tl-reason">{_esc(call["reasoning"])}</div>' if call["reasoning"] else ""
        )
        rows.append(
            f'<div class="tl-row {state}"><div class="tl-head">'
            f'<span class="tl-idx">{i:02d}</span><span class="tl-name">{_esc(call["tool"])}</span>'
            f'<span class="tl-tag">{tag}</span></div>{reason}'
            f'<details><summary>Input / output</summary>'
            f'<pre>{_esc(json.dumps(call["input"], indent=2, default=str))}</pre>'
            f'<pre>{_esc(call["output"])}</pre></details></div>'
        )
    st.markdown(f'<div class="timeline">{"".join(rows)}</div>', unsafe_allow_html=True)

    st.subheader("Final QC report")
    st.markdown(result["report"])


def render_rag_tab():
    st.header("Ask About This Project")
    st.markdown(
        "Ask a question about the project's methodology, models, or documentation. "
        "Answers are grounded in the project docs via RAG — if nothing relevant is "
        "found, you'll be told rather than getting a fabricated answer."
    )

    question = st.text_input(
        "Your question",
        key="rag_question",
        placeholder="e.g. What is the training set size?",
    )
    ask = st.button("Ask", key="rag_ask_button")

    if not ask:
        return
    if not question.strip():
        st.info("Type a question above, then click Ask.")
        return

    with st.spinner("Retrieving relevant docs and asking Gemini..."):
        try:
            response = requests.post(
                f"{API_URL}/rag/ask",
                json={"question": question},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            st.error(f"Request to the API failed: {exc}")
            return

    if response.status_code != 200:
        st.error(f"❌ {_error_detail(response)}")
        return

    result = response.json()

    st.divider()
    st.subheader("Answer")
    grounded = result["grounded"]
    badge = "grounded: true" if grounded else "grounded: false — fallback, not from the docs"
    pills = "".join(f'<span class="pill">{_esc(s)}</span>' for s in result["sources"])
    st.markdown(
        f'<div class="answer {"" if grounded else "ungrounded"}">'
        f'<div class="badge">{badge}</div>{_esc(result["answer"])}'
        f'{f"<div>{pills}</div>" if pills else ""}</div>',
        unsafe_allow_html=True,
    )
    if not grounded:
        st.caption(
            "No relevant information was found in the project documentation for this question."
        )


tab_predict, tab_agent, tab_rag = st.tabs(
    ["📊 Predictions", "🤖 Agentic QC Pipeline", "📚 Ask About This Project"]
)

with tab_predict:
    render_predictions_tab()

with tab_agent:
    render_agent_tab()

with tab_rag:
    render_rag_tab()
