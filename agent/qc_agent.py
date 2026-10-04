"""LangGraph tool-calling QC agent for surface roughness prediction.

Builds a small ReAct-style graph — one LLM node bound to the 6 tools in
agent/tools.py, one tool-execution node, and a conditional edge that loops
back to the LLM as long as it keeps requesting tools. The LLM decides which
tools to call, with what arguments, in what order, and how many times;
nothing here hardcodes a call sequence. See run_qc_pipeline() for the entry
point and _build_graph() for the actual graph wiring.
"""

import logging
import operator
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.tools import (
    check_training_range,
    get_signal_noise_level,
    predict_cnn,
    predict_fuzzy,
    predict_mlr,
    query_rag,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_NAME = "claude-sonnet-5"
_PREDICTION_TOOL_NAMES = {"predict_mlr", "predict_fuzzy", "predict_cnn"}
_LLM_TIMEOUT_SECONDS = 45.0

# Lets a caller (api/main.py) watch progress from outside the worker thread
# running run_qc_pipeline, e.g. to report the last completed tool call if an
# external wall-clock deadline fires before the pipeline itself returns.
_progress_var: ContextVar[dict | None] = ContextVar("qc_progress", default=None)

TOOLS = [
    get_signal_noise_level,
    predict_mlr,
    predict_fuzzy,
    predict_cnn,
    check_training_range,
    query_rag,
]
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# claude-sonnet-5 has no temperature parameter, so without an explicit
# tie-breaking rule, the agent's final weighting between outlier-exclusion
# and equal-averaging was observed to vary run-to-run on identical input
# (2.3776 vs 2.4516 for part 28) despite fully deterministic tool outputs.
_SYSTEM_PROMPT = """\
You are a Quality Control (QC) analyst agent for a CNC machining surface \
roughness prediction system. You are given the 36 extracted vibration \
features for one machined part and have tools to predict its surface \
roughness (Ra, in micrometres), assess input signal noise, and search the \
project's documentation.

Use your own judgment about which tools to call, with what arguments, and \
in what order — there is no fixed checklist to follow mechanically. That \
said, a sound QC process generally looks like this:

1. Start by checking the signal noise level with get_signal_noise_level. \
Noisy input should make you more cautious about trusting any single \
prediction.
2. Based on the noise level and your own reasoning, decide which of the \
three prediction models (predict_mlr, predict_fuzzy, predict_cnn) to call, \
in whatever order and combination makes sense to you. If the signal looks \
noisy you may want more than one model to cross-check each other; if it \
looks clean, fewer calls may be enough. Use real judgment, not a rigid \
rule — you decide.
3. Call check_training_range on every numeric Ra value you get back. A \
model's R² or accuracy figure only describes how well it fits the range of \
parts it was trained on (~1.9-2.8 µm) — it says nothing about how that \
model behaves on a part outside that range. If check_training_range flags \
a prediction as outside the training range, treat that prediction as a \
potential extrapolation error and downweight it rather than trusting it by \
default — do not pick it as your final answer just because it came from \
the model with the highest R², and be especially skeptical of it when the \
*other* models agree closely with each other and fall inside the known \
range. In that situation the in-range consensus is more likely correct \
than the out-of-range outlier.
4. If the resulting Ra values are borderline near the 2.3 µm \
smooth/average tolerance boundary, or if two or more models disagree by \
more than roughly 0.2 µm, call query_rag ONCE to check what the project \
documentation says (e.g. about thresholds, model reliability, or \
methodology) before finalizing your judgment. If the models agree closely \
and are not borderline, you do not need to consult the documentation. If \
query_rag comes back "Not grounded" or otherwise fails to answer, that \
means the documentation does not cover it — proceed with your own \
reasoning from the tool descriptions and the evidence already gathered. Do \
NOT call query_rag again with a rephrased or narrower version of the same \
question; a second attempt essentially never finds something the first \
one missed, and it only adds latency for no benefit.
5. Any R² or accuracy figures you encounter — whether in a tool's own \
description or in a query_rag answer — are fits measured on the same \
27-part training set the models were built from, not held-out or \
test-set performance on unseen parts. The dataset is too small to hold \
out meaningfully, so these numbers show how well a model matches data it \
already saw, not how accurate it will be here. Do not treat a higher \
training-set R² as proof a model is more accurate on this specific part, \
and never let it override a check_training_range warning.
6. When the three model predictions disagree, apply this rule, in order, \
to decide the final numeric estimate:
   a. If any model's prediction was flagged by check_training_range as an \
extrapolation risk (outside 1.931-2.812 µm), exclude it from the final \
estimate and use the average of the remaining in-range models.
   b. If all three are in-range but one diverges from the other two by \
more than 0.15 µm, treat it as an outlier relative to consensus and \
weight the final estimate toward the average of the two closer models, \
not an equal three-way average.
   c. Only use an equal-weighted average of all three when no model \
qualifies as an outlier under rules a or b (i.e. all three are \
reasonably close together).
7. When you have gathered enough evidence, stop calling tools and write a \
final QC report as plain text. Explain: which models you consulted and \
their Ra values, the noise assessment and how it affected your confidence, \
whether any prediction was flagged as an out-of-range extrapolation and \
how that affected which value you trusted, whether you consulted the \
documentation and what it added (if anything), and your overall \
conclusion about this part's surface quality category (Smooth / Average / \
Rough).

End your final report with a line of the exact form:
Final Ra Estimate: X.XXXX µm
using the Ra value, in micrometres to 4 decimal places, that you judge \
most trustworthy for this part given everything you found.

Do not call the same tool with identical arguments more than once. Do not \
call query_rag more than once in total, regardless of phrasing. Do not \
fabricate tool results — only report numbers actually returned by the tools.
"""

_FINAL_RA_RE = re.compile(r"final ra estimate:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TOOL_RA_RE = re.compile(r"ra\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


class QCState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_log: Annotated[list, operator.add]


def _load_llm():
    """Return a ChatAnthropic instance, same model as rag/qa_engine.py."""
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-key-here":
        raise OSError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Add your key to the .env file at the project root:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=_MODEL_NAME,
            anthropic_api_key=api_key,
            # claude-sonnet-5 rejects `temperature` outright (400: "temperature
            # is deprecated for this model") — sampling params were removed
            # for this model family, so there's no equivalent knob to set.
            # Without this, a stalled response hangs the request
            # indefinitely — there's no default timeout.
            timeout=_LLM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise ChatAnthropic: {exc}") from exc


def _extract_text(content) -> str:
    """Flatten a LangChain message .content into plain text.

    Claude (and Gemma before it) returns a list of blocks such as
    [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]
    rather than a plain string.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
            elif block.get("type") == "thinking" and block.get("thinking"):
                parts.append(block["thinking"])
        return "\n".join(p.strip() for p in parts if p).strip()
    return ""


_MAX_RATE_LIMIT_RETRIES = 6
_RATE_LIMIT_BACKOFF_SECONDS = 60  # matched a per-minute quota window under Gemini's free tier


def _invoke_with_rate_limit_retry(llm_with_tools, messages):
    """Retry on 429s with a fixed backoff.

    Originally sized for Gemini's free-tier per-minute input-token quota,
    which this agent hit reliably once the conversation grew past a few
    tool round-trips — the google-genai client's own internal backoff
    topped out well under 60s and gave up before the quota actually reset.
    Claude's paid tier doesn't have the same per-minute ceiling, so this
    should rarely trigger now, but it's left in as a harmless defensive
    fallback for any transient 429.
    """
    last_exc = None
    for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return llm_with_tools.invoke(messages)
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "The AI service timed out — please try again."
            ) from exc
        except Exception as exc:
            if "timeout" in str(exc).lower():
                raise RuntimeError(
                    "The AI service timed out — please try again."
                ) from exc
            if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                raise
            last_exc = exc
            logger.warning(
                "LLM call rate-limited (attempt %d/%d); backing off %ds",
                attempt, _MAX_RATE_LIMIT_RETRIES, _RATE_LIMIT_BACKOFF_SECONDS,
            )
            time.sleep(_RATE_LIMIT_BACKOFF_SECONDS)
    raise last_exc


def _agent_node(state: QCState) -> dict:
    llm = _load_llm().bind_tools(TOOLS)
    response = _invoke_with_rate_limit_retry(llm, state["messages"])
    return {"messages": [response]}


def _tools_node(state: QCState) -> dict:
    last: AIMessage = state["messages"][-1]
    reasoning = _extract_text(last.content) or None

    tool_messages = []
    log_entries = []
    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        tool = _TOOLS_BY_NAME.get(name)
        try:
            if tool is None:
                raise KeyError(f"Unknown tool requested by the model: {name!r}")
            output = tool.invoke(args)
        except Exception as exc:
            output = f"Error invoking {name}: {exc}"

        logger.info("Tool call: %s(%s) -> %s", name, args, output)
        log_entries.append(
            {"tool": name, "input": args, "output": output, "reasoning": reasoning}
        )
        tool_messages.append(
            ToolMessage(content=str(output), tool_call_id=call["id"])
        )
        progress = _progress_var.get()
        if progress is not None:
            progress["last_tool_call"] = {
                "tool": name, "input": args, "output": str(output)[:300]
            }

    return {"messages": tool_messages, "tool_log": log_entries}


def _route_after_agent(state: QCState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def _build_graph():
    graph = StateGraph(QCState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def _parse_final_ra(report: str, tool_log: list) -> float | None:
    match = _FINAL_RA_RE.search(report)
    if match:
        return float(match.group(1))

    logger.warning("No 'Final Ra Estimate' line found in report; falling back to tool outputs.")
    values = []
    for entry in tool_log:
        if entry["tool"] in _PREDICTION_TOOL_NAMES:
            m = _TOOL_RA_RE.search(str(entry["output"]))
            if m:
                values.append(float(m.group(1)))
    if not values:
        return None
    return sum(values) / len(values)


def run_qc_pipeline(features: dict, progress: dict | None = None) -> dict:
    """Run the QC agent end-to-end on one part's 36 extracted features.

    Args:
        features: the 36 extracted vibration features for one part.
        progress: optional mutable dict a caller running this on another
            thread can poll for {"last_tool_call": {...}} while waiting —
            useful for reporting how far the agent got if an external
            wall-clock deadline fires before this function returns.

    Returns:
        ra            (float | None) — the agent's final trusted Ra (µm)
        models_used   (list[str])    — prediction tool names the agent called
        tool_calls    (list[dict])   — {tool, input, output, reasoning} per call,
                                        in the order they actually happened
        noise_level   (str | None)   — raw get_signal_noise_level output, if called
        report        (str)          — the agent's final free-text QC report
        rag_consulted (bool)         — whether query_rag was called
    """
    import json

    if progress is not None:
        _progress_var.set(progress)

    human_content = (
        "Run a QC assessment on this part using its 36 extracted vibration "
        f"features:\n{json.dumps(features, indent=2)}"
    )
    initial_state: QCState = {
        "messages": [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ],
        "tool_log": [],
    }

    final_state = _get_graph().invoke(initial_state, config={"recursion_limit": 25})

    tool_log = final_state["tool_log"]
    models_used = [e["tool"] for e in tool_log if e["tool"] in _PREDICTION_TOOL_NAMES]
    rag_consulted = any(e["tool"] == "query_rag" for e in tool_log)
    noise_level = next(
        (e["output"] for e in tool_log if e["tool"] == "get_signal_noise_level"), None
    )

    report = _extract_text(final_state["messages"][-1].content)
    ra = _parse_final_ra(report, tool_log)

    return {
        "ra": ra,
        "models_used": models_used,
        "tool_calls": tool_log,
        "noise_level": noise_level,
        "report": report,
        "rag_consulted": rag_consulted,
    }
