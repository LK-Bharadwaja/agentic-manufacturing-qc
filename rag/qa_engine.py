"""RAG Q&A engine: retrieve relevant chunks, then answer with Claude.

Distance threshold reasoning
-----------------------------
gemini-embedding-001 cosine distances observed in this project:
  - Clearly on-topic query ("CNN model architecture"): best distance 0.18,
    all top-3 results in the 0.18-0.20 band.
  - Threshold chosen: 0.35
    * Leaves ~0.15 of headroom above the "good" band — absorbs slight wording
      variation without misclassifying real questions.
    * Completely off-topic queries (e.g. general trivia with no overlap with
      CNC/ML concepts) are expected to land at 0.40+ because the embedding
      space for general knowledge is far from the project's domain cluster.
    * 0.35 is conservative enough to keep false negatives low while reliably
      catching out-of-domain questions before wasting an LLM call.
"""

import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rag.vector_store import query as _retrieve

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DISTANCE_THRESHOLD = 0.35
_LLM_TIMEOUT_SECONDS = 45.0

_SYSTEM_PROMPT = (
    "You are a technical assistant for a university Capstone project on CNC surface "
    "roughness prediction using machine learning.\n"
    "Answer the user's question using ONLY the document excerpts provided below.\n"
    "If the excerpts do not contain enough information to answer the question, "
    'respond with exactly: "I don\'t know"\n'
    "Do not use any outside knowledge. Be concise and precise."
)


def _load_llm():
    """Return a ChatAnthropic instance using ANTHROPIC_API_KEY from .env.

    Note: embeddings for the vector store stay on Gemini
    (GoogleGenerativeAIEmbeddings, see rag/vector_store.py) — only the
    generation/reasoning LLM here is Claude.
    """
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
            model="claude-sonnet-5",
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


def answer_question(question: str) -> dict:
    """Retrieve top-3 chunks and answer using Claude, grounded in the docs.

    Returns:
        answer   (str)       — Claude's answer, or the "not found" fallback
        sources  (list[str]) — unique source filenames used (empty if not grounded)
        grounded (bool)      — False when best chunk distance exceeds threshold
    """
    hits = _retrieve(question, top_k=3)

    if not hits:
        return {
            "answer": "I couldn't find relevant information in the project documentation.",
            "sources": [],
            "grounded": False,
        }

    best_distance = hits[0]["distance"]
    logger.info(
        "Top-3 distances for %r: %s", question, [h["distance"] for h in hits]
    )

    if best_distance > _DISTANCE_THRESHOLD:
        logger.info(
            "Best distance %.4f > threshold %.2f — returning not-grounded",
            best_distance,
            _DISTANCE_THRESHOLD,
        )
        return {
            "answer": "I couldn't find relevant information in the project documentation.",
            "sources": [],
            "grounded": False,
        }

    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[Excerpt {i} — {hit['source_filename']} chunk {hit['chunk_index']}]\n"
            f"{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _load_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Document excerpts:\n\n{context}\n\nQuestion: {question}"),
        ])
    except httpx.TimeoutException as exc:
        raise RuntimeError("The AI service timed out — please try again.") from exc
    except Exception as exc:
        if "timeout" in str(exc).lower():
            raise RuntimeError("The AI service timed out — please try again.") from exc
        raise
    content = response.content
    if isinstance(content, list):
        # Claude returns [{type: "thinking", ...}, {type: "text", ...}]
        text_parts = [c["text"] for c in content if isinstance(c, dict) and c.get("type") == "text"]
        answer = " ".join(text_parts).strip()
    else:
        answer = content.strip()
    sources = list(dict.fromkeys(h["source_filename"] for h in hits))

    return {"answer": answer, "sources": sources, "grounded": True}
