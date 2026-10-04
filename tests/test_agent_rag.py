"""Agent / RAG tests. No real Claude or Gemini calls: LLM entry points are mocked
and key-dependent code is exercised with the keys removed."""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agent import qc_agent
from agent.tools import check_training_range
from modules.config import TRAINING_RA_MAX, TRAINING_RA_MIN
from rag import qa_engine


@pytest.fixture
def client():
    with TestClient(api_main.app) as c:
        yield c


def test_check_training_range_inside():
    out = check_training_range.invoke({"ra": (TRAINING_RA_MIN + TRAINING_RA_MAX) / 2})
    assert "within the known training range" in out
    assert not out.startswith("WARNING")


@pytest.mark.parametrize("ra", [TRAINING_RA_MIN - 0.5, TRAINING_RA_MAX + 0.5])
def test_check_training_range_flags_extrapolation(ra):
    assert check_training_range.invoke({"ra": ra}).startswith("WARNING")


def test_rag_ask_returns_mocked_answer(client, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "answer_question",
        lambda q: {"answer": "27 parts", "sources": ["report.pdf"], "grounded": True},
    )
    r = client.post("/rag/ask", json={"question": "Training set size?"})
    assert r.status_code == 200
    assert r.json()["grounded"] is True


def test_rag_ask_missing_key_is_502(client, monkeypatch):
    def boom(_q):
        raise OSError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(api_main, "answer_question", boom)
    r = client.post("/rag/ask", json={"question": "anything"})
    assert r.status_code == 502
    assert r.json()["error"] == "rag_unavailable"


def test_llm_loaders_raise_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(qa_engine, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(qc_agent, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
        qa_engine._load_llm()
    with pytest.raises(OSError, match="ANTHROPIC_API_KEY"):
        qc_agent._load_llm()
