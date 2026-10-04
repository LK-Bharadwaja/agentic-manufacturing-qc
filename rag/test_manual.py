"""Manual integration test for the RAG Q&A pipeline.

Runs three questions and prints real distances + answers.
Expected outcomes:
  Q1 "Why 81 fuzzy rules?"    — grounded: True  (fuzzy logic design is in the reports)
  Q2 "What was the MLR R2 score?" — grounded: True  (metrics documented in reports)
  Q3 "What is the capital of France?" — grounded: False (out of domain, distance > 0.35)
"""

import logging
import sys

# Claude's answer text can include non-ASCII characters (e.g. "µm", "R²")
# that the default Windows console encoding (cp1252) can't print — force
# UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from rag.qa_engine import _DISTANCE_THRESHOLD, answer_question  # noqa: E402
from rag.vector_store import query as retrieve  # noqa: E402

QUESTIONS = [
    "Why 81 fuzzy rules?",
    "What was the MLR R2 score?",
    "What is the capital of France?",
    "What is the difference between the 91.94% accuracy figure and the R2 score for MLR?",
]

for question in QUESTIONS:
    print(f"\n{'=' * 65}")
    print(f"Q: {question}")

    hits = retrieve(question, top_k=3)
    distances = [h["distance"] for h in hits]
    print(f"Raw distances : {distances}")
    print(f"Threshold     : {_DISTANCE_THRESHOLD}  (best={'PASS' if distances[0] <= _DISTANCE_THRESHOLD else 'FAIL — not grounded'})")

    result = answer_question(question)
    print(f"Grounded      : {result['grounded']}")
    print(f"Sources       : {result['sources']}")
    print(f"Answer        :\n  {result['answer']}")
