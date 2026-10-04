"""Manual, human-inspected run of the QC agent on two real parts.

Not a pytest suite — this is meant to be run directly and read: it prints
the full tool_calls trace so a person can check whether the agent actually
reasoned about each part (different tool choices/order/counts driven by the
input) or just replayed a fixed sequence regardless of what it saw.

Case 1: part 28 — the three models roughly agree (spread ~0.25 µm).
Case 2: part 41 — the models disagree sharply (CNN=4.75 vs MLR/Fuzzy~2.36,
spread ~2.4 µm), found by scanning the test parts 28-43 in api/service.py's
run_all() for the largest MLR/Fuzzy/CNN spread.
"""

import json
import logging
import sys

from agent.qc_agent import run_qc_pipeline
from modules.feature_extraction import extract_features_from_file

# Claude's report text can include non-ASCII characters (e.g. "≈") that the
# default Windows console encoding (cp1252) can't print — force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

PARTS_DIR = r"C:\Capstone Project\data\parts"

CASES = {
    "28": ("part 28 — models roughly agree (spread ~0.25 um)", "part 28.xlsx"),
    "41": ("part 41 — models disagree sharply (spread ~2.4 um)", "part 41.xlsx"),
}


def _run_case(label: str, part_file: str) -> None:
    print("\n" + "=" * 88)
    print(f"CASE: {label}  ({part_file})")
    print("=" * 88)

    features_df = extract_features_from_file(f"{PARTS_DIR}\\{part_file}")
    features = features_df.iloc[0].to_dict()

    result = run_qc_pipeline(features)

    print(f"\n--- tool_calls trace ({len(result['tool_calls'])} calls) ---")
    for i, call in enumerate(result["tool_calls"], 1):
        print(f"\n[{i}] tool: {call['tool']}")
        print(f"    input: {json.dumps(call['input'])[:200]}")
        print(f"    output: {call['output']}")
        print(f"    reasoning: {call['reasoning']}")

    print("\n--- summary ---")
    print(f"ra:            {result['ra']}")
    print(f"models_used:   {result['models_used']}")
    print(f"noise_level:   {result['noise_level']}")
    print(f"rag_consulted: {result['rag_consulted']}")
    print("\n--- final report ---")
    print(result["report"])


def main():
    # Optional: `python -m agent.test_manual 41` runs only that case.
    keys = sys.argv[1:] or list(CASES)
    for key in keys:
        _run_case(*CASES[key])


if __name__ == "__main__":
    main()
