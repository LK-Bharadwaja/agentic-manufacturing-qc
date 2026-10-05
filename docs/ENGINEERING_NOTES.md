# Engineering Notes

**How determinism actually works now.** Claude Sonnet 5 has no `temperature` parameter at all — sending one returns a 400 (`temperature is deprecated for this model`), so unlike the project's original Gemma-based agent, there's no sampling knob to pin to 0. The three prediction models' own numeric outputs are still exactly reproducible (they're plain deterministic ML inference, untouched by the LLM), but early testing showed the *agent's final synthesis* — specifically, whether it excluded/downweighted a diverging model or fell back to an equal three-way average — could vary between runs on identical input (e.g. 2.3776 µm vs. 2.4516 µm for the same part). Rather than relying on a sampling parameter that no longer exists, the system prompt (`agent/qc_agent.py`) now states an explicit, three-step tie-breaking rule for the final estimate: (1) exclude any model flagged by `check_training_range` as an extrapolation risk and average the rest; (2) if all three are in-range but one diverges from the other two by more than 0.15 µm, weight toward the average of the two closer models instead of an outlier-including three-way average; (3) only average all three equally when none qualifies as an outlier under (1) or (2). This was verified with 5 back-to-back runs on the same part (part 28) after adding the rule — all 5 produced the identical `Final Ra Estimate: 2.3776 µm`, with the report explicitly citing the rule each time. The LLM's free-text reasoning and report wording can still vary slightly run to run — a hosted API isn't guaranteed bit-for-bit deterministic even with an explicit rule — but the numeric outcome the rule governs is now stable in practice.

**Historical issue: LLM latency and occasional hangs (resolved by the Claude switch, machinery kept as a safety net).** During development, the agent ran on Gemini's free tier (16,000 input tokens/minute), so a multi-tool-call run routinely got rate-limited and had to back off — a full run could take anywhere from ~30 seconds to several minutes, and consecutive timeouts on the same part weren't unusual. Separately, the underlying LLM call was occasionally observed to hang well past its own configured timeout (a slow trickle of bytes on the response can outlast a per-read `httpx` timeout without ever triggering it). This motivated `/agent/predict` running the whole agent under a **hard 90-second wall-clock deadline**, independent of and on top of the per-call timeout, on its own thread pool; if the deadline fires, the API returns a clean `504` rather than hanging the request, and writes a diagnostic dump (the last tool call completed, plus a full stack trace of the stuck thread) to `logs/agent_hang_*.log`. Python has no safe way to kill a running thread, so a timed-out run keeps executing in the background rather than being cancelled — harmless, but it did mean a burst of back-to-back timeouts could compound the same rate limit for a little while afterward. **Since switching to Claude, this has not recurred**: repeated real `/agent/predict` runs (single runs and back-to-back batches of 3-5) have logged zero rate-limit retries, and typical run time is a fairly consistent ~50-90 seconds. The deadline/hang-detection/logging machinery stays in place regardless — it's a general safety net against any LLM provider stalling mid-response, not a Gemini-specific workaround, so there's no reason to remove it.

## How the agent and RAG layers were verified

These are results observed during development, reproducible with `agent/test_manual.py` and `rag/test_manual.py`. They are not stored logs.

### Agent

- **Tool coverage (part 28).** Every run called `get_signal_noise_level`, the three prediction tools, `check_training_range` three times, and `query_rag`. The underlying predictions were identical every run (MLR 2.4041, Fuzzy 2.3511, CNN 2.5995 µm).
- **Self-correction (part 41).** The agent's first `predict_mlr` call used wrong feature key names and got a 400. Its next step diagnosed the key mismatch and retried correctly. Nothing in the code handles that specific error.
- **Extrapolation guard (part 41).** The CNN predicted 4.7493 µm, far outside the 1.931 to 2.812 µm training range. Before the guard existed, the agent trusted it because of the CNN's higher training R². After adding `check_training_range` and the R²-provenance wording to the system prompt, it flagged the CNN as an extrapolation artifact and used the MLR/Fuzzy consensus (2.3663 µm).
- **Tie-breaking rule.** Before the rule, the final Ra on part 28 varied between 2.3776 and 2.4516 µm across runs. After it, 5 of 5 runs gave 2.3776 µm.

### RAG

- **Distances.** Relevant questions scored 0.17 to 0.32 cosine distance; an off-topic question ("capital of France") scored 0.46 or more. The threshold is 0.35, so off-topic questions never reach the LLM.
- **Corpus fixes.** "What was the MLR R²?" originally returned the stale 91.94% figure from the reports; adding `docs/reports/Metrics_Clarification.md` fixed that to 0.4644. "Why 81 fuzzy rules?" originally answered "I don't know" until `Fuzzy_Rules_Explanation.md` was added.

### Limits

- This was hand-checked on a small number of parts.
- There is no automated eval harness for agent decisions.
- Free-text reasoning varies between runs, even though the numeric outcome the tie-breaking rule governs is stable.
