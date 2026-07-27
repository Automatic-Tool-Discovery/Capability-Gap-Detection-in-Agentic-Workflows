# Paper Plan: Capability Gap Detection in Agentic Workflows

## Step 1: Build the capability-gap dataset
Reuse AgentRx + MCP-Atlas traces, relabeled into the F0–F8 taxonomy (F6 = missing capability).
**Status: done.**

## Step 2: Find the best detection approach
Compare candidate methods for spotting F6 (missing capability) failures:
- Pure symbolic/rule-based matching — tried, removed (misses semantic gaps)
- Pure LLM-as-judge (baseline, mirrors AgentRx)
- Hybrid: LLM extracts required capabilities → deterministic set comparison against available tools
**Status: hybrid approach implemented (`capability_matcher.py`). On the six labeled
live F6 traces, the fair hybrid scored gap-detection F1 = 1.000 versus 0.909 for
the pure-LLM baseline (July 28 run; see `outputs/evaluation/summary_all.json`).**

## Step 3: Evaluate the detection approach
Score all methods with the same metrics (accuracy, F1, binary gap-detection F1) under fair conditions (no answer-key hints).
**Status: all-split live result recorded.** Broader random/CV results still require
a dataset containing enough labeled non-F6 examples; the current live controls
are intentionally unlabeled and therefore excluded from scoring.

## Step 4: Produce a capability request
On a detected gap, output a structured spec of the missing tool (name, inputs, outputs, rationale) — not just a label.
**Status: implemented, but only logged, not scored.**

## Step 5: Evaluate capability request quality
Check whether the generated capability spec is actually correct and usable (not just whether F6 was detected).
**Status: implemented.** For traces with benchmark-known withheld capabilities,
report capability precision/recall/F1, exact set match, request coverage, and
schema completeness. Natural-language quality and executable-tool validation
remain limitations requiring human review or a tool-synthesis experiment.

## Step 6: Build a demo system
A small interactive tool with two ways in, feeding the same classify → evaluate pipeline:

**6a. Live mode (end-to-end)**
User asks a question, with some tools available → `src/live_agent.py` runs the agent
live and produces a trace → pipeline classifies it (F0–F8) → if F6, shows the generated
capability request.
- Proves the whole system works, not just the classifier
- Good for paper demo / screenshots
- Reuses the existing live-agent generator; needs a single-question entry point
  instead of a batch task file (`data/live_tasks.json`)

**6b. Trace-replay mode (debugging)**
User picks or pastes an existing trace (with a known gold label) → pipeline classifies it
→ shows predicted label + capability request next to the gold answer.
- No LLM call needed to regenerate an agent run — faster, deterministic
- Lets you replay a specific failing case to debug *why* it was misclassified
- Needed for Step 5: comparing generated capability requests against known-good ones

**Status: implemented in `src/demo.py`.** It accepts either one recorded trace
(`--trace`, optionally `--trace-id`) or one live question (`--question`, optionally
`--available-tools`) and displays the prediction and capability request.
