# Research project

## Capability Gap Detection in Agentic Workflows

## Project Overview
This project investigates how execution traces from tool-using AI agents can be analyzed to identify capability gaps.

A capability gap occurs when an agent fails because the required functionality is unavailable in the current toolset. Detecting such gaps is an important step toward automated tool synthesis and adaptive agent systems.

The project is inspired by AgentRx, which diagnoses failures from execution traces, but extends the idea toward identifying missing capabilities and generating structured capability specifications.

## Research Goal

Given:

- A user task
- Available tools
- Agent execution trace

Determine:

1. Why the agent failed
2. Whether the failure is caused by a capability gap
3. What missing capability would be required to complete the task

## Failure Taxonomy

| Label | Description |
|---------|---------|
| F0 | Success (No Failure) |
| F1 | Reasoning or Planning Error |
| F2 | Wrong Tool Selected |
| F3 | Wrong Tool Parameters |
| F4 | Tool Runtime Error |
| F5 | Tool Documentation or Schema Error |
| F6 | Missing Capability Gap |
| F7 | Insufficient User Information |
| F8 | Environment or State Error |

## Dataset

The project uses two trace sources:

1. **Manual traces** (`data/traces.jsonl`) — 20 hand-constructed execution traces covering all failure categories.
2. **MCP traces** (`data/mcp_traces.jsonl`) — traces collected by running scenarios against real MCP servers.

Each trace contains:

- User task
- Available tools (from `list_tools()` when collected via MCP)
- Agent plan
- Tool calls
- Observations and errors
- Final response
- Gold failure label
- Failure explanation
- MCP metadata (`mcp_servers`, `tool_schemas`) for MCP-generated traces

## MCP Integration

Tools are exposed through [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers instead of a static JSON registry.

### MCP server

All tools live in one MCP server:

| Server | Path | Tools |
|--------|------|-------|
| `research_tools` | `mcp_servers/research_tools/server.py` | `read_file`, `calculator`, `csv_reader`, `sql_query`, `summarizer`, `weather_api`, `send_email`, and others (17 total) |

Capability gaps (F6) come from **missing tool types** in the server (e.g. no OCR or PDF extractor), not from connecting multiple servers.

### Collect MCP traces

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.trace_collector
```

Scenarios are defined in `data/mcp_scenarios.json` (12 scenarios covering all failure categories F0–F8). Each scenario specifies:

- which MCP servers to connect
- a scripted sequence of tool calls (simulating agent behavior)
- the expected gold failure label

The collector connects to MCP servers over stdio, calls tools for real, and writes traces to `data/mcp_traces.jsonl`.

### Run classification and evaluation

```bash
python -m src.main
python -m src.evaluate
```

By default, both commands use manual and MCP traces.

### Use MCP servers in Cursor

Add this to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "research_tools": {
      "command": "python",
      "args": ["mcp_servers/research_tools/server.py"]
    }
  }
}
```

You can then run tasks in Cursor with real MCP tools and export the resulting tool-call history into the trace format.

## Current Pipeline

User Task
    ↓
MCP Servers (tools) + Agent Execution
    ↓
Trace Collector → `data/mcp_traces.jsonl`
    ↓
Heuristic Classifier
    ↓
Failure Category

## Baseline Method

Two rule-based baselines are implemented in `src/heuristic_classifier.py`:

| Method | CLI name | Uses `failure_explanation`? | Role |
|--------|----------|-------------------------------|------|
| **Heuristic oracle** | `heuristic-oracle` | Yes | Upper bound (uses human-written explanations) |
| **Heuristic fair** | `heuristic-fair` | No | Honest deployable baseline |

An **LLM baseline** is available in `src/llm_classifier.py` via the [TUD:AI API](https://llm.scads.ai/docs/usage/api/) (`llm-fair`, `llm-oracle`).

## Evaluation

Evaluation lives in `src/evaluate.py`. It compares predicted labels against human `gold_label` values.

### What gets measured

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Exact match on F0–F8 |
| **Macro / weighted F1** | Multi-class performance |
| **F6 precision/recall/F1** | Capability-gap class only |
| **Binary gap detection F1** | F6 vs all other failures |

### Splits (train/test)

| Split | Train | Test | Use when |
|-------|-------|------|----------|
| `all` | all | all | Quick sanity check (not generalization) |
| `holdout-mcp` | 20 synthetic | 12 MCP | **Recommended** — rules trained on hand-written, tested on real MCP |
| `holdout-synthetic` | 12 MCP | 20 synthetic | Reverse generalization check |
| `random` | 75% | 25% | Stratified random split |
| `loo` | n−1 | 1 | Leave-one-out CV (mean metrics) |
| `cv5` | 80% | 20% | 5-fold stratified CV |

```bash
# Recommended: test on MCP traces, train/dev on synthetic
python -m src.evaluate --split holdout-mcp

# Compare fair vs oracle heuristics
python -m src.evaluate --split holdout-mcp --method heuristic-oracle heuristic-fair

# 5-fold cross-validation
python -m src.evaluate --split cv5 --method heuristic-fair
```

Results are written to `outputs/evaluation/summary_<split>.json`.

### LLM baseline (TUD:AI)

```bash
export SCADS_API_KEY="your-tud-ai-key"   # from https://llm.scads.ai/docs/
export SCADS_MODEL="alias-ha"            # optional; see https://llm.scads.ai/status/

python -m src.evaluate --split holdout-mcp --method llm-fair
```

Uses OpenAI-compatible API at `https://llm.scads.ai/v1`. Pick a model with **Tools? ✅** on the [status page](https://llm.scads.ai/status/) for future agent runs; classification only needs chat.

### External benchmark: MCP-Atlas

[MCP-Atlas](https://huggingface.co/datasets/ScaleAI/MCP-Atlas) provides `PROMPT`, `ENABLED_TOOLS`, and `TRAJECTORY` (required tools). We derive synthetic F6 cases by withholding one required tool from the enabled set.

```bash
pip install datasets pyarrow openai
python -m src.evaluate --benchmark mcp-atlas --atlas-limit 50 --method heuristic-fair
```

First run downloads a sample to `data/benchmarks/mcp_atlas_sample.jsonl`. MCPMark is a separate agent benchmark (task success); use it later for full agent runs, not failure taxonomy labels.

### Baseline benchmark: AgentRx (τ-bench)

[AgentRx](https://github.com/microsoft/AgentRx) is the baseline paper: it diagnoses *why* an agent failed from its trajectory, using a 10-category taxonomy on real **τ-bench** retail traces. The adapter (`src/evaluation/benchmarks/agentrx.py`) loads AgentRx trajectories, maps their taxonomy to our F0–F8, and lets us run our classifiers on the same data.

```bash
# Public GitHub sample trajectories (ungated, label from filename)
python -m src.evaluate --benchmark agentrx --agentrx-source samples --agentrx-only \
  --method heuristic-fair --split all

# Full gated benchmark (115 failures) — requires HF auth first:
#   1. Accept terms at https://huggingface.co/datasets/microsoft/AgentRx
#   2. huggingface-cli login   (or export HF_TOKEN=...)
python -m src.evaluate --benchmark agentrx --agentrx-source hf --agentrx-only \
  --method llm-fair --split all
```

Results are written to `outputs/evaluation/summary_<split>_agentrx.json`.

**Taxonomy mapping (AgentRx → ours).** The mapping is intentionally lossy and is itself a finding — AgentRx splits reasoning failures more finely than we do, and we have classes (F0, F5) it lacks.

| AgentRx category | Our label |
|------------------|-----------|
| Intent Not Supported | **F6 missing capability gap** |
| Invalid Invocation | F3 wrong tool parameters |
| Underspecified User Intent | F7 insufficient user information |
| System Failure | F4 tool runtime error |
| Guardrails Triggered | F8 environment or state error |
| Instruction/Plan Adherence, Invention of New Info, Misinterpretation of Tool Output, Intent-Plan Misalignment, Inconclusive | F1 reasoning or planning error |

**Key result (full gated set, 29 τ-retail failures).** Both baselines collapse on real trajectories — in different ways:

| Method | Accuracy | F6 precision | F6 recall | F6 F1 |
|--------|----------|--------------|-----------|-------|
| heuristic-fair | **0.207** | 0.333 | 0.500 | 0.400 |
| llm-fair (`alias-ha`, small) | 0.069 | 0.118 | 1.000 | 0.211 |
| llm-fair (`alias-huge-no-thinking`) | **0.207** | 0.143 | 0.500 | 0.222 |

`alias-huge` (with thinking) hangs on the long real trajectories — use `alias-huge-no-thinking` for classification.

**Why the heuristic fails (from per-trace inspection).** The dominant error is not misfiring rules — it is **blindness to semantic failures**. On 20 of 29 real traces the heuristic predicts **F0 (success)**, because every tool call returned without an "Error:" string. Real τ-bench failures are semantic (didn't authenticate, premature transfer, wrong reasoning), not tool crashes, so an error-string-based classifier sees a clean run and declares success:

| Gold → Predicted | Count | Why |
|------------------|-------|-----|
| F1 → F0 | 11 | reasoning errors, tools ran fine |
| F7 → F0 | 7 | underspecified intent, tools ran fine |
| F6 → F0 | 1 | policy gap, tools ran fine |
| F4 → F0 | 1 | — |

The one "correct" F6 (`tau_retail_105`) is a **false positive that happened to be right**: the rule matched the word *email* in the customer's email address and concluded "no email tool → F6". The same bogus rule misfired on an F1 and an F7 trace. The genuine F6 (`tau_retail_47`, premature transfer to human) was labeled F0.

**Heuristic vs strong LLM — same score (21%), opposite failures.** The strong LLM ties the heuristic on accuracy but for completely different reasons:

- **Heuristic**: blind — labels 20/29 as F0 (success) because no tool threw an error.
- **Strong LLM**: actually reads the trajectory (its evidence is specific and often correct, e.g. *"agent only modified the keyboard and never changed the shoes"*), but **disagrees with the gold taxonomy** — especially on F7 (Underspecified User Intent), which it reads as F1 (reasoning) or F6 (capability gap).

**The deeper finding: much of the "error" is taxonomy disagreement, not model failure.** Example: `tau_retail_91` is gold **F7**, but the LLM labels it **F6** with a sound justification — *"no return/exchange/refund tools exist in available_tools."* That is a defensible capability-gap call. This means (a) F6 detection by the LLM is **semantically grounded** (unlike the heuristic's accidental email-string match), and (b) fair comparison requires either reconciling the two taxonomies or evaluating on the cleaner **binary capability-gap task**. Reconciling F0–F8 with AgentRx's 10 categories is itself a thesis contribution.

**Takeaways for the thesis:**
1. Error-string heuristics cannot diagnose semantic/policy failures — they need to reason over the trajectory and policy.
2. A strong LLM reasons well but is bottlenecked by taxonomy alignment, motivating an F6-focused, binary-first evaluation.
3. Use `--save-predictions` to regenerate the per-trace comparisons in `outputs/evaluation/predictions_*.jsonl`.

## Evaluation Results

Re-run after collecting MCP traces:

```bash
python -m src.evaluate --split all --method heuristic-oracle heuristic-fair
```

On the combined 32-trace dataset, heuristic-oracle reaches ~97% accuracy. Use `holdout-mcp` for a more honest generalization estimate.
