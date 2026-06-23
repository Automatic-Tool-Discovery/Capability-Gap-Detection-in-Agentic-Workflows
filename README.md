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

### MCP servers

| Server | Path | Tools |
|--------|------|-------|
| `research_tools` | `mcp_servers/research_tools/server.py` | `calculator`, `read_file`, `csv_reader`, `text_search`, `sql_query`, `web_search`, `run_python` |
| `extended_tools` | `mcp_servers/extended_tools/server.py` | `summarizer`, `translate_text`, `weather_api`, `currency_converter`, `send_email`, and others |

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
    },
    "extended_tools": {
      "command": "python",
      "args": ["mcp_servers/extended_tools/server.py"]
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

## Evaluation Results

Re-run after collecting MCP traces:

```bash
python -m src.evaluate --split all --method heuristic-oracle heuristic-fair
```

On the combined 32-trace dataset, heuristic-oracle reaches ~97% accuracy. Use `holdout-mcp` for a more honest generalization estimate.
