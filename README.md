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

A heuristic rule-based classifier is implemented as the initial baseline.

The classifier analyzes:

- User task
- Tool call errors
- Failure explanations
- Available tools

and predicts one of the defined failure categories.

The baseline serves as a reference point for future LLM-based approaches.

## Evaluation Results

Dataset Size: 20 traces

Overall Accuracy: 90%

| Metric | Value |
|----------|----------|
| Accuracy | 90% |
| Macro F1 | 0.89 |
| Weighted F1 | 0.89 |

The baseline performs well on the current manually curated dataset and establishes a benchmark for future experiments.

Re-run `python -m src.evaluate` after collecting MCP traces to evaluate on the combined dataset.
