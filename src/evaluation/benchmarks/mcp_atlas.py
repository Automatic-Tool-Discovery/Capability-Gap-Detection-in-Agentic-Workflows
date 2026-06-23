"""MCP-Atlas adapter for external capability-gap evaluation.

MCP-Atlas (https://huggingface.co/datasets/ScaleAI/MCP-Atlas) exposes, per task:
- PROMPT: user task
- ENABLED_TOOLS: tools visible to the agent
- TRAJECTORY: minimal required tool-call sequence

We derive synthetic F6 cases by withholding one required tool from ENABLED_TOOLS and
asking whether a classifier detects a missing capability.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from src.schemas import AgentTrace, ToolCall
from src.taxonomy import FailureType

ATLAS_CACHE = Path("data/benchmarks/mcp_atlas_sample.jsonl")
F6_LABEL = FailureType.MISSING_CAPABILITY_GAP.value


def _parse_tool_list(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in re.split(r"[,\n]+", raw) if part.strip()]


def _parse_trajectory_tools(raw: str) -> list[str]:
    tools: list[str] = []
    raw = raw.strip()
    if not raw:
        return tools

    # JSON list of step dicts
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for step in parsed:
                if isinstance(step, dict):
                    name = step.get("tool") or step.get("tool_name") or step.get("name")
                    if name:
                        tools.append(str(name))
            if tools:
                return list(dict.fromkeys(tools))
    except json.JSONDecodeError:
        pass

    # Fallback: extract tool-like tokens from serialized trajectory text
    for match in re.finditer(r"'tool(?:_name)?':\s*'([^']+)'", raw):
        tools.append(match.group(1))
    for match in re.finditer(r'"tool(?:_name)?":\s*"([^"]+)"', raw):
        tools.append(match.group(1))
    return list(dict.fromkeys(tools))


def row_to_gap_traces(row: dict[str, Any], *, max_per_task: int = 2) -> list[AgentTrace]:
    prompt = str(row.get("PROMPT") or row.get("prompt") or "")
    enabled = _parse_tool_list(str(row.get("ENABLED_TOOLS") or row.get("enabled_tools") or ""))
    required = _parse_trajectory_tools(str(row.get("TRAJECTORY") or row.get("trajectory") or ""))
    task_id = str(row.get("TASK") or row.get("task") or row.get("id") or "atlas")

    if not prompt or not enabled or not required:
        return []

    traces: list[AgentTrace] = []
    withheld = [tool for tool in required if tool in enabled][:max_per_task]
    for index, missing_tool in enumerate(withheld):
        available = [tool for tool in enabled if tool != missing_tool]
        traces.append(
            AgentTrace(
                trace_id=f"atlas_{task_id}_{index}",
                user_task=prompt,
                available_tools=available,
                agent_plan="Complete the task using the enabled MCP tools.",
                tool_calls=[
                    ToolCall(
                        tool_name=available[0] if available else "unknown_tool",
                        arguments={},
                        observation=None,
                        error=(
                            f"Tool '{missing_tool}' is required for this task but is not "
                            "available in the enabled tool set."
                        ),
                    )
                ],
                final_response="I could not complete the task with the available tools.",
                gold_label=F6_LABEL,
                failure_explanation=(
                    f"Required tool '{missing_tool}' from the reference trajectory is missing "
                    "from ENABLED_TOOLS."
                ),
                mcp_servers=["mcp-atlas"],
                tool_schemas={},
            )
        )
    return traces


def load_atlas_rows(
    *,
    limit: int = 50,
    cache_path: Path = ATLAS_CACHE,
) -> list[dict[str, Any]]:
    if cache_path.exists():
        rows = []
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows[:limit]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install datasets to download MCP-Atlas: pip install datasets pyarrow"
        ) from exc

    dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
    rows = [dict(row) for row in dataset.select(range(min(limit, len(dataset))))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def build_atlas_gap_dataset(*, limit: int = 50) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    for row in load_atlas_rows(limit=limit):
        traces.extend(row_to_gap_traces(row))
    return traces
