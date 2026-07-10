"""Build a small MCP-Atlas tool-removal plan for capability-gap experiments.

This does not run the MCP-Atlas harness. It prepares candidate ablation records
from cached MCP-Atlas rows so the actual runs can be executed with Omar's
handoff/runbook or the official MCP-Atlas scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmarks.mcp_atlas import (  # noqa: E402
    _parse_tool_list,
    _parse_trajectory_tools,
)

DEFAULT_INPUT = Path("data/benchmarks/mcp_atlas_sample.jsonl")
DEFAULT_OUTPUT = Path("outputs/mcp_atlas_ablation_plan.jsonl")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_plan(
    rows: list[dict[str, Any]],
    *,
    max_tasks: int,
    max_tools_per_task: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row.get("TASK") or row.get("task") or row.get("id") or "")
        prompt = str(row.get("PROMPT") or row.get("prompt") or "")
        enabled_tools = _parse_tool_list(
            str(row.get("ENABLED_TOOLS") or row.get("enabled_tools") or "")
        )
        required_tools = _parse_trajectory_tools(
            str(row.get("TRAJECTORY") or row.get("trajectory") or "")
        )
        removable_tools = [tool for tool in required_tools if tool in enabled_tools]
        if not task_id or not prompt or not removable_tools:
            continue

        for hidden_tool in removable_tools[:max_tools_per_task]:
            records.append(
                {
                    "case_id": f"mcp_atlas_{task_id}_remove_{hidden_tool}",
                    "source_task_id": task_id,
                    "prompt": prompt,
                    "hidden_required_tools": [hidden_tool],
                    "original_enabled_tools": enabled_tools,
                    "visible_tools_in_gap_run": [
                        tool for tool in enabled_tools if tool != hidden_tool
                    ],
                    "baseline_coverage": None,
                    "gap_run_coverage": None,
                    "candidate_gap_label": "tool_removal",
                    "validation_status": "needs_paired_mcp_atlas_run",
                    "strict_labeling_rule": (
                        "Promote only if original succeeds repeatedly, ablated run "
                        "fails repeatedly, and the hidden capability is not reasonably "
                        "replaceable by model knowledge or another visible tool."
                    ),
                }
            )
            if len({record["source_task_id"] for record in records}) >= max_tasks:
                break
        if len({record["source_task_id"] for record in records}) >= max_tasks:
            break

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MCP-Atlas tool-removal candidates for paired runs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--max-tools-per-task", type=int, default=1)
    args = parser.parse_args()

    rows = _load_rows(args.input)
    records = build_plan(
        rows,
        max_tasks=args.max_tasks,
        max_tools_per_task=args.max_tools_per_task,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} ablation candidates to {args.output}")


if __name__ == "__main__":
    main()
