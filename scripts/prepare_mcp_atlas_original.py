"""Sample original MCP-Atlas records and inventory evidence without assigning labels.

No tool removal, generated trajectories, or classifier calls. Hash-ranked sampling
is stable across source row ordering. Source fields are preserved in the JSONL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "8c563b55d7c967755f474299848049834d624617"
DEFAULT_ROOT = Path("data/benchmarks/mcp_atlas_original")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(rows: list[dict], count: int, seed: str) -> list[dict]:
    ids = [row["TASK"] for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("All source tasks need nonempty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source task IDs")
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} records from {len(rows)} source rows")
    return sorted(rows, key=lambda r: (
        hashlib.sha256(f"{seed}:{r['TASK']}".encode()).hexdigest(), r["TASK"]
    ))[:count]


def enabled_names(raw: str) -> list[str]:
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("Expected tool list")
    names = [v if isinstance(v, str) else v.get("name") if isinstance(v, dict) else None for v in values]
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Invalid tool name")
    return names


def audit_row(row: dict) -> dict:
    issues = []
    try:
        messages = json.loads(row["TRAJECTORY"])
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("Expected list of message objects")
    except (ValueError, TypeError):
        messages = []
        issues.append("unparseable_trajectory")
    try:
        enabled = enabled_names(row["ENABLED_TOOLS"])
    except (ValueError, TypeError):
        enabled = []
        issues.append("unparseable_enabled_tools")
    calls = [c for m in messages for c in (m.get("tool_calls") or [])]
    results = [m for m in messages if m.get("role") == "tool"]
    call_ids = [c.get("id") for c in calls]
    result_ids = [m.get("tool_call_id") for m in results]
    names = [c.get("function", {}).get("name") for c in calls]
    missing_results = [i for i in call_ids if i is None or i not in result_ids]
    orphan_results = [i for i in result_ids if i is None or i not in call_ids]
    outside = sorted({n for n in names if n and n not in enabled})
    if missing_results:
        issues.append("calls_without_linked_results")
    if orphan_results:
        issues.append("results_without_linked_calls")
    if outside:
        issues.append("calls_outside_declared_inventory")
    if len([i for i in call_ids if i is not None]) != len(set(i for i in call_ids if i is not None)):
        issues.append("duplicate_call_ids")
    # Terminal assistant text is only a structural candidate, not proof of an answer.
    terminal_answer = bool(messages and messages[-1].get("role") == "assistant"
                           and not messages[-1].get("tool_calls") and messages[-1].get("content"))
    if not terminal_answer:
        issues.append("no_terminal_assistant_text")
    error_blocks = sum(
        block.get("isError") is True
        for m in results
        for block in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(block, dict)
    )
    return {
        "source_task_id": row["TASK"], "message_count": len(messages),
        "tool_call_count": len(calls), "tool_result_count": len(results),
        "enabled_tool_count": len(enabled), "called_tools": sorted(set(n for n in names if n)),
        "last_role": messages[-1].get("role") if messages else None,
        "has_terminal_assistant_text": terminal_answer,
        "explicit_error_block_count": error_blocks,
        "calls_outside_declared_inventory": outside,
        "issues": issues,
        "annotation_status": "unreviewed", "task_outcome": None,
        "gap_status": None, "gold_label": None, "required_capabilities": [],
        "evidence": [], "reviewer": None,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(source: Path, output: Path, count: int, seed: str, revision: str) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset/review directory: {output}")
    rows = pq.read_table(source).to_pylist()
    selected = select_rows(rows, count, seed)
    audits = [audit_row(row) for row in selected]
    inventory = Counter()
    for row in selected:
        inventory.update(enabled_names(row["ENABLED_TOOLS"]))
    report = {
        "n_source": len(rows), "n_selected": len(selected),
        "labels_assigned": 0,
        "terminal_assistant_text_records": sum(a["has_terminal_assistant_text"] for a in audits),
        "last_roles": dict(Counter(a["last_role"] for a in audits)),
        "issue_counts": dict(Counter(i for a in audits for i in a["issues"])),
        "traces_with_explicit_error_blocks": sum(a["explicit_error_block_count"] > 0 for a in audits),
        "total_tool_calls": sum(a["tool_call_count"] for a in audits),
        "unique_enabled_tool_names": len(inventory),
        "tool_inventory_frequency": dict(sorted(inventory.items())),
        "interpretation": "Structural audit only. Error blocks do not establish task failure or F6. Missing final answers prevent automatic outcome labeling. Tool names alone do not establish semantic coverage.",
    }
    output.mkdir(parents=True)
    write_jsonl(output / "records.jsonl", selected)
    write_jsonl(output / "review_queue.jsonl", audits)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "dataset": "ScaleAI/MCP-Atlas", "revision": revision,
        "source_url": f"https://huggingface.co/datasets/ScaleAI/MCP-Atlas/resolve/{revision}/MCP-Atlas.parquet",
        "source_sha256": sha256(source), "source_rows": len(rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": "ascending SHA256(seed:source_task_id), independent of labels and content",
        "seed": seed, "sample_size": count, "selected_ids": [r["TASK"] for r in selected],
        "script_sha256": sha256(Path(__file__)),
        "records_sha256": sha256(output / "records.jsonl"),
        "original_fields_preserved": True, "tool_removal": False,
        "generated_trajectories": False, "gold_labels_assigned": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_ROOT / "MCP-Atlas.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "sample_200")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", default="2026-09-07")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.count, args.seed, args.revision)
    print(json.dumps({k: v for k, v in report.items() if k != "tool_inventory_frequency"}, indent=2))


if __name__ == "__main__":
    main()
