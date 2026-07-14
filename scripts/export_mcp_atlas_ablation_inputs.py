"""Export MCP-Atlas baseline/ablated CSV inputs from an ablation plan.

The official MCP-Atlas completion script expects CSV rows with the original
dataset columns. This script preserves each source row and changes only
``ENABLED_TOOLS`` in the ablated copy.

It is the bridge between the JSONL ablation plan produced by
``build_mcp_atlas_ablation_plan.py`` and the paired CSV files consumed by the
external MCP-Atlas harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ROWS = Path("data/benchmarks/mcp_atlas_sample.jsonl")
DEFAULT_PLAN = Path("outputs/mcp_atlas_ablation_plan.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/mcp_atlas_runs")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _rows_by_task(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    return {str(row.get("TASK") or row.get("task") or row.get("id")): row for row in rows}


def _safe_case_name(case_id: str) -> str:
    keep = [char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id]
    return "".join(keep)[:180]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def export_inputs(
    *,
    source_rows_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    source_rows = _rows_by_task(source_rows_path)
    plan_records = _read_jsonl(plan_path)
    if not plan_records:
        raise ValueError(f"No ablation records found in {plan_path}")

    first_source = next(iter(source_rows.values()))
    fieldnames = list(first_source.keys())
    manifest: list[dict[str, Any]] = []
    all_baseline_rows: list[dict[str, Any]] = []
    all_ablated_rows: list[dict[str, Any]] = []

    for record in plan_records:
        task_id = str(record["source_task_id"])
        source = source_rows.get(task_id)
        if source is None:
            raise KeyError(f"Task {task_id} from plan not found in {source_rows_path}")

        baseline_row = dict(source)
        ablated_row = dict(source)
        ablated_row["ENABLED_TOOLS"] = json.dumps(
            record["visible_tools_in_gap_run"],
            ensure_ascii=False,
        )

        case_name = _safe_case_name(str(record["case_id"]))
        baseline_path = output_dir / f"baseline_{case_name}.csv"
        ablated_path = output_dir / f"ablated_{case_name}.csv"

        _write_csv(baseline_path, [baseline_row], fieldnames)
        _write_csv(ablated_path, [ablated_row], fieldnames)

        all_baseline_rows.append(baseline_row)
        all_ablated_rows.append(ablated_row)
        manifest.append(
            {
                "case_id": record["case_id"],
                "source_task_id": task_id,
                "hidden_required_tools": record["hidden_required_tools"],
                "baseline_csv": str(baseline_path),
                "ablated_csv": str(ablated_path),
                "baseline_output": f"completion_results/baseline_{case_name}.csv",
                "ablated_output": f"completion_results/ablated_{case_name}.csv",
                "status": "ready_for_mcp_atlas_harness",
            }
        )

    _write_csv(output_dir / "baseline_all.csv", all_baseline_rows, fieldnames)
    _write_csv(output_dir / "ablated_all.csv", all_ablated_rows, fieldnames)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MCP-Atlas baseline and ablated CSV inputs."
    )
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest = export_inputs(
        source_rows_path=args.rows,
        plan_path=args.plan,
        output_dir=args.output_dir,
    )
    print(f"Wrote {len(manifest)} paired MCP-Atlas CSV inputs to {args.output_dir}")
    print(f"Combined baseline: {args.output_dir / 'baseline_all.csv'}")
    print(f"Combined ablated:  {args.output_dir / 'ablated_all.csv'}")


if __name__ == "__main__":
    main()
