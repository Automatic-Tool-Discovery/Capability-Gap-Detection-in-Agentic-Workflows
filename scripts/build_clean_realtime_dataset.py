"""Build the cleaned real-time Qwen dataset.

This removes the flaky India holidays pair and, when available, replaces it with
the Canada holidays pair generated from data/live_realtime_replacement_tasks.json.
"""

from __future__ import annotations

import json
from pathlib import Path

RAW_PATH = Path("data/live_realtime_traces_qwen3.jsonl")
REPLACEMENT_PATH = Path("data/live_realtime_replacement_traces_qwen3.jsonl")
CLEAN_PATH = Path("data/live_realtime_traces_qwen3_clean.jsonl")
DROP_PREFIXES = {
    "live_rt_holidays_in_2026_control",
    "live_rt_holidays_in_2026_gap",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    rows = [row for row in read_jsonl(RAW_PATH) if row["trace_id"] not in DROP_PREFIXES]
    if REPLACEMENT_PATH.exists():
        rows.extend(read_jsonl(REPLACEMENT_PATH))
    rows_by_id = {row["trace_id"]: row for row in rows}
    rows = list(rows_by_id.values())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    controls = [row for row in rows if not row.get("gold_label")]
    gaps = [row for row in rows if row.get("gold_label")]
    bad_controls = [
        row["trace_id"]
        for row in controls
        if not row.get("tool_calls") or any(call.get("error") for call in row.get("tool_calls", []))
    ]
    gap_with_calls = [
        row["trace_id"]
        for row in gaps
        if row.get("tool_calls")
    ]
    print(f"wrote {CLEAN_PATH}")
    print(f"traces={len(rows)} controls={len(controls)} gaps={len(gaps)}")
    print(f"bad_controls={bad_controls}")
    print(f"gap_with_calls={gap_with_calls}")


if __name__ == "__main__":
    main()
