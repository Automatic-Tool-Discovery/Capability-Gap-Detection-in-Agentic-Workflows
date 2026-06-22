import argparse
import json
from pathlib import Path

from src.schemas import AgentTrace
from src.heuristic_classifier import classify_trace


DEFAULT_TRACE_PATHS = [
    Path("data/traces.jsonl"),
    Path("data/mcp_traces.jsonl"),
]
OUTPUT_PATH = Path("outputs/predictions.jsonl")


def load_traces(paths: list[Path]) -> list[AgentTrace]:
    traces = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    traces.append(AgentTrace.model_validate_json(line))
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run failure classification on traces.")
    parser.add_argument(
        "--traces",
        nargs="*",
        type=Path,
        default=None,
        help="Trace files to classify. Defaults to data/traces.jsonl and data/mcp_traces.jsonl.",
    )
    args = parser.parse_args()

    trace_paths = args.traces if args.traces else DEFAULT_TRACE_PATHS
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    traces = load_traces(trace_paths)
    if not traces:
        raise SystemExit(
            "No traces found. Run `python -m src.trace_collector` first or provide --traces."
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for trace in traces:
            prediction = classify_trace(trace)
            f.write(json.dumps(prediction.model_dump(), ensure_ascii=False) + "\n")

            print(f"{trace.trace_id}: {prediction.predicted_label}")


if __name__ == "__main__":
    main()
