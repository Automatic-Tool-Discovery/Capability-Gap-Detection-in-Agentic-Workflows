import json
from pathlib import Path

from src.schemas import AgentTrace
from src.heuristic_classifier import classify_trace


DATA_PATH = Path("data/traces.jsonl")
OUTPUT_PATH = Path("outputs/predictions.jsonl")


def load_traces(path: Path) -> list[AgentTrace]:
    traces = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(AgentTrace.model_validate_json(line))
    return traces


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    traces = load_traces(DATA_PATH)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for trace in traces:
            prediction = classify_trace(trace)
            f.write(json.dumps(prediction.model_dump(), ensure_ascii=False) + "\n")

            print(f"{trace.trace_id}: {prediction.predicted_label}")


if __name__ == "__main__":
    main()
