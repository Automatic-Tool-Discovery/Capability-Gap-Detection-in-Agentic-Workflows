from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from src.schemas import AgentTrace
from src.heuristic_classifier import classify_trace


DATA_PATH = Path("data/traces.jsonl")


def load_traces(path: Path) -> list[AgentTrace]:
    traces = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(AgentTrace.model_validate_json(line))
    return traces


def main() -> None:
    traces = load_traces(DATA_PATH)

    y_true = []
    y_pred = []

    for trace in traces:
        prediction = classify_trace(trace)

        if trace.gold_label is None:
            continue

        y_true.append(trace.gold_label)
        y_pred.append(prediction.predicted_label)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
