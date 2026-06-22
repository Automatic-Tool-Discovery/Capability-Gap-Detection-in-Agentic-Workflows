from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

from src.main import DEFAULT_TRACE_PATHS, load_traces
from src.heuristic_classifier import classify_trace


def main() -> None:
    traces = load_traces(DEFAULT_TRACE_PATHS)

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
