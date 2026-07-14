"""Evaluation metrics for failure and capability-gap classification.

This module compares ``Prediction`` objects against gold ``AgentTrace`` labels.
It reports ordinary multiclass metrics over F0-F8 plus capability-specific
numbers for F6, including binary gap detection F1. ``src.evaluate`` calls these
helpers after running either the LLM baseline or the capability matcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from src.schemas import AgentTrace, Prediction
from src.taxonomy import FailureType


F6_LABEL = FailureType.MISSING_CAPABILITY_GAP.value


@dataclass
class EvaluationResult:
    method: str
    split: str
    n_train: int
    n_test: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    f6_precision: float
    f6_recall: float
    f6_f1: float
    gap_detection_f1: float
    report: str
    predictions: list[Prediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "split": self.split,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "f6_precision": self.f6_precision,
            "f6_recall": self.f6_recall,
            "f6_f1": self.f6_f1,
            "gap_detection_f1": self.gap_detection_f1,
        }


def evaluate_predictions(
    traces: list[AgentTrace],
    predictions: list[Prediction],
    *,
    method: str,
    split: str,
    n_train: int,
) -> EvaluationResult:
    by_id = {prediction.trace_id: prediction for prediction in predictions}
    y_true: list[str] = []
    y_pred: list[str] = []
    gap_true: list[int] = []
    gap_pred: list[int] = []

    for trace in traces:
        if trace.gold_label is None:
            continue
        prediction = by_id[trace.trace_id]
        y_true.append(trace.gold_label)
        y_pred.append(prediction.predicted_label)
        gap_true.append(1 if trace.gold_label == F6_LABEL else 0)
        gap_pred.append(1 if prediction.predicted_label == F6_LABEL else 0)

    if not y_true:
        raise ValueError("No labeled traces in evaluation set.")

    precision, recall, f6_f1, _ = precision_recall_fscore_support(
        [1 if value == F6_LABEL else 0 for value in y_true],
        [1 if value == F6_LABEL else 0 for value in y_pred],
        average="binary",
        zero_division=0,
    )
    gap_f1 = f1_score(gap_true, gap_pred, zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    return EvaluationResult(
        method=method,
        split=split,
        n_train=n_train,
        n_test=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        f6_precision=float(precision),
        f6_recall=float(recall),
        f6_f1=float(f6_f1),
        gap_detection_f1=float(gap_f1),
        report=report,
        predictions=predictions,
    )


def format_result(result: EvaluationResult) -> str:
    lines = [
        f"Method: {result.method}",
        f"Split: {result.split}",
        f"Train size: {result.n_train} | Test size: {result.n_test}",
        f"Accuracy: {result.accuracy:.3f}",
        f"Macro F1: {result.macro_f1:.3f}",
        f"Weighted F1: {result.weighted_f1:.3f}",
        f"F6 precision/recall/F1: {result.f6_precision:.3f} / "
        f"{result.f6_recall:.3f} / {result.f6_f1:.3f}",
        f"Binary gap detection F1 (F6 vs rest): {result.gap_detection_f1:.3f}",
        "",
        result.report,
    ]
    return "\n".join(lines)
