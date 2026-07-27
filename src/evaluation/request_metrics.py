"""Deterministic quality metrics for generated capability requests.

Correctness is measured against benchmark-known withheld capabilities. Usability
is measured as schema completeness; it deliberately does not claim to measure
natural-language quality or whether generated code would execute successfully.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.capabilities import normalize_tool_name, tool_to_capability
from src.schemas import AgentTrace, CapabilityRequest, Prediction
from src.taxonomy import FailureType

F6_LABEL = FailureType.MISSING_CAPABILITY_GAP.value
GENERIC_CAPABILITY_TOKENS = {"api", "current", "get", "tool"}


def _canonical(name: str) -> str:
    return tool_to_capability(normalize_tool_name(name))


def _tokens(name: str) -> set[str]:
    tokens = set(_canonical(name).split("_")) - GENERIC_CAPABILITY_TOKENS
    normalized = set()
    for token in tokens:
        if token.endswith("ies") and len(token) > 3:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token in {"conversion", "converter"}:
            token = "convert"
        normalized.add(token)
    return normalized


def capabilities_match(left: str, right: str) -> bool:
    """Deterministic lexical equivalence for differently phrased slugs."""
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return _canonical(left) == _canonical(right)
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    return overlap >= 2 / 3


def _matched_count(gold: set[str], predicted: set[str]) -> int:
    remaining = set(predicted)
    matched = 0
    for expected in sorted(gold):
        candidate = next(
            (value for value in sorted(remaining) if capabilities_match(expected, value)),
            None,
        )
        if candidate is not None:
            remaining.remove(candidate)
            matched += 1
    return matched


def _request_completeness(request: CapabilityRequest) -> float:
    checks = [
        bool(request.name.strip()),
        bool(request.capability.strip()),
        bool(request.description.strip()),
        bool((request.rationale or "").strip()),
        bool(request.inputs),
        bool(request.outputs),
    ]
    return sum(checks) / len(checks)


@dataclass
class CapabilityRequestResult:
    n_eligible: int
    capability_precision: float
    capability_recall: float
    capability_f1: float
    exact_match_rate: float
    request_coverage: float
    schema_completeness: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "request_n_eligible": self.n_eligible,
            "request_capability_precision": self.capability_precision,
            "request_capability_recall": self.capability_recall,
            "request_capability_f1": self.capability_f1,
            "request_exact_match_rate": self.exact_match_rate,
            "request_coverage": self.request_coverage,
            "request_schema_completeness": self.schema_completeness,
        }


def evaluate_capability_requests(
    traces: list[AgentTrace], predictions: list[Prediction]
) -> CapabilityRequestResult | None:
    """Score only F6 traces with explicit missing-capability ground truth."""
    by_id = {prediction.trace_id: prediction for prediction in predictions}
    eligible = [
        trace
        for trace in traces
        if trace.gold_label == F6_LABEL and trace.gold_missing_capabilities
    ]
    if not eligible:
        return None

    true_positive = false_positive = false_negative = exact = covered = 0
    total_gold = 0
    completeness: list[float] = []

    for trace in eligible:
        prediction = by_id[trace.trace_id]
        gold = {_canonical(value) for value in trace.gold_missing_capabilities}
        predicted = {_canonical(value) for value in prediction.missing_capabilities}
        request_caps = {
            _canonical(request.capability) for request in prediction.capability_requests
        }

        predicted_matches = _matched_count(gold, predicted)
        request_matches = _matched_count(gold, request_caps)
        true_positive += predicted_matches
        false_positive += len(predicted) - predicted_matches
        false_negative += len(gold) - predicted_matches
        exact += int(predicted_matches == len(gold) == len(predicted))
        covered += request_matches
        total_gold += len(gold)
        completeness.extend(
            _request_completeness(request)
            for request in prediction.capability_requests
            if any(
                capabilities_match(request.capability, expected) for expected in gold
            )
        )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CapabilityRequestResult(
        n_eligible=len(eligible),
        capability_precision=precision,
        capability_recall=recall,
        capability_f1=f1,
        exact_match_rate=exact / len(eligible),
        request_coverage=covered / total_gold if total_gold else 0.0,
        schema_completeness=sum(completeness) / len(completeness) if completeness else 0.0,
    )
