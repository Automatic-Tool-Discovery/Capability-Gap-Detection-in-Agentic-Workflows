"""Small trace-replay interface for inspecting one pipeline result.

This reuses a recorded run, so only classification is performed; the agent and
its tool calls are not regenerated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from src import capability_matcher, llm_classifier
from src.live_agent import DEFAULT_MODEL, run_single_question
from src.evaluation.request_metrics import evaluate_capability_requests
from src.evaluation.splits import load_traces
from src.schemas import AgentTrace, Prediction


def classify(trace: AgentTrace, method: str) -> Prediction:
    if method == "llm-fair":
        return llm_classifier.classify_trace(trace, use_failure_explanation=False)
    if method == "capmatch-fair":
        return capability_matcher.classify_trace(trace, use_failure_explanation=False)
    raise ValueError(f"Unsupported method: {method}")


def replay_trace(trace: AgentTrace, *, method: str) -> dict:
    prediction = classify(trace, method)
    result = {
        "trace_id": trace.trace_id,
        "user_task": trace.user_task,
        "method": method,
        "gold_label": trace.gold_label,
        "gold_missing_capabilities": trace.gold_missing_capabilities,
        "predicted_label": prediction.predicted_label,
        "correct": (
            trace.gold_label == prediction.predicted_label
            if trace.gold_label is not None
            else None
        ),
        "confidence": prediction.confidence,
        "evidence": prediction.evidence,
        "missing_capabilities": prediction.missing_capabilities,
        "capability_requests": [
            request.model_dump() for request in prediction.capability_requests
        ],
    }
    request_score = evaluate_capability_requests([trace], [prediction])
    if request_score is not None:
        result["capability_request_scores"] = request_score.to_dict()
    return result


def _select_trace(path: Path, trace_id: str | None) -> AgentTrace:
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        traces = load_traces([path])
    else:
        if not isinstance(document, dict):
            raise SystemExit("A JSON trace file must contain one object.")
        traces = [AgentTrace.model_validate(document)]
    if trace_id is None:
        if len(traces) > 1:
            raise SystemExit(
                f"{path} contains {len(traces)} traces; pass --trace-id to choose one."
            )
        return traces[0]
    for trace in traces:
        if trace.trace_id == trace_id:
            return trace
    raise SystemExit(f"Trace id {trace_id!r} was not found in {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify one recorded trace or run and classify one live question."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path, help="JSON or JSONL trace file.")
    source.add_argument("--question", help="Run this question against the live MCP server.")
    parser.add_argument("--trace-id", help="Required when the file contains multiple traces.")
    parser.add_argument(
        "--available-tools",
        help="Comma-separated live-mode tool allowlist (default: all tools).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SCADS_MODEL", DEFAULT_MODEL),
        help="Live agent model.",
    )
    parser.add_argument(
        "--method",
        choices=["llm-fair", "capmatch-fair"],
        default="capmatch-fair",
    )
    args = parser.parse_args()
    if args.trace is not None:
        trace = _select_trace(args.trace, args.trace_id)
    else:
        selected = (
            {value.strip() for value in args.available_tools.split(",") if value.strip()}
            if args.available_tools
            else None
        )
        trace = asyncio.run(
            run_single_question(
                args.question,
                available_tools=selected,
                model=args.model,
            )
        )
    print(json.dumps(replay_trace(trace, method=args.method), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
