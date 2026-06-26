"""Run evaluation across baselines, splits, and optional external benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from src import heuristic_classifier, llm_classifier
from src.evaluation.benchmarks.agentrx import build_agentrx_dataset
from src.evaluation.benchmarks.mcp_atlas import build_atlas_gap_dataset
from src.evaluation.metrics import EvaluationResult, evaluate_predictions, format_result
from src.evaluation.splits import (
    MCP_PATH,
    SYNTHETIC_PATH,
    get_splitter,
    iter_leave_one_out,
    iter_stratified_kfold,
    load_traces,
    split_random,
)
from src.schemas import AgentTrace, Prediction

DEFAULT_TRACE_PATHS = [SYNTHETIC_PATH, MCP_PATH]
OUTPUT_DIR = Path("outputs/evaluation")

ClassifierFn = Callable[[AgentTrace], Prediction]


def _heuristic_oracle(trace: AgentTrace) -> Prediction:
    return heuristic_classifier.classify_trace(trace, use_failure_explanation=True)


def _heuristic_fair(trace: AgentTrace) -> Prediction:
    return heuristic_classifier.classify_trace(trace, use_failure_explanation=False)


def _llm_fair(trace: AgentTrace) -> Prediction:
    return llm_classifier.classify_trace(trace, use_failure_explanation=False)


def _llm_oracle(trace: AgentTrace) -> Prediction:
    return llm_classifier.classify_trace(trace, use_failure_explanation=True)


METHODS: dict[str, ClassifierFn] = {
    "heuristic-oracle": _heuristic_oracle,
    "heuristic-fair": _heuristic_fair,
    "llm-fair": _llm_fair,
    "llm-oracle": _llm_oracle,
}


def run_evaluation(
    traces: list[AgentTrace],
    *,
    method: str,
    split_name: str,
    train_traces: list[AgentTrace],
    test_traces: list[AgentTrace],
) -> EvaluationResult:
    classifier = METHODS[method]
    predictions = [classifier(trace) for trace in test_traces]
    return evaluate_predictions(
        test_traces,
        predictions,
        method=method,
        split=split_name,
        n_train=len(train_traces),
    )


def write_predictions(
    path: Path,
    test_traces: list[AgentTrace],
    predictions: list[Prediction],
    *,
    method: str,
    split_name: str,
) -> None:
    """Write a per-trace comparison (gold vs predicted) for error inspection."""
    by_id = {prediction.trace_id: prediction for prediction in predictions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in test_traces:
            if trace.gold_label is None:
                continue
            prediction = by_id[trace.trace_id]
            correct = trace.gold_label == prediction.predicted_label
            handle.write(
                json.dumps(
                    {
                        "method": method,
                        "split": split_name,
                        "trace_id": trace.trace_id,
                        "gold_label": trace.gold_label,
                        "predicted_label": prediction.predicted_label,
                        "correct": correct,
                        "confidence": prediction.confidence,
                        "evidence": prediction.evidence,
                        "user_task": (trace.user_task or "")[:200],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _aggregate_results(results: list[EvaluationResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "f6_precision",
        "f6_recall",
        "f6_f1",
        "gap_detection_f1",
    ]
    return {key: sum(getattr(result, key) for result in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate failure classifiers with splits and optional MCP-Atlas benchmark."
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS),
        nargs="+",
        default=["heuristic-oracle", "heuristic-fair"],
        help="Classifier(s) to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "holdout-mcp", "holdout-synthetic", "random", "loo", "cv5"],
        default="all",
        help="Evaluation split strategy.",
    )
    parser.add_argument(
        "--traces",
        nargs="*",
        type=Path,
        default=None,
        help="Local trace files (default: synthetic + MCP traces).",
    )
    parser.add_argument(
        "--benchmark",
        choices=["none", "mcp-atlas", "agentrx"],
        default="none",
        help="Optional external benchmark dataset.",
    )
    parser.add_argument(
        "--atlas-limit",
        type=int,
        default=50,
        help="Number of MCP-Atlas tasks to load when --benchmark mcp-atlas.",
    )
    parser.add_argument(
        "--agentrx-source",
        choices=["samples", "hf"],
        default="samples",
        help="AgentRx data source: public GitHub samples or gated HF benchmark.",
    )
    parser.add_argument(
        "--agentrx-only",
        action="store_true",
        help="Evaluate only on AgentRx traces (ignore local synthetic/MCP traces).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for JSON evaluation summaries.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write per-trace gold-vs-predicted comparisons for error inspection.",
    )
    args = parser.parse_args()

    trace_paths = args.traces if args.traces else DEFAULT_TRACE_PATHS
    traces = [] if args.agentrx_only else load_traces(trace_paths)
    if args.benchmark == "mcp-atlas":
        atlas_traces = build_atlas_gap_dataset(limit=args.atlas_limit)
        traces.extend(atlas_traces)
        print(f"Loaded {len(atlas_traces)} synthetic F6 traces from MCP-Atlas.")
    elif args.benchmark == "agentrx":
        oracle = any(method.endswith("oracle") for method in args.method)
        agentrx_traces = build_agentrx_dataset(
            source=args.agentrx_source,
            use_failure_explanation=oracle,
        )
        traces.extend(agentrx_traces)
        print(
            f"Loaded {len(agentrx_traces)} AgentRx traces "
            f"(source={args.agentrx_source})."
        )

    if not traces:
        raise SystemExit("No traces found for evaluation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    summary_suffix = f"_{args.benchmark}" if args.benchmark != "none" else ""

    for method in args.method:
        if args.split in {"loo", "cv5"}:
            iterator = (
                iter_leave_one_out(traces)
                if args.split == "loo"
                else iter_stratified_kfold(traces, k=5)
            )
            fold_results: list[EvaluationResult] = []
            for fold_index, (train, test) in enumerate(iterator, start=1):
                result = run_evaluation(
                    traces,
                    method=method,
                    split_name=f"{args.split}-fold{fold_index}",
                    train_traces=train,
                    test_traces=test,
                )
                fold_results.append(result)
                print(format_result(result))
                print("-" * 60)

            aggregate = _aggregate_results(fold_results)
            summary = {
                "method": method,
                "split": args.split,
                "folds": len(fold_results),
                **aggregate,
            }
            summaries.append(summary)
            print(f"Mean metrics for {method} ({args.split}):")
            for key, value in aggregate.items():
                print(f"  {key}: {value:.3f}")
            continue

        if args.split == "random":
            train, test = split_random(traces)
            split_name = "random-75-25"
        else:
            train, test = get_splitter(args.split)(traces)
            split_name = args.split

        result = run_evaluation(
            traces,
            method=method,
            split_name=split_name,
            train_traces=train,
            test_traces=test,
        )
        print(format_result(result))
        print("-" * 60)
        summaries.append(result.to_dict())

        if args.save_predictions:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = (
                args.output_dir
                / f"predictions_{method}_{split_name}{summary_suffix}.jsonl"
            )
            write_predictions(
                predictions_path,
                test,
                result.predictions,
                method=method,
                split_name=split_name,
            )
            print(f"Wrote per-trace predictions to {predictions_path}")

    summary_path = args.output_dir / f"summary_{args.split}{summary_suffix}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
