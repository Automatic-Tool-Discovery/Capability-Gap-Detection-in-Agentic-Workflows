"""Evaluate gap detection on paired, deliberately withheld-tool pilot traces.

Gold here means the assigned controlled tool-availability condition. It does
not mean a naturally observed F6 failure or prove task completion quality.
Experiment metadata is read only by this evaluator; classifier payloads use the
shared trace evidence builder, which excludes condition and label metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.evaluate import METHODS
from src.evaluation.splits import load_traces
from src.model_config import get_model_config
from src.taxonomy import FailureType


def family(task_id: str) -> str:
    parts = task_id.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def evaluate(path: Path, output_dir: Path, methods: list[str]) -> dict:
    traces = load_traces([path])
    if not traces:
        raise ValueError("No traces loaded")
    if any(t.gold_label is not None or t.gold_missing_capabilities for t in traces):
        raise ValueError("Input already contains gold labels; expected unreviewed generated traces")
    if any(t.experiment is None or t.experiment.condition not in {"control", "gap"} for t in traces):
        raise ValueError("Every trace must carry a controlled condition in experiment metadata")
    if any(t.inventory_complete is not True for t in traces):
        raise ValueError("Controlled pilot requires complete tool inventory records")
    if len({t.trace_id for t in traces}) != len(traces):
        raise ValueError("Duplicate trace IDs")

    pairs: dict[str, dict[str, object]] = defaultdict(dict)
    for trace in traces:
        exp = trace.experiment
        assert exp is not None
        if exp.condition in pairs[exp.task_id]:
            raise ValueError(f"Duplicate {exp.condition} run for task {exp.task_id}")
        pairs[exp.task_id][exp.condition] = trace
    incomplete = {task: sorted(set(conditions) ^ {"control", "gap"})
                  for task, conditions in pairs.items()
                  if set(conditions) != {"control", "gap"}}
    if incomplete:
        raise ValueError(f"Incomplete paired runs: {incomplete}")
    if any(not pair["gap"].experiment.withheld_tools for pair in pairs.values()):
        raise ValueError("Gap condition must record at least one withheld tool")
    if any(pair["control"].experiment.withheld_tools for pair in pairs.values()):
        raise ValueError("Control condition unexpectedly withholds tools")
    if any(pair["control"].experiment.pair_id != pair["gap"].experiment.pair_id
           for pair in pairs.values()):
        raise ValueError("Control and gap traces must share their pair ID")

    output_dir.mkdir(parents=True, exist_ok=True)
    gold_by_id = {t.trace_id: int(t.experiment.condition == "gap") for t in traces}
    family_by_id = {t.trace_id: family(t.experiment.task_id) for t in traces}
    model_config = get_model_config(default_model="alias-ha")
    summary = {
        "evaluation_type": "controlled_tool_withholding_pilot",
        "interpretation": "Scores measure detection of assigned withheld-tool conditions; they are not estimates on natural failures or natural prevalence.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "classifier_model": model_config.model,
        "classifier_provider": model_config.provider,
        "n_traces": len(traces),
        "n_paired_tasks": len(pairs),
        "task_families": sorted({family(task) for task in pairs}),
        "methods": {},
    }
    f6 = FailureType.MISSING_CAPABILITY_GAP.value
    for method in methods:
        predict = METHODS[method]
        predictions = []
        for trace in traces:
            try:
                pred = predict(trace)
                predicted_gap = int(pred.predicted_label == f6)
                error_type = None
                confidence = pred.confidence
                evidence = pred.evidence
            except Exception as exc:
                predicted_gap, confidence, evidence = None, None, []
                error_type = type(exc).__name__
            gold = gold_by_id[trace.trace_id]
            predictions.append({
                "trace_id": trace.trace_id,
                "task_id": trace.experiment.task_id,
                "family": family_by_id[trace.trace_id],
                "condition": trace.experiment.condition,
                "gold_withheld_condition": gold,
                "predicted_gap": predicted_gap,
                "correct": predicted_gap == gold,
                "confidence": confidence,
                "evidence": evidence,
                "error_type": error_type,
            })
        valid = [p for p in predictions if p["predicted_gap"] is not None]
        y_true = [p["gold_withheld_condition"] for p in valid]
        y_pred = [p["predicted_gap"] for p in valid]
        # A failed classifier call is a failed prediction. Treat it as the
        # opposite class for the conservative, failure-inclusive confusion
        # matrix and report completed-case metrics separately.
        all_y_true = [p["gold_withheld_condition"] for p in predictions]
        all_y_pred = [p["predicted_gap"] if p["predicted_gap"] is not None
                      else 1 - p["gold_withheld_condition"] for p in predictions]
        per_family = {}
        for fam in sorted({p["family"] for p in valid}):
            group = [p for p in valid if p["family"] == fam]
            gy = [p["gold_withheld_condition"] for p in group]
            py = [p["predicted_gap"] for p in group]
            per_family[fam] = {
                "n": len(group),
                "accuracy": float(accuracy_score(gy, py)),
                "gap_precision": float(precision_score(gy, py, zero_division=0)),
                "gap_recall": float(recall_score(gy, py, zero_division=0)),
                "gap_f1": float(f1_score(gy, py, zero_division=0)),
            }
        predictions_by_task: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in predictions:
            predictions_by_task[row["task_id"]][row["condition"]] = row
        pair_correct = sum(
            group["control"]["predicted_gap"] == 0
            and group["gap"]["predicted_gap"] == 1
            for group in predictions_by_task.values()
        )
        summary["methods"][method] = {
            "n_attempted": len(predictions),
            "n_scored": len(valid),
            "n_errors": len(predictions) - len(valid),
            "coverage": len(valid) / len(predictions),
            "accuracy": float(accuracy_score(y_true, y_pred)) if valid else None,
            "gap_precision": float(precision_score(y_true, y_pred, zero_division=0)) if valid else None,
            "gap_recall": float(recall_score(y_true, y_pred, zero_division=0)) if valid else None,
            "gap_f1": float(f1_score(y_true, y_pred, zero_division=0)) if valid else None,
            "confusion_matrix_labels_0_no_gap_1_withheld": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() if valid else [],
            "failure_inclusive_accuracy": float(accuracy_score(all_y_true, all_y_pred)),
            "failure_inclusive_gap_precision": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
            "failure_inclusive_gap_recall": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
            "failure_inclusive_gap_f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
            "failure_inclusive_confusion_matrix_labels_0_no_gap_1_withheld": confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]).tolist(),
            "n_pairs_correct_both_conditions": pair_correct,
            "paired_exact_accuracy": pair_correct / len(pairs),
            "per_family": per_family,
        }
        with (output_dir / f"predictions_{method}.jsonl").open("w") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/controlled_gap_pilot"))
    parser.add_argument("--method", nargs="+", choices=sorted(METHODS), default=["llm-fair", "capmatch-fair"])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.traces, args.output_dir, args.method), indent=2))


if __name__ == "__main__":
    main()
