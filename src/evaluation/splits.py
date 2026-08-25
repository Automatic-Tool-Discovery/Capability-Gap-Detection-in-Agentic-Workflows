"""Trace loading and train/test split strategies.

Evaluation begins here when ``src.evaluate`` needs local trace data. The loader
reads JSONL files into ``AgentTrace`` models, then the split helpers decide which
records are used for training context vs. testing. The current classifiers are
mostly zero/few-shot, but these split names keep experiments comparable across
live MCP traces and external benchmark adapters.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable, Iterator

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split

from src.schemas import AgentTrace
from src.evaluation.capabilities import tools_to_capabilities
from src.taxonomy import FailureType

LIVE_PATH = Path("data/live_traces.jsonl")


def _add_legacy_gap_ground_truth(trace: AgentTrace) -> AgentTrace:
    """Backfill old live traces written before gold capability fields existed."""
    if trace.gold_missing_capabilities or not trace.failure_explanation:
        return trace
    match = re.search(r"Required tool\(s\) (\[[^\]]*\]) were withheld", trace.failure_explanation)
    if not match:
        return trace
    try:
        tools = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return trace
    if isinstance(tools, list):
        trace.gold_missing_capabilities = tools_to_capabilities(
            [str(tool) for tool in tools]
        )
    return trace


def load_traces(paths: list[Path]) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trace = AgentTrace.model_validate_json(line)
                    traces.append(_add_legacy_gap_ground_truth(trace))
    return traces


def label_clean_controls_as_success(traces: list[AgentTrace]) -> list[AgentTrace]:
    """Treat unlabeled clean control traces as F0 for paired live evaluations."""
    labeled: list[AgentTrace] = []
    for trace in traces:
        if trace.gold_label is None and trace.source == "mcp-live":
            has_tool_error = any(call.error for call in trace.tool_calls)
            if trace.tool_calls and not has_tool_error:
                trace = trace.model_copy(
                    update={"gold_label": FailureType.SUCCESS_NO_FAILURE.value}
                )
        labeled.append(trace)
    return labeled


def labeled_traces(traces: list[AgentTrace]) -> list[AgentTrace]:
    return [trace for trace in traces if trace.gold_label is not None]


Split = tuple[list[AgentTrace], list[AgentTrace]]


def split_all(traces: list[AgentTrace]) -> Split:
    return traces, traces


def split_random(
    traces: list[AgentTrace],
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> Split:
    labeled = labeled_traces(traces)
    if len(labeled) < 4:
        raise ValueError("Need at least 4 labeled traces for a random split.")
    labels = [trace.gold_label for trace in labeled]
    train, test = train_test_split(
        labeled,
        test_size=test_size,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    return list(train), list(test)


def iter_leave_one_out(traces: list[AgentTrace]) -> Iterator[Split]:
    labeled = labeled_traces(traces)
    if len(labeled) < 2:
        raise ValueError("Need at least 2 labeled traces for leave-one-out.")
    indices = list(range(len(labeled)))
    for train_idx, test_idx in LeaveOneOut().split(indices):
        train = [labeled[i] for i in train_idx]
        test = [labeled[i] for i in test_idx]
        yield train, test


def iter_stratified_kfold(
    traces: list[AgentTrace],
    *,
    k: int = 5,
    seed: int = 42,
) -> Iterator[Split]:
    labeled = labeled_traces(traces)
    if len(labeled) < k:
        raise ValueError(f"Need at least {k} labeled traces for {k}-fold CV.")
    labels = [trace.gold_label for trace in labeled]
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    indices = list(range(len(labeled)))
    for train_idx, test_idx in splitter.split(indices, labels):
        train = [labeled[i] for i in train_idx]
        test = [labeled[i] for i in test_idx]
        yield train, test


SPLITTERS: dict[str, Callable[[list[AgentTrace]], Split]] = {
    "all": split_all,
}


def get_splitter(name: str) -> Callable[[list[AgentTrace]], Split]:
    if name not in SPLITTERS:
        raise ValueError(f"Unknown split '{name}'. Options: {', '.join(SPLITTERS)}")
    return SPLITTERS[name]
