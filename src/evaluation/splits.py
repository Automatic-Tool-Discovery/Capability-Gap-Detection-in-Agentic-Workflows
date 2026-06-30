"""Train/test split strategies for trace evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split

from src.schemas import AgentTrace

LIVE_PATH = Path("data/live_traces.jsonl")


def load_traces(paths: list[Path]) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    traces.append(AgentTrace.model_validate_json(line))
    return traces


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
