# Experiment and submission plan (October 6–9, 2026)

## Decision

Do not wait for a large natural F6 dataset. The source review indexed 2,414 records but found **zero verified natural F6 positives**. Twelve media/file cases remain unresolved. This means none currently has enough evidence to confidently attribute failure to a necessary ability missing from the agent's complete toolset. Do not count those cases as positives or negatives.

The main experiment is a **controlled tool-withholding pilot**, expanded to 120 prompt/task IDs in `realtime_120`. It contains 20 prompts in each of six families: live weather, currency rates, earthquakes, ISS position, public holidays, and book search. Each task runs with the full toolset and with its target capability withheld, for up to 240 executions. This tests whether the methods detect the assigned intervention. It does not test natural F6 detection or prove that the task actually failed. The prompts cluster within six capabilities; this is not 120 independent capability types.

## What prior results mean

An older project note reported a six-task paired live pilot (`llm-fair` gap F1 0.909; `capmatch-fair` gap F1 1.000). Twelve raw traces are preserved in `outputs/archive/live_trace_artifacts_2026-08-30.tar.gz`, but they have no verified outcome labels, so that score cannot be reproduced from the archived traces as they stand. Treat it as an unverified historical report, not a submission result.

The archived `outputs/evaluation/summary_all.json` reports 60 examples in both `n_train` and `n_test` for the `all` split. Those are not disjoint sets; its perfect classification numbers are **not held-out evaluation results**. The archive is retained for history and audit only.

No archived metric establishes natural-F6 performance. The current pilot evaluator instead uses the recorded withholding assignment after inference to score detection of that controlled condition.

## Run path

1. Check that the Slurm project allocation, model runtime, model files, and live MCP endpoints are available. The last documented HPC check said job submission was blocked by a file-access-only allocation; confirm current access before budgeting time around HPC.
2. Generate paired traces using `hpc/qwen_realtime_dataset.sbatch`. It writes `outputs/traces/live_realtime_120_traces_qwen3.jsonl`. The output begins unreviewed; an assigned condition is not a task-outcome annotation.
3. Evaluate using `hpc/qwen_eval_apptainer.sbatch` or run `scripts/evaluate_controlled_gap_pilot.py` on the generated JSONL. The evaluator checks pairs, predicts before using assignment metadata for scoring, and saves per-method predictions and summary metrics under `outputs/controlled_gap_pilot_120/`.
4. Inspect missing/failed runs, paired confusion matrices, per-family outcomes, prediction errors, and example disagreements. Report both 120 prompts and the six capability families; repeated prompt forms and shared tools mean these are not 120 independent capability types.
5. Write the report and slides from saved artifacts. If a complete rerun cannot finish, show the older six-task result as an unverified historical report only; its raw files have no verified outcome labels.

See the [HPC runbook](hpc_qwen_runbook.md) for commands and output locations. Do not silently relabel controls as successful or gaps as failures.

## Submission claims

The report can claim that the project implements a detector and a controlled evaluation design. It can report natural-source screening as a dataset-feasibility result and the new pilot as detection of an assigned tool-withholding condition. Keep these results in separate sections and tables.

Do not claim a natural F6 benchmark, 30–50 verified natural positives, generalization across 120 independent capabilities, or a held-out improvement from the archived all-split results. A high controlled-pilot score is evidence about this constructed setup only.

## Dataset screening snapshot

The assistant desk-screened 124 keyword retrieval hits; this was not a line-by-line review of all 2,414 records and was not independent human adjudication.

| Screening outcome | Records | Interpretation |
|---|---:|---|
| No F6 evidence in retrieved material | 55 | Not verified negatives |
| Unresolved media/file boundary cases | 12 | Not verified positives |
| Different process-quality target | 46 | Not F6 labels |
| Insufficient task/rationale evidence | 11 | Unresolved |

Full source notes and evidence pointers are in [the natural-gap collection report](natural_gap_collection_2026-10-06.md). Original source data and review artifacts remain under `outputs/review/`.
