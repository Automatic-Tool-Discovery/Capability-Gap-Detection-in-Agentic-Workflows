# Capability Gap Detection in Agentic Workflows

This project studies whether an agent's task failed because a necessary ability is missing from its available tools. It compares an LLM failure judge (`llm-fair`) with a capability matcher (`capmatch-fair`), which identifies required abilities, checks them against available tools, and can produce a structured request for a missing capability.

## Current evidence

- The detector, shared model-input builder, trace schema, and offered-tool enforcement are implemented. See [the implementation note](docs/day_1_2_implementation.md).
- A review indexed 2,414 records from public sources. It found **zero verified natural F6 examples**: none had sufficient evidence that a missing, non-substitutable ability caused the failure. The 12 remaining candidates are unresolved, not positives. See the [collection audit](docs/natural_gap_collection_2026-10-06.md).
- The current evaluation is a project-authored, paired tool-withholding pilot with 120 prompts across six tool families (up to 240 traces). It measures detection of a known experimental condition; it does not establish performance on natural failures or 120 independent capabilities. See the [current experiment plan](docs/week_experiment_decision_2026-10-06.md).
- Archived evaluation summaries include an `all` split with the same 60 examples listed for training and testing. Do not present those scores as held-out performance. Archived files are retained for provenance, not as current validated results.

## Start here

- New to the project: [KNOWLEDGE.md](KNOWLEDGE.md)
- Experiment and submission plan: [docs/week_experiment_decision_2026-10-06.md](docs/week_experiment_decision_2026-10-06.md)
- Dataset review: [docs/natural_gap_collection_2026-10-06.md](docs/natural_gap_collection_2026-10-06.md)
- Reproducing the Qwen pilot: [docs/hpc_qwen_runbook.md](docs/hpc_qwen_runbook.md)
- Results and limitations: [docs/paper_plan.md](docs/paper_plan.md)

## Main code

- `src/llm_classifier.py`: direct LLM failure-classification baseline.
- `src/capability_matcher.py`: required-versus-available capability detector.
- `src/trace_evidence.py`: shared, allowlisted model-visible evidence.
- `src/live_agent.py`: paired live-tool trace generation and enforcement.
- `scripts/evaluate_controlled_gap_pilot.py`: controlled withheld-tool evaluation.
- `data/live_task_suites.json`: project-authored pilot tasks.
- `data/benchmarks/mcp_atlas_original/`: source-preserving MCP-Atlas sample and audit. It has no verified F6 labels.

## Run locally

Install dependencies with `uv sync`. Run the offline suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Live generation and model evaluation require configured model access and reachable MCP endpoints. Read the runbook before starting a batch. New generated traces are unreviewed until their outcomes are checked; experimental withholding metadata is not itself a gold failure label.

## Scope of claims

Report natural-source screening and controlled tool-withholding results separately. Do not call the controlled pilot a natural F6 benchmark, count unresolved cases as positives, or report archived all-split scores as held-out results. The current submission plan describes the defensible scope and remaining steps.
