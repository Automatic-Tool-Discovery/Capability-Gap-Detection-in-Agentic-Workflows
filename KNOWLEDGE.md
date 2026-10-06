# Project guide

This guide is a short map of the repository. The [README](README.md) gives the current status; the [experiment plan](docs/week_experiment_decision_2026-10-06.md) is the source of truth for the submission work.

## What the project asks

A capability gap (F6) means a task requires an ability that the agent cannot get from any available tool or valid substitute. A failed tool call alone does not prove a gap: it could be a bad argument, outage, missing user information, policy restriction, or other issue. A **verified natural F6** therefore needs evidence about the task, actual available tools, alternatives, and why the agent failed.

The project compares two methods:

- `llm-fair`: an LLM classifies the trace using the shared evidence.
- `capmatch-fair`: an LLM identifies task requirements, then the system checks them against the capabilities in the offered toolset and can emit a capability request.

## What the data says

The natural-source index contains 2,414 records, but the review found zero verified natural F6 positives. Twelve media/file cases remain uncertain. MCP-Atlas contributes original reference trajectories, not confirmed failures or F6 labels. Keep these sources as feasibility and diagnostic evidence; do not turn uncertain cases into labels.

The archived evaluation files are historical. In particular, summaries using the `all` split show the same 60 examples as training and testing, so they are not held-out test results. A controlled tool-withholding pilot can still evaluate whether methods recognize an assigned intervention, but it answers a narrower question than natural failure detection.

## Repository map

| Path | Purpose |
|---|---|
| `src/` | Trace schemas, classifiers, live agent, and evaluation code |
| `data/live_task_suites.json` | Original pilot tasks plus the expanded `realtime_120` controlled suite |
| `data/benchmarks/` | Benchmark snapshots, samples, and source-preserving audits |
| `scripts/` | Dataset preparation, source screening, and pilot evaluation |
| `mcp_servers/research_tools/` | MCP tools used by the live pilot |
| `hpc/` | Slurm/Apptainer job scripts |
| `outputs/` | Generated traces, evaluation files, and archived artifacts; some outputs are gitignored |
| `docs/` | Current experiment, dataset, implementation, and reporting notes |

## Useful commands

Install project dependencies:

```bash
uv sync
```

Run the offline test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Start live generation or model scoring only after checking [the runbook](docs/hpc_qwen_runbook.md) and configuring model access. New traces begin unreviewed. Withheld-tool assignments describe the experiment, not verified task outcomes.

## Read next

- [Current experiment and submission plan](docs/week_experiment_decision_2026-10-06.md)
- [Natural-source collection audit](docs/natural_gap_collection_2026-10-06.md)
- [Implementation record](docs/day_1_2_implementation.md)
- [MCP-Atlas source audit](docs/mcp_atlas_200_audit.md)
- [Report outline and claim limits](docs/paper_plan.md)
