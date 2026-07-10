# Interim Progress Status

Date: 2026-07-11

## Current Status

This project is now suitable for an interim progress presentation, not a final
thesis/results claim.

Implemented:

- F0-F8 failure taxonomy.
- LLM baseline classifier.
- Capability matcher that detects missing capabilities and emits structured
  capability requests.
- AgentRx loader for tau-retail and sample traces.
- MCP-Atlas adapter for synthetic missing-tool cases.
- Live MCP trace generator with paired control/gap task definitions.
- Offline unit tests for core parsing, metrics, and matcher behavior.
- MCP-Atlas ablation planning script for Omar handoff integration.
- MCP-Atlas CSV exporter for official harness/HPC paired runs.

Verified locally:

- `ruff check .` passes.
- `python -m unittest discover -s tests` passes.
- Live controlled-gap evaluation completed with API-backed LLM calls.
- MCP-Atlas ablation inputs exported and packaged for HPC.

Current blockers:

- Romeo HPC login and file access work, but job execution is blocked because the
  current project account is file-access-only:
  `p_scads_lv_llm has been locked: Lecture ended. Project open for fileaccess only. No Jobs.`
- MCP-Atlas paired ablation execution needs an active CPU job allocation or Omar's
  active MCP-Atlas environment.

## Current Local Result

Dataset:

- 6 controlled live MCP capability-gap tasks.
- Each task has a full-tool control run and a gap run where one required
  non-substitutable capability is withheld.
- This is live execution, but the gap label is induced by experimental design.

| method | accuracy | F6 precision | F6 recall | F6 F1 | binary gap F1 |
|---|---:|---:|---:|---:|---:|
| `llm-fair` | 0.833 | 1.000 | 0.833 | 0.909 | 0.909 |
| `capmatch-fair` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Observed improvement:

- `capmatch-fair` catches the `currency_conversion` gap that the generic LLM
  judge mislabeled as success.
- `capmatch-fair` also emits structured `capability_requests`, which the baseline
  LLM judge does not.

## What Changed For Interim Quality

Weak live tasks were removed:

- `translate_de`
- `percent_calc`

These were weak because the model can answer from internal knowledge, so they do
not prove a missing external capability.

The live task set now emphasizes non-substitutable external capabilities:

- current weather lookup;
- currency conversion;
- email sending;
- calendar event creation;
- restaurant booking;
- private email lookup.

## Omar / MCP-Atlas Integration

Omar's repo is infrastructure, not a finished capability-gap dataset. Use it to
execute paired MCP-Atlas runs:

1. Baseline: original `ENABLED_TOOLS`.
2. Ablation: same task, one required tool hidden.
3. Score both with official MCP-Atlas evaluator.
4. Promote only stable score drops to candidate gaps.
5. Promote to validated capability gaps only after checking substitutability.

Local planning command:

```bash
.venv/bin/python scripts/build_mcp_atlas_ablation_plan.py --max-tasks 5
```

Output:

```text
outputs/mcp_atlas_ablation_plan.jsonl
```

Export runnable MCP-Atlas CSV inputs:

```bash
.venv/bin/python scripts/export_mcp_atlas_ablation_inputs.py
```

Outputs:

```text
outputs/mcp_atlas_runs/baseline_all.csv
outputs/mcp_atlas_runs/ablated_all.csv
outputs/mcp_atlas_runs/manifest.json
outputs/mcp_atlas_hpc_inputs.tar.gz
```

HPC status:

```text
Romeo environment check:
- login: OK
- apptainer/singularity/bwrap: OK
- uv: installed locally
- input packet unpacked: OK
- srun: BLOCKED by file-access-only project allocation
```

## Commands To Run With API Key

```bash
export SCADS_API_KEY=...

.venv/bin/python -m src.evaluate \
  --method llm-fair capmatch-fair \
  --split all \
  --save-predictions

.venv/bin/python -m src.evaluate \
  --benchmark agentrx \
  --agentrx-source hf \
  --agentrx-only \
  --method llm-fair capmatch-fair \
  --split all \
  --save-predictions
```

## Strict Interim Claim

Correct claim:

> We implemented a prototype capability-gap detector. On controlled live MCP
> gap traces, it improves over an LLM-as-judge baseline and emits structured
> missing-capability requests. MCP-Atlas paired ablation inputs are prepared;
> execution is pending active HPC job access.

Incorrect claim:

> We have proven generalization across real-world benchmarks.

## Tomorrow Plan

1. Ask Omar or supervisor for an active CPU job allocation, or ask Omar to run
   the one-task MCP-Atlas smoke with our prepared CSVs.
2. Execute one MCP-Atlas baseline/ablated pair, not all five.
3. Score both with the official MCP-Atlas evaluator.
4. If the first pair works, run the remaining four pairs with `concurrency=1`.
5. Add a result table: task id, hidden tool, baseline score, ablated score,
   candidate gap status, capability request.
