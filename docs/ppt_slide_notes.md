# Interim PPT Slide Notes

## Problem

Tool-using agents fail for different reasons. This project focuses on one
specific class: capability gaps.

## Definition

A capability gap is present when the task requires a non-substitutable external
capability that is absent from the agent's available toolset, and the task cannot
be completed reliably by model knowledge or another available tool.

Includes:

- missing tool/API;
- missing live or private data access;
- missing authenticated action;
- missing external service.

Excludes:

- wrong tool choice;
- wrong parameters;
- tool runtime error;
- missing user information;
- simple tasks solvable from model knowledge.

## Method

Baseline:

- `llm-fair`: generic LLM-as-judge over F0-F8.

Proposed method:

- `capmatch-fair`: extract required capabilities, compare against available
  capabilities, label F6 if missing, and emit a structured capability request.

## Current Result

Controlled live MCP gap traces:

| method | accuracy | F6 F1 | binary gap F1 |
|---|---:|---:|---:|
| `llm-fair` | 0.833 | 0.909 | 0.909 |
| `capmatch-fair` | 1.000 | 1.000 | 1.000 |

Main observed failure of baseline:

- `llm-fair` mislabeled a missing currency-conversion capability as success
  because an unrelated tool call reported successful execution.
- `capmatch-fair` correctly identified `currency_conversion` as missing.

## Contribution

The method does not only diagnose F6. It emits a structured missing-capability
request: name, capability, inputs, outputs, and rationale.

## MCP-Atlas Plan

MCP-Atlas does not directly provide capability-gap labels. We generate candidate
gaps by paired ablation:

1. Run original MCP-Atlas task with original tools.
2. Remove one required tool from `ENABLED_TOOLS`.
3. Run the same task again.
4. Compare official evaluator coverage.
5. Promote only stable score drops to validated candidate gaps after checking
   non-substitutability.

Prepared files:

- `outputs/mcp_atlas_runs/baseline_all.csv`
- `outputs/mcp_atlas_runs/ablated_all.csv`
- `outputs/mcp_atlas_runs/manifest.json`

## Current Blocker

Romeo HPC setup succeeded for file access and environment checks, but Slurm job
execution is blocked:

```text
p_scads_lv_llm has been locked: Lecture ended. Project open for fileaccess only. No Jobs.
```

Next action: obtain active CPU allocation or ask Omar to run the one-task
MCP-Atlas smoke using the prepared input packet.
