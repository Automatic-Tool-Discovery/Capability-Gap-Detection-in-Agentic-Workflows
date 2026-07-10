# MCP-Atlas HPC Input Packet

This folder contains paired MCP-Atlas inputs for capability-gap ablation runs.

## Files

- `baseline_all.csv`  
  Five MCP-Atlas tasks with the original `ENABLED_TOOLS`.

- `ablated_all.csv`  
  The same five tasks with one required tool removed per task.

- `manifest.json`  
  Case IDs, hidden tools, and expected output filenames.

## First Smoke Run

Run only one task first:

```bash
uv run python mcp_completion_script.py \
  --model openai/alias-code \
  --input /path/to/baseline_all.csv \
  --output ramya_baseline_smoke.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter

uv run python mcp_completion_script.py \
  --model openai/alias-code \
  --input /path/to/ablated_all.csv \
  --output ramya_ablated_smoke.csv \
  --num-tasks 1 \
  --concurrency 1 \
  --no-filter
```

Then score:

```bash
uv run python mcp_evals_scores.py \
  --input-file completion_results/ramya_baseline_smoke.csv \
  --model-label ramya_baseline_smoke \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1

uv run python mcp_evals_scores.py \
  --input-file completion_results/ramya_ablated_smoke.csv \
  --model-label ramya_ablated_smoke \
  --evaluator-model openai/alias-ha \
  --num-tasks 1 \
  --concurrency 1
```

## Success Criterion

A useful candidate gap has:

```text
baseline score > ablated score
```

It becomes a stronger validated capability-gap case only after checking that the
hidden tool/capability was not reasonably replaceable by another visible tool or
model knowledge.
