# HPC Qwen Controlled Pilot Runbook

Date: 2026-10-06

## Goal

Run the expanded Qwen3 controlled pilot on TU Dresden HPC. The `realtime_120`
suite contains 120 prompts, balanced as 20 per family across six families. Each
task is run once with all tools and once with its target capability withheld,
for up to 240 raw traces. The 120 prompts are clustered within six capabilities
and should not be reported as 120 independent capability families.

## Local Preparation

Commit or copy these project files to HPC:

- `data/live_task_suites.json`
- `mcp_servers/research_tools/server.py`
- `src/`
- `scripts/`
- `hpc/qwen_realtime_dataset.sbatch`
- `hpc/qwen_eval_apptainer.sbatch`
- `scripts/evaluate_controlled_gap_pilot.py`
- `pyproject.toml`

## First Login

Use the login host provided by your TU Dresden/ZIH account. After login, move to
your project/work directory and clone or copy this repository there.

Do not run the full dataset on a login node.

## Environment Check

Run:

```bash
python --version
ollama --version
sbatch --version
```

If `ollama` is missing, check whether the cluster provides it as a module or
container. If not, we need an alternate runner for local HF/transformers models.

On Barnard, Ollama may be unavailable as a module while Apptainer is available.
Use the Apptainer job in that case.

## Interactive Smoke

Use an interactive allocation first:

```bash
srun --pty --cpus-per-task=8 --mem=32G --time=00:30:00 bash
```

Inside the allocated shell:

```bash
source .venv/bin/activate
export MODEL_PROVIDER=ollama
export MODEL_NAME=qwen3:8b

ollama serve > outputs/hpc/ollama-smoke.log 2>&1 &
ollama pull qwen3:8b

python scripts/smoke_model_tool_calling.py
```

If the smoke test calls `realtime_weather` successfully, exit the interactive
session and submit the batch job.

## Batch Run

```bash
mkdir -p outputs/hpc
sbatch hpc/qwen_realtime_dataset.sbatch
```

Watch:

```bash
squeue -u "$USER"
tail -f outputs/hpc/qwen-realtime-<jobid>.out
```

## Expected Result

The generation job requests up to eight hours and should produce the paired,
unreviewed trace file:

```text
outputs/traces/live_realtime_120_traces_qwen3.jsonl
```

Expected structure:

```text
120 task IDs, each with one control and one gap trace (240 total)
```

These traces do not have F0/F6 outcome labels. The assigned condition records
which tool was withheld; it does not prove a natural failure. Evaluate this
controlled intervention with:

```bash
python scripts/evaluate_controlled_gap_pilot.py \
  --traces outputs/traces/live_realtime_120_traces_qwen3.jsonl \
  --output-dir outputs/controlled_gap_pilot_120 \
  --method llm-fair capmatch-fair
```

The evaluator measures detection of the assigned withheld-tool condition, not
task success or natural F6 prevalence. It saves per-method predictions and
completed-case and failure-inclusive scores, confusion matrices, and
per-family results. The evaluation job requests up to twelve hours; actual
runtime depends on the model and cluster.

## Barnard Apptainer Fallback

If `ollama --version` fails but `apptainer --version` works, build/pull an Ollama
container in the project directory:

```bash
apptainer pull ollama.sif docker://ollama/ollama:latest
```

Then submit:

```bash
mkdir -p outputs/hpc data/ollama
sbatch hpc/qwen_ollama_apptainer.sbatch
```

Watch:

```bash
squeue -u "$USER"
tail -f outputs/hpc/qwen-ollama-apptainer-<jobid>.out
tail -f outputs/hpc/ollama-apptainer-<jobid>.log
```

If `apptainer pull` fails because compute/login nodes cannot reach Docker Hub,
pull/build the image on a machine with network access and copy `ollama.sif` to
the HPC project directory.
