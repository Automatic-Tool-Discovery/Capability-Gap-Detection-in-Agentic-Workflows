# HPC Qwen Real-Time Dataset Runbook

Date: 2026-08-22

## Goal

Run the Qwen3 real-time trace generation job on TU Dresden HPC because one local
full run takes about 45 minutes.

## Local Preparation

Commit or copy these project files to HPC:

- `data/live_realtime_tasks.json`
- `data/live_realtime_replacement_tasks.json`
- `mcp_servers/research_tools/server.py`
- `src/`
- `scripts/`
- `hpc/qwen_realtime_dataset.sbatch`
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

The job should produce a Qwen trace file under `data/` and then build:

```text
data/live_realtime_traces_qwen3_clean.jsonl
```

Expected quality:

```text
traces=60 controls=30 gaps=30
bad_controls=[]
gap_with_calls=[]
```

## Then Evaluate

```bash
python -m src.evaluate \
  --traces data/live_realtime_traces_qwen3_clean.jsonl \
  --method llm-fair capmatch-fair \
  --split all \
  --save-predictions
```

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
