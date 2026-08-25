#!/bin/bash
#SBATCH --job-name=qwen-realtime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/hpc/qwen-realtime-%j.out
#SBATCH --error=outputs/hpc/qwen-realtime-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/hpc

export MODEL_PROVIDER=ollama
export MODEL_NAME="${MODEL_NAME:-qwen3:8b}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not on PATH. Load/install Ollama before submitting this job."
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is not on PATH. Activate the project virtualenv before submitting or load a Python module."
  exit 1
fi

ollama serve > "outputs/hpc/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill ${OLLAMA_PID} || true' EXIT

sleep 10
ollama pull "${MODEL_NAME}"

python scripts/smoke_model_tool_calling.py

python -m src.live_agent \
  --tasks data/live_realtime_tasks.json \
  --output data/live_realtime_traces_qwen3.jsonl \
  --mode both

python -m src.live_agent \
  --tasks data/live_realtime_replacement_tasks.json \
  --output data/live_realtime_replacement_traces_qwen3.jsonl \
  --mode gap \
  --append

python scripts/build_clean_realtime_dataset.py
