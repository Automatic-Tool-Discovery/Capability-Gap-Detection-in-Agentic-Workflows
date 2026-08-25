#!/bin/bash
#SBATCH --job-name=qwen-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/hpc/qwen-eval-%j.out
#SBATCH --error=outputs/hpc/qwen-eval-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/hpc outputs/evaluation data/ollama

export MODEL_PROVIDER=ollama
export MODEL_NAME="${MODEL_NAME:-qwen3:8b}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${SLURM_SUBMIT_DIR}/data/ollama"

IMAGE="${OLLAMA_IMAGE:-ollama.sif}"

if [[ ! -f "${IMAGE}" ]]; then
  echo "Missing ${IMAGE}."
  echo "Build/pull it first with:"
  echo "  apptainer pull ollama.sif docker://ollama/ollama:latest"
  exit 1
fi

apptainer exec --nv \
  --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
  --env OLLAMA_HOST="${OLLAMA_HOST}" \
  --env OLLAMA_MODELS="${OLLAMA_MODELS}" \
  "${IMAGE}" \
  ollama serve > "outputs/hpc/ollama-eval-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill ${OLLAMA_PID} || true' EXIT

sleep 15

apptainer exec --nv \
  --bind "${SLURM_SUBMIT_DIR}:${SLURM_SUBMIT_DIR}" \
  --env OLLAMA_HOST="${OLLAMA_HOST}" \
  --env OLLAMA_MODELS="${OLLAMA_MODELS}" \
  "${IMAGE}" \
  ollama pull "${MODEL_NAME}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx capgap-hpc; then
    conda create -y -n capgap-hpc python=3.11
  fi
  conda activate capgap-hpc
else
  python -m venv .venv-hpc
  source .venv-hpc/bin/activate
fi

unset PIP_REQUIRE_VIRTUALENV
python --version
python -m pip install -e .

python -m src.evaluate \
  --traces data/live_realtime_traces_qwen3_clean.jsonl \
  --method llm-fair capmatch-fair \
  --split all \
  --label-clean-controls \
  --save-predictions
