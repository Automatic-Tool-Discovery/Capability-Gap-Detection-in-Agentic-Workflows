#!/bin/bash
#SBATCH --job-name=capgap-v2
#SBATCH --account=p_scads_gnn_electricity
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-3%1
#SBATCH --output=outputs/hpc/followup-%A_%a.out
#SBATCH --error=outputs/hpc/followup-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.conda-pilot/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PROVIDER=ollama
export MODEL_NAME="${MODEL_NAME:-qwen3:8b}"
export OLLAMA_HOST=127.0.0.1:11439
export MODEL_BASE_URL=http://127.0.0.1:11439/v1
export OLLAMA_MODELS="$(readlink -f data/ollama)"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

RUN_DIR="$PWD/outputs/followup-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir "$RUN_DIR"
cp data/live_task_suites_v2.json "$RUN_DIR/tasks.json"
cp hpc/controlled_followup.sbatch "$RUN_DIR/job.sbatch"
python -m pip freeze > "$RUN_DIR/packages.txt"
printf '%s\n' "$MODEL_NAME" > "$RUN_DIR/model.txt"
nvidia-smi > "$RUN_DIR/gpu.txt"
apptainer exec --nv --bind "$OLLAMA_MODELS:$OLLAMA_MODELS" \
  --env OLLAMA_HOST="$OLLAMA_HOST" --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  ollama.sif ollama serve > "$RUN_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import json, os, time, urllib.request
for _ in range(60):
    try:
        with urllib.request.urlopen('http://127.0.0.1:11439/api/tags', timeout=2) as response:
            data = json.load(response)
        break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit('Ollama startup failed')
print(json.dumps(data))
if os.environ['MODEL_NAME'] not in {m['name'] for m in data['models']}:
    raise SystemExit('Requested model is not cached; stage it before submitting.')
PY

python -m src.live_agent --tasks "$RUN_DIR/tasks.json" \
  --suite realtime_120_v2 --mode both --output "$RUN_DIR/traces.jsonl"
sha256sum "$RUN_DIR/traces.jsonl" > "$RUN_DIR/input.sha256"
python scripts/evaluate_controlled_gap_pilot.py --traces "$RUN_DIR/traces.jsonl" \
  --output-dir "$RUN_DIR/evaluation" --method llm-fair capmatch-fair
echo "Follow-up complete: $RUN_DIR"
