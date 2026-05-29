#!/bin/bash
#SBATCH --job-name=gin-llmtrain
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/llmtrain_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/llmtrain_%j.err

# RL-vs-LLM training (CPU): PPO trains against the LLM opponent served by the
# master + GPU worker pool. Finds the master via the shared-FS registry
# (runtime/master.json) and waits until at least one worker is healthy.

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1          # RL + requests stack lives in the coev env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# find the master (written by slurm/master.slurm to runtime/master.json)
echo "waiting for master + a healthy worker..."
URL=""
for i in $(seq 1 240); do
  URL=$("$PY" -m llm.discovery 2>/dev/null || true)
  if [ -n "$URL" ]; then
    H=$(curl -sf "$URL/health" 2>/dev/null || true)
    NH=$(echo "$H" | grep -oE '"n_healthy":[0-9]+' | grep -oE '[0-9]+' || echo 0)
    [ "${NH:-0}" -ge 1 ] && { echo "master=$URL n_healthy=$NH after ~$((i*5))s"; break; }
  fi
  sleep 5
done
[ -n "$URL" ] || { echo "ERROR: no master found in runtime/master.json"; exit 1; }
export GINLLM_MASTER_URL="$URL"

export NUM_ENV="${NUM_ENV:-64}"
export TOTAL_STEPS="${TOTAL_STEPS:-40000}"
export N_STEPS="${N_STEPS:-128}"
export INIT_MODEL="${INIT_MODEL:-game/model/ppo_gin_rummy_selfplay.zip}"
export SAVE_PATH="${SAVE_PATH:-game/model/ppo_gin_rummy_llm.zip}"

echo "host=$(hostname) master=$GINLLM_MASTER_URL num_env=$NUM_ENV total=$TOTAL_STEPS"
"$PY" sweep/llmplay_one.py
echo "=== master stats ==="; curl -sf "$URL/stats" 2>/dev/null || true; echo
