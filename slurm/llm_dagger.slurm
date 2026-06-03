#!/bin/bash
#SBATCH --job-name=gin-dagger
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/dagger_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/dagger_%j.err

# Phase-4 LLM-DAgger (CPU trainer). Collects (obs, llm_action) labels from the
# Qwen3-30B served by the master + GPU worker pool, behavioural-clones the masked
# policy, then PPO self-play. Set BC_STATES=0 for the no-LLM baseline (no master
# needed). The gold-standard agent is only used to score the result.
#
#   sbatch slurm/master.slurm
#   sbatch --gres=gpu:v100:2 --export=ALL,WORKER_PRESET=qwen3-30b,WORKER_MAXTOK=48 slurm/worker.slurm
#   sbatch slurm/llm_dagger.slurm                                  # 30B-guided run
#   sbatch --export=ALL,BC_STATES=0,WANDB_NAME=dagger_baseline slurm/llm_dagger.slurm  # baseline

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}"

export LLM_MODEL="${LLM_MODEL:-qwen3-30b}"
export BC_STATES="${BC_STATES:-20000}"
export BC_EPOCHS="${BC_EPOCHS:-8}"
export RL_STEPS="${RL_STEPS:-2000000}"
export NUM_ENV="${NUM_ENV:-64}"
export N_STEPS="${N_STEPS:-256}"
export COLLECT_THREADS="${COLLECT_THREADS:-48}"
export EVAL_GAMES="${EVAL_GAMES:-400}"
export INIT_MODEL="${INIT_MODEL:-}"          # empty => behavioural-clone from scratch
export SEED="${SEED:-0}"
export SAVE_PATH="${SAVE_PATH:-game/model/ppo_gin_rummy_dagger.zip}"

# The LLM run needs the master; the baseline (BC_STATES=0) does not.
if [ "${BC_STATES}" != "0" ]; then
  echo "waiting for master + a healthy worker..."
  URL=""; HEALTHY=""
  for i in $(seq 1 360); do
    URL=$("$PY" -m llm.discovery 2>/dev/null || true)
    if [ -n "$URL" ]; then
      H=$(curl -sf "$URL/health" 2>/dev/null || true)
      NH=$(echo "$H" | grep -oE '"n_healthy":[0-9]+' | grep -oE '[0-9]+' || echo 0)
      [ "${NH:-0}" -ge 1 ] && { echo "master=$URL n_healthy=$NH after ~$((i*5))s"; HEALTHY=1; break; }
    fi
    sleep 5
  done
  # Fail loudly if no HEALTHY worker -- never collect on the random fallback.
  [ -n "$HEALTHY" ] || { echo "ERROR: no healthy worker in time; aborting (no garbage labels)"; exit 1; }
  export GINLLM_MASTER_URL="$URL"
fi

echo "host=$(hostname) llm=$LLM_MODEL bc_states=$BC_STATES rl_steps=$RL_STEPS init=${INIT_MODEL:-scratch}"
"$PY" sweep/llm_dagger_one.py
[ "${BC_STATES}" != "0" ] && { echo "=== master stats ==="; curl -sf "${GINLLM_MASTER_URL}/stats" 2>/dev/null || true; echo; } || true
