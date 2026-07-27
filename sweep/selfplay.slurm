#!/bin/bash
#SBATCH --job-name=ginrummy-selfplay
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/selfplay_%a_%A.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/selfplay_%a_%A.err

# Self-play fine-tuning sweep over (knock_reward, gin_reward) configs, on CPU
# (the PPO MLP trains fine on CPU, like the Phase-1 sweep, and this avoids GPU
# contention with the LLM workers). Each task resumes run_5 and trains vs a
# frozen run_5, then evals vs run_5 and vs random.

set -euo pipefail

cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1           # RL stack lives in the coev env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# W&B: wandb authenticates with WANDB_API_KEY. The user's token is named
# WANDB_TOKEN, so map it (from the env or ~/.bashrc) onto WANDB_API_KEY.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}"
if [ -n "$WANDB_API_KEY" ]; then echo "W&B: key resolved (len=${#WANDB_API_KEY})"; else echo "W&B: no key found — running without logging"; fi

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID cpus=$SLURM_CPUS_PER_TASK python=$("$PY" --version 2>&1)"
"$PY" sweep/selfplay_one.py --combo "$SLURM_ARRAY_TASK_ID"
