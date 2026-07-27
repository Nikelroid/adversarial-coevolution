#!/bin/bash
#SBATCH --job-name=ginrummy-poolplay
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/poolplay_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/poolplay_%j.err

# AlphaZero-style pool self-play: resume run_5, train vs a growing pool of past
# selves, checkpoint new generations in. CPU (the PPO MLP trains fine on CPU).

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# W&B: map WANDB_TOKEN (env or ~/.bashrc) onto WANDB_API_KEY.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}"
if [ -n "$WANDB_API_KEY" ]; then echo "W&B: key resolved (len=${#WANDB_API_KEY})"; else echo "W&B: no key"; fi

echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK python=$("$PY" --version 2>&1)"
"$PY" sweep/poolplay_one.py --steps "${POOL_STEPS:-12000000}" --save-freq "${POOL_SAVE_FREQ:-1500000}"
