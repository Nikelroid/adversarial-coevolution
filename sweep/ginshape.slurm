#!/bin/bash
#SBATCH --job-name=ginshape
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/ginshape_%a_%A.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/ginshape_%a_%A.err

# Phase-3 reward-shaping sweep: can we close the gin headroom Phase 1 exposed?
# 6 reward configs x 3 seeds = 18 runs (array 0-17). Each resumes the Phase-1
# champion, trains 8M steps of self-play under its (knock,gin) shaping (with a
# periodic gin-rate eval every 1M steps -> curve + best-checkpoint, so the longer
# run can't be hurt by late divergence), then evals vs random + vs the frozen
# champion with FIXED rewards so the gin rate is comparable across configs.
# CPU-only (epyc-64), no GPU contention with workers. Override depth with STEPS=.
#
#   idx -> combo = idx % 6 ; seed = SEEDS[idx / 6]

set -euo pipefail
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

# W&B key (token stored as WANDB_TOKEN)
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}"

SEEDS=(100 200 300)
COMBO=$(( SLURM_ARRAY_TASK_ID % 6 ))
SIDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
SEED=${SEEDS[$SIDX]}

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID combo=$COMBO seed=$SEED cpus=$SLURM_CPUS_PER_TASK"
"$PY" sweep/selfplay_one.py --combo "$COMBO" --seed "$SEED"
