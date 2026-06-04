#!/bin/bash
#SBATCH --job-name=gin-phase5
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/phase5_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/phase5_%j.err

# Phase 5: train PPO with one observation representation, eval vs gold/champion/random.
#   sbatch --export=ALL,EMBEDDER=sparse,TAG=sparse slurm/phase5.slurm
#   sbatch --export=ALL,EMBEDDER=/scratch1/.../temporal_dataset_embedder.pt,TAG=temporal slurm/phase5.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY

export EMBEDDER="${EMBEDDER:-sparse}"
export TAG="${TAG:-sparse}"
export RL_STEPS="${RL_STEPS:-2000000}"
export NUM_ENV="${NUM_ENV:-64}"
export EVAL_GAMES="${EVAL_GAMES:-400}"
echo "host=$(hostname) tag=$TAG embedder=$EMBEDDER rl_steps=$RL_STEPS"
"$PY" sweep/phase5_compare.py
