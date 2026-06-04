#!/bin/bash
#SBATCH --job-name=gin-algo
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/algo_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/algo_%j.err

# Algorithm comparison (CPU): PPO vs TRPO under identical masked self-play.
#   sbatch --export=ALL,ALGO=ppo,TAG=ppo_s0,SEED=0  slurm/algo.slurm
#   sbatch --export=ALL,ALGO=trpo,TAG=trpo_s0,SEED=0 slurm/algo.slurm

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

export ALGO="${ALGO:-ppo}"
export TAG="${TAG:-${ALGO}_s${SEED:-0}}"
export RL_STEPS="${RL_STEPS:-2000000}"
export NUM_ENV="${NUM_ENV:-64}"
export EVAL_GAMES="${EVAL_GAMES:-400}"
export SEED="${SEED:-0}"
echo "host=$(hostname) algo=$ALGO tag=$TAG seed=$SEED rl_steps=$RL_STEPS"
"$PY" sweep/algo_compare.py
