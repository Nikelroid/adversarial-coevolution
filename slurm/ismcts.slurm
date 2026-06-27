#!/bin/bash
#SBATCH --job-name=gin-ismcts
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ismcts_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ismcts_%j.err

# Grade the PIMC/ISMCTS search baseline vs gold / random / champion (no training, CPU search).
# Rollout-budget sweep:
#   for r in 10 30 60 120; do
#     sbatch --export=ALL,OPP=gold,N=300,ISMCTS_ROLLOUTS=$r,NAME=ismcts_vs_gold_r$r slurm/ismcts.slurm; done

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# W&B (group phase8-ismcts): key from env or ~/.bashrc WANDB_TOKEN; logging is fully guarded.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"
export WANDB_GROUP="${WANDB_GROUP:-phase8-ismcts}"

export OPP="${OPP:-gold}" N="${N:-300}" ISMCTS_ROLLOUTS="${ISMCTS_ROLLOUTS:-20}"
export NAME="${NAME:-ismcts_vs_${OPP}_r${ISMCTS_ROLLOUTS}}"
echo "host=$(hostname) opp=$OPP N=$N rollouts=$ISMCTS_ROLLOUTS name=$NAME"
"$PY" sweep/ismcts_eval.py
