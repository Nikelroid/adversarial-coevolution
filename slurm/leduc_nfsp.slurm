#!/bin/bash
#SBATCH --job-name=gin-nfsp
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/nfsp_%A_%a.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/nfsp_%A_%a.err

# Phase-8 Wave 4: NFSP (Heinrich & Silver 2016) on Leduc Hold'em, graded vs the CFR-optimal expert.
# The canonical imperfect-information learning baseline. SEED = array task id (multi-seed for CIs).
#   sbatch --array=0-3 --export=ALL,EPISODES=600000,GRADE_GAMES=2000 slurm/leduc_nfsp.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# W&B (group phase8-leduc): key from env or ~/.bashrc WANDB_TOKEN; logging is fully guarded.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"
export WANDB_GROUP="${WANDB_GROUP:-phase8-leduc}"

export SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export EPISODES="${EPISODES:-600000}" CFR_ITERS="${CFR_ITERS:-400}" GRADE_GAMES="${GRADE_GAMES:-2000}"
echo "host=$(hostname) seed=$SEED episodes=$EPISODES grade_games=$GRADE_GAMES"
"$PY" sweep/leduc_nfsp.py
