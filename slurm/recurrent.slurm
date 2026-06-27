#!/bin/bash
#SBATCH --job-name=gin-rec
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/rec_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/rec_%j.err

# Phase-8 recurrence side-study (ISOLATED). One job = one (ARCH, SEED). Trains a masked PPO-LSTM
# (ARCH=lstm) or its matched PPO-MLP control (ARCH=mlp) vs the frozen champion, grades vs gold/
# champion/random, writes sweep/curriculum/rec_*.json. Separate from the running arch sweep.
#
#   for a in lstm mlp; do for s in 0 1; do
#     sbatch --export=ALL,ARCH=$a,SEED=$s,WANDB_GROUP=phase8-recurrent slurm/recurrent.slurm
#   done; done

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
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"
export WANDB_GROUP="${WANDB_GROUP:-phase8-recurrent}"
export ARCH="${ARCH:-lstm}" SEED="${SEED:-0}"
export STEPS="${STEPS:-4000000}" NUM_ENV="${NUM_ENV:-8}" EVAL_GAMES="${EVAL_GAMES:-600}"

echo "host=$(hostname) arch=$ARCH seed=$SEED steps=$STEPS group=$WANDB_GROUP"
"$PY" sweep/recurrent_one.py
