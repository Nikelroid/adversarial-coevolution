#!/bin/bash
#SBATCH --job-name=gin-ppoarch
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ppoarch_%A_%a.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ppoarch_%A_%a.err

# Attention (and a matched MLP anchor) under PPO. Attention is incompatible with the TRPO
# recipe (second-order HVP over a transformer: ~20 steps/s on CPU, and an autograd LayerNorm
# double-backward crash on CUDA), so we grade it under PPO, where it is transformer-safe and
# fast (~270 steps/s at OMP=8/16-env -> ~10h for 10M). Unlike curriculum.slurm this sets
# OMP_NUM_THREADS=8 so the transformer matmuls are multi-threaded.
#
#   sbatch --array=0-3 --export=ALL,CFG_DIR=ppoarch_cfgs,NUM_ENV=16,WANDB_GROUP=phase8-arch slurm/ppoarch.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

# W&B (same pattern as curriculum.slurm): key from env or ~/.bashrc WANDB_TOKEN; guarded in-harness.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"
export WANDB_GROUP="${WANDB_GROUP:-phase8-arch}"

CFG_DIR="${CFG_DIR:-ppoarch_cfgs}"
IDX_FILE="sweep/${CFG_DIR}/INDEX.txt"
NAME=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0}+1))p" "$IDX_FILE")
if [ -z "$NAME" ]; then echo "no config for array index ${SLURM_ARRAY_TASK_ID:-0} in $IDX_FILE"; exit 1; fi
export CFG="sweep/${CFG_DIR}/${NAME}.json"

export NUM_ENV="${NUM_ENV:-16}"
[ -n "${STEPS:-}" ] && export STEPS
[ -n "${EVAL_EVERY:-}" ] && export EVAL_EVERY
[ -n "${CKPT_EVAL_GAMES:-}" ] && export CKPT_EVAL_GAMES
[ -n "${FINAL_EVAL_GAMES:-}" ] && export FINAL_EVAL_GAMES
[ -n "${SAVE_FREQ:-}" ] && export SAVE_FREQ

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} name=$NAME cfg=$CFG num_env=$NUM_ENV omp=$OMP_NUM_THREADS"
"$PY" sweep/curriculum_train.py
