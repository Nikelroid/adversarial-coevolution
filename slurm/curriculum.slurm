#!/bin/bash
#SBATCH --job-name=gin-curr
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/curr_%A_%a.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/curr_%A_%a.err

# Phase-6 curriculum/algorithm/reward sweep, as a SLURM ARRAY (one cell per task).
# Each task reads its config name from sweep/curriculum_cfgs/INDEX.txt by array index.
#
#   python sweep/curriculum_configs.py                       # (once) emit configs + INDEX.txt
#   N=$(wc -l < sweep/curriculum_cfgs/INDEX.txt)
#   sbatch --array=0-$((N-1)) --partition=oneweek --cpus-per-task=16 \
#          --export=ALL,NUM_ENV=16 slurm/curriculum.slurm
#
# Smoke a single cell fast (tiny steps/envs):
#   sbatch --array=0 --export=ALL,NUM_ENV=4,STEPS=8000,EVAL_EVERY=4000,\
#          CKPT_EVAL_GAMES=10,FINAL_EVAL_GAMES=10,SAVE_FREQ=3000 slurm/curriculum.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# W&B tracking (group phase6-curriculum). Pull the key from env or ~/.bashrc WANDB_TOKEN, like
# the other CPU jobs. Logging is fully guarded in the harness, so a missing key just disables it.
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"

IDX_FILE=sweep/curriculum_cfgs/INDEX.txt
NAME=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0}+1))p" "$IDX_FILE")
if [ -z "$NAME" ]; then echo "no config for array index ${SLURM_ARRAY_TASK_ID:-0}"; exit 1; fi
export CFG="sweep/curriculum_cfgs/${NAME}.json"

# optional overrides (smoke / tuning) flow through env -> _load_config()
export NUM_ENV="${NUM_ENV:-16}"
[ -n "${STEPS:-}" ] && export STEPS
[ -n "${EVAL_EVERY:-}" ] && export EVAL_EVERY
[ -n "${CKPT_EVAL_GAMES:-}" ] && export CKPT_EVAL_GAMES
[ -n "${FINAL_EVAL_GAMES:-}" ] && export FINAL_EVAL_GAMES
[ -n "${SAVE_FREQ:-}" ] && export SAVE_FREQ

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} name=$NAME cfg=$CFG num_env=$NUM_ENV"
"$PY" sweep/curriculum_train.py
