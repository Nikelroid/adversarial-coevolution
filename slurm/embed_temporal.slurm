#!/bin/bash
#SBATCH --job-name=gin-tempemb
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/tempemb_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/tempemb_%j.err

# Phase-4(alt2): self-supervised TEMPORAL-context embedding (word2vec-style), CPU only,
# no LLM -> no GPU wait. Builds the dataset then trains the embedder, autonomously.

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY

export N_GAMES="${N_GAMES:-6000}"
export N_PAIRS="${N_PAIRS:-200000}"
export WINDOW="${WINDOW:-4}"
export OUT="${OUT:-/scratch1/kelidari/advcoev_store/embed/temporal_dataset.npz}"
mkdir -p "$(dirname "$OUT")"
DS="$OUT"
echo "host=$(hostname) n_games=$N_GAMES n_pairs=$N_PAIRS out=$DS"
"$PY" sweep/embed_temporal.py

if [ -f "$DS" ]; then
  echo "=== training temporal embedder on $DS ==="
  DATASET="$DS" OUT="${DS%.npz}_embedder.pt" EMB_DIM="${EMB_DIM:-20}" \
    EPOCHS="${EPOCHS:-80}" "$PY" sweep/embed_train.py
fi
