#!/bin/bash
#SBATCH --job-name=gin-h2h
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/h2h_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/h2h_%j.err

# Head-to-head: a trained model vs the (determinized) ISMCTS search agent.
#   sbatch --export=ALL,MODEL_PATH=game/model/gin_ace.zip,MODEL_LABEL=ace,ISMCTS_ROLLOUTS=60,N=300 slurm/model_vs_ismcts.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}" WANDB_ENTITY="${WANDB_ENTITY:-VLAvengers}"
export WANDB_GROUP="${WANDB_GROUP:-phase8-head2head}"

export MODEL_PATH="${MODEL_PATH:-game/model/gin_ace.zip}" MODEL_LABEL="${MODEL_LABEL:-ace}"
export ISMCTS_ROLLOUTS="${ISMCTS_ROLLOUTS:-60}" ISMCTS_DETERMINIZE="${ISMCTS_DETERMINIZE:-1}" N="${N:-300}"
echo "host=$(hostname) model=$MODEL_LABEL rollouts=$ISMCTS_ROLLOUTS det=$ISMCTS_DETERMINIZE N=$N"
"$PY" sweep/model_vs_ismcts.py
