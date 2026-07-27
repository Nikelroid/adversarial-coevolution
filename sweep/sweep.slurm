#!/bin/bash
#SBATCH --job-name=ginrummy-sweep
#SBATCH --account=liu32_1378
#SBATCH --partition=epyc-64
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/run_%a_%A.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/sweep/logs/run_%a_%A.err

set -euo pipefail

cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# W&B: leave unset to enable when WANDB_API_KEY is exported in your shell;
# uncomment the next line to fully disable W&B for the array.
# export WANDB_DISABLED=true
# limit per-process BLAS threads so 96 env subprocesses don't fight over cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)  task=$SLURM_ARRAY_TASK_ID  cpus=$SLURM_CPUS_PER_TASK"
python sweep/run_one.py --combo "$SLURM_ARRAY_TASK_ID"
