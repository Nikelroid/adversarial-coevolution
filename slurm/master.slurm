#!/bin/bash
#SBATCH --job-name=ginllm-master
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/master_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/master_%j.err

# CPU-only router: holds the cache + queue and load-balances over the GPU
# worker array. No GPU. It writes its address to runtime/master.json so the
# training session and the workers find it through the shared filesystem.

set -uo pipefail

cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
# `conda activate` is unreliable in pristine batch shells; call python directly.
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
# Do NOT set PYTHONNOUSERSITE=1: the fastapi/httpx serving stack lives in
# ~/.local (user site), unlike the RL stack which is in the coev env. Disabling
# user site (as sweep.slurm does) makes the master fail to import httpx/fastapi.

PORT="${MASTER_PORT:-11434}"

echo "=========================================================="
echo "host=$(hostname)  job=$SLURM_JOB_ID  port=$PORT"
echo "python=$("$PY" --version 2>&1)"
echo "RL clients: read runtime/master.json, or  http://$(hostname):$PORT"
echo "=========================================================="

"$PY" -m llm.master --host 0.0.0.0 --port "$PORT"
