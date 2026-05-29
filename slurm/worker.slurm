#!/bin/bash
#SBATCH --job-name=ginllm-worker
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-15%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/worker_%A_%a.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/worker_%A_%a.err

# GPU worker pool. Each array task loads one OLMoE on one V100 and registers in
# the shared-FS registry; the master discovers and load-balances over them.
# Resize the fleet with --array (default 0-15, max 16 concurrent via %16).
# Qwen3-30B-A3B needs 2 GPUs/task: submit with
#   sbatch --gres=gpu:v100:2 --export=WORKER_PRESET=qwen3-30b slurm/worker.slurm

set -uo pipefail

cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
# Do NOT set PYTHONNOUSERSITE=1: vllm/transformers/fastapi live in ~/.local
# (user site); disabling it breaks the worker import.
export TOKENIZERS_PARALLELISM=false
# Load weights from scratch (BeeGFS): home NFS reads a 7B at ~11 MB/s (~28 min)
# and times out; scratch loads it in ~30 s. Pre-stage with HF_HOME=... snapshot_download.
export HF_HOME=/scratch1/kelidari/hf_cache
export HF_HUB_OFFLINE=1

PRESET="${WORKER_PRESET:-olmoe}"
MAXTOK="${WORKER_MAXTOK:-1024}"   # cap generation; set low (e.g. 64) for fast training opponents
# Unique port per array task: two tasks can share a 2-GPU node, so they must not
# both bind the same port. The worker registers (host, port) so the master finds
# whichever node/port it actually landed on.
PORT=$(( ${WORKER_PORT_BASE:-8001} + SLURM_ARRAY_TASK_ID ))

echo "=========================================================="
echo "host=$(hostname)  job=${SLURM_JOB_ID}  array=${SLURM_ARRAY_TASK_ID}"
echo "preset=$PRESET  port=$PORT  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "python=$("$PY" --version 2>&1)"
echo "=========================================================="

"$PY" -m llm.worker --preset "$PRESET" --host 0.0.0.0 --port "$PORT" --max-output-tokens "$MAXTOK"
