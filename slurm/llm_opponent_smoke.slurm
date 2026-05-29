#!/bin/bash
#SBATCH --job-name=ginllm-oppsmoke
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/oppsmoke_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/oppsmoke_%j.err

# Verify the full Phase-2 RL<->LLM path on one GPU node: master (router) + one
# HF OLMoE worker + an LLM-opponent Gin Rummy run, with an isolated registry.

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python   # serving stack is in ~/.local (user site)

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PRESET="${MASTER_PRESET:-olmoe}"
MPORT=11434
WPORT=8001
export GINLLM_RUNTIME_DIR="$(mktemp -d)"
export GINLLM_MASTER_URL="http://127.0.0.1:${MPORT}"
MLOG="slurm/logs/oppsmoke_master_${SLURM_JOB_ID}.log"
WLOG="slurm/logs/oppsmoke_worker_${SLURM_JOB_ID}.log"

echo "host=$(hostname) preset=$PRESET runtime=$GINLLM_RUNTIME_DIR"

"$PY" -m llm.master --host 127.0.0.1 --port "$MPORT" --scan-interval 2 > "$MLOG" 2>&1 &
MA=$!
"$PY" -m llm.worker --preset "$PRESET" --host 0.0.0.0 --port "$WPORT" > "$WLOG" 2>&1 &
WK=$!
trap 'kill $MA $WK 2>/dev/null || true' EXIT

echo "waiting for a healthy worker (OLMoE is cached, so ~1-3 min to load)..."
OK=""
for i in $(seq 1 240); do
    H=$(curl -sf "http://127.0.0.1:${MPORT}/health" 2>/dev/null || true)
    if echo "$H" | grep -q '"n_healthy":1'; then OK=1; echo "healthy after ~$((i*2))s"; break; fi
    kill -0 $MA 2>/dev/null || { echo "ERROR: master died"; tail -n 40 "$MLOG"; exit 1; }
    kill -0 $WK 2>/dev/null || { echo "ERROR: worker died"; tail -n 40 "$WLOG"; exit 1; }
    sleep 2
done
[ -n "$OK" ] || { echo "ERROR: no healthy worker in time"; tail -n 40 "$WLOG"; exit 1; }

echo "=== LLM-opponent Gin Rummy run ==="
N_EP="${N_EP:-3}" "$PY" -m llm.llm_opponent_smoke

echo "=== worker log tail ==="; tail -n 12 "$WLOG"
echo "=== done ==="
