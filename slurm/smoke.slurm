#!/bin/bash
#SBATCH --job-name=ginllm-smoke
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/smoke_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/smoke_%j.err

# Full-stack single-node check: real master (router) + ONE real vLLM worker +
# client smoke, all on one GPU node, with an isolated registry. Proves the live
# path end to end (discovery -> health -> route -> vLLM generate -> cache hit)
# before going multi-node.

set -uo pipefail

cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
# Do NOT set PYTHONNOUSERSITE=1: the vllm/fastapi/httpx serving stack lives in
# ~/.local (user site); disabling it breaks the master/worker imports.
export TOKENIZERS_PARALLELISM=false

PRESET="${MASTER_PRESET:-olmoe}"
MPORT=11434
WPORT=8001
# Isolate this smoke's registry so it never collides with a real running stack.
export GINLLM_RUNTIME_DIR="$(mktemp -d)"
MLOG="slurm/logs/smoke_master_${SLURM_JOB_ID}.log"
WLOG="slurm/logs/smoke_worker_${SLURM_JOB_ID}.log"

echo "host=$(hostname)  preset=$PRESET  runtime=$GINLLM_RUNTIME_DIR"

# Worker binds 0.0.0.0 (master reaches it via the registered hostname); master
# on 127.0.0.1 is enough since only the local client talks to it.
"$PY" -m llm.master --host 127.0.0.1 --port "$MPORT" --scan-interval 2 \
    > "$MLOG" 2>&1 &
MA=$!
"$PY" -m llm.worker --preset "$PRESET" --host 0.0.0.0 --port "$WPORT" \
    > "$WLOG" 2>&1 &
WK=$!
trap 'kill $MA $WK 2>/dev/null || true' EXIT

echo "waiting for the master to report a healthy worker (model load ~1-3 min)..."
OK=""
for i in $(seq 1 180); do
    H=$(curl -sf "http://127.0.0.1:${MPORT}/health" 2>/dev/null || true)
    if echo "$H" | grep -q '"n_healthy":1'; then OK=1; echo "healthy after ~$((i*2))s"; break; fi
    kill -0 $MA 2>/dev/null || { echo "ERROR: master died"; tail -n 40 "$MLOG"; exit 1; }
    kill -0 $WK 2>/dev/null || { echo "ERROR: worker died"; tail -n 40 "$WLOG"; exit 1; }
    sleep 2
done
if [ -z "$OK" ]; then
    echo "ERROR: no healthy worker in time"; tail -n 40 "$WLOG"; exit 1
fi

echo "=== client smoke (cold miss vs warm hit) ==="
"$PY" -m llm.master_smoke --url "http://127.0.0.1:${MPORT}" --model "$PRESET"

echo "=== /stats ==="
curl -sf "http://127.0.0.1:${MPORT}/stats"; echo
echo "=== worker log tail ==="
tail -n 15 "$WLOG"
echo "=== smoke complete ==="
