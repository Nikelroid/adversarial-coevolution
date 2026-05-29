#!/bin/bash
#SBATCH --job-name=ginllm-eval
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/llmeval_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/llmeval_%j.err

# Measure LLM-opponent STRENGTH on one GPU node: master (router) + one HF worker
# + N full Gin Rummy games with the LLM as opponent. Reports its win-rate.
#   sbatch --export=ALL,MASTER_PRESET=qwen2.5-7b,N_EP=16,HERO=random slurm/llm_eval.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python   # serving stack is in ~/.local (user site)

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1                       # model is pre-cached; don't hit the network
# Load weights from scratch (BeeGFS) -- home NFS reads the 7B at ~11 MB/s (~28 min);
# scratch is far faster. Pre-stage there with snapshot_download (HF_HOME=...).
export HF_HOME=/scratch1/kelidari/hf_cache
PRESET="${MASTER_PRESET:-qwen2.5-7b}"
MPORT=11434
WPORT=8001
export GINLLM_RUNTIME_DIR="$(mktemp -d)"
export GINLLM_MASTER_URL="http://127.0.0.1:${MPORT}"
MLOG="slurm/logs/llmeval_master_${SLURM_JOB_ID}.log"
WLOG="slurm/logs/llmeval_worker_${SLURM_JOB_ID}.log"

echo "host=$(hostname) preset=$PRESET hero=${HERO:-random} n_ep=${N_EP:-16} runtime=$GINLLM_RUNTIME_DIR"

"$PY" -m llm.master --host 127.0.0.1 --port "$MPORT" --scan-interval 2 > "$MLOG" 2>&1 &
MA=$!
"$PY" -m llm.worker --preset "$PRESET" --host 0.0.0.0 --port "$WPORT" > "$WLOG" 2>&1 &
WK=$!
trap 'kill $MA $WK 2>/dev/null || true' EXIT

echo "waiting for a healthy worker (7B load from scratch, allow generous time)..."
OK=""
for i in $(seq 1 600); do
    H=$(curl -sf "http://127.0.0.1:${MPORT}/health" 2>/dev/null || true)
    if echo "$H" | grep -q '"n_healthy":1'; then OK=1; echo "healthy after ~$((i*2))s"; break; fi
    kill -0 $MA 2>/dev/null || { echo "ERROR: master died"; tail -n 40 "$MLOG"; exit 1; }
    kill -0 $WK 2>/dev/null || { echo "ERROR: worker died"; tail -n 40 "$WLOG"; exit 1; }
    sleep 2
done
[ -n "$OK" ] || { echo "ERROR: no healthy worker in time"; tail -n 40 "$WLOG"; exit 1; }

echo "=== LLM-opponent strength eval (preset=$PRESET) ==="
N_EP="${N_EP:-16}" HERO="${HERO:-random}" "$PY" -m llm.eval_opponent

echo "=== worker log tail ==="; tail -n 15 "$WLOG"
echo "=== done ==="
