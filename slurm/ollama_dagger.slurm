#!/bin/bash
#SBATCH --job-name=gin-olladag
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/olladag_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/olladag_%j.err

# Phase-4 LLM-DAgger on OLLAMA (single GPU, quantized, fast). Self-contained: runs
# `ollama serve` on the GPU node and points the dagger client straight at it
# (llm/api.py speaks the Ollama API natively, so NO master/worker stack). The
# gold-standard agent only scores the result.
#
#   sbatch slurm/ollama_dagger.slurm
#   sbatch --export=ALL,LLM_MODEL=gpt-oss:20b,BC_STATES=8000 slurm/ollama_dagger.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution

source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT="${WANDB_PROJECT:-Adversarial-CoEvolution}"

# ---- Ollama (quantized, on the one GPU) ----
export OLLAMA_HOME=/scratch1/kelidari/ollama
export OLLAMA_MODELS=/scratch1/kelidari/ollama/models
export OLLAMA_TMPDIR=/scratch1/kelidari/ollama/tmp
export PATH="$OLLAMA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
# Throughput tuning: batch concurrent requests + a small context (our prompts are
# short). The default 32k context + NUM_PARALLEL=1 made bulk collection crawl.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OPORT=$(( 11434 + (SLURM_JOB_ID % 1000) ))
# Default to a FAST non-reasoning model: gpt-oss:20b reasoning is 2-5 min/call on a
# V100 (too slow for thousands of labels). qwen2.5:7b emits the action directly.
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"

OLLAMA_HOST=127.0.0.1:$OPORT ollama serve > "slurm/logs/olladag_serve_${SLURM_JOB_ID}.log" 2>&1 &
SV=$!
trap 'kill $SV 2>/dev/null || true' EXIT

echo "waiting for ollama serve on :$OPORT ..."
up=""
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$OPORT/api/version" >/dev/null 2>&1 && { up=1; break; }
  sleep 2
done
[ -n "$up" ] || { echo "ERROR: ollama serve did not start"; tail -20 "slurm/logs/olladag_serve_${SLURM_JOB_ID}.log"; exit 1; }

echo "preloading $LLM_MODEL onto the GPU ..."
OLLAMA_HOST=127.0.0.1:$OPORT ollama run "$LLM_MODEL" "Reply with: ok" >/dev/null 2>&1 || true
# verify a real generation works before committing to collection
TEST=$(curl -s -m 120 -X POST "http://127.0.0.1:$OPORT/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"prompt\":\"Reply with exactly: ok\",\"stream\":false}" 2>/dev/null)
echo "test generation: $(echo "$TEST" | head -c 160)"
echo "$TEST" | grep -q '"response"' || { echo "ERROR: $LLM_MODEL not serving"; exit 1; }

export GINLLM_MASTER_URL="http://127.0.0.1:$OPORT"
export LLM_MODEL
export BC_STATES="${BC_STATES:-8000}"
export BC_EPOCHS="${BC_EPOCHS:-8}"
export RL_STEPS="${RL_STEPS:-2000000}"
export NUM_ENV="${NUM_ENV:-32}"
export N_STEPS="${N_STEPS:-256}"
export COLLECT_THREADS="${COLLECT_THREADS:-24}"
export EVAL_GAMES="${EVAL_GAMES:-400}"
export INIT_MODEL="${INIT_MODEL:-}"
export SEED="${SEED:-0}"
export SAVE_PATH="${SAVE_PATH:-game/model/ppo_gin_rummy_dagger_ollama.zip}"
export WANDB_NAME="${WANDB_NAME:-dagger_ollama_${LLM_MODEL//[:\/]/_}}"

echo "host=$(hostname) ollama=$GINLLM_MASTER_URL model=$LLM_MODEL bc_states=$BC_STATES rl_steps=$RL_STEPS"
"$PY" sweep/llm_dagger_one.py
