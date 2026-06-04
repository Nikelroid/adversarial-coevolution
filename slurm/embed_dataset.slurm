#!/bin/bash
#SBATCH --job-name=gin-embed
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/embed_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/embed_%j.err

# Build the LLM-judged state-similarity dataset on an L40S (native FP4 -> gpt-oss
# is fast there). Self-contained: ollama serve + embed_dataset.py.
#   sbatch slurm/embed_dataset.slurm
#   sbatch --export=ALL,N_PAIRS=3000,LLM_MODEL=gpt-oss:20b slurm/embed_dataset.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

export OLLAMA_HOME=/scratch1/kelidari/ollama
export OLLAMA_MODELS=/scratch1/kelidari/ollama/models
export OLLAMA_TMPDIR=/scratch1/kelidari/ollama/tmp
export PATH="$OLLAMA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-4096}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OPORT=$(( 11434 + (SLURM_JOB_ID % 1000) ))
LLM_MODEL="${LLM_MODEL:-gpt-oss:20b}"

OLLAMA_HOST=127.0.0.1:$OPORT ollama serve > "slurm/logs/embed_serve_${SLURM_JOB_ID}.log" 2>&1 &
SV=$!; trap 'kill $SV 2>/dev/null || true' EXIT
up=""
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$OPORT/api/version" >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
[ -n "$up" ] || { echo "ERROR: ollama serve down"; tail -20 "slurm/logs/embed_serve_${SLURM_JOB_ID}.log"; exit 1; }
OLLAMA_HOST=127.0.0.1:$OPORT ollama run "$LLM_MODEL" "ok" >/dev/null 2>&1 || true
TEST=$(curl -s -m 120 -X POST "http://127.0.0.1:$OPORT/api/generate" -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"prompt\":\"Reply with: 3\",\"stream\":false}" 2>/dev/null)
echo "test: $(echo "$TEST" | head -c 120)"
echo "$TEST" | grep -q '"response"' || { echo "ERROR: $LLM_MODEL not serving"; exit 1; }

export GINLLM_MASTER_URL="http://127.0.0.1:$OPORT"
export LLM_MODEL
export N_STATES="${N_STATES:-5000}"
export N_PAIRS="${N_PAIRS:-20000}"
export NEAR_FRAC="${NEAR_FRAC:-0.35}"
export COLLECT_THREADS="${COLLECT_THREADS:-24}"
# Big dataset lives on SCRATCH (home1 is tight). Shards: pass OUT + SEED per job.
export OUT="${OUT:-/scratch1/kelidari/advcoev_store/embed/sim_dataset.npz}"
export TIME_BUDGET="${TIME_BUDGET:-39600}"   # collect ~11h, leave margin in a 12h job
mkdir -p "$(dirname "$OUT")"
DS="$OUT"
echo "host=$(hostname) model=$LLM_MODEL n_pairs=$N_PAIRS out=$DS"
"$PY" sweep/embed_dataset.py

# Train the embedder on the dataset we just built, in the SAME job, so the val
# correlation lands autonomously even if the interactive session has ended.
if [ -f "$DS" ]; then
  echo "=== training embedder on $DS ==="
  DATASET="$DS" OUT="${DS%.npz}_embedder.pt" EMB_DIM="${EMB_DIM:-20}" \
    EPOCHS="${EPOCHS:-80}" "$PY" sweep/embed_train.py
fi
