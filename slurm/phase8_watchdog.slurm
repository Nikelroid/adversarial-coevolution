#!/bin/bash
#SBATCH --job-name=gin-p8-wd
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/p8wd_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/p8wd_%j.err

# Phase-8 overnight babysitter for the NEW waves (Stage-C, Leduc, NFSP, ISMCTS). Every CHECK_EVERY
# seconds it: (1) for each wave whose SLURM array has fully DRAINED while results are still missing,
# resubmits just the missing pieces (capped, so it never loops forever); (2) refreshes the local
# collected summary (sweep/curriculum/_phase8_summary.{md,json}). It does NOT git-push (the arch
# watchdog owns publishing -> no git races). Self-resubmits before its own wall limit.
#
#   sbatch --export=ALL,NLEDUC=8,NNFSP=4,ISM_ROLLOUTS="10 30 60 120" slurm/phase8_watchdog.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

# W&B key for any resubmits (same extraction as the wave wrappers).
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
if [ -z "$WANDB_API_KEY" ]; then
  line=$(grep -E '\bWANDB_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; WANDB_API_KEY=${v%% *}
fi
export WANDB_API_KEY
export WANDB_PROJECT=Adversarial-CoEvolution WANDB_ENTITY=VLAvengers

RES=sweep/curriculum
CHECK_EVERY="${CHECK_EVERY:-1200}"
NLEDUC="${NLEDUC:-8}"; NNFSP="${NNFSP:-4}"; ISM_ROLLOUTS="${ISM_ROLLOUTS:-10 30 60 120}"
LEDUC_EP="${LEDUC_EP:-500000}"; NFSP_EP="${NFSP_EP:-600000}"; ISM_N="${ISM_N:-300}"
SC_PART="${SC_PART:-main}"; SC_CPUS="${SC_CPUS:-16}"
START=$(date +%s); WALL=$((42*3600))           # self-resubmit ~6h before the 2-day limit
ATT="sweep/.p8wd_attempts"; mkdir -p "$ATT"; MAX_TRIES="${MAX_TRIES:-2}"

jobs_active() { squeue -u "$USER" -h -r -n "$1" -o "%T" 2>/dev/null | grep -qiE 'RUNNING|PENDING'; }
tries() { local f="$ATT/$1"; local n=0; [ -f "$f" ] && n=$(cat "$f"); echo "$n"; }
bump()  { local f="$ATT/$1"; echo $(( $(tries "$1") + 1 )) > "$f"; }

while true; do
  ts=$(date -u +%FT%H:%MZ)

  # ---- Stage-C training cells (curriculum.slurm, CFG_DIR=stagec_cfgs) ----
  if [ -f sweep/stagec_cfgs/INDEX.txt ] && ! jobs_active gin-stagec; then
    mapfile -t cells < sweep/stagec_cfgs/INDEX.txt
    miss=()
    for i in "${!cells[@]}"; do
      c="${cells[$i]}"; [ -z "$c" ] && continue
      if [ ! -f "$RES/${c}.json" ] && [ "$(tries sc_$c)" -lt "$MAX_TRIES" ]; then miss+=("$i"); bump "sc_$c"; fi
    done
    if [ "${#miss[@]}" -gt 0 ]; then
      list=$(IFS=,; echo "${miss[*]}")
      sbatch --array="$list" --job-name=gin-stagec --partition="$SC_PART" --cpus-per-task="$SC_CPUS" \
        --export=ALL,NUM_ENV="$SC_CPUS",CFG_DIR=stagec_cfgs,WANDB_GROUP=phase8-stagec slurm/curriculum.slurm \
        >/dev/null && echo "[p8wd $ts] resubmitted Stage-C idx {$list}"
    fi
  fi

  # ---- Leduc tabular-Q seeds ----
  if ! jobs_active gin-leduc; then
    miss=()
    for s in $(seq 0 $((NLEDUC-1))); do
      [ -f "$RES/leduc_rl_vs_cfr_s${s}.json" ] || { [ "$(tries leduc_$s)" -lt "$MAX_TRIES" ] && { miss+=("$s"); bump "leduc_$s"; }; }
    done
    for s in "${miss[@]}"; do
      sbatch --job-name=gin-leduc --export=ALL,SEED="$s",EPISODES="$LEDUC_EP",GRADE_GAMES=2000 slurm/leduc.slurm \
        >/dev/null && echo "[p8wd $ts] resubmitted Leduc seed $s"
    done
  fi

  # ---- NFSP seeds ----
  if ! jobs_active gin-nfsp; then
    miss=()
    for s in $(seq 0 $((NNFSP-1))); do
      [ -f "$RES/leduc_nfsp_vs_cfr_s${s}.json" ] || { [ "$(tries nfsp_$s)" -lt "$MAX_TRIES" ] && { miss+=("$s"); bump "nfsp_$s"; }; }
    done
    for s in "${miss[@]}"; do
      sbatch --job-name=gin-nfsp --export=ALL,SEED="$s",EPISODES="$NFSP_EP",GRADE_GAMES=2000 slurm/leduc_nfsp.slurm \
        >/dev/null && echo "[p8wd $ts] resubmitted NFSP seed $s"
    done
  fi

  # ---- ISMCTS rollout budgets vs gold ----
  if ! jobs_active gin-ismcts; then
    for r in $ISM_ROLLOUTS; do
      n="ismcts_vs_gold_r${r}"
      if [ ! -f "$RES/${n}.json" ] && [ "$(tries $n)" -lt "$MAX_TRIES" ]; then
        bump "$n"
        sbatch --job-name=gin-ismcts \
          --export=ALL,OPP=gold,N="$ISM_N",ISMCTS_ROLLOUTS="$r",NAME="$n",WANDB_GROUP=phase8-ismcts slurm/ismcts.slurm \
          >/dev/null && echo "[p8wd $ts] resubmitted ISMCTS r$r"
      fi
    done
  fi

  # ---- refresh local summary (no git push) ----
  PYTHONNOUSERSITE=1 "$PY" sweep/collect_phase8.py >/dev/null 2>&1 || true

  # ---- done? (all expected results present) ----
  need=0; have=0
  if [ -f sweep/stagec_cfgs/INDEX.txt ]; then
    while IFS= read -r c; do [ -z "$c" ] && continue; need=$((need+1)); [ -f "$RES/${c}.json" ] && have=$((have+1)); done < sweep/stagec_cfgs/INDEX.txt
  fi
  for s in $(seq 0 $((NLEDUC-1))); do need=$((need+1)); [ -f "$RES/leduc_rl_vs_cfr_s${s}.json" ] && have=$((have+1)); done
  for s in $(seq 0 $((NNFSP-1))); do need=$((need+1)); [ -f "$RES/leduc_nfsp_vs_cfr_s${s}.json" ] && have=$((have+1)); done
  for r in $ISM_ROLLOUTS; do need=$((need+1)); [ -f "$RES/ismcts_vs_gold_r${r}.json" ] && have=$((have+1)); done
  echo "[p8wd $ts] $have/$need new-wave results present"
  [ "$have" -ge "$need" ] && { echo "[p8wd $ts] all new-wave results in -> exit"; break; }

  # ---- self-resubmit before wall limit ----
  if [ $(( $(date +%s) - START )) -gt "$WALL" ]; then
    echo "[p8wd $ts] nearing wall -> self-resubmit"
    sbatch --export=ALL,NLEDUC="$NLEDUC",NNFSP="$NNFSP",ISM_ROLLOUTS="$ISM_ROLLOUTS" slurm/phase8_watchdog.slurm >/dev/null || true
    break
  fi
  sleep "$CHECK_EVERY"
done
