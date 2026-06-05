#!/bin/bash
#SBATCH --job-name=gin-curr-wd
#SBATCH --account=liu32_1378
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/curr_wd_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/curr_wd_%j.err

# Babysits the Phase-6 sweep so it runs to completion with NO human in the loop -- it survives
# this chat session ending. Every CHECK_EVERY seconds it:
#   - counts which cells (sweep/curriculum_cfgs/INDEX.txt) still lack a result JSON
#   - if the array has DRAINED (no gin-curr jobs queued/running) but cells are missing, those
#     cells failed/timed-out -> it resubmits just those (capped at MAX_TRIES each; after that it
#     marks them given-up so it never loops forever)
#   - publishes a partial update (aggregate -> figures -> report -> push) whenever new cells land
#   - when every cell has a result-or-giveup, runs the final publish and exits
#   - self-resubmits a fresh watchdog before nearing its own 7-day wall limit
#
#   sbatch --export=ALL,PART=oneweek,CPUS=16 slurm/curriculum_watchdog.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev

IDX=sweep/curriculum_cfgs/INDEX.txt
RES=sweep/curriculum
ATT=sweep/curriculum_cfgs/.wd_attempts          # per-cell resubmit counters / giveup markers
PART="${PART:-oneweek}"
CPUS="${CPUS:-16}"
CHECK_EVERY="${CHECK_EVERY:-1200}"
MAX_TRIES="${MAX_TRIES:-3}"
START=$(date +%s)
WALL=$((6*24*3600))                              # self-resubmit ~1 day before the 7-day limit
mkdir -p "$ATT"
last_done=-1

resubmit_one() {                                 # $1 = cell name; resubmit its array index
  local name="$1" cf="$ATT/$1"
  local n=0; [ -f "$cf" ] && n=$(cat "$cf")
  if [ "$n" -ge "$MAX_TRIES" ]; then
     touch "$ATT/$name.giveup"; echo "  !! $name gave up after $n tries"; return
  fi
  local ln; ln=$(grep -nxF "$name" "$IDX" | head -1 | cut -d: -f1)
  [ -z "$ln" ] && { echo "  ?? $name not in INDEX"; return; }
  echo $((n+1)) > "$cf"
  sbatch --array=$((ln-1)) --partition="$PART" --cpus-per-task="$CPUS" \
         --export=ALL,NUM_ENV="$CPUS" slurm/curriculum.slurm >/dev/null \
    && echo "  resubmitted $name (idx $((ln-1)), try $((n+1)))"
}

while true; do
  total=$(grep -cve '^[[:space:]]*$' "$IDX")
  missing=(); pending=()
  while IFS= read -r name; do
     [ -z "$name" ] && continue
     if [ ! -f "$RES/${name}.json" ]; then
        missing+=("$name")
        [ -f "$ATT/$name.giveup" ] || pending+=("$name")
     fi
  done < "$IDX"
  have=$(( total - ${#missing[@]} ))
  echo "[wd $(date -u +%FT%H:%MZ)] $have/$total results; ${#missing[@]} missing (${#pending[@]} retryable)"

  # publish whenever progress advanced
  if [ "$have" -gt "$last_done" ]; then
     bash sweep/curriculum_publish.sh || true
     last_done="$have"
  fi

  # terminal: everything is either done or given-up
  if [ "${#pending[@]}" -eq 0 ]; then
     echo "[wd] no retryable cells left ($have/$total real results) -> final publish + exit"
     bash sweep/curriculum_publish.sh || true
     exit 0
  fi

  active=$(squeue -u "$USER" -h -r -n gin-curr -t PENDING,RUNNING 2>/dev/null | wc -l)
  if [ "$active" -eq 0 ]; then
     echo "[wd] array drained, ${#pending[@]} cells still missing -> resubmitting"
     for name in "${pending[@]}"; do resubmit_one "$name"; sleep 1; done
  else
     echo "[wd] $active array task(s) still active; waiting"
  fi

  if [ $(( $(date +%s) - START )) -ge "$WALL" ]; then
     echo "[wd] nearing wall limit -> self-resubmitting a fresh watchdog"
     sbatch --export=ALL,PART="$PART",CPUS="$CPUS" slurm/curriculum_watchdog.slurm >/dev/null || true
     exit 0
  fi
  sleep "$CHECK_EVERY"
done
