#!/bin/bash
#SBATCH --job-name=gin-ppoarch-wd
#SBATCH --account=liu32_1378
#SBATCH --partition=oneweek
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ppoarch_wd_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ppoarch_wd_%j.err

# Babysits the PPO attention / MLP-anchor cells (job gin-ppoarch) to completion, no human in
# the loop. IMPORTANT: on this cluster `squeue -u`/`-n` intermittently returns EMPTY even when
# jobs are running, so this watchdog NEVER uses those. It tracks the array job IDs explicitly and
# queries them with `squeue -j` (reliable), prunes purged IDs, and requires TWO consecutive
# "drained" readings before resubmitting anything -> it cannot fire false duplicates.
#
# Each cycle: refresh a small summary; if the array has genuinely drained (2 clean zero-active
# readings) and some cells still lack a result JSON, those failed -> resubmit via ppoarch.slurm
# (OMP=8!), capped MAX_TRIES. Exits when every cell has a result-or-giveup. Never git-pushes.
#
#   sbatch --export=ALL,WATCH_JIDS="9893420 9893679" slurm/ppoarch_watchdog.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python

CFG_DIR=ppoarch_cfgs
IDX="sweep/${CFG_DIR}/INDEX.txt"
RES=sweep/curriculum
STATE="sweep/${CFG_DIR}/.wd_state"; mkdir -p "$STATE"
ATT="$STATE/attempts"; mkdir -p "$ATT"
JIDS="$STATE/jids"                                   # tracked array job IDs (one per line)
CHECK_EVERY="${CHECK_EVERY:-900}"
MAX_TRIES="${MAX_TRIES:-3}"
ERRTMP="$STATE/squeue_err"
# seed tracked job IDs from WATCH_JIDS (first run only)
if [ ! -s "$JIDS" ]; then printf '%s\n' ${WATCH_JIDS:-} | grep -vE '^\s*$' > "$JIDS"; fi
drain_strikes=0

summarize() {
  "$PY" - <<'PYEOF' 2>/dev/null || true
import json, glob, statistics as st
rows=[]
for p in sorted(glob.glob("sweep/curriculum/ppoarch_*.json")):
    d=json.load(open(p)); rows.append((d["arch"], d["name"], d["vs_gold"]["win_rate"], d["vs_champion"]["win_rate"]))
out=["# PPO arch cells (attention vs MLP anchor) vs GOLD","",
     "| arch | cell | vs_gold | vs_champ |","|---|---|---:|---:|"]
for a,n,g,c in sorted(rows): out.append(f"| {a} | {n} | {g*100:.1f} | {c*100:.1f} |")
def iqm(xs):
    xs=sorted(xs); k=len(xs)//4; core=xs[k:len(xs)-k] if len(xs)>=4 else xs
    return sum(core)/len(core) if core else float('nan')
out.append("")
for a in ("attn","mlp"):
    gs=[g for aa,_,g,_ in rows if aa==a]
    if gs: out.append(f"- **{a}**: n={len(gs)} IQM={iqm(gs)*100:.1f}%  mean={st.mean(gs)*100:.1f}%  range={min(gs)*100:.1f}-{max(gs)*100:.1f}")
open("sweep/curriculum/_ppoarch_summary.md","w").write("\n".join(out)+"\n")
PYEOF
}

# count PENDING/RUNNING across tracked IDs; prune purged IDs; sets globals ACTIVE and FLAKY
count_active() {
  ACTIVE=0; FLAKY=0; local keep=""
  for jid in $(cat "$JIDS" 2>/dev/null); do
    local out rc; out=$(squeue -j "$jid" -h -o "%T" 2>"$ERRTMP"); rc=$?
    if grep -qi "invalid job id\|not found" "$ERRTMP" 2>/dev/null; then continue; fi   # purged/done -> drop
    if [ "$rc" -ne 0 ]; then FLAKY=1; keep="$keep $jid"; continue; fi                  # controller hiccup -> keep
    local c; c=$(printf '%s\n' "$out" | grep -cE "PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING|REQUEUED" || true)
    ACTIVE=$((ACTIVE+c)); keep="$keep $jid"
  done
  printf '%s\n' $keep | grep -vE '^\s*$' > "$JIDS" 2>/dev/null || true
}

resubmit_one() {
  local name="$1" cf="$ATT/$1"; local n=0; [ -f "$cf" ] && n=$(cat "$cf")
  if [ "$n" -ge "$MAX_TRIES" ]; then touch "$ATT/$name.giveup"; echo "  !! $name gave up after $n"; return; fi
  local ln; ln=$(grep -nxF "$name" "$IDX" | head -1 | cut -d: -f1)
  [ -z "$ln" ] && { echo "  ?? $name not in INDEX"; return; }
  echo $((n+1)) > "$cf"
  local jid; jid=$(sbatch --parsable --array=$((ln-1)) --job-name=gin-ppoarch \
         --export=ALL,CFG_DIR=ppoarch_cfgs,NUM_ENV=16,WANDB_GROUP=phase8-arch slurm/ppoarch.slurm 2>/dev/null)
  jid=${jid%%;*}                                       # strip any ";cluster" suffix
  [ -n "$jid" ] && { echo "$jid" >> "$JIDS"; echo "  resubmitted $name (idx $((ln-1)), try $((n+1)), job $jid)"; }
}

while true; do
  mapfile -t cells < <(grep -vE '^\s*$' "$IDX")
  missing=(); for c in "${cells[@]}"; do
    [ -f "$RES/$c.json" ] && continue
    [ -f "$ATT/$c.giveup" ] && continue
    missing+=("$c")
  done
  count_active
  summarize
  echo "[$(date '+%m-%d %H:%M')] missing=${#missing[@]} active=$ACTIVE flaky=$FLAKY strikes=$drain_strikes tracked=$(wc -l < "$JIDS" 2>/dev/null)"
  if [ "${#missing[@]}" -eq 0 ]; then echo "all ppoarch cells done (or gave up) -> exit"; break; fi

  if [ "$FLAKY" -eq 1 ]; then
    drain_strikes=0; echo "  squeue flaky this cycle -> hold (no resubmit)"
  elif [ "$ACTIVE" -gt 0 ]; then
    drain_strikes=0
  else
    drain_strikes=$((drain_strikes+1))
    if [ "$drain_strikes" -ge 2 ]; then
      echo "  array drained (2 clean readings) & ${#missing[@]} missing -> resubmitting"
      for c in "${missing[@]}"; do resubmit_one "$c"; done
      drain_strikes=0
    else
      echo "  first drained reading -> waiting one more cycle to confirm"
    fi
  fi
  sleep "$CHECK_EVERY"
done
