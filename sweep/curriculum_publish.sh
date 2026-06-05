#!/bin/bash
# Phase-6 publish: aggregate the cell results, regenerate the curriculum figures + report,
# and commit + push. Idempotent and safe to run on a partial sweep -- it pushes only when
# something new landed. Run by the watchdog as cells finish, or manually any time.
set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

echo "[publish] aggregating cells..."
"$PY" sweep/collect_curriculum.py || true
echo "[publish] regenerating figures + report..."
"$PY" paper/make_figures.py        || echo "[publish] make_figures failed (continuing)"
"$PY" paper/make_report_html.py    || echo "[publish] make_report_html failed (continuing)"

# identity MUST be Nikelroid for the commit to count on the profile
git config user.name  "Nima Kelidari"
git config user.email "68930046+Nikelroid@users.noreply.github.com"
git pull --rebase --autostash origin main 2>&1 | tail -1 || true   # stay current with remote
git add -A sweep/curriculum sweep/collect_curriculum.py sweep/curriculum_train.py \
           sweep/curriculum_configs.py sweep/curriculum_cfgs slurm/curriculum*.slurm \
           sweep/curriculum_publish.sh docs/ paper/ 2>/dev/null || true
if git diff --cached --quiet; then
  echo "[publish] nothing new to commit"; exit 0
fi
git commit -q -m "Phase-6 sweep: curriculum/algorithm/reward results + docs ($(date -u +%Y-%m-%dT%H:%MZ))" || true

TOK="${ADVERSCO_TOKEN:-}"
if [ -z "$TOK" ]; then
  line=$(grep -E '\bADVERSCO_TOKEN=' ~/.bashrc 2>/dev/null | tail -1 || true)
  v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}; TOK=${v%% *}
fi
SLUG=$(git remote get-url origin | sed -E 's#https://[^@]*@#https://#; s#https://github.com/##; s#\.git$##')
git push "https://x-access-token:${TOK}@github.com/${SLUG}.git" HEAD:main 2>&1 \
  | sed -E 's/(ghp_|github_pat_|x-access-token:)[A-Za-z0-9_]+/\1REDACTED/g' | tail -2
echo "[publish] pushed"
