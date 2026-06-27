#!/bin/bash
#SBATCH --job-name=gin-ismcts
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ismcts_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/ismcts_%j.err

# Grade the PIMC/ISMCTS search baseline vs gold / random / champion (no training, CPU search).
#   for o in gold random champion; do
#     sbatch --export=ALL,OPP=$o,N=200,ISMCTS_ROLLOUTS=20 slurm/ismcts.slurm; done

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPP="${OPP:-gold}" N="${N:-200}" ISMCTS_ROLLOUTS="${ISMCTS_ROLLOUTS:-20}"
export NAME="${NAME:-ismcts_vs_${OPP}}"
echo "host=$(hostname) opp=$OPP N=$N rollouts=$ISMCTS_ROLLOUTS"
"$PY" sweep/ismcts_eval.py
