#!/bin/bash
#SBATCH --job-name=gin-leduc
#SBATCH --account=liu32_1378
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/leduc_%j.out
#SBATCH --error=/home1/kelidari/Adversarial-CoEvolution/slurm/logs/leduc_%j.err

# Phase-8 Wave 3: train an RL agent on Leduc Hold'em, grade it vs the CFR-optimal expert.
#   sbatch --export=ALL,EPISODES=500000,GRADE_GAMES=2000 slurm/leduc.slurm

set -uo pipefail
cd /home1/kelidari/Adversarial-CoEvolution
source ~/miniconda/etc/profile.d/conda.sh
conda activate /scratch1/kelidari/envs/coev
PY=/scratch1/kelidari/envs/coev/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EPISODES="${EPISODES:-500000}" CFR_ITERS="${CFR_ITERS:-400}" GRADE_GAMES="${GRADE_GAMES:-2000}"
echo "host=$(hostname) episodes=$EPISODES grade_games=$GRADE_GAMES"
"$PY" sweep/leduc_train.py
