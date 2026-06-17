#!/bin/bash
#SBATCH --job-name=coupled_smoke
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=logs/coupled_smoke_%j.out
#SBATCH --error=logs/coupled_smoke_%j.err

# First GPU validation of the coupled fjord-ocean case: SMOKE resolution (160²),
# capped at 300 coupled iterations. Self-contained env (no --export).
REPO=/shared/home/greg/Projects/BjerknesWorkshop
export THURSDAY_REPO_ROOT=$REPO
export COUPLED_STOP_ITERATION=300
# RUN_CLASS unset -> SMOKE (160²); set =production for the 768² run.

mkdir -p $REPO/output/day4/coupled_fjord_smoke
cd $REPO/output/day4/coupled_fjord_smoke

echo "=== $(date) host=$(hostname) ==="
nvidia-smi -L
/shared/home/greg/.juliaup/bin/julia --project=$REPO $REPO/tutorials/day4/scripts/10_coupled_fjord_ocean.jl
echo "=== done exit $? ==="
