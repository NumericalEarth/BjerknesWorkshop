#!/bin/bash
#SBATCH --job-name=coupled_shake
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/coupled_shake_%j.out
#SBATCH --error=logs/coupled_shake_%j.err

# Production-resolution (384^2), bigger domain, capped at 300 coupled iterations to
# confirm stability before the full 1/2-inertial run.
REPO=/shared/home/greg/Projects/BjerknesWorkshop
export THURSDAY_REPO_ROOT=$REPO
export RUN_CLASS=production
export COUPLED_STOP_ITERATION=300

mkdir -p $REPO/output/day4/coupled_fjord_shake
cd $REPO/output/day4/coupled_fjord_shake
echo "=== $(date) host=$(hostname) ==="
nvidia-smi -L
/shared/home/greg/.juliaup/bin/julia --project=$REPO $REPO/tutorials/day4/scripts/10_coupled_fjord_ocean.jl
echo "=== done exit $? ==="
