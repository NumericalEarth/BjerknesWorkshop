#!/bin/bash
#SBATCH --job-name=day4_run
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=logs/day4_run_%j.out
#SBATCH --error=logs/day4_run_%j.err
cd /shared/home/greg/Projects/BjerknesWorkshop
export RUN_CASES="lead_atmosphere,lead_ocean_waves,norway_100m"
export RUN_CLASS=production
export ALLOW_CASE_FAILURES=true
export STRICT_CASES=false
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
/shared/home/greg/.juliaup/bin/julia --project=. scripts/run_tutorials.jl
echo "=== done exit $? ==="
