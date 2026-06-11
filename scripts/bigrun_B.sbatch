#!/bin/bash
#SBATCH --job-name=bigrunB
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bigrunB_%j.out
#SBATCH --error=logs/bigrunB_%j.err
cd /shared/home/greg/Projects/BjerknesWorkshop
export RUN_CASES="lead_ocean_waves,norway_100m"
export RUN_CLASS=production
export FORCE_RERUN=1
export ALLOW_CASE_FAILURES=true
export STRICT_CASES=false
echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L
/shared/home/greg/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --project=. scripts/run_tutorials.jl
echo "=== done exit $? ==="
