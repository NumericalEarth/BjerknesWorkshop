#!/bin/bash
#SBATCH --job-name=diag_gpu
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
REPO=/shared/home/greg/Projects/BjerknesWorkshop
export THURSDAY_REPO_ROOT=$REPO
export DIAG_N=${1:-384}
echo "=== $(date) host=$(hostname) DIAG_N=$DIAG_N ==="; nvidia-smi -L
/shared/home/greg/.juliaup/bin/julia --project=$REPO /tmp/diag_gpu.jl
echo "=== done exit $? ==="
