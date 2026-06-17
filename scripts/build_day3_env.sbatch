#!/bin/bash
#SBATCH --job-name=day3_env
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/day3_env_%j.out
#SBATCH --error=logs/day3_env_%j.err

# Build (resolve + instantiate + precompile) the merged day3 environment, which now
# carries both the ML stack (SpeedyWeather/Lux/Zygote) and the Breeze differentiable
# stack (Breeze/Reactant/Enzyme/Oceananigans) needed by 03_acoustic_wave_gradient.
# CUDA is required to load the Reactant extension, so we build on a real GPU node.

cd /shared/home/greg/Projects/BjerknesWorkshop/tutorials/day3
JULIA=/shared/home/greg/.juliaup/bin/julia

echo "=== $(date) host=$(hostname) ==="; nvidia-smi -L

echo "=== resolve + instantiate + precompile ==="
$JULIA --project=. -e '
using Pkg
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
Pkg.status()
'
rc=$?
echo "=== instantiate exit $rc ==="
[ $rc -ne 0 ] && { echo "RESOLVE/INSTANTIATE FAILED"; exit $rc; }

echo "=== load test: full acoustic-wave stack ==="
$JULIA --project=. -e '
using Breeze
using Oceananigans
using Reactant, CUDA
using Enzyme
using CairoMakie
using Printf
using Statistics
@info "All packages loaded" Breeze=pkgversion(Breeze) Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
println("LOAD_TEST_OK")
'
echo "=== done exit $? ==="
