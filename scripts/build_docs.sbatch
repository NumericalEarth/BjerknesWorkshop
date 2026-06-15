#!/bin/bash
#SBATCH --job-name=build_docs
#SBATCH --partition=gpu-prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
cd /shared/home/greg/Projects/BjerknesWorkshop
rm -f docs/src/day4/*_viz.md
echo "=== Phase A: render_all_viz (workshop env) ==="
/shared/home/greg/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --project=. scripts/render_all_viz.jl
echo "=== Phase B: make.jl (docs env) ==="
/shared/home/greg/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --project=docs docs/make.jl
echo "=== done exit $? ==="
