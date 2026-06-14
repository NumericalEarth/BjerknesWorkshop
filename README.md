# Bjerknes Workshop

## setting up your Julia environment

The tutorials listed below run either on your laptop or in the Olivia supercomputer.
Please set up a running julia installation and environment in both. 
To set up julia in your local enviroment check out

```
https://github.com/JuliaLang/juliaup
```

To set up on Olivia, use the following in your enviroment setup

```
module purge
module load NRIS/GPU
module load JupyterNotebook/7.4.7-GCCcore-14.3.0
module load Julia
export DATA_DIR=/cluster/projects/nn9984k
mkdir /cluster/work/projects/nn9984k/$USER
cd /cluster/work/projects/nn9984k/$USER
git clone https://github.com/NumericalEarth/BjerknesWorkshop.git
export JULIA_DEPOT_PATH=/cluster/work/projects/nn9984k/$USER/julia_depot
env -u LD_LIBRARY_PATH julia -e 'import Pkg; Pkg.add("IJulia")'
env -u LD_LIBRARY_PATH julia -e 'using IJulia; installkernel("Julia 1.12 (clean env)", "-t 8"; env = Dict("LD_LIBRARY_PATH" => "/cluster/software/NRIS/neoverse_v2/software/Julia/1.12.2/lib"))'
```

## Schedule

### Monday
- **Morning (10:00–12:30):** Introduction to Julia and script-based & interactive ESM modelling
- **Afternoon (13:30–16:00):** Introduction to GPU-based modelling

### Tuesday
- **Morning (09:00–12:30):** Hands-on experiments using Oceananigans (ocean), SpeedyWeather (atmosphere), and Terrarium (land surface)
- **Afternoon (13:30–15:30):** Discussion session

### Wednesday
- **Morning (09:00–12:30):** Hybrid physics–ML modelling, differentiable ESMs, and related topics
- **Afternoon (13:30–15:30):** Continuation and discussion

### Thursday
- **Morning (09:00–12:30):** Coupled ocean–atmosphere simulations using interactive ESM frameworks (it can be easier than you think!)
- **Afternoon (13:30–15:30):** Coupled atmosphere–ocean LES / nonhydrostatic modelling, including examples such as sea-ice leads and complex topography

### Friday
- **Morning (09:30–12:30):** Biogeochemistry (BGC)
- **Afternoon (13:30–15:30):** Open discussion

## Tutorials & website

The LES tutorials (Thursday = `tutorials/day4/`) are built into a website served
at <https://numericalearth.github.io/BjerknesWorkshop/>.

To generate the docs, run cases, publish to GitHub Pages, or regenerate just part
of the site, see the operator guide: [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Related projects

Part of [NumericalEarth](https://github.com/NumericalEarth):

- [PolarPlunge.jl](https://github.com/NumericalEarth/PolarPlunge.jl) — Swim lessons in Scottish waters
- [SwimLessons.jl](https://github.com/NumericalEarth/SwimLessons.jl) — Tutorials and scripts that teach ocean-flavored fluid dynamics with Oceananigans
