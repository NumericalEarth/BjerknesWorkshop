# Bjerknes Workshop

## Tools & docs

- **Install Julia** — [installer](https://julialang.org/install/) · [docs](https://docs.julialang.org/)
- **Oceananigans.jl** — [repo](https://github.com/CliMA/Oceananigans.jl) · [docs](https://clima.github.io/OceananigansDocumentation/stable/)
- **SpeedyWeather.jl** — [repo](https://github.com/SpeedyWeather/SpeedyWeather.jl) · [docs](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/)
- **NumericalEarth.jl** — [repo](https://github.com/NumericalEarth/NumericalEarth.jl) · [docs](https://numericalearth.github.io/NumericalEarthDocumentation/stable/)
- **Breeze.jl** — [repo](https://github.com/NumericalEarth/Breeze.jl) · [docs](https://numericalearth.github.io/BreezeDocumentation/dev/)
- **Terrarium.jl** — [repo](https://github.com/NumericalEarth/Terrarium.jl) · [docs](https://numericalearth.github.io/Terrarium.jl/dev/)

## Setting up your Julia environment

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

The tutorials are built into a website served at
<https://numericalearth.github.io/BjerknesWorkshopDocs/> (the rendered docs live in
the companion [BjerknesWorkshopDocs](https://github.com/NumericalEarth/BjerknesWorkshopDocs)
repo so this repo stays lightweight).

To generate the docs, run cases, publish to GitHub Pages, or regenerate just part
of the site, see the operator guide: [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Related projects

Part of [NumericalEarth](https://github.com/NumericalEarth):

- [PolarPlunge.jl](https://github.com/NumericalEarth/PolarPlunge.jl) — Swim lessons in Scottish waters
- [SwimLessons.jl](https://github.com/NumericalEarth/SwimLessons.jl) — Tutorials and scripts that teach ocean-flavored fluid dynamics with Oceananigans

## Table of contents

Every tutorial is rendered (with figures and movies) at
<https://numericalearth.github.io/BjerknesWorkshopDocs/>. Sources live under
`tutorials/dayN/src/`.

### Day 1 — Julia & GPU foundations
- [Interactive climate modelling basics](https://numericalearth.github.io/BjerknesWorkshopDocs/day1/01_intro_interactive_climate/) — SpeedyWeather.jl + Oceananigans.jl
- [Mesoscale eddies: baroclinic instability in a channel](https://numericalearth.github.io/BjerknesWorkshopDocs/day1/02_oceananigans_baroclinic_adjustment/)
- [Implementing new physics: a capsizing iceberg](https://numericalearth.github.io/BjerknesWorkshopDocs/day1/03_capsizing_iceberg/)
- [Hydrostatic ocean modelling: internal tides over a sill](https://numericalearth.github.io/BjerknesWorkshopDocs/day1/04_optional_hydrostatic_internal_tide/) *(optional)*
- [GPU computing in Julia: from arrays to a turbulence solver](https://numericalearth.github.io/BjerknesWorkshopDocs/day1/05_optional_gpu_computing/) *(optional)*

### Day 2 — High-latitude ocean & sea ice
- [Introduction to Breeze](https://numericalearth.github.io/BjerknesWorkshopDocs/day2/03_breeze_tutorial/) — thermal bubble → free convection → split-explicit compressible → mountain lee waves → clouds & drizzle over a 3D mountain
- [Sea ice in the Arctic: a pan-Arctic simulation](https://numericalearth.github.io/BjerknesWorkshopDocs/day2/01_arctic_sea_ice/)
- [The Barents Sea: a regional coupled ocean–sea ice simulation](https://numericalearth.github.io/BjerknesWorkshopDocs/day2/02_barents_sea_regional/)

### Day 3 — Hybrid physics & differentiable ESMs
- Hybrid ML — learning a surface-roughness parameterization, end to end:
  [introduction](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01a_hybrid_ml_introduction/) ·
  [preprocessing](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01b_preprocessing/) ·
  [dataloaders](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01c_dataloaders/) ·
  [training](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01d_training/) ·
  [parameterization](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01e_parameterization/) ·
  [run in SpeedyWeather](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01f_run_parameterization/) ·
  [Pangaea](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01g_run_pangaea/) ·
  [GPU](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/01h_run_parameterization_gpu/)
- [Differentiable Earth-system models](https://numericalearth.github.io/BjerknesWorkshopDocs/day3/02_differentiable_esms/)

### Day 4 — Boundary heterogeneity & turbulence
- [A crack in the ice: atmospheric turbulence over a sea-ice lead](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/01_atmospheric_turbulence_over_a_sea_ice_lead/)
- [Beneath the crack: ocean turbulence & Langmuir structures](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/02_ocean_turbulence_below_a_lead_with_surface_waves/)
- [Steep island mountains: 100 m coupled air–land flow over Lofoten](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/03_norway_100m_prescribed_fluxes/)
- [Two fluids, one interface: 2D coupled air–sea convection](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/07_intro_coupled_convection/)
- [A warm filament writes a cloud street: 3D coupled LES](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/08_coupled_warm_filament/)
- [A realistic global ocean–sea ice simulation](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/09_global_ocean/)
- [Gallery & discussion](https://numericalearth.github.io/BjerknesWorkshopDocs/day4/04_gallery_and_discussion/)
