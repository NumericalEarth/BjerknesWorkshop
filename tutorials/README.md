# Bjerknes Workshop tutorials — days 1 & 2

Literate.jl sources for the Monday GPU session and the Tuesday hands-on ocean / sea-ice
sessions. Each `.jl` file under `dayN/src/` is simultaneously a runnable Julia script
and the canonical source of a rendered tutorial page (the `# ` comment lines are
markdown; everything else is code).

## Day 1 — GPU computing

| File | Content | Runs on a laptop? |
|---|---|---|
| `day1/src/01_gpu_computing.jl` | GPU computing, kernels, and a 2D Navier–Stokes solver from scratch | yes (~2 min, CPU fallback) |
| `day1/src/02_distributed_nonhydrostatic.jl` | Multi-GPU LES of polar deep convection | demo yes; LES needs GPUs |

## Day 2 — one day in the high-latitude ocean

The Tuesday tutorials are stations along a single transect, from a Norwegian fjord sill
out to the Barents Sea:

| Part | File | Content | Runs on a laptop? |
|---|---|---|---|
| 1 | `day2/src/01_hydrostatic_internal_tide.jl` | Tidal energy at the sill: `HydrostaticFreeSurfaceModel`, immersed boundaries, internal tides | yes (~1 min) |
| 2 | `day2/src/02_baroclinic_instability.jl` | The eddies that carry the heat north: eddying channel, custom drag BC, adaptive Δt | yes (~2 min) |
| 3 | `day2/src/03_sea_ice_thermodynamics.jl` | The freezing surface: ClimaSeaIce slab thermodynamics, freezing bucket → Semtner seasonal cycle | yes (seconds) |
| 4 | `day2/src/04_sea_ice_dynamics.jl` | The pack in motion: EVP rheology, leads and linear kinematic features | yes (~4 min) |
| 5 | `day2/src/05_capsizing_iceberg.jl` | The calving front: implementing new physics — a penalized rigid iceberg, two-way coupled, GPU-compatible | yes (~5 min) |
| 6 | `day2/src/06_barents_sea_coupled.jl` | All of it together: a regional coupled ocean–sea ice simulation of the Barents Sea — GLORYS12-fed open boundary conditions (Flather + Orlanski radiation) with a sponge behind them, JRA55 atmosphere | no: GPU + a few GB of data + Copernicus account |
| 7 | `day2/src/07_distributed_hydrostatic_ocean.jl` | Epilogue — scaling up: the eddy-resolving channel on many GPUs; route to GB-25 | smoke-test yes; science needs GPUs |

`extras/global_ocean_simulation.jl` holds the one-degree *global* ocean–sea ice
configuration (the OMIP-style cousin of part 6), kept as supporting material for the
Thursday coupled sessions.

## A note on the day-2 environment

Part 6 uses open boundary conditions from the `ss/open-boundary-conditions` branch of
Oceananigans (version 0.110), which the registered ClimaSeaIce and NumericalEarth do not
yet support. The day-2 environment therefore `dev`s a stack: the local Oceananigans
checkout on that branch, plus compat-widened ClimaSeaIce and NumericalEarth clones under
`.deps/` (gitignored; the NumericalEarth copy also carries a fix that merges
user-supplied lateral boundary conditions side-by-side with `ocean_simulation`'s
defaults — both candidates for upstreaming). To reconstruct on another machine:

```julia
# in tutorials/day2
using Pkg
Pkg.develop([PackageSpec(path=".deps/ClimaSeaIce.jl"),       # clone + widen compat first
             PackageSpec(path=".deps/NumericalEarth.jl"),
             PackageSpec(path="path/to/Oceananigans.jl")])   # ss/open-boundary-conditions
```

Once the branch and compat bumps land upstream, the environment goes back to registered
versions and this section disappears.

## Running

Each day has its own environment:

```bash
julia --project=tutorials/day1 -e 'using Pkg; Pkg.instantiate()'
julia --project=tutorials/day1 tutorials/day1/src/01_gpu_computing.jl
```

All scripts run top-to-bottom on a CPU-only laptop except where the table says
otherwise; GPU sections are guarded or commented, and the iceberg tutorial documents
the host-truth/device-mirror pattern that makes its coupled physics GPU-compatible
(tested on CUDA-style architectures and Apple Metal with `FT = Float32`). The
distributed tutorials write their MPI drivers to the working directory and document
the `mpiexec`/Slurm launch lines; both drivers accept environment-variable overrides
(grid size, architecture, duration) so the identical script smoke-tests on two laptop
CPU ranks. GPU scripts are best launched in fresh Julia sessions: method
redefinitions from re-`include`-ing in a long-lived session can invalidate device
code.

## Rendering the website

The docs pipeline on the `workshop-docs-deployment` branch
(`docs/make.jl` + `src/TutorialWorkflow.jl`) renders any `tutorials/day$d/src/*.jl`
through `Literate.markdown(...; execute = false)` and demotes executable fences, so
nothing runs at build time. To include these tutorials in the site:

1. merge (or rebase onto) `workshop-docs-deployment`;
2. set `DOC_DAYS=1,2` (or extend `selected_doc_days`) so `render_day(1)` and
   `render_day(2)` are invoked — the renderer skips files starting with `00_`;
3. optionally register the cheap cases (`day2` 01–05, `day1` 01) in `case_registry`
   inside `src/TutorialWorkflow.jl` so the runner executes them and the site embeds
   their figures/movies in a Results section, following the day-3/day-4 pattern;
4. add day-1/day-2 labels in `_nav_for_day` (it falls back to "Day N" otherwise).

Every figure/movie referenced by a `![](...)` line is produced by the script itself in
the working directory, so `Literate.markdown` with `execute = true` (or running the
script before a plain Documenter build) also works for a standalone site.

## Status

All sources parse; the laptop-runnable ones have been executed end-to-end on CPU
(macOS, Julia 1.12, Oceananigans 0.109/0.110, ClimaSeaIce 0.5.5, NumericalEarth 0.5.4),
including 2-rank MPI smoke tests of both distributed drivers and a Metal (Apple GPU)
execution test of the iceberg tutorial. `06_barents_sea_coupled.jl` follows the
NumericalEarth regional-simulation APIs (`DatasetRestoring`, `ocean_simulation`,
`sea_ice_simulation`, `OceanSeaIceModel`) verified against the installed package
source, but needs a GPU machine with the datasets staged — run it once on the cluster
before the workshop.
