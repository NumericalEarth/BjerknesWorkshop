# Bjerknes Workshop Tutorials

*Boundary heterogeneity writes turbulence into the fluid.*

This site collects the Bjerknes workshop tutorial suite. Each tutorial is
authored as a [Literate.jl](https://github.com/fredrikekre/Literate.jl) source
under `tutorials/dayN/src/`, rendered to the page you are reading, and *run
separately* by a deployment workflow that records per-run status under
`output/dayN/<slug>/`.

!!! note "How these pages are built"
    The documentation build renders the tutorial narrative and code **without
    executing it** — the large-eddy simulations run on an H100 through the
    deployment workflow, not during the doc build. The figures and movies in
    each page's *Results* section are embedded from the most recent **successful**
    cached run. When a run has not completed, a status card is shown in place of
    the missing artifact.

## Tutorials by day

### Day 1 — Julia & GPU foundations for Earth-system modeling

Getting productive with the Julia modeling stack, from interactive climate models
to GPU kernels:

- **Interactive climate modeling** — first steps with SpeedyWeather.jl and
  Oceananigans.jl.
- **Mesoscale eddies** — baroclinic instability in a channel.
- **Implementing new physics** — a capsizing iceberg, two-way coupled.
- **Hydrostatic ocean modeling** *(optional)* — internal tides over a sill.
- **GPU computing in Julia** *(optional)* — from arrays to a turbulence solver.
- **A first taste of the atmosphere (Breeze)** — a five-act tour from a dry
  thermal bubble to free convection, the split-explicit compressible solver,
  mountain lee waves, and finally clouds and drizzle over a 3D mountain.

### Day 2 — The high-latitude ocean & sea ice

Coupled ocean–sea ice simulation at regional and pan-Arctic scale:

- **Sea ice in the Arctic** — a pan-Arctic simulation over a slab ocean.
- **The Barents Sea** — a regional coupled ocean–sea ice simulation.

### Day 3 — Hybrid physics & differentiable Earth-system models

An end-to-end **hybrid machine-learning** tutorial — learning a surface-roughness
parameterization and running it inside a climate model — plus differentiable ESMs:

- **Hybrid ML, start to finish** — introduction, data preprocessing, dataloaders,
  offline training, defining the learned parameterization, and running it in
  SpeedyWeather.jl (including on the GPU and generalized to the Pangaea
  supercontinent).
- **Differentiable ESMs** — gradient-based calibration of an Earth-system
  component.

### Day 4 — Boundary heterogeneity & turbulence

GPU large-eddy and coupled simulations, each driven by a heterogeneous *boundary*
that organizes turbulence in the fluid:

1. **Two fluids, one interface** — 2D coupled air–sea convection over a warm
   ocean filament, with fluxes computed at the interface.
2. **A crack in the ice** — atmospheric turbulence over a sea-ice lead (Breeze).
3. **Beneath the crack** — ocean turbulence below the lead, no-waves control vs.
   Craik–Leibovich surface-wave (Langmuir) forcing (Oceananigans).
4. **A warm filament writes a cloud street** — 3D two-way-coupled air–sea LES.
5. **Steep island mountains as boundary conditions** — 100 m coupled air–land
   flow over the Lofoten islands (terrain-following Breeze).
6. **A realistic global ocean** — a NumericalEarth global ocean–sea ice simulation.

A **gallery & discussion** page and a trivial **smoke case** (full deploy pipeline
end-to-end on CPU) round out the day.

## Run status

The [Run status](status/index.md) pages summarize the state of the most recent
deployment for every case: whether the latest attempt succeeded, whether its
cached outputs are still current (matching the current parameters, source, and
manifest), and when it ran.
