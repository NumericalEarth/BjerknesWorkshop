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

### Day 3 — Hybrid physics & differentiable Earth-system models

Lightweight placeholder tutorials establishing the deployment pipeline:

- **Hybrid physics + ML** — a physical core with a learned closure correction.
- **Differentiable ESMs** — gradient-based calibration of an Earth-system
  component.

### Day 4 — Boundary heterogeneity & turbulence

Three GPU large-eddy simulation case studies, each driven by a heterogeneous
*boundary* that organizes turbulence in the fluid:

1. **A crack in the ice** — atmospheric turbulence over a sea-ice lead
   (Breeze atmosphere-only LES with prescribed surface fluxes).
2. **Beneath the crack** — ocean turbulence below the lead, comparing a
   no-waves control against Craik–Leibovich surface-wave forcing (Oceananigans).
3. **Fjords as boundary conditions** — 100 m terrain-following atmospheric flow
   over coastal Norway (Breeze).

A trivial **smoke case** exercises the full deploy pipeline end-to-end on CPU.

## Run status

The [Run status](status/index.md) pages summarize the state of the most recent
deployment for every case: whether the latest attempt succeeded, whether its
cached outputs are still current (matching the current parameters, source, and
manifest), and when it ran.
