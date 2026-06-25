# 8-hour autonomous work plan — Nærøyfjord coupled LES

**Goal:** without intervention, get from "modules + bathymetry" to a **stability-tested
coupled run** (LES atmosphere + hydrostatic/CATKE ocean) over the Nærøyfjord with the
rotating-wind forcing, iterating on the GPU as needed.

**Hardware:** on GPU node `gpu-1-8` — **4× NVIDIA GH200 120 GB**, runnable directly (no
Slurm needed). Julia 1.12.2, `JULIA_DEPOT_PATH=/cluster/work/projects/nn9984k/glwagner/julia_depot`.
Env: `tutorials/day4/Project.toml` (pinned to cached Breeze 0.6.0 / NumericalEarth 0.5.6
/ Oceananigans 0.110.1 / CUDA 6.2.0).

## Operating protocol (how the autonomy works)
- **GPU jobs run in the background** (`run_in_background`), each writing a log under
  `logs/`. When one finishes I'm re-invoked, I analyze the log, and launch the next —
  this chain spans hours unattended. If ever idle with nothing pending, schedule a
  wake-up.
- **State file `logs/autonomous_state.md`** is rewritten every cycle (what ran, result,
  decision, next action) so progress survives context summarization.
- **Stability gate:** every config gets a *short* GPU smoke run first
  (`erroring_NaNChecker!`, ~200–500 steps, capped wall-time). Log `max|w|`, `max|u|`,
  `Δt`, CFL. **Do not scale up until the short run is clean.** On blow-up: lower `Δt` /
  raise smoothing / deepen sponge / refine vertical coord, then re-run.
- **Fallback ladder** (never get stuck): real bathymetry → synthetic fjord; hydrostatic
  ocean coupling → thin nonhydrostatic → `SlabOcean`; steep terrain → more smoothing /
  finer Δ / gentler walls. Each fallback is logged, not silent.

## Timeline (checkpoints)
- **H0 0:00–0:30** — env instantiation (bg); write `10a`; write this plan. *(active)*
- **H1 0:30–1:30** — run `10a`: synthetic + real (Kartverket walls + EMODnet seafloor)
  → fused topo/bathy artifact + validation figure. **Smoothing study (CPU, cheap):**
  pick the *smallest* wall smoothing that keeps the channel ≥ ~6 cells wide while
  bounding the atmosphere's max resolved terrain slope (target ≲ 1.5). Report the
  fjord width and slope vs smoothing length. *(user: "smooth the fjord, hopefully not
  too much")*
- **H2 1:30–3:00** — write `10`. **Atmosphere-only** over the fjord terrain at a modest
  grid; **short GPU stability test** (the key risk: terrain-following over near-vertical
  walls). Tune smoothing/Δt/sponge/vertical-coord until clean. This is the main
  stability gate the user asked for.
- **H3 3:00–4:30** — **ocean-only** hydrostatic + CATKE in the fjord basin, prescribed
  rotating wind stress; short GPU stability test; tune CFL/free-surface.
- **H4 4:30–6:00** — **couple** (`AtmosphereOceanModel`); verify hydrostatic-ocean
  coupling early (fallback ladder if unsupported); short coupled stability test.
- **H5 6:00–7:30** — longer coupled run with the rotating-wind schedule
  (cross-fjord → down-fjord); write outputs.
- **H6 7:30–8:00** — viz, results summary, update `NAEROYFJORD_PLAN.md §7` + memory.

## Success criteria
1. Fused bathymetry artifact + figure showing a recognizable fjord with steep walls.
2. A short atmosphere-only run that is **NaN-free and CFL-stable** over the real terrain.
3. A coupled run that steps stably and shows the **stagnant → wind-driven mixing**
   transition as the wind rotates onto the fjord axis.
4. All decisions/fallbacks logged; plan + memory updated for a clean handoff.

## Risks held in view
- Terrain-following stability over ~1200 m near-vertical walls (primary).
- Narrow fjord (~250–500 m) vs ocean resolution (need ≥ ~6 cells across).
- Hydrostatic-ocean ↔ atmosphere coupling support (experimental upstream).
- Env/version drift (mitigated by pinning to the cached, mutually-compatible set).
