# Plan: coupled atmosphere–ocean LES of the Nærøyfjord (Gudvangen)

Branch: `glw/naeroyfjord-coupled-les`

A new Thursday/day-4 case study: a **two-way coupled atmosphere–ocean LES** of the
Nærøyfjord — the narrow, UNESCO-listed branch of the Sognefjord system that ends at
**Gudvangen** (≈ 60.876° N, 6.845° E). The scientific hook is exactly the scenario
the user described: a wind whose direction **rotates relative to the fjord axis**, so
that the steep fjord walls first *block* a cross-fjord wind (the fjord water stays
stagnant) and then, as the wind swings to blow **down-valley**, a gap/down-fjord jet
develops and drives surface mixing in the ocean.

This case reuses the machinery already in `tutorials/day4/`:
- `03_norway_100m_prescribed_fluxes.jl` — real-terrain atmospheric LES on Breeze
  terrain-following coordinates, with topography from the Kartverket DTM
  (`src/KartverketDEM.jl`) cached by `03a_prepare_norway_topography.jl`.
- `08_coupled_warm_filament.jl` / `07_intro_coupled_convection.jl` — the
  `AtmosphereOceanModel` (`EarthSystemModel`) coupling pattern: two grids sharing one
  sea surface, fluxes computed by similarity theory every step.

---

## 1. The site and the physics

**Geometry.** Nærøyfjord is ~17 km long, **very narrow** (≈ 250 m at its tightest,
~500 m typical), with walls rising **1000–1400 m almost vertically** from the water.
The axis runs roughly N–S in its inner reach down to Gudvangen at the head. This is
the extreme end of the orographic-blocking regime — far steeper and narrower than the
Lofoten case (`03`).

**Wind regime (the experiment).** Two competing forcings:
- A **cross-fjord** wind is almost entirely blocked by the 1000 m+ walls. Little
  surface stress reaches the water → the fjord surface stays quiescent ("stagnant").
- A **down-fjord / down-valley** wind is funnelled and accelerated along the axis
  (Bernoulli gap-jet, the same mechanism as the Lofoten fjord jets in `03`). Strong
  along-axis surface stress → wind-driven mixing, set-up/set-down, and upwelling at
  the head near Gudvangen.

The novel ingredient versus `03`/`08` is a **time-rotating background wind**: we sweep
the large-scale wind direction from cross-fjord toward down-fjord over the run and let
the terrain decide how much of it reaches the water. The ocean's response is the
diagnostic — stagnant → mixing as the wind aligns.

**Nondimensional control.** Same `M = N h / U` framing as `03`. With `h ≈ 1200 m` and
a stratified airmass, even modest `U` puts us deep in the blocking/flow-splitting
regime — which is the point.

---

## 2. Bathymetry: *can we get it?* — yes, in tiers

This was the key open question. Findings from probing the data services from the
cluster (network access confirmed working):

| Source | What | Access | Res | Verdict |
|---|---|---|---|---|
| **Kartverket national DTM** (`wcs.hoyde-dtm-nhm-25833`) | land + fjord **walls** above water | **WCS → NetCDF, works today** (already wired in `KartverketDEM.jl`) | 1 m native | ✅ use for terrain |
| **Kartverket Dybdedata** (`wms.dybdedata2`) | **seafloor** depth, multibeam | only **WMS** images + contour layers (`Dybdelag`, `Dybdekontur`); **no open coverage/WCS raster** | high | ⚠️ raster gated |
| **EMODnet Bathymetry** (`ows.emodnet-bathymetry.eu/wcs`, coverage `emodnet__mean`) | seafloor DTM | **open WCS, works today** | ~115 m | ✅ recommended seafloor source |
| Synthetic parametric channel | idealized fjord | none | any | ✅ dependency-free fallback |

So the plan splits the surface into **two rasters fused into one signed-elevation
field** `h(x,y)` (land > 0, sea < 0):
1. **Above water** (walls, head valley): Kartverket DTM via WCS — high-res, reliable.
2. **Below water** (the fjord channel): **EMODnet `emodnet__mean` via WCS** as the
   default; an optional Kartverket-Dybdedata upgrade path (if/when raster access is
   arranged); and a **synthetic fjord channel** fallback (V/U-shaped cross-section,
   sill near the mouth, deepening basin) fit to published Nærøyfjord depths so the
   whole workflow runs with zero data access — mirroring `synthetic_lofoten` in `03a`.

**Resolution reality check.** EMODnet ~115 m smoothed onto a 20–50 m ocean grid means
the channel cross-section is interpolated, not truly resolved; for a ~500 m-wide fjord
that's ~10 cells across — marginal but usable. The synthetic channel gives clean,
fully-controlled bathymetry for a first working run; EMODnet adds realism; Kartverket
multibeam is the eventual "real" upgrade.

### 2a. Required code change — UTM zone

`KartverketDEM.jl` is **hardwired to UTM zone 33N** (`_UTM33_λ₀ = 15° E`, EPSG:25833,
endpoint `…-25833`). **Gudvangen at 6.845° E is UTM zone 32N (EPSG:25832).** Using
zone 33 there incurs ~8° of longitude distortion — too much for a metric LES box. So
we must **generalize the projection** (parametrize central meridian / false-easting /
EPSG and the WCS endpoint) and add a zone-32 path. The depth WMS confirmed it serves
EPSG:25832, so the zone-32 family is available on Geonorge.

---

## 3. Model configuration

**Coupling.** `AtmosphereOceanModel` (`EarthSystemModel`), exactly as `08`: a Breeze
`AtmosphereModel` above an Oceananigans ocean, sharing horizontal extent and cell
count, fluxes by Monin–Obukhov similarity theory each step.

> ⚠️ Caveat carried over from `07`/`08`: coupling to a **nonhydrostatic** ocean is
> experimental upstream (tested path is `SlabOcean`). We additionally intend a
> **hydrostatic** ocean (see below) — coupling support for that must be verified early
> (Phase 4). If hydrostatic coupling is not yet supported, fall back to a thin
> nonhydrostatic ocean or a `SlabOcean` mixed-layer for the first coupled run.

**Ocean — hydrostatic + CATKE, high-res.** **Only the atmosphere is LES.** The ocean
is a **`HydrostaticFreeSurfaceModel`** (the `ocean_simulation` default) with **CATKE**
parameterizing the vertical mixing — so it does *not* resolve 3-D turbulence and is far
cheaper than an LES ocean, which is what affords **20–50 m horizontal** resolution over
the fjord with fine near-surface vertical spacing (≈ 1–5 m) coarsening with depth.
Land/closed boundaries from the fused bathymetry mask (the fjord is a thin wet channel
inside a mostly-dry box). Stratified initial `T`/`S` with a fresh, buoyant surface lens
(fjords are strongly stratified by riverine input at the head) — this stratification,
plus CATKE, is what makes the "stagnant vs wind-driven mixing" contrast visible.

**Grid type — rectilinear (metric) for both.** Because the ocean is hydrostatic+CATKE
it *could* run on a `LatitudeLongitudeGrid`, but we use a **metric `RectilinearGrid`
for both** components: `AtmosphereOceanModel` maps surface columns 1:1 (matching
horizontal grids is simplest), the atmosphere's terrain-following coordinate is
inherently metric, and over a ~20 km box at 61° N a lat–lon grid only buys anisotropic
cells and a coordinate mismatch with the UTM bathymetry. Lat–lon remains feasible if a
later, larger domain wants it.

**Atmosphere.** Breeze compressible dynamics on **terrain-following coordinates** with
acoustic substepping and an upper sponge, as in `03`. Domain to ~6–8 km with a 3–4 km
sponge for mountain waves. `SmagorinskyLilly` LES closure, `FPlane(latitude = 60.9)`.

**The rotating wind.** Drive the atmosphere with a **time-dependent background wind
direction** — implemented as relaxation/forcing toward a large-scale wind vector
`U(t) = U₀ (cos θ(t), sin θ(t))` whose angle `θ(t)` sweeps from **cross-fjord** to
**down-fjord** over the run (e.g. a quarter rotation over a few hours, or a hold–rotate
–hold schedule). The terrain does the blocking; we only rotate the far-field forcing.
Diagnose surface wind stress over the water and the ocean's TKE / mixed-layer depth as
`θ(t)` aligns with the axis.

**Grid sketch (starting point, to tune to one H100).**
- Box ≈ 20 km (cross) × 24 km (along-fjord), elongated along the axis.
- Atmosphere: terrain-following, e.g. 400×480 × ~80 vertical (tune).
- Ocean: same horizontal extent & cell count as atmosphere surface (coupler maps
  columns 1:1), depth ~600 m (or local max), ~60–100 vertical levels stretched.
- Horizontal Δ ≈ 50 m to start; push toward 20–30 m if it fits.

---

## 4. Deliverables (files)

1. **`src/KartverketDEM.jl`** — generalize projection to arbitrary UTM zone
   (parametrize central meridian + EPSG + WCS endpoint); add zone-32N support.
2. **`src/FjordBathymetry.jl`** *(new)* — EMODnet Bathymetry WCS fetch (coverage
   `emodnet__mean`, windowed, NetCDF via `NCDatasets`, no GDAL); fuse land DTM (>0) and
   seafloor (<0) into one signed `h(x,y)`; synthetic-fjord fallback; ocean mask.
3. **`tutorials/day4/src/10a_prepare_naeroyfjord_topography.jl`** — preprocessing →
   cached artifact `thursday/data/naeroyfjord_topography.jld2`
   `(x, y, h, land_mask, ocean_mask, depth, taper_mask, source_metadata)` + validation
   figure. Modes: `kartverket+emodnet` / `synthetic` (default), like `03a`.
4. **`tutorials/day4/src/10_naeroyfjord_coupled_les.jl`** — the coupled run with the
   rotating-wind forcing.
5. **`tutorials/day4/src/10_naeroyfjord_coupled_les_viz.jl`** — figures/movie
   (surface wind stress over water, ocean SST/TKE/MLD, along-fjord transect, the
   stagnant→mixing transition vs wind angle).
6. **`scripts/bigrun_naeroyfjord.sbatch`** — GPU batch script (clone of
   `bigrun_norway.sbatch`).

---

## 5. Phasing (cheap validation first)

- **Phase 0 — data probe** ✅ done: network OK; Kartverket DTM WCS OK; EMODnet WCS OK
  (`emodnet__mean`); Kartverket depth is WMS-only; Gudvangen is UTM 32N.
- **Phase 1 — bathymetry artifact + figure** (cheap, CPU): fuse land+sea, build the
  fjord channel, validate the cross-section and masks in a figure *before* any GPU run.
- **Phase 2 — atmosphere-only** over the fjord with the rotating wind; confirm blocking
  (cross) → gap jet (along) and the surface-stress signature over the water.
- **Phase 3 — ocean-only** in the fjord basin: stratified rest state + a prescribed
  rotating wind stress; confirm stagnant → mixing response and CFL/grid affordability.
- **Phase 4 — couple** (`AtmosphereOceanModel`): verify the hydrostatic-ocean coupling
  path early; run the full case; render.

## 6. Open questions / risks

- **Hydrostatic-ocean coupling** support in NumericalEarth (verify in Phase 4; have
  nonhydrostatic-thin / `SlabOcean` fallback).
- **Zone-32 WCS endpoint**: confirm a `…-25832` national-DTM coverage exists, or
  whether the national grid is published in 25833 everywhere (then reproject the box).
- **Fjord narrowness vs ocean resolution**: ~500 m channel at 50 m = ~10 cells; may
  need 20–30 m, which costs.
- **EMODnet adequacy** for a fjord (115 m) — synthetic channel covers the gap; real
  Kartverket multibeam is the upgrade.
- **Rotating-wind implementation** in Breeze: relaxation forcing vs prescribed inflow —
  pick the cleanest hook.

---

## 7. Progress / resume notes (live)

**Branch:** `glw/naeroyfjord-coupled-les`. Work is in the working tree (not yet
committed — commit when ready).

**Done & validated (CPU, cheap):**
- ✅ `src/KartverketDEM.jl` — generalized to **arbitrary UTM zone** (was hardwired to
  zone 33). New `latlon_to_utm(...; zone)`, `utm_to_latlon(...; zone)` (inverse), zone
  carried through `KartverketDTM`/`KartverketWindow`/`kartverket_metadatum`, endpoint &
  EPSG derived per zone. Backward-compatible (`zone = 33` default; `latlon_to_utm33n`
  alias kept). **Round-trip tested**: fwd∘inv ≈ 6e-9° (sub-mm) for zones 32 & 33.
  - ⚠️ gotcha fixed: inverse used a var `e1`; `3e1`/`27e1`/… lex as float literals in
    Julia, so it's renamed `e₁` with explicit `*`. Watch for this if editing.
- ✅ `src/FjordBathymetry.jl` *(new)* — EMODnet Bathymetry via WCS 2.0.1 `GetCoverage`
  in **`text/plain`** (GDAL-free ASCII grid; `format=image/tiff` also works but needs a
  TIFF reader). `read_emodnet` parses the affine header + `Band 0:` data; tested on a
  real Nærøyfjord window (134×106, −75.6 m … +1533.8 m; `emodnet__mean` is **combined
  land+sea**, so we keep only the negative seafloor part). Plus `synthetic_naeroyfjord`
  parametric channel (tested: walls ~1257 m, depth ~319 m).

**Key data facts (probed live, network works from cluster):**
- Kartverket DTM WCS zone-32 endpoint: `wcs.hoyde-dtm-nhm-25832`, coverage
  `nhm_dtm_topo_25832`, EPSG:25832. Gudvangen (60.876° N, 6.845° E) → zone 32, central
  meridian 9° E, UTM ≈ E 383005, N 6 750 898.
- EMODnet: `https://ows.emodnet-bathymetry.eu/wcs`, coverage `emodnet__mean`, EPSG:4326,
  1/16′ (~115 m N–S / ~55 m E–W at 61° N). Request `subset=Lat(a,b)&subset=Long(a,b)`.

**Next (in order):**
1. **Task 3** — `tutorials/day4/src/10a_prepare_naeroyfjord_topography.jl`: build the
   metric UTM-32 grid centred on the fjord, sample Kartverket land (>0) via WCS and
   EMODnet seafloor (<0) — projecting grid nodes to lat/lon with `utm_to_latlon` —
   fuse into one signed `h`, smooth/taper, write `naeroyfjord_topography.jld2` + a
   validation figure. Modes `synthetic` (default) / `kartverket+emodnet`. Pattern:
   clone `03a_prepare_norway_topography.jl`.
2. **Task 4** — run `10a` on CPU; confirm real fetch + figure; tune synthetic.
3. **Task 5** — `10_naeroyfjord_coupled_les.jl`: terrain-following Breeze LES +
   hydrostatic/CATKE rectilinear ocean, `AtmosphereOceanModel`, time-rotating wind
   (verify hydrostatic-ocean coupling support **early** — fallback: thin nonhydrostatic
   or `SlabOcean`). Plus `_viz.jl` and `scripts/bigrun_naeroyfjord.sbatch`.

**Env:** day4 has no `Project.toml`; cases run with `julia --project=.` against the
run-machine env (Breeze/NumericalEarth/Oceananigans/CUDA/NCDatasets/CairoMakie). Julia
1.12.2 at `/cluster/software/NRIS/neoverse_v2/software/Julia/1.12.2/bin/julia`. Heavy
deps take minutes to load; pure-stdlib parser tests run fast (used above).

---

## 8. FINAL RESULTS (autonomous 8-hour run)

All on GPU node `gpu-1-8` (GH200). The day4 env was instantiated by cloning the sibling
workshop's git-pinned manifest (Breeze 0.6.0 / NumericalEarth 0.5.6 / Oceananigans 0.110.1).

**Delivered & validated:**
- **Bathymetry** (`src/KartverketDEM.jl` zone-generalized + `src/FjordBathymetry.jl`):
  real Kartverket UTM-32 DTM (walls) + EMODnet seafloor, fused to one signed field; also
  a synthetic fjord. Figures: `naeroyfjord_topography_real.png`.
- **Atmosphere** (`10_naeroyfjord_atmosphere.jl`): terrain-following compressible LES over
  the **real** Nærøyfjord. **Stable config = 600 m wall smoothing (slope 1.48) + CFL 0.5**
  (a 350 m/CFL-0.7 run went NaN at ~17 min — caught by running longer). Rotating-wind run
  is stable through the full cross→down-fjord sweep and shows **cross-fjord blocking →
  down-fjord gap jet** (|u|→16.5 m/s along axis). Fig/movie: `naeroyfjord_atmosphere_rotate600.*`.
- **Ocean** (`10_naeroyfjord_ocean.jl`): hydrostatic **+ CATKE** on the immersed real
  fjord, rotating wind stress. 4 h run is stable and shows the **stagnant → mixing**
  transition: cross-fjord does little; down-fjord wind erodes the fresh surface lens and
  mixes salty deep water to the surface. Fig/movie: `naeroyfjord_ocean_ocean4h.*`.

**Two-way coupling — WORKS (after a 2-line upstream fix).** Initially blocked: the air–sea
coupler assumed the *anelastic* reference state, so a `CompressibleDynamics` atmosphere hit
`FieldError: Nothing has no field density` in `interpolate_state!` (and then a
scalar-indexing error in `surface_layer_height` on the terrain-following GPU grid). Both
are fixed — fall back to `terrain_reference_density`/`surface_pressure`, and wrap the scalar
`zspacing` read in `@allowscalar`. Submitted upstream as **NumericalEarth.jl PR #350**;
reproduced in-script (`10_naeroyfjord_coupled_les.jl`) as two method overrides so it runs
today. **Verified on GPU:** the terrain-following compressible atmosphere + hydrostatic
CATKE immersed-fjord ocean steps stably, air–sea fluxes (Qsens ≈ 15 W/m²) computed at the
interface, ocean responding to the coupled wind stress. A 90-min capstone run (full
cross→down-fjord rotation) is the headline result.

**Bugs found & fixed by running (7):** in-function `include` world-age (10a real path);
forcing re-keyed to prognostic names (`model.forcing.ρu.forcing.geostrophic_velocity`);
`3e1` float-literal lexing in the inverse projection; missing `using NumericalEarth`;
`estimate_maximum_Δt` spherical-grid assumption on immersed grids (pass explicit `Δt`);
ocean WENO halo ≥ (7,7,4); short stability gate masking a 17-min blow-up.

**Run logs:** `logs/`. Live state: `logs/autonomous_state.md`.
