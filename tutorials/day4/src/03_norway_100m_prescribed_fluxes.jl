# # Fjords as boundary conditions: 100 m atmospheric flow over coastal Lofoten
#
# *Boundary heterogeneity writes turbulence into the fluid — case 3 of 3.*
#
# The flagship visual example: a real-terrain atmospheric LES over a
# 100 km × 100 km patch of coastal northern Norway (Lofoten) at a 100 m horizontal
# *production target*. Lofoten is a ~160 km chain of granite peaks that rise
# *directly from the sea* — the highest, Higravstinden, reaches 1161 m, and dozens
# of summits stand 600–1300 m above steep, narrow fjords such as Trollfjord. There
# is essentially no coastal plain to soften the transition: the ocean meets a wall
# of rock cut by fjord gaps. That geometry, plus the land/sea flux contrast, is the
# entire forcing of this experiment.
#
# ## Why this terrain produces interesting flow
#
# The single number that organizes stratified flow over a peak is the
# **nondimensional mountain height** `M = N h / U` (an inverse Froude number;
# Smith 1989, Bauer et al. 2000). For our peaks (`h ≈ 1100` m), free-tropospheric
# stratification `N ≈ 0.0122 s⁻¹` (`N² = 1.5e-4`), and inflow `U = 12 m/s`, we get
# **M ≈ 1.35** — squarely in the nonlinear regime where the flow does *all four* of
# the things we want to show at once:
#
#   - **Flow splitting / windward stagnation** — incoming air cannot all go over the
#     ridges, so it splits and runs around the islands.
#   - **Gap / fjord jets** — the split flow is funneled and accelerated through the
#     fjords (Bernoulli acceleration in orographic descent), the high-latitude cousin
#     of Greenland tip and barrier jets (Doyle & Shapiro 1999; Moore & Renfrew 2005),
#     where observed jets reach tens of m/s.
#   - **Lee eddies / vortex shedding** — wakes and counter-rotating vortex pairs form
#     downstream of the islands.
#   - **Gravity / mountain waves** — the fraction of flow that *does* climb the ridges
#     launches vertically propagating mountain waves, with vertical wavelength
#     `λ_z = 2π U / N ≈ 6 km`, which is why the domain is 12 km deep with a 4 km sponge
#     to absorb them before they reflect off the top.
#
# Keep `M` in roughly 1–2: too small (weak `N` or fast `U`) and everything just flows
# over with weak wakes; too large (strong `N` or slow `U`) and the flow blocks
# completely and the over-mountain waves disappear. `M` is *sensitive near this
# transition* and depends on the effective (resolution-limited) peak height — see the
# resolution note below.
#
# This case depends on Breeze's **terrain-following coordinates** and **acoustic
# substepping** (the `glw/terrain-following-substepping` branch — the "PR #712
# equivalent"). It loads a cached topography artifact produced by
# `03a_prepare_norway_topography.jl`; it does **not** download or reproject DEM
# data live. We run a single dry configuration sized for one H100.

using Breeze
using Breeze: BulkDrag
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization,
                                             TwoLevelDecay, SlopeInsideInterpolation,
                                             materialize_terrain!
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

# Per-phase wall-clock timing with a forced flush, for visibility into the
# (compile-dominated) startup of this large compressible terrain run.
const _t0 = Ref(time_ns())
checkpoint(msg) = (@info @sprintf("⏱ %-26s %8.1f s", msg, 1e-9 * (time_ns() - _t0[])); flush(stderr))

Random.seed!(100)

config = RunConfig("03_norway_100m")
arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float32
FT = Float32
nothing #hide

# ## Load the cached topography
#
# Produced by `03a`. If it is missing, run that source first (the synthetic
# fallback needs no network or GDAL).

const topo_path = joinpath("thursday", "data", "norway_lofoten_100m_topography.jld2")
isfile(topo_path) || error("Missing topography artifact $topo_path — run 03a_prepare_norway_topography.jl first.")

topo = load(topo_path)
xt, yt = topo["x"], topo["y"]
h_data = topo["h"]
land_data = topo["land_mask"]
@info "Loaded topography" topo_path size_h = size(h_data) source = topo["source_metadata"].source

# A bilinear interpolation of the cached arrays, evaluated on the CPU by
# `materialize_terrain!` and for building static output fields.

function bilinear(arr, xs, ys)
    x0, x1 = first(xs), last(xs); y0, y1 = first(ys), last(ys)
    nx, ny = length(xs), length(ys)
    dx = (x1 - x0) / (nx - 1); dy = (y1 - y0) / (ny - 1)
    return function (x, y)
        fx = clamp((x - x0) / dx, 0, nx - 1 - 1e-6)
        fy = clamp((y - y0) / dy, 0, ny - 1 - 1e-6)
        i = floor(Int, fx) + 1; j = floor(Int, fy) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        @inbounds (arr[i, j]   * (1 - tx) * (1 - ty) + arr[i+1, j]   * tx * (1 - ty) +
                   arr[i, j+1] * (1 - tx) * ty       + arr[i+1, j+1] * tx * ty)
    end
end

h_fun    = bilinear(h_data, xt, yt)
land_fun = bilinear(land_data, xt, yt)

# ## Grid and terrain-following vertical coordinate
#
# 100 km × 100 km × 12 km at the **100 m production resolution**:
# 1000 × 1000 × 160 ≈ 160 million cells. This resolves the O(1–3 km) fjord gaps with
# ≥4–6 cells across, so the gap jets, windward flow splitting, and lee eddies are
# sharp rather than blurred.
#
# !!! note "Runs on one H100 in ≈ 30 minutes"
#     At 160 M cells this uses ≈ 14 GiB and completes 10 min of simulated time in
#     ≈ 29 min wall on one H100 (model construction ≈ 46 s). This is only practical
#     because the terrain reference-state (Exner) build was moved from a per-cell
#     scalar host loop to a GPU column kernel in Breeze — without that fix,
#     constructing the model at this size took *hours*. For a quick teaching run,
#     coarsen to e.g. `Nx = Ny = 256, Nz = 64` (≈ 4 M cells, a few minutes), at the
#     cost of blurring the narrowest fjord jets and lowering the effective `M`.

const Lx = 100kilometers
const Ly = 100kilometers
const Lz = 12kilometers

## 100 m production grid (≈160M cells, ~30 min on one H100). Coarsen for a quick run.
const Nx = 1000
const Ny = 1000
const Nz = 160

function stretched_z_faces(Lz, Nz; stretching = 1.1)
    σ(k) = (k - 1) / Nz
    return [Lz * expm1(stretching * σ(k)) / expm1(stretching) for k in 1:Nz+1]
end

r_faces = stretched_z_faces(Lz, Nz)
z_faces = TerrainFollowingVerticalDiscretization(r_faces;
                  formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                              small_scale_height = Lz / 8))

memory_report(Nx, Ny, Nz; FT, nfields = 6)

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
                       topology = (Periodic, Periodic, Bounded))

# Carve the terrain into the grid. The outer rim was tapered to flat in `03a`, so
# the periodic boundaries see a clean buffer; the central ≈70 km is the science window.
#
# **Terrain data.** The cached topography (built by `03a`) comes from real DEMs. Use
# **Copernicus GLO-30** (30 m global DSM; WGS84/EPSG:4326 horizontal, EGM2008 vertical)
# — over Norway it is itself infilled with Kartverket data, so it is both globally
# consistent and locally faithful. For maximum fidelity, source the native
# **Kartverket DTM** from hoydedata.no (down to 1 m LiDAR). Reproject to **UTM zone 33N
# (EPSG:32633)** — the correct projected CRS for Lofoten's longitudes (~13–16°E) — to
# get metric x/y for the 100 km box, resample to the grid (bilinear/cubic), and taper
# the outer rim flat (done in `03a`). Note the DSM-vs-DTM and EGM2008-geoid-vs-ellipsoid
# distinctions can bias peak heights by tens of meters — small for 1100 m peaks but
# worth confirming.
checkpoint("start")
materialize_terrain!(grid, (x, y) -> h_fun(x, y))
@info "Terrain materialized into grid."
checkpoint("terrain materialized")

# ## Compressible dynamics with acoustic substepping
#
# Terrain-following compressible dynamics use a split-explicit time discretization
# with acoustic substeps and an upper sponge to absorb vertically propagating waves.

const θ₀ = FT(285)          # K, surface reference potential temperature
const p₀ = FT(1e5)          # Pa
## Stratification sets M = N h / U. N² = 1.5e-4 ⇒ N ≈ 0.0122 s⁻¹; with h ≈ 1100 m and
## U = 12 m/s this gives M ≈ 1.35 — firmly in the Smith (1989) / Bauer et al. (2000)
## nonlinear regime (wave breaking + windward stagnation/flow splitting + lee vortices).
## High-latitude winter is often more stable, so 1.5–2.5e-4 is defensible; staying below
## M ≈ 2 keeps enough flow going *over* the ridges to launch mountain waves.
const N² = FT(1.5e-4)       # s⁻², free-tropospheric stratification (N ≈ 0.0122 s⁻¹)
const g  = FT(9.81)
const cₚ = FT(1004)

potential_temperature_profile(z) = θ₀ * exp(N² * z / g)

## Hydrostatic mountain-wave vertical wavelength λ_z = 2π U / N ≈ 6.2 km (U = 12, N = 0.0122).
## The 12 km domain fits ~2 wavelengths; the 4 km sponge (base ≈ 8 km) absorbs upward wave
## energy before top reflection without reaching down into the wave-breaking layer. A too-thin
## or too-weak sponge reflects waves and produces spurious standing waves in the lee.
sponge_depth = 4kilometers
time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = sponge_depth))

dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)
checkpoint("dynamics built")

# ## Prescribed surface fluxes over land and ocean: pick a regime and commit to it
#
# The land/sea mask imposes flux heterogeneity on top of the geometry. There are
# **two physically distinct high-latitude regimes, and their heat-flux signs are
# opposite — do not mix them** (mixing land-heated and ocean-heated values gives an
# unphysical, internally inconsistent contrast):
#
#   1. **Fair-weather summer day (land-heated).** Land heats more than the cool sea:
#      `Qʰ_land ≈ +150`, `Qʰ_ocean ≈ +40 W/m²`.
#   2. **Winter marine cold-air outbreak (ocean-heated) — the flagship choice here.**
#      Cold air advects off snow-covered land over a much warmer sea, and the *ocean*
#      becomes the dominant heat source: `Qʰ_ocean ≈ +200 W/m²` (Nordic-Seas
#      observations reach O(100–300) W/m² sensible alone; Papritz et al. 2015;
#      Renfrew et al. 2019), while the cold land is near-neutral to slightly stable
#      (`Qʰ_land ≈ 0 to -10 W/m²`). This lights up vigorous convective plumes in the
#      island wakes and pairs naturally with the terrain-forced jets.
#
# For momentum we use Breeze's `BulkDrag`, which applies the velocity-dependent
# quadratic surface stress `Jᵘ = -Cᴰ |U| ρu` (opposing the *local* wind, so it
# vanishes in calm air and lee eddies — unlike a constant prescribed stress). We
# use a single representative neutral drag coefficient; the land/ocean contrast is
# carried by the heat flux, which is the dominant driver of the cold-air outbreak.
# (A land/ocean-varying `Cᴰ` would need a spatially varying coefficient, which
# `BulkDrag` does not currently take.)
#
# We default to the **cold-air-outbreak** regime below. To switch to the summer case,
# set `Qʰ_land = 150`, `Qʰ_ocean = 40` and keep both positive (land > ocean). The
# heat flux is precomputed as a 2D array on the grid (GPU-safe).

## --- Cold-air-outbreak (winter, ocean-heated) regime ---
const Qʰ_ocean = FT(200);  const Qʰ_land = FT(-10)    # W m⁻²  (ocean is the heat source)
## --- Summer (land-heated) alternative: Qʰ_land = FT(150); Qʰ_ocean = FT(40) ---
const Cᴰ = FT(1.5e-3)         # neutral bulk drag coefficient (uniform)
const gustiness = FT(0.5)     # m s⁻¹, floor on |U| so calm air does not divide by zero

q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
constants = ThermodynamicConstants()
const ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)

xc = [xnode(i, grid, Center()) for i in 1:Nx]
yc = [ynode(j, grid, Center()) for j in 1:Ny]
ℵ = [land_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]   # land fraction

ρθ_surface = Oceananigans.on_architecture(arch, FT.((ℵ .* Qʰ_land .+ (1 .- ℵ) .* Qʰ_ocean) ./ cₚ))

# !!! note "Sign convention (verified)"
#     A bottom flux is *added* to the tendency in Breeze, so a positive `ρθ` flux
#     heats the atmosphere (warm ocean `Qʰ_ocean > 0` drives the cold-air-outbreak
#     plumes; cold land `Qʰ_land < 0` stabilizes). `BulkDrag` supplies the negative
#     (wind-opposing) momentum flux — matching `bomex.jl` / the prescribed-SST example.

## Velocity-dependent surface drag via Breeze's BulkDrag (the same object works for
## both ρu and ρv; the momentum direction is inferred from the field location).
drag = BulkDrag(coefficient = Cᴰ, gustiness = gustiness)
ρθ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρθ_surface))
ρu_bcs = FieldBoundaryConditions(bottom = drag)
ρv_bcs = FieldBoundaryConditions(bottom = drag)

# ## Model

advection = WENO(order = 9)
closure = SmagorinskyLilly()

model = AtmosphereModel(grid; dynamics, advection, closure,
                        timestepper = :AcousticRungeKutta3,
                        boundary_conditions = (; ρθ = ρθ_bcs, ρu = ρu_bcs, ρv = ρv_bcs))
checkpoint("model built")

# ## Initial conditions
#
# The stratified reference profile plus small near-surface perturbations and a
# mean westerly inflow the terrain can deflect.
#
# Compressible models on terrain-following grids must be initialized in discrete
# hydrostatic balance: we seed the density from `terrain_reference_density`
# (available because we passed `reference_potential_temperature`). Without this the
# prognostic density is zero and velocities are NaN at iteration 0.

## U₀ = 12 m/s keeps M = N h / U ≈ 1.35 in the 1–2 sweet spot and represents the wind
## regime that produces fjord gap jets and Greenland-type tip/barrier jets (gap-jet
## enhancement is typically 1.5–2× upstream). Onshore westerly flow (u = U₀, v = 0)
## drives flow into the steep seaward faces; rotate ~30° (southwesterly) for asymmetric,
## more realistic storm-track wakes.
const U₀ = FT(12)   # m s⁻¹ mean wind (westerly, onshore)
## Larger near-surface θ perturbation (0.3 K) seeds 3-D turbulence and speeds up spin-up
## of the cold-air-outbreak convective plumes without contaminating the wave field.
const δθ = FT(0.3)
const zδ = FT(500)

ϵ() = rand(FT) - FT(0.5)
uᵢ(x, y, z) = U₀
θᵢ(x, y, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)

## Report the *nominal* nondimensional mountain height (resolution lowers the achieved value).
let N = sqrt(N²), h_peak = maximum(h_data)
    @info @sprintf("Nondimensional mountain height M = N h / U = %.2f  (N = %.4f s⁻¹, h_peak = %.0f m, U = %.0f m/s)",
                   N * h_peak / U₀, N, h_peak, U₀)
end

set!(model, ρ = model.dynamics.terrain_reference_density,
     θ = θᵢ, u = uᵢ, v = 0, w = 0, enforce_mass_conservation = false)
checkpoint("set! done")
Oceananigans.TimeSteppers.update_state!(model)
checkpoint("update_state! done")

# ## Simulation
#
# The split-explicit scheme advances on a large advective outer step (≈10 s) while
# acoustic substeps handle the fast sound waves. We run **10 minutes** of simulated
# time so the wave field and gap jets begin to establish over the terrain.
#
# !!! note "The wake is still in spin-up"
#     Advective transit across a ~30 km island at 12 m/s is ~40 min, so 10 min lets
#     the mountain-wave field and the leading gap jets appear but leaves the lee
#     wakes and vortex shedding *still developing*. For statistically meaningful
#     wakes/eddies run 1–2 hours; this short run is the visual smoke, not a
#     converged-statistics experiment. (10 min also respects the performance caveat
#     above — see the grid section.)

simulation = Simulation(model; Δt = 1.0, stop_time = 10minutes)
conjure_time_step_wizard!(simulation, cfl = 0.5, max_Δt = 10.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), maximum(abs, sim.model.velocities.w))
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# ## Outputs
#
# Static terrain/mask fields; high-frequency near-surface wind and a vertical
# transect through a fjord/mountain line; sparse 3D fields.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)

# Build static surface fields from precomputed host arrays (the bilinear
# interpolators close over host arrays and cannot run in a GPU `set!` kernel).
h_surface = reshape(FT.([h_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]), Nx, Ny, 1)
ℵ_surface = reshape(FT.(ℵ), Nx, Ny, 1)
h_field = Field{Center, Center, Nothing}(grid)
ℵ_field = Field{Center, Center, Nothing}(grid)
interior(h_field) .= Oceananigans.on_architecture(arch, h_surface)
interior(ℵ_field) .= Oceananigans.on_architecture(arch, ℵ_surface)

jmid = Ny ÷ 2 + 1
k_surface = 2

slice_outputs = (
    u_xy = view(u, :, :, k_surface),
    v_xy = view(v, :, :, k_surface),
    w_xy = view(w, :, :, k_surface),
    w_xz = view(w, :, jmid, :),
    θ_xz = view(θ, :, jmid, :),
)

simulation.output_writers[:statics] = JLD2Writer(model, (; h = h_field, ℵ = ℵ_field);
    filename = output_name(config, "statics"), schedule = IterationInterval(typemax(Int)),
    overwrite_existing = true)
simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = slice_name(config), schedule = TimeInterval(15seconds), overwrite_existing = true)
## (No full-3D field output — the near-surface slices and transect drive the visualization.)

write_once!(simulation.output_writers[:statics], model)
checkpoint("statics written; starting run!")

# ## Go time
run!(simulation)

# ## Visualization
#
# A near-surface wind-speed map over the terrain and a vertical `w` transect
# through the fjord/mountain line.
#
# **What to watch.** The near-surface wind-speed map should show bright jets threading
# the fjord gaps and accelerating around island tips, with calmer split-flow
# stagnation on the windward seaward faces and turbulent wakes downstream. The
# vertical `w` transect through the fjord/mountain line should show coherent
# mountain-wave phase tilting upstream with height, capped by the sponge, and — if `M`
# is large enough — overturning/wave-breaking signatures above the lee slopes. At this
# 625 m smoke grid you see island-scale wakes and waves; the narrowest fjord jets only
# fully resolve at the 100 m production target (≥4–6 cells across a ~1–3 km gap).

using CairoMakie

if isfile(slice_name(config))
    u_xy = FieldTimeSeries(slice_name(config), "u_xy")
    v_xy = FieldTimeSeries(slice_name(config), "v_xy")
    w_xz = FieldTimeSeries(slice_name(config), "w_xz")
    times = w_xz.times
    Nt = length(times)

    xs, ys, _ = nodes(u_xy)
    xz_x, _, xz_z = nodes(w_xz)

    n = Observable(Nt)
    speed = @lift sqrt.(interior(u_xy[$n], :, :, 1).^2 .+ interior(v_xy[$n], :, :, 1).^2)
    wn = @lift interior(w_xz[$n], :, 1, :)
    title = @lift "Norway 100 m — t = " * prettytime(times[$n])

    fig = Figure(size = (1200, 520))
    Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
    ax1 = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "near-surface wind speed (m s⁻¹)", aspect = 1)
    ax2 = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "z (km)", title = "w transect (m s⁻¹)")
    hm1 = heatmap!(ax1, xs ./ 1e3, ys ./ 1e3, speed, colormap = :speed)
    wlim = max(1e-3, maximum(abs, interior(w_xz[Nt])))
    hm2 = heatmap!(ax2, xz_x ./ 1e3, xz_z ./ 1e3, wn, colormap = :balance, colorrange = (-wlim, wlim))
    Colorbar(fig[1, 0], hm1); Colorbar(fig[1, 3], hm2)

    save(figure_name(config, "norway_final_w_slice"), fig)
    if Nt > 1
        record(fig, movie_name(config, "norway_100m_prescribed_fluxes"), 1:Nt; framerate = 12) do i
            n[] = i
        end
    end
end

@info "Case 3 complete" run_stamp(config)...

# ## References
#
#   - Smith, R. B. (1989). Hydrostatic airflow over mountains. *Adv. Geophys.*, 31, 1–41.
#     doi:10.1016/S0065-2687(08)60052-7 — defines `M = N h / U` and the regime diagram for
#     wave breaking, windward stagnation/flow splitting, and lee-vortex onset.
#   - Bauer, M. H., Mayr, G. J., Vergeiner, I., & Pichler, H. (2000). Strongly nonlinear flow
#     over and around a 3-D mountain as a function of horizontal aspect ratio. *J. Atmos. Sci.*,
#     57, 3971–3991. doi:10.1175/1520-0469(2001)058<3971:SNFOAA>2.0.CO;2 — nonlinear 3-D regime
#     curves supporting the `M ≈ 1–2` target.
#   - Smith, R. B. (1980). Linear theory of stratified hydrostatic flow past an isolated mountain.
#     *Tellus*, 32, 348–364. doi:10.3402/tellusa.v32i4.10590 — `λ_z = 2π U / N`; sizes domain depth
#     and sponge.
#   - Doyle, J. D., & Shapiro, M. A. (1999). Flow response to large-scale topography: the Greenland
#     tip jet. *Tellus A*, 51, 728–748. doi:10.3402/tellusa.v51i5.14471 — tip/barrier jets from flow
#     splitting with Bernoulli acceleration; the high-latitude analog for fjord/gap jets.
#   - Moore, G. W. K., & Renfrew, I. A. (2005). Tip jets and barrier winds: a QuikSCAT climatology…
#     *J. Climate*, 18, 3713–3725. doi:10.1175/JCLI3455.1 — observed terrain-forced jet magnitudes
#     (peaks up to ~50 m/s); supports U = 12–15 m/s.
#   - Mayr, G. J., et al. (2007). Gap flows: results from MAP. *Quart. J. Roy. Meteor. Soc.*, 133,
#     881–896. doi:10.1002/qj.66 — gap-flow dynamics, hydraulic control, and the resolution
#     requirement (several cells across a gap).
#   - Vosper, S. B. (2004). Inversion effects on mountain lee waves. *Quart. J. Roy. Meteor. Soc.*,
#     130, 1723–1748. doi:10.1256/qj.03.63 — boundary-layer/inversion control of lee-wave and rotor
#     response.
#   - Papritz, L., et al. (2015). A Lagrangian climatology of wintertime cold-air outbreaks in the
#     Irminger and Nordic Seas… *J. Climate*, 28, 342–364. doi:10.1175/JCLI-D-14-00482.1 —
#     O(100s) W/m² ocean heat loss in CAOs; justifies the ocean-dominant flux contrast.
#   - Renfrew, I. A., et al. (2019). The Iceland Greenland Seas Project. *Bull. Amer. Meteor. Soc.*,
#     100, 1795–1817. doi:10.1175/BAMS-D-18-0217.1 — in-situ CAO sensible heat fluxes O(100–300) W/m².
#   - Copernicus DEM GLO-30 product handbook (ESA / Copernicus Data Space Ecosystem) —
#     https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model — 30 m global
#     DSM, WGS84/EPSG:4326 horizontal, EGM2008 vertical; Norway infilled with Kartverket data.
#   - Kartverket (Norwegian Mapping Authority), national DTM — https://hoydedata.no/ — highest-res
#     native Norwegian DTM (down to 1 m LiDAR), CC BY 4.0.

nothing #hide
