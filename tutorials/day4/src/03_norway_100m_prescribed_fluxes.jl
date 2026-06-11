# # Fjords as boundary conditions: 100 m coupled air–land flow over coastal Lofoten
#
# *Boundary heterogeneity writes turbulence into the fluid — case 3 of 3.*
#
# The flagship visual example: a real-terrain atmospheric LES over a
# 100 km × 100 km patch of coastal northern Norway (Lofoten) at a 100 m horizontal
# *production target*. Lofoten is a ~160 km chain of granite peaks that rise
# *directly from the sea* — the highest, Higravstinden, reaches 1161 m, and dozens
# of summits stand 600–1300 m above steep, narrow fjords such as Trollfjord. There
# is essentially no coastal plain to soften the transition: the ocean meets a wall
# of rock cut by fjord gaps. That geometry, **plus the land/sea surface contrast**,
# is the entire forcing of this experiment.
#
# ## Two kinds of boundary heterogeneity, at once
#
# This case layers two heterogeneities the atmosphere must respond to:
#
#  1. **Orography.** The single number that organizes stratified flow over a peak is
#     the **nondimensional mountain height** `M = N h / U` (an inverse Froude number;
#     Smith 1989, Bauer et al. 2000). For our peaks (`h ≈ 1100` m), free-tropospheric
#     stratification `N ≈ 0.0122 s⁻¹` (`N² = 1.5e-4`), and inflow `U = 12 m/s`, we get
#     **M ≈ 1.35** — the nonlinear regime with flow splitting / windward stagnation,
#     **gap / fjord jets** (Bernoulli acceleration; the high-latitude cousin of the
#     Greenland tip jet — Doyle & Shapiro 1999), lee eddies / vortex shedding, and
#     vertically propagating **mountain waves** (`λ_z = 2π U / N ≈ 6 km`, which is why
#     the domain is 12 km deep with a 4 km sponge).
#
#  2. **Surface moisture.** Rather than *prescribe* the surface fluxes, we couple the
#     atmosphere to a **land-surface model** (`AtmosphereLandModel`) whose surface
#     wetness varies in space: the **fjords and sea are wet** (saturated), the **land
#     is dry**. In a winter marine cold-air-outbreak regime — cold air over a warmer
#     surface — the wet water bodies lose heat as strong *latent + sensible* flux and
#     light up convective plumes, while the dry land exchanges far less. The fluxes
#     are *computed by Monin–Obukhov similarity theory from the instantaneous surface
#     state*, so the wet/dry pattern (and its interaction with the terrain jets) is an
#     emergent boundary condition, not a number we typed in.
#
# This case depends on Breeze's **terrain-following coordinates** and **acoustic
# substepping**, and on NumericalEarth's **`AtmosphereLandModel`** coupling. It loads a
# cached topography artifact produced by `03a_prepare_norway_topography.jl` (real
# Kartverket DTM); it does **not** download or reproject DEM data live.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
using Breeze: PiecewiseStretchedDiscretization, CompressibleDynamics,
              SplitExplicitTimeDiscretization, UpperSponge
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization,
                                             TwoLevelDecay, SlopeInsideInterpolation,
                                             materialize_terrain!
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

## Per-phase wall-clock timing with a forced flush, for visibility into the
## (compile-dominated) startup of this large compressible terrain run.
const _t0 = Ref(time_ns())
checkpoint(msg) = (@info @sprintf("⏱ %-26s %8.1f s", msg, 1e-9 * (time_ns() - _t0[])); flush(stderr))

Random.seed!(100)

config = RunConfig("03_norway_100m")
arch = choose_architecture()
gpu_report()
## The coupled `EarthSystemModel` clock follows the atmosphere; we run the whole
## stack in Float32 (the terrain compressible solver's native precision).
Oceananigans.defaults.FloatType = Float32
FT = Float32
nothing #hide

# ## Load the cached topography
#
# Produced by `03a` from the real Kartverket DTM (or a synthetic Lofoten fallback).
# `land_mask` is 1 over land, 0 over water — we use its complement as the
# **fjord/sea (wet) fraction**.

const topo_path = joinpath("thursday", "data", "norway_lofoten_100m_topography.jld2")
isfile(topo_path) || error("Missing topography artifact $topo_path — run 03a_prepare_norway_topography.jl first.")

topo = load(topo_path)
xt, yt = topo["x"], topo["y"]
h_data = topo["h"]
land_data = topo["land_mask"]
@info "Loaded topography" topo_path size_h = size(h_data) source = topo["source_metadata"].source

## A bilinear interpolation of the cached arrays, evaluated on the CPU when building
## the terrain and the surface land/water fields.
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
# 100 km × 100 km × 12 km. We use Breeze's built-in `PiecewiseStretchedDiscretization`
# to build the vertical coordinate — fine ~120 m cells through the boundary layer and
# lower troposphere where the convection and gap jets live, coarsening to ~800 m aloft
# where only the mountain waves matter — instead of hand-coding a stretching formula.
# The faces are then wrapped in a `TerrainFollowingVerticalDiscretization`, whose
# `TwoLevelDecay` relaxes the terrain-following surfaces back to flat with height so
# the coordinate is smooth under the sponge.

const Lx = 100kilometers
const Ly = 100kilometers
const Lz = 12kilometers

## ~200 m horizontal grid for a long run (≈34 M cells). Production: 1000×1000;
## quick teaching run: 256×256.
const Nx = 512
const Ny = 512

z_faces = PiecewiseStretchedDiscretization(z  = [0, 3000, 6000, Int(Lz)],
                                           Δz = [120, 120, 400, 800])
const Nz = length(z_faces) - 1

z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                          small_scale_height = Lz / 8))

memory_report(Nx, Ny, Nz; FT, nfields = 6)

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_coord,
                       topology = (Periodic, Periodic, Bounded))

## Carve the real terrain into the grid. The outer rim was tapered to flat in `03a`,
## so the periodic boundaries see a clean buffer; the central ≈70 km is the science window.
checkpoint("start")
materialize_terrain!(grid, (x, y) -> h_fun(x, y))
@info "Terrain materialized into grid."
checkpoint("terrain materialized")

# ## Compressible dynamics with acoustic substepping
#
# Terrain-following compressible dynamics use a split-explicit time discretization
# with acoustic substeps and an upper sponge to absorb the vertically propagating
# mountain waves before they reflect off the model top. The free-tropospheric
# stratification `N²` sets `M = N h / U`; staying near `M ≈ 1.35` keeps enough flow
# going *over* the ridges to launch waves while splitting the rest through the fjords.

const θ₀ = FT(272)          # K, cold-airmass surface potential temperature (the inflow)
const p₀ = FT(1e5)          # Pa
const N² = FT(1.5e-4)       # s⁻², free-tropospheric stratification (N ≈ 0.0122 s⁻¹)
const g  = FT(9.81)

potential_temperature_profile(z) = θ₀ * exp(N² * z / g)

sponge_depth = 4kilometers
time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = sponge_depth))

dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)
checkpoint("dynamics built")

# ## The atmosphere (coupling-ready, moist)
#
# `atmosphere_simulation` builds a Breeze `AtmosphereModel` wrapped in a `Simulation`,
# pre-wired for coupling: the bottom boundary conditions on momentum, energy, and
# moisture are blank 2D fields the `AtmosphereLandModel` coupler fills each step from
# the similarity-theory surface fluxes. We do **not** set surface fluxes by hand. The
# atmosphere is moist (warm-phase saturation-adjustment microphysics) so the wet fjords
# can actually evaporate into it.

atmosphere = atmosphere_simulation(grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = 68))
checkpoint("atmosphere built")

# ## The land surface: wet fjords, dry land
#
# A 2D `SlabLand` over the same horizontal extent carries a skin temperature, soil
# water, and a diagnostic surface saturation `𝒮 ∈ [0, 1]`. We use the conservative
# `VariablySaturatedHydrology` and a `WaterCoupledEnergy` closure with a **warm deep
# reservoir** (`deep_temperature = 280 K`): the surface is held warm relative to the
# cold inflowing air, so the land model supplies a cold-air-outbreak surface flux.
# The **wet/dry contrast is set spatially**: water storage is near-saturation over the
# fjords/sea (`land_mask = 0`) and near-zero over the land (`land_mask = 1`).

land_grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (grid.Hx, grid.Hy),
                            x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                            topology = (Periodic, Periodic, Flat))

hydrology = VariablySaturatedHydrology(eltype(land_grid);
    slab_depth = 1.0, porosity = 0.4, residual_liquid_fraction = 0.05,
    storage_height = 1000, critical_saturation = 0.5,
    retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

const T_surface = FT(280)   # K, warm surface (sea / coast) under the cold airmass
energy = WaterCoupledEnergy(eltype(land_grid);
    dry_heat_capacity = 1480 * 1500 * 0.10, liquid_heat_capacity = 4186,
    reference_temperature = 273.15, deep_temperature = T_surface,
    deep_time_scale = 12hours,
    advect_deep_liquid_energy = false, advect_surface_liquid_energy = false)

land = SlabLand(land_grid; hydrology, energy)

## Wet (near-saturated) over water, dry over land. Water storage Mˡᵃ⁺ = ρˡ ν D.
Mˡᵃ⁺  = hydrology.porosity * hydrology.slab_depth * 1000
M_wet = FT(0.95) * Mˡᵃ⁺
M_dry = FT(0.02) * Mˡᵃ⁺
ocean_fraction(x, y) = 1 - land_fun(x, y)          # 1 over fjords/sea, 0 over land
M_init(x, y) = M_dry + (M_wet - M_dry) * ocean_fraction(x, y)

set!(land; T = T_surface, M = M_init)
Oceananigans.TimeSteppers.update_state!(land)
checkpoint("land built")

# ## Couple the atmosphere and land
#
# The surface specific humidity is solved by `DryLayerHumidity`: a Fickian vapor-flux
# balance through an unresolved dry layer whose depth grows as the surface dries. Wet
# fjords (`𝒮 ≥ 𝒮ᶜ`) present a saturated skin and evaporate freely; dry land has a deep
# dry layer that throttles evaporation to near zero. `AtmosphereLandModel` wires the
# turbulent fluxes (sensible, latent, momentum) between the two components by
# similarity theory and owns the coupled time step.

al_interface = atmosphere_land_interface(land_grid, atmosphere, land;
    specific_humidity = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth = 0.05,
                                                    critical_saturation = 0.5,
                                                    dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-4,
                                                     molecular_diffusivity = 2.5e-5,
                                                     tortuosity_model = MillingtonQuirk()),
        thermal_exchange_depth = 0.10,
        porosity = hydrology.porosity))

# ## Initial conditions
#
# A stratified, cold reference profile with a mean westerly inflow the terrain
# deflects, plus small near-surface θ perturbations to seed 3-D turbulence and a
# modest near-surface humidity. Compressible models on terrain-following grids must be
# initialized in discrete hydrostatic balance, so we seed the density from the
# dynamics' `terrain_reference_density`.

const U₀  = FT(12)     # m s⁻¹ mean wind (westerly, onshore)
const δθ  = FT(0.3)    # K, near-surface θ perturbation
const zδ  = FT(500)
const qᵗ₀ = FT(2e-3)   # kg/kg, modest cold-air humidity

ϵ() = rand(FT) - FT(0.5)
uᵢ(x, y, z) = U₀
θᵢ(x, y, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀

let N = sqrt(N²), h_peak = maximum(h_data)
    @info @sprintf("Nondimensional mountain height M = N h / U = %.2f  (N = %.4f s⁻¹, h_peak = %.0f m, U = %.0f m/s)",
                   N * h_peak / U₀, N, h_peak, U₀)
end

set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = uᵢ, v = 0, w = 0, qᵗ = qᵢ, enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)
checkpoint("set! done")

model = AtmosphereLandModel(atmosphere, land;
                            atmosphere_land_interface = al_interface,
                            clock = Oceananigans.TimeSteppers.Clock{FT}(time = 0))
checkpoint("coupled model built")

# ## Simulation
#
# The split-explicit scheme advances on a large advective outer step while acoustic
# substeps handle the fast sound waves. We run **90 min** so the mountain-wave field,
# the gap jets, and the convective plumes over the wet fjords develop.

simulation = Simulation(model; Δt = 1.0, stop_time = 90minutes)
conjure_time_step_wizard!(simulation, cfl = 0.5, max_Δt = 10.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    a = sim.model.atmosphere.model
    𝒮 = sim.model.land.saturation
    Q = sim.model.interfaces.atmosphere_land_interface.fluxes.sensible_heat
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s, 𝒮∈[%.2f,%.2f], max|Qsens| %.0f W/m²",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, a.velocities.w), minimum(𝒮), maximum(𝒮), maximum(abs, interior(Q)))
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# ## Outputs
#
# Static terrain + land/water fields (for the terrain visualization); high-frequency
# near-surface wind and a vertical transect; and the evolving land surface state.

u, v, w = atmosphere.model.velocities
θ = liquid_ice_potential_temperature(atmosphere.model)

xc = [xnode(i, grid, Center()) for i in 1:Nx]
yc = [ynode(j, grid, Center()) for j in 1:Ny]
h_surface = reshape(FT.([h_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]), Nx, Ny, 1)
water_surface = reshape(FT.([ocean_fraction(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]), Nx, Ny, 1)
h_field = Field{Center, Center, Nothing}(grid)
water_field = Field{Center, Center, Nothing}(grid)
interior(h_field) .= Oceananigans.on_architecture(arch, h_surface)
interior(water_field) .= Oceananigans.on_architecture(arch, water_surface)

jmid = Ny ÷ 2 + 1
k_surface = 2

slice_outputs = (
    u_xy = view(u, :, :, k_surface),
    v_xy = view(v, :, :, k_surface),
    w_xy = view(w, :, :, k_surface),
    w_xz = view(w, :, jmid, :),
    θ_xz = view(θ, :, jmid, :),
)

simulation.output_writers[:statics] = JLD2Writer(atmosphere.model, (; h = h_field, water = water_field);
    filename = output_name(config, "statics"), schedule = IterationInterval(typemax(Int)),
    overwrite_existing = true)
simulation.output_writers[:slices] = JLD2Writer(atmosphere.model, slice_outputs;
    filename = slice_name(config), schedule = TimeInterval(15seconds), overwrite_existing = true)
simulation.output_writers[:land] = JLD2Writer(model, (; 𝒮 = land.saturation, T = land.temperature);
    filename = output_name(config, "land"), schedule = TimeInterval(15seconds), overwrite_existing = true)

write_once!(simulation.output_writers[:statics], atmosphere.model)
checkpoint("statics written; starting run!")

# ## Go time
run!(simulation)

# ## Visualization
#
# Four panels make the coupled flow interpretable: (top-left) the **terrain with the
# land/water mask** — the boundary the flow reads; (top-right) the near-surface
# wind-speed, showing gap jets threading the fjords and wakes downstream of the
# islands; (bottom-left) a vertical `w` transect through the mountain line, showing the
# mountain-wave train capped by the sponge; (bottom-right) the surface saturation `𝒮`
# (wet fjords vs dry land), the moisture heterogeneity that drives the differential
# surface flux.

using CairoMakie

stat = output_name(config, "statics")
if isfile(slice_name(config)) && isfile(stat)
    u_xy = FieldTimeSeries(slice_name(config), "u_xy")
    v_xy = FieldTimeSeries(slice_name(config), "v_xy")
    w_xz = FieldTimeSeries(slice_name(config), "w_xz")
    𝒮_ts = FieldTimeSeries(output_name(config, "land"), "𝒮")
    times = w_xz.times
    Nt = length(times)

    h_ts     = FieldTimeSeries(stat, "h")
    water_ts = FieldTimeSeries(stat, "water")

    xs, ys, _ = nodes(u_xy)
    xz_x, _, xz_z = nodes(w_xz)
    xkm, ykm = xs ./ 1e3, ys ./ 1e3

    n = Observable(Nt)
    speed = @lift sqrt.(interior(u_xy[$n], :, :, 1).^2 .+ interior(v_xy[$n], :, :, 1).^2)
    wn    = @lift interior(w_xz[$n], :, 1, :)
    𝒮n    = @lift interior(𝒮_ts[$n], :, :, 1)
    title = @lift "Coupled flow over Lofoten — t = " * prettytime(times[$n])

    hh    = interior(h_ts[1], :, :, 1)
    water = interior(water_ts[1], :, :, 1)

    fig = Figure(size = (1150, 1000))
    Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)

    ax_terr = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "terrain (m) + water (cyan)", aspect = 1)
    hmt = heatmap!(ax_terr, xkm, ykm, hh, colormap = :terrain, colorrange = (0, maximum(hh)))
    ## Overlay water as a translucent cyan mask (fjords/sea).
    contourf!(ax_terr, xkm, ykm, water; levels = [0.5, 1.0], colormap = [(:cyan, 0.0), (:cyan, 0.55)])
    Colorbar(fig[1, 0], hmt, label = "elevation (m)")

    ax_spd = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "y (km)", title = "near-surface wind speed (m s⁻¹)", aspect = 1)
    hms = heatmap!(ax_spd, xkm, ykm, speed, colormap = :speed)
    Colorbar(fig[1, 3], hms)

    ax_w = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "w transect (m s⁻¹)")
    wlim = max(1e-3, maximum(abs, interior(w_xz[Nt])))
    hmw = heatmap!(ax_w, xz_x ./ 1e3, xz_z ./ 1e3, wn, colormap = :balance, colorrange = (-wlim, wlim))
    Colorbar(fig[2, 0], hmw)

    ax_𝒮 = Axis(fig[2, 2], xlabel = "x (km)", ylabel = "y (km)", title = "surface saturation 𝒮 (wet fjords / dry land)", aspect = 1)
    hm𝒮 = heatmap!(ax_𝒮, xkm, ykm, 𝒮n, colormap = :dense, colorrange = (0, 1))
    Colorbar(fig[2, 3], hm𝒮)

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
#     doi:10.1016/S0065-2687(08)60052-7 — `M = N h / U` and the nonlinear-regime diagram.
#   - Bauer, M. H., et al. (2000). Strongly nonlinear flow over and around a 3-D mountain.
#     *J. Atmos. Sci.*, 57, 3971–3991 — 3-D regime curves supporting `M ≈ 1–2`.
#   - Doyle, J. D., & Shapiro, M. A. (1999). The Greenland tip jet. *Tellus A*, 51, 728–748 —
#     tip/barrier jets from flow splitting; the high-latitude analog for fjord/gap jets.
#   - Papritz, L., et al. (2015). Wintertime cold-air outbreaks in the Irminger and Nordic Seas.
#     *J. Climate*, 28, 342–364 — O(100s) W/m² surface heat loss in CAOs.
#   - Manabe, S. (1969). Climate and the ocean circulation: the atmospheric circulation and the
#     hydrology of the earth's surface. *Mon. Wea. Rev.*, 97, 739–774 — bucket land-surface hydrology.
#   - Kartverket (Norwegian Mapping Authority) national DTM — https://hoydedata.no/ — the real
#     Lofoten terrain (and land/water mask), CC BY 4.0.

nothing #hide
