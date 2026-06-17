# # Wind-driven fjord mixing: a coupled atmosphere–land–ocean simulation over Sunnmøre
#
# *Boundary heterogeneity writes turbulence into the fluid — and the fluid on the
# other side of the surface is a living, mixing ocean.*
#
# A two-way coupled **atmosphere + land + ocean** simulation over the real Sunnmøre
# coast of western Norway, on a **latitude–longitude grid** with real **ETOPO2022**
# relief. One signed relief field gives both the terrain (the Sunnmøre Alps, carved
# into the terrain-following atmosphere) and the bathymetry (the fjords and shelf,
# an immersed bottom under the hydrostatic ocean). The atmosphere and ocean exchange
# momentum, heat, and moisture every step by similarity theory; over the coastline
# the coupler blends the air–land and air–sea fluxes by a per-cell `land_fraction`.
#
# ## The experiment
#
# We drive the atmosphere with a large-scale geostrophic wind whose **direction
# rotates over the run, from cross-valley to along-valley**, and watch how the
# **ocean mixed layer** responds. At ~62° N the inertial period is `2π/f ≈ 13.5 h`,
# so the rotation rate relative to that timescale (`t_rotate`) is the key knob.
#
# Both fluids start from **horizontally-uniform idealized profiles** (no data
# initialization): a stratified atmosphere with a mean wind, and an ocean at rest
# with a warm mixed layer over a stratified interior.

using Breeze
using NumericalEarth
using NumericalEarth: regrid_bathymetry, ETOPO2022
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: λnode, φnode, λnodes, φnodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using CUDA
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

_t0 = Ref(time_ns())
checkpoint(msg) = (@info @sprintf("⏱ %-28s %8.1f s", msg, 1e-9 * (time_ns() - _t0[])); flush(stderr))

Random.seed!(62)

arch = GPU()
gpu_report()
## Coupled runs: the EarthSystemModel clock is Float64, so all component grids match.
Oceananigans.defaults.FloatType = Float64
FT = Float64
nothing #hide

# ## Domain & grid (latitude–longitude)
#
# An ~90 km box over the Sunnmøre coast: south edge ≈ 62.0° N, east edge ≈ 7.5° E, so it
# opens onto the open Norwegian Sea in the NW (more ocean fetch) while the SE reaches the
# *heads* of several fjords — Hjørundfjorden and the Storfjorden→Sunnylvsfjorden/
# Geirangerfjorden/Tafjord system. `RUN_CLASS=production` selects the fine grid.

const PROD = get(ENV, "RUN_CLASS", "smoke") == "production"

center_lat = 62.40
center_lon = 6.63
## Half-spans in degrees for an ~90 km square (latitude ~111 km/°, longitude shrinks by
## cos) ⇒ south ≈ 62.0° N, east ≈ 7.5° E, NW corner ≈ (62.8° N, 5.76° E) in open water.
dlat = 45kilometers / 111320
dlon = 45kilometers / (111320 * cosd(center_lat))

Nλ = Nφ = PROD ? 384 : 160
Lz_a = 12kilometers   # atmosphere depth
Lz_o = 700meters      # ocean depth (deeper bathymetry is truncated to a flat floor)
Nz_o = PROD ? 40 : 20

longitude = (center_lon - dlon, center_lon + dlon)
latitude  = (center_lat - dlat, center_lat + dlat)

# ## Real relief from ETOPO2022 (terrain + bathymetry in one field)
#
# `regrid_bathymetry` maps the signed ETOPO2022 relief (land > 0, ocean < 0) straight
# onto our lat–lon grid — the canonical NumericalEarth pattern. We taper the relief to
# flat (sea level) within `taper_width` of the edges so the domain rim is a clean
# buffer, then split it into the ocean's immersed bottom and the atmosphere's terrain.

taper_width = 6kilometers

ocean_underlying = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ, Nz_o), halo = (7, 7, 7),
                                         longitude, latitude, z = (-Lz_o, 0),
                                         topology = (Bounded, Bounded, Bounded))

checkpoint("start")
relief = regrid_bathymetry(ocean_underlying; dataset = ETOPO2022(),
                           major_basins = Inf, height_above_water = nothing)
checkpoint("ETOPO relief regridded")

## Taper the signed relief to 0 (sea level) at the domain edges, using the metric
## offset of each lat–lon node from the box center. `edge_taper` (ThursdayLES) returns
## 1 in the interior and 0 at the rim.
λc = Array(λnodes(ocean_underlying, Center()))
φc = Array(φnodes(ocean_underlying, Center()))
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * cosd(center_lat)
Lx = 2 * dlon * m_per_deg_lon
Ly = 2 * dlat * m_per_deg_lat
taper = [edge_taper((λc[i] - center_lon) * m_per_deg_lon,
                    (φc[j] - center_lat) * m_per_deg_lat, Lx, Ly; taper_width)
         for i in 1:Nλ, j in 1:Nφ]

relief_cpu = Array(interior(relief, :, :, 1)) .* taper      # signed, tapered
land_elev  = max.(relief_cpu, 0.0)                          # terrain height (≥0)
bottom_cpu = min.(relief_cpu, 0.0)                          # sea floor z (≤0)
land_frac_cpu = FT.(relief_cpu .> 0)                        # 1 land, 0 ocean

@info "Relief" max_terrain = maximum(land_elev) max_depth = -minimum(bottom_cpu) water_fraction = sum(land_frac_cpu .== 0) / length(land_frac_cpu)

# ## Ocean grid: immersed bottom from the bathymetry

bottom_field = Field{Center, Center, Nothing}(ocean_underlying)
set!(bottom_field, bottom_cpu)
ocean_grid = ImmersedBoundaryGrid(ocean_underlying, GridFittedBottom(bottom_field))
checkpoint("ocean immersed grid")

# ## Atmosphere grid: terrain-following, real terrain carved in
#
# Fine ~120 m cells through the boundary layer, coarsening aloft; terrain-following
# surfaces relax to flat with height (`TwoLevelDecay`) under a 4 km sponge.

z_faces = PiecewiseStretchedDiscretization(z = [0, 3000, 6000, Int(Lz_a)], Δz = [120, 120, 400, 800])
Nz_a = length(z_faces) - 1
z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz_a/2, small_scale_height = Lz_a/8))

atmos_grid = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ, Nz_a), halo = (5, 5, 5),
                                   longitude, latitude, z = z_coord,
                                   topology = (Bounded, Bounded, Bounded))

## Terrain height as a function of (λ, φ): bilinear interpolation of the tapered land
## elevation over the lat–lon nodes (materialize_terrain! evaluates it on the CPU).
terrain_fun = bilinear(land_elev, λc, φc)
materialize_terrain!(atmos_grid, (λ, φ) -> terrain_fun(λ, φ))
checkpoint("terrain materialized")

land_grid = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ), halo = (atmos_grid.Hx, atmos_grid.Hy),
                                  longitude, latitude, topology = (Bounded, Bounded, Flat))

## Per-cell land fraction (drives the coupler's flux blending) on the exchange grid.
land_fraction = Field{Center, Center, Nothing}(ocean_underlying)
set!(land_fraction, land_frac_cpu)

# ## Atmosphere component (compressible, terrain-following, coupling-ready)

θ₀ = 280; p₀ = 1e5; N²_a = 1.0e-4; g = 9.81
potential_temperature_profile(z) = θ₀ * exp(N²_a * z / g)

time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = 4kilometers))
dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)

# ### Rotating large-scale wind (geostrophic; sweeps cross → along valley)
#
# `geostrophic_forcings(uᵍ, vᵍ)` drives the flow toward a geostrophic wind under
# Coriolis. We rotate it from cross-valley to along-valley over `t_rotate` by updating
# the stored geostrophic velocity each step (callback below).

## Inertial period at this latitude (2π/f ≈ 13.5 h at 62.35° N) sets the natural
## timescale: we rotate the wind cross→along over ¼ inertial period and run for ½.
f_coriolis = 2 * 7.2921e-5 * sind(center_lat)
inertial_period = 2π / f_coriolis

U_mag    = 12.0
α_valley = 0.0                  # fjord-axis angle (rad from east); 0 ⇒ along-valley = eastward
t_rotate = PROD ? inertial_period / 4 : 3hours
wind_params = (; U_mag, α_valley, t_rotate)
wind_angle(t, p) = (p.α_valley + π/2) + clamp(t / p.t_rotate, 0, 1) * (p.α_valley - (p.α_valley + π/2))
target_u(t, p) = p.U_mag * cos(wind_angle(t, p))
target_v(t, p) = p.U_mag * sin(wind_angle(t, p))

uᵍ0(z) = target_u(0.0, wind_params)
vᵍ0(z) = target_v(0.0, wind_params)
geostrophic = geostrophic_forcings(uᵍ0, vᵍ0)

atmosphere = atmosphere_simulation(atmos_grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = center_lat),
                                   forcing = geostrophic)
checkpoint("atmosphere built")

δθ = 0.3; zδ = 500; qᵗ₀ = 2e-3
ϵ() = rand() - 0.5
θᵢ(λ, φ, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)
set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = target_u(0.0, wind_params), v = target_v(0.0, wind_params), w = 0, qᵗ = qᵗ₀,
     enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)
checkpoint("atmosphere initialized")

# ## Land component (dry mountain soil)

hydrology = VariablySaturatedHydrology(eltype(land_grid);
    slab_depth = 1.0, porosity = 0.4, residual_liquid_fraction = 0.05,
    storage_height = 1000, critical_saturation = 0.5,
    retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))
energy = WaterCoupledEnergy(eltype(land_grid);
    dry_heat_capacity = 1480 * 1500 * 0.10, liquid_heat_capacity = 4186,
    reference_temperature = 273.15, deep_temperature = 278, deep_time_scale = 12hours,
    advect_deep_liquid_energy = false, advect_surface_liquid_energy = false)
land = SlabLand(land_grid; hydrology, energy)
set!(land; T = 278, M = 0.15 * hydrology.porosity * hydrology.slab_depth * 1000)
Oceananigans.TimeSteppers.update_state!(land)
checkpoint("land built")

# ## Ocean component (hydrostatic free-surface + CATKE mixing)
#
# `ocean_simulation` returns a `HydrostaticFreeSurfaceModel` with (T, S), TEOS-10, a
# split-explicit free surface, and CATKE vertical mixing — all by default. Its top BCs
# are the coupling fields. Idealized init: warm mixed layer over a stratified interior,
# at rest. Temperature in **°C** (TEOS-10), not Kelvin.

ocean = ocean_simulation(ocean_grid)
checkpoint("ocean built")

T_surface = 10.0; h_ml = 15.0; N²_o = 1e-4; S₀ = 35.0
α_T = 2e-4; g_oce = 9.81
dTdz = N²_o / (g_oce * α_T)
Tᵢ(λ, φ, z) = T_surface + (z > -h_ml ? 0.0 : dTdz * (z + h_ml))
set!(ocean.model, T = Tᵢ, S = S₀)
checkpoint("ocean initialized")

# ## Couple atmosphere + land + ocean
#
# The atmosphere feels both land and ocean, blended per cell by `land_fraction` via the
# weighted-flux coupling.

model = NumericalEarth.EarthSystemModels.EarthSystemModel(nothing, atmosphere, land, nothing, ocean;
            clock = Oceananigans.TimeSteppers.Clock{FT}(time = 0), land_fraction = land_fraction)
checkpoint("coupled model built")

# ## Simulation

stop_time = PROD ? inertial_period / 2 : 4hours
simulation = Simulation(model; Δt = 1.0, stop_time)
conjure_time_step_wizard!(simulation, cfl = 1.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
haskey(ENV, "COUPLED_STOP_ITERATION") && (simulation.stop_iteration = parse(Int, ENV["COUPLED_STOP_ITERATION"]))

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    cm = sim.model
    a = cm.atmosphere.model; o = cm.ocean.model
    φ = rad2deg(wind_angle(cm.clock.time, wind_params))
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, wind∠ %.0f°, max|w_atm| %.2e, max|u_oce| %.2e m/s, SST∈[%.2f,%.2f]°C",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed), φ,
                   maximum(abs, a.velocities.w), maximum(abs, o.velocities.u),
                   minimum(interior(o.tracers.T)), maximum(interior(o.tracers.T)))
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

## Rotate the geostrophic wind cross → along valley (u-forcing stores vᵍ, v-forcing uᵍ).
function rotate_geostrophic_wind!(sim)
    t = sim.model.clock.time
    a = sim.model.atmosphere.model
    set!(a.forcing.ρu.forcing.geostrophic_velocity, target_v(t, wind_params))
    set!(a.forcing.ρv.forcing.geostrophic_velocity, target_u(t, wind_params))
    return nothing
end
add_callback!(simulation, rotate_geostrophic_wind!, IterationInterval(10))

# ## Outputs (names are the contract the viz reads — see 10_..._viz.jl)

u_a, v_a, w_a = atmosphere.model.velocities
u_o, v_o, w_o = ocean.model.velocities
T_o, S_o = ocean.model.tracers.T, ocean.model.tracers.S
e_o = hasproperty(ocean.model.tracers, :e) ? ocean.model.tracers.e : nothing

jmid = Nφ ÷ 2 + 1
k_a_surface = 2
k_o_surface = Nz_o

## Statics for the viz (terrain, bathymetry, water mask + lon/lat) as a plain JLD2 file —
## simpler and more robust than an output-writer snapshot of a terrain-following field.
jldsave("coupled_fjord_statics.jld2"; lon = λc, lat = φc,
        h = land_elev, depth = -bottom_cpu, water = FT.(land_frac_cpu .== 0))

atmos_outputs = (u_xy = view(u_a, :, :, k_a_surface), v_xy = view(v_a, :, :, k_a_surface),
                 w_xy = view(w_a, :, :, k_a_surface), w_xz = view(w_a, :, jmid, :))
ocean_outputs = (T_xy = view(T_o, :, :, k_o_surface), S_xy = view(S_o, :, :, k_o_surface),
                 u_xy = view(u_o, :, :, k_o_surface), v_xy = view(v_o, :, :, k_o_surface),
                 T_xz = view(T_o, :, jmid, :), w_xz = view(w_o, :, jmid, :))
ocean_outputs = e_o === nothing ? ocean_outputs : merge(ocean_outputs, (; e_xz = view(e_o, :, jmid, :)))

ao = model.interfaces.atmosphere_ocean_interface.fluxes
flux_outputs = (tau_x = ao.x_momentum, tau_y = ao.y_momentum,
                Q_sensible = ao.sensible_heat, Q_latent = ao.latent_heat)

out_schedule = TimeInterval(PROD ? 5minutes : 2minutes)
simulation.output_writers[:atmos] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = "coupled_fjord_atmos.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = "coupled_fjord_ocean.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "coupled_fjord_fluxes.jld2", schedule = out_schedule, overwrite_existing = true)

checkpoint("starting run!")

# ## Go time
run!(simulation)

@info "Case 10 (coupled fjord ocean) complete"
nothing #hide
