# # Wind-driven fjord mixing: a 3-way coupled atmosphere–land–ocean LES over Sunnmøre
#
# *Boundary heterogeneity writes turbulence into the fluid — and now the fluid on the
# other side of the surface is a living, mixing ocean.*
#
# This case puts a **prognostic ocean in the real Sunnmøre fjords** beneath the
# terrain-following atmosphere of case 3, and couples all three components two-way:
#
#  - the **atmosphere** (Breeze, compressible, terrain-following) over the real
#    Kartverket terrain,
#  - the **land** surface (a `SlabLand`) over the mountainous cells, and
#  - a **prognostic ocean** (Oceananigans hydrostatic free-surface model with CATKE
#    vertical mixing) filling the fjord/sea cells with an idealized bathymetry.
#
# The surface fluxes (momentum, sensible + latent heat, evaporation) are computed
# **every step from the instantaneous state of whichever fluid sits below each
# column** — land or ocean — via similarity theory. Over the coastline this needs a
# coupler that blends the atmosphere–land and atmosphere–ocean fluxes by a per-cell
# `land_fraction`; see the note on the coupled-model construction below.
#
# ## The experiment: a wind that rotates from cross-valley to along-valley
#
# We drive the atmosphere with a large-scale wind whose **direction rotates over the
# run**, from *cross-valley* (across the main fjord axis) to *along-valley* (down the
# fjord). The question is how the **ocean mixing** responds: a cross-fjord wind piles
# water against one shore and drives a shallow, sheltered overturning; an along-fjord
# wind has a long fetch, builds surface stress and waves along the channel, and drives
# stronger entrainment and a **deeper mixed layer**. At 62° N the inertial period is
# `2π/f ≈ 13.5 h`, so the *rate* of rotation relative to that timescale also matters —
# rotate near-resonantly and the wind keeps feeding the inertial currents it just set
# up, maximizing mixing.
#
# ## Idealized, horizontally-uniform initial state
#
# Per the project decision, both fluids are initialized from **horizontally-uniform,
# idealized profiles** (no reanalysis/data initialization): a stratified atmosphere
# with a mean wind, and an ocean at rest with a mixed layer over a stratified
# interior. The *geometry* (terrain + fjord planform) is real; the *state* is clean.
#
# !!! warning "Experimental coupling path"
#     This is a 3-way `atmosphere + land + ocean` coupling over real terrain — beyond
#     what the upstream test suite exercises. It depends on the per-cell
#     `land_fraction` weighted-flux assembly added to NumericalEarth's Breeze coupling
#     extension. Treat as experimental; validate incrementally (atmosphere-only, then
#     ocean-only, then coupled) on GPU.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
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

## Coupled runs: the EarthSystemModel clock is Float64, so every component grid and
## field must be Float64 too (see case 08).
Oceananigans.defaults.FloatType = Float64
FT = Float64
nothing #hide

# ## Resolution config
#
# A coupled 3-way LES over terrain is expensive; iterate on the SMOKE config and only
# switch to PROD once the pipeline is validated end-to-end. `RUN_CLASS=production`
# selects the large grid.

const PROD = get(ENV, "RUN_CLASS", "smoke") == "production"

Lx = 50kilometers
Ly = 50kilometers
Lz_a = 12kilometers     # atmosphere depth (matches case 3)

Nx = PROD ? 768 : 160
Ny = PROD ? 768 : 160
Nz_o = PROD ? 48 : 24   # ocean vertical levels

# ## Load the real geometry: terrain + idealized fjord bathymetry
#
# `03a` produces the Kartverket terrain; `03b` produces the idealized fjord bathymetry
# on the same grid. Both are on the 100 m native grid; we bilinearly sample them onto
# the LES grid (the same pattern case 3 uses for the terrain).

repo_root = get(ENV, "THURSDAY_REPO_ROOT", pwd())
topo_path = joinpath(repo_root, "thursday", "data", "sunnmore_50km_100m_topography.jld2")
bathy_path = joinpath(repo_root, "thursday", "data", "sunnmore_50km_bathymetry.jld2")
isfile(topo_path)  || error("Missing topography artifact $topo_path — run 03a first.")
isfile(bathy_path) || error("Missing bathymetry artifact $bathy_path — run 03b first.")

topo = load(topo_path)
xt, yt = topo["x"], topo["y"]
h_data    = topo["h"]
land_data = topo["land_mask"]

bathy = load(bathy_path)
depth_data = bathy["depth"]   # ≥ 0 m, 0 on land

h_fun     = bilinear(h_data, xt, yt)
land_fun  = bilinear(land_data, xt, yt)
depth_fun = bilinear(depth_data, xt, yt)

Lz_o = maximum(depth_data)    # ocean grid depth = deepest fjord (idealized)
@info "Loaded geometry" topo_path bathy_path max_terrain = maximum(h_data) max_depth = Lz_o
checkpoint("geometry loaded")

# ## Three grids that share the sea/land surface
#
# Atmosphere: `z ∈ [0, Lz_a]`, terrain-following, real terrain carved in (case 3).
# Ocean:      `z ∈ [-Lz_o, 0]`, immersed bottom from the bathymetry.
# Land:       2-D (`Flat` in z) over the same horizontal extent.
# All three share `Nx × Ny` and the same horizontal extent so the coupler maps surface
# columns one-to-one.

## Atmosphere vertical coordinate + terrain (identical recipe to case 3).
z_faces = PiecewiseStretchedDiscretization(z  = [0, 3000, 6000, Int(Lz_a)],
                                           Δz = [120, 120, 400, 800])
Nz_a = length(z_faces) - 1
z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz_a / 2,
                                          small_scale_height = Lz_a / 8))

memory_report(Nx, Ny, Nz_a; FT, nfields = 6)

atmos_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz_a), halo = (5, 5, 5),
                             x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_coord,
                             topology = (Periodic, Periodic, Bounded))
materialize_terrain!(atmos_grid, (x, y) -> h_fun(x, y))
checkpoint("atmosphere terrain materialized")

## Ocean grid with an immersed bottom. The bathymetry gives sea-floor depth (≥0);
## the immersed bottom sits at z = -depth(x,y), so dry (land) columns where depth≈0
## have the bottom at the surface (no wet cells) — exactly the fjord planform.
ocean_base_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz_o), halo = (5, 5, 5),
                                  x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = (-Lz_o, 0),
                                  topology = (Periodic, Periodic, Bounded))
bottom(x, y) = -depth_fun(x, y)
ocean_grid = ImmersedBoundaryGrid(ocean_base_grid, GridFittedBottom(bottom))
checkpoint("ocean immersed grid built")

land_grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (atmos_grid.Hx, atmos_grid.Hy),
                            x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2),
                            topology = (Periodic, Periodic, Flat))

# ## The per-cell land fraction (drives the coupler's flux blending)
#
# `land_fraction = 1` over the mountains (atmosphere ↔ land), `0` over the fjords
# (atmosphere ↔ ocean), smoothly varying along the coastline. This is the static mask
# the patched coupler uses to blend the two interfaces' fluxes:
# `net = (1 - fₗ)·ocean_flux + fₗ·land_flux`.

xc = [xnode(i, atmos_grid, Center()) for i in 1:Nx]
yc = [ynode(j, atmos_grid, Center()) for j in 1:Ny]
land_fraction_data = FT.([clamp(land_fun(xc[i], yc[j]), 0, 1) for i in 1:Nx, j in 1:Ny])
land_fraction = Field{Center, Center, Nothing}(land_grid)
interior(land_fraction) .= Oceananigans.on_architecture(arch, reshape(land_fraction_data, Nx, Ny, 1))

# ## Atmosphere component (compressible, terrain-following, coupling-ready)
#
# Same compressible terrain dynamics as case 3. `atmosphere_simulation` pre-wires the
# surface BCs as blank coupling fields the coupler fills — we do NOT set surface fluxes
# by hand.

θ₀ = 280          # K, reference near-surface potential temperature (mild marine airmass)
p₀ = 1e5          # Pa
N²_a = 1.0e-4     # s⁻², free-tropospheric stratification
g  = 9.81
potential_temperature_profile(z) = θ₀ * exp(N²_a * z / g)

sponge_depth = 4kilometers
time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = sponge_depth))
dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)

# ### Rotating large-scale wind (geostrophic; direction sweeps cross → along valley)
#
# We impose the synoptic wind as a **geostrophic forcing** (Breeze `geostrophic_forcings`):
# the flow adjusts toward `(uᵍ, vᵍ)` under Coriolis, leaving the terrain-driven
# turbulence free to develop. To ROTATE the wind from cross-valley to along-valley we
# reuse that tested machinery and **update the stored geostrophic velocity each step**
# via a callback (added after the simulation is built). `α_valley` is the fjord-axis
# angle (rad from +x); the wind sweeps from `α + 90°` (cross) to `α` (along) over
# `t_rotate`. Rotating near the ~13.5 h inertial period (62° N) maximizes the inertial
# (mixing) response — `t_rotate` is the key scientific knob.

U_mag    = 12.0                 # m s⁻¹ large-scale wind speed
α_valley = 0.0                  # rad, fjord axis from +x (0 ⇒ along-valley = x)
t_rotate = PROD ? 12hours : 3hours
wind_params = (; U_mag, α_valley, t_rotate)

wind_angle(t, p) = (p.α_valley + π/2) + clamp(t / p.t_rotate, 0, 1) * (p.α_valley - (p.α_valley + π/2))
target_u(t, p) = p.U_mag * cos(wind_angle(t, p))
target_v(t, p) = p.U_mag * sin(wind_angle(t, p))

## Steady baseline at the initial (cross-valley) direction; the rotation callback below
## sweeps it toward along-valley. `geostrophic_forcings(uᵍ, vᵍ)` takes profiles of z.
uᵍ0(z) = target_u(0.0, wind_params)
vᵍ0(z) = target_v(0.0, wind_params)
geostrophic = geostrophic_forcings(uᵍ0, vᵍ0)

atmosphere = atmosphere_simulation(atmos_grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = 62),
                                   forcing = geostrophic)
checkpoint("atmosphere built")

## Idealized, horizontally-uniform atmospheric initial state: the reference θ profile,
## the initial (cross-valley) wind, a modest humidity, small near-surface perturbations.
δθ = 0.3; zδ = 500; qᵗ₀ = 2e-3
ϵ() = rand() - 0.5
θᵢ(x, y, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)
uᵢ(x, y, z) = target_u(0.0, wind_params)
vᵢ(x, y, z) = target_v(0.0, wind_params)
qᵢ(x, y, z) = qᵗ₀
set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = uᵢ, v = vᵢ, w = 0, qᵗ = qᵢ, enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)
checkpoint("atmosphere initialized")

# ## Land component (real mountainous land surface)
#
# A `SlabLand` over the terrain cells, moderately dry (rock/soil), so the land–air
# exchange contrasts with the wet ocean. (Case 3's wet-land trick is no longer needed —
# the ocean is now explicit.)

hydrology = VariablySaturatedHydrology(eltype(land_grid);
    slab_depth = 1.0, porosity = 0.4, residual_liquid_fraction = 0.05,
    storage_height = 1000, critical_saturation = 0.5,
    retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))
energy = WaterCoupledEnergy(eltype(land_grid);
    dry_heat_capacity = 1480 * 1500 * 0.10, liquid_heat_capacity = 4186,
    reference_temperature = 273.15, deep_temperature = 278,
    deep_time_scale = 12hours,
    advect_deep_liquid_energy = false, advect_surface_liquid_energy = false)
land = SlabLand(land_grid; hydrology, energy)

Mˡᵃ⁺ = hydrology.porosity * hydrology.slab_depth * 1000
set!(land; T = 278, M = 0.15 * Mˡᵃ⁺)
Oceananigans.TimeSteppers.update_state!(land)
checkpoint("land built")

# ## Ocean component (prognostic, hydrostatic free-surface + CATKE mixing)
#
# `ocean_simulation` builds an Oceananigans `HydrostaticFreeSurfaceModel` with `(T, S)`
# and TEOS-10, pre-wired for coupling (its top BCs are the coupling fields). The CATKE
# vertical-mixing closure is what we watch respond to the rotating wind stress. Its top
# BCs are filled by the coupler — we do NOT set surface fluxes by hand. An `FPlane` at
# 62° N gives the ocean rotation (inertial currents).

ocean = ocean_simulation(ocean_grid; coriolis = FPlane(latitude = 62))
checkpoint("ocean built")

# ### Idealized, horizontally-uniform ocean initial state
#
# A warm mixed layer of depth `h_ml` over a stratified interior, at rest. Temperature
# in **°C** (TEOS-10 / coupler convention — NOT Kelvin; see case 08's warning).
# `S` uniform. The mixed layer is what the wind will deepen.

T_surface = 10.0   # °C, surface mixed-layer temperature
h_ml      = 15.0   # m, initial ocean mixed-layer depth
N²_o      = 1e-4   # s⁻², interior stratification
S₀        = 35.0   # g/kg
α_T = 2e-4; g_oce = 9.81
dTdz = N²_o / (g_oce * α_T)
Tᵒᵢ(x, y, z) = T_surface + (z > -h_ml ? 0.0 : dTdz * (z + h_ml))
set!(ocean.model, T = Tᵒᵢ, S = S₀)
checkpoint("ocean initialized")

# ## Couple atmosphere + land + ocean
#
# The 3-way assembly is `EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean)`
# with `sea_ice = nothing`. The atmosphere feels BOTH land and ocean, blended per cell
# by `land_fraction` via the patched Breeze coupling extension.
#
# NOTE: the exact keyword that passes `land_fraction` into the coupled-model
# constructor is finalized by the NumericalEarth coupling patch (workstream A1); the
# line below uses the planned signature and will be reconciled with the merged API.

radiation = nothing   # no radiation in this idealized run (turbulent fluxes only)
model = NumericalEarth.EarthSystemModels.EarthSystemModel(radiation, atmosphere, land, nothing, ocean;
            clock = Oceananigans.TimeSteppers.Clock{FT}(time = 0),
            land_fraction = land_fraction)   # << A1 API: per-cell land/ocean blend
checkpoint("coupled model built")

# ## Simulation
#
# A single outer `Simulation` steps the coupled system. We integrate long enough for
# the wind to complete its cross→along rotation and the ocean mixed layer to respond.

stop_time = PROD ? 18hours : 4hours
simulation = Simulation(model; Δt = 1.0, stop_time)
conjure_time_step_wizard!(simulation, cfl = 1.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    cm = sim.model
    a = cm.atmosphere.model
    o = cm.ocean.model
    φ = rad2deg(wind_angle(cm.clock.time, wind_params))
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, wind∠ %.0f°, max|w_atm| %.2e, max|u_oce| %.2e m/s, SST∈[%.2f,%.2f]°C",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed), φ,
                   maximum(abs, a.velocities.w), maximum(abs, o.velocities.u),
                   minimum(interior(o.tracers.T)), maximum(interior(o.tracers.T)))
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

## Rotate the large-scale geostrophic wind cross → along valley. `geostrophic_forcings`
## stores vᵍ under the u-forcing and uᵍ under the v-forcing; we update both z-profile
## fields in place each step. (Reaches into forcing internals — validate live.)
function rotate_geostrophic_wind!(sim)
    t = sim.model.clock.time
    a = sim.model.atmosphere.model
    set!(a.forcing.ρu.forcing.geostrophic_velocity, target_v(t, wind_params))
    set!(a.forcing.ρv.forcing.geostrophic_velocity, target_u(t, wind_params))
    return nothing
end
add_callback!(simulation, rotate_geostrophic_wind!, IterationInterval(10))

# ## Outputs (variable names are the contract the viz reads — see 10_..._viz.jl)

u_a, v_a, w_a = atmosphere.model.velocities
u_o, v_o, w_o = ocean.model.velocities
T_o, S_o = ocean.model.tracers.T, ocean.model.tracers.S

## CATKE turbulent kinetic energy (the mixing diagnostic), if present in the closure.
e_o = hasproperty(ocean.model.tracers, :e) ? ocean.model.tracers.e : nothing

jmid = Ny ÷ 2 + 1
k_a_surface = 2
k_o_surface = Nz_o

## Static fields for the viz (terrain, bathymetry, water fraction).
water_data = FT.([1 - land_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny])
h_field     = Field{Center, Center, Nothing}(atmos_grid)
depth_field = Field{Center, Center, Nothing}(atmos_grid)
water_field = Field{Center, Center, Nothing}(atmos_grid)
interior(h_field)     .= Oceananigans.on_architecture(arch, reshape(FT.([h_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]), Nx, Ny, 1))
interior(depth_field) .= Oceananigans.on_architecture(arch, reshape(FT.([depth_fun(xc[i], yc[j]) for i in 1:Nx, j in 1:Ny]), Nx, Ny, 1))
interior(water_field) .= Oceananigans.on_architecture(arch, reshape(water_data, Nx, Ny, 1))

atmos_outputs = (u_xy = view(u_a, :, :, k_a_surface), v_xy = view(v_a, :, :, k_a_surface),
                 w_xy = view(w_a, :, :, k_a_surface), w_xz = view(w_a, :, jmid, :))

ocean_outputs = (T_xy = view(T_o, :, :, k_o_surface), S_xy = view(S_o, :, :, k_o_surface),
                 u_xy = view(u_o, :, :, k_o_surface), v_xy = view(v_o, :, :, k_o_surface),
                 T_xz = view(T_o, :, jmid, :), w_xz = view(w_o, :, jmid, :))
ocean_outputs = e_o === nothing ? ocean_outputs : merge(ocean_outputs, (; e_xz = view(e_o, :, jmid, :)))

## Interface fluxes (2-D along the surface), computed by similarity theory.
ao = model.interfaces.atmosphere_ocean_interface.fluxes
flux_outputs = (tau_x = ao.x_momentum, tau_y = ao.y_momentum,
                Q_sensible = ao.sensible_heat, Q_latent = ao.latent_heat)

out_schedule = TimeInterval(PROD ? 5minutes : 2minutes)
simulation.output_writers[:statics] = JLD2Writer(atmosphere.model, (; h = h_field, depth = depth_field, water = water_field);
    filename = "coupled_fjord_statics.jld2", schedule = IterationInterval(typemax(Int)), overwrite_existing = true)
simulation.output_writers[:atmos] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = "coupled_fjord_atmos.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = "coupled_fjord_ocean.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "coupled_fjord_fluxes.jld2", schedule = out_schedule, overwrite_existing = true)

write_once!(simulation.output_writers[:statics], atmosphere.model)
checkpoint("statics written; starting run!")

# ## Go time
run!(simulation)

@info "Case 10 (coupled fjord ocean) complete"
nothing #hide
