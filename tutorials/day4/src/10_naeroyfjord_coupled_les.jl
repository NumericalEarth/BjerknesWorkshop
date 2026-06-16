# # Nærøyfjord: two-way coupled atmosphere–ocean over a real fjord
#
# *The capstone — a rotating wind, blocked cross-fjord and funnelled down-fjord, writes
# mixing into a stratified fjord, with the air–sea fluxes computed at the interface.*
#
# This couples the two components validated separately in `10_naeroyfjord_atmosphere.jl`
# (terrain-following compressible LES over the real Nærøyfjord walls) and
# `10_naeroyfjord_ocean.jl` (hydrostatic + CATKE ocean on the immersed fjord bathymetry)
# via NumericalEarth's **`AtmosphereOceanModel`**. The wind stress on the ocean is no
# longer prescribed — it is **computed every step** from the coupled atmosphere, so the
# story emerges: when the geostrophic wind blows *cross-fjord* the walls block it and
# little stress reaches the water (the fjord stays stratified/stagnant); as the wind
# rotates *down-fjord*, a gap jet drives surface stress along the axis and mixes the
# fresh surface lens.
#
# !!! note "Coupling a compressible (terrain-following) atmosphere"
#     Out of the box, **NumericalEarth ≤ 0.5.6 could not couple a `CompressibleDynamics`
#     atmosphere**: the air–sea coupler's Breeze interface assumed the *anelastic* reference
#     state in two spots — `interpolate_state!` reads `dynamics.reference_state.density`
#     (`nothing` for compressible → `FieldError: Nothing has no field density`), and
#     `surface_layer_height` does a scalar `zspacing` read that trips the GPU scalar-indexing
#     guard on a terrain-following grid. Both are fixed upstream in **NumericalEarth.jl
#     PR #350** (fall back to `terrain_reference_density`/`surface_pressure`; wrap the scalar
#     read in `@allowscalar`). Until that release is picked up, the two small method
#     overrides below reproduce the fix in-script so this run works today. (Delete them once
#     the env has the fix.) Verified on GPU: the coupled run steps stably with air–sea fluxes
#     computed at the interface.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
using CUDA
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(2606)
arch = GPU()
gpu_report()
Oceananigans.defaults.FloatType = Float64   # coupled ESM clock is Float64 → components match
FT = Float64

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OCEAN_MODE = Symbol(get(ENV, "OCEAN_MODE", "full"))   # :full (hydrostatic+CATKE) or :slab

# ## Load the fjord terrain + bathymetry (one artifact carries both)

topo_name = get(ENV, "FJORD_ARTIFACT", "naeroyfjord_topography_real600.jld2")
topo = load(joinpath(repo_root, "thursday", "data", topo_name))
xt, yt = topo["x"], topo["y"]
h_fun = bilinear(topo["h_terrain"], xt, yt)
depth_fun = bilinear(topo["depth"], xt, yt)
meta = topo["source_metadata"]
maxdepth = maximum(topo["depth"])
@info "Loaded fjord" topo_name maxdepth max_wall = maximum(topo["h_terrain"]) source = meta.source

Lx = meta.Lx; Ly = meta.Ly
Nx = parse(Int, get(ENV, "FJORD_NX", "80"))
Ny = parse(Int, get(ENV, "FJORD_NY", "160"))

# ## Atmosphere: terrain-following compressible LES (the stable real600 / CFL-0.5 config)

Lz_a = 8kilometers
z_faces = PiecewiseStretchedDiscretization(z = [0, 2000, 4000, Int(Lz_a)], Δz = [100, 150, 350, 700])
Nz_a = length(z_faces) - 1
z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz_a / 2, small_scale_height = Lz_a / 8))

atmos_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz_a), halo = (5, 5, 5),
                             x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_coord,
                             topology = (Periodic, Periodic, Bounded))
materialize_terrain!(atmos_grid, (x, y) -> h_fun(x, y))

θ₀ = 283; N² = 1.0e-4; g = 9.81
potential_temperature_profile(z) = θ₀ * exp(N² * z / g)
time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = 3kilometers))
dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = 1e5,
                                reference_potential_temperature = potential_temperature_profile)

U_geo = parse(Float64, get(ENV, "FJORD_UGEO", "8"))
uᵍ(z) = U_geo; vᵍ(z) = 0.0
geo = geostrophic_forcings(uᵍ, vᵍ)

atmosphere = atmosphere_simulation(atmos_grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = 60.9),
                                   forcing = geo)

ϵ() = rand() - 0.5
θᵢ(x, y, z) = potential_temperature_profile(z) + 0.2 * ϵ() * (z < 400)
set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = (x,y,z)->U_geo, v = 0, w = 0, qᵗ = 3e-3, enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)

# ## Ocean: hydrostatic + CATKE on the immersed fjord (coupling-ready: no manual stress)

if OCEAN_MODE === :slab
    sst_grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (5, 5),
                               x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), topology = (Periodic, Periodic, Flat))
    ocean = SlabOcean(sst_grid, depth = 30, density = 1025, heat_capacity = 4000)
    set!(ocean, T = 8.0)
else
    Lz_o = ceil(maxdepth / 10) * 10 + 20
    Nz_o = parse(Int, get(ENV, "FJORD_NZ", "48"))
    refinement = 12; stretching = 5
    _h(k) = (k - 1) / Nz_o; _ζ(k) = 1 + (_h(k) - 1) / refinement
    _Σ(k) = (1 - exp(-stretching * _h(k))) / (1 - exp(-stretching))
    zf(k) = Lz_o * (_ζ(k) * _Σ(k) - 1)
    ug_o = RectilinearGrid(arch; size = (Nx, Ny, Nz_o), halo = (7, 7, 7),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = zf,
                           topology = (Periodic, Periodic, Bounded))
    ocean_grid = ImmersedBoundaryGrid(ug_o, GridFittedBottom((x, y) -> -depth_fun(x, y)))
    ocean = ocean_simulation(ocean_grid; model = :hydrostatic, Δt = 1.0,
                             coriolis = FPlane(latitude = 60.9))
    ## Stratified estuarine IC: fresh brackish lens over salty deep (salinity-dominated).
    S_surf = 18.0; S_deep = 33.5; h_lens = 6.0; dS = 4.0
    Sᵢ(x, y, z) = S_surf + (S_deep - S_surf) * (1 + tanh(-(z + h_lens) / dS)) / 2
    set!(ocean.model, S = Sᵢ, T = 8.0)
end

# ## Make the air–sea coupler work with the compressible (terrain-following) atmosphere
#
# Upstream's `interpolate_state!` reads the *anelastic* reference state for the surface air
# density/pressure (`atmosphere.dynamics.reference_state.density/.surface_pressure`), which
# is `nothing` for `CompressibleDynamics`. We override that one method to fall back to the
# compressible dynamics' `terrain_reference_density` (a 3-D field; `[i,j,1]` is the surface
# density — exactly what the kernel indexes) and scalar `surface_pressure`. The branch keeps
# the anelastic path intact, and we reuse upstream's own kernel/helpers so the behaviour is
# identical apart from where ρ₀/p₀ come from. (Remove once upstream supports this directly.)

const NEBExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@assert NEBExt !== nothing "NumericalEarthBreezeExt not loaded"

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid,
                                                             atmosphere::NEBExt.BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    ρqᵛᵉ = atmosphere.moisture_density
    dyn = atmosphere.dynamics
    ref = dyn.reference_state
    if ref === nothing                       # compressible (terrain-following) dynamics
        ρ₀ = dyn.terrain_reference_density   # 3-D field; [i,j,1] is the surface density
        p₀ = dyn.surface_pressure            # scalar p₀
    else                                     # anelastic dynamics (upstream default)
        ρ₀ = ref.density
        p₀ = ref.surface_pressure
    end
    arch = Oceananigans.architecture(exchange_grid)
    kp = NEBExt.interface_kernel_parameters(exchange_grid)
    Oceananigans.Utils.launch!(arch, exchange_grid, kp, NEBExt._interpolate_breeze_state!,
                               state, u, v, T, ρqᵛᵉ, ρ₀, p₀)
    return nothing
end

# Second override: `surface_layer_height` reads the lowest cell's `zspacing` as a single
# scalar; on a terrain-following GPU grid that getindex trips the scalar-indexing guard.
# Upstream's own comment notes this "may require allowscalar" — it is one scalar per step
# (host-side), so we wrap it. (Also remove once upstream handles terrain-following grids.)
NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::NEBExt.BreezeAtmosphere) =
    CUDA.@allowscalar(Oceananigans.zspacing(1, 1, 1, atmosphere.grid, Center(), Center(), Center())) / 2

# ## Couple

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Rotating-wind schedule (cross-fjord → down-fjord), applied to the coupled atmosphere

const t_hold = 30minutes; const t_rot = 60minutes
@inline function wind_angle(t)
    s = clamp((t - t_hold) / t_rot, 0, 1)
    return (-π/2) * (3s^2 - 2s^3)
end
function rotate_wind!(sim)
    a = sim.model.atmosphere.model
    θ = wind_angle(sim.model.clock.time)
    set!(a.forcing.ρu.forcing.geostrophic_velocity, U_geo * sin(θ))   # holds vᵍ
    set!(a.forcing.ρv.forcing.geostrophic_velocity, U_geo * cos(θ))   # holds uᵍ
    return nothing
end

# ## Simulation

stop_minutes = parse(Float64, get(ENV, "FJORD_STOPMIN", "20"))
Δt0 = parse(Float64, get(ENV, "FJORD_DT", "1.0"))
simulation = Simulation(model; Δt = Δt0, stop_time = stop_minutes * 60)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
get(ENV, "WIND_ROTATE", "1") == "1" && add_callback!(simulation, rotate_wind!, IterationInterval(5))

wall = Ref(time_ns())
function progress(sim)
    cm = sim.model
    a = cm.atmosphere.model
    elapsed = 1e-9 * (time_ns() - wall[])
    wa = maximum(abs, a.velocities.w)
    Q = cm.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
    θw = rad2deg(wind_angle(cm.clock.time))
    msg = @sprintf("Iter %d  t %s  wall %s  max|w_atm| %.2e  max|u_atm| %.1f  max|Qsens| %.1f  wind∠ %.0f°",
                   cm.clock.iteration, prettytime(cm.clock.time), prettytime(elapsed),
                   wa, maximum(abs, a.velocities.u), maximum(abs, interior(Q)), θw)
    if OCEAN_MODE !== :slab
        msg *= @sprintf("  max|u_oce| %.3f", maximum(abs, cm.ocean.model.velocities.u))
    end
    @info msg
    wall[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(20))

# ## Outputs: atmosphere surface wind, air–sea fluxes, and (full ocean) surface salinity
run_tag = get(ENV, "RUN_TAG", "coupled")
_dir = joinpath(repo_root, "thursday", "data")
u_a, v_a, w_a = atmosphere.model.velocities
atmos_out = (u_xy = view(u_a, :, :, 2), v_xy = view(v_a, :, :, 2), w_xy = view(w_a, :, :, 2))
simulation.output_writers[:atmos] = JLD2Writer(atmosphere.model, atmos_out;
    filename = joinpath(_dir, "naeroyfjord_coupled_atmos_$(run_tag).jld2"),
    schedule = TimeInterval(1minute), overwrite_existing = true)

aoflux = model.interfaces.atmosphere_ocean_interface.fluxes
flux_out = (Qsens = aoflux.sensible_heat, Qlat = aoflux.latent_heat, taux = aoflux.x_momentum, tauy = aoflux.y_momentum)
simulation.output_writers[:flux] = JLD2Writer(atmosphere.model, flux_out;
    filename = joinpath(_dir, "naeroyfjord_coupled_flux_$(run_tag).jld2"),
    schedule = TimeInterval(1minute), overwrite_existing = true)

if OCEAN_MODE !== :slab
    S_o = ocean.model.tracers.S
    uo, vo, wo = ocean.model.velocities
    ko = ocean.model.grid.Nz
    ocean_out = (S_xy = view(S_o, :, :, ko), u_xy = view(uo, :, :, ko), w_xy = view(wo, :, :, ko),
                 S_yz = view(S_o, Nx ÷ 2 + 1, :, :))
    simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_out;
        filename = joinpath(_dir, "naeroyfjord_coupled_ocean_$(run_tag).jld2"),
        schedule = TimeInterval(1minute), overwrite_existing = true)
end

@info "Starting coupled run" OCEAN_MODE stop_minutes Nx Ny
run!(simulation)
@info "Coupled run complete" iterations = iteration(simulation)
nothing #hide
