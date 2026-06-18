# # Phase 2: atmosphere + prognostic ocean over Sunnmøre
#
# *The second half of a controlled comparison. The water surface is now a living,
# mixing ocean: a hydrostatic free-surface model with CATKE vertical mixing whose
# mixed layer responds to the same wind that Phase 1 saw only as a fixed warm skin.*
#
# This phase couples the shared terrain-following atmosphere (with its rotating
# geostrophic wind and open lateral boundaries) to a prognostic, **CLOSED** ocean via
# `AtmosphereOceanModel`. There is **no land model** and **no `land_fraction` blending**
# — the single ocean interface IS the surface. Compare against Phase 1
# (`10a_atmosphere_land.jl`), which replaces this prognostic ocean with a land-as-ocean
# slab while keeping the atmosphere, terrain, and wind identical.

include(joinpath(@__DIR__, "10_fjord_setup.jl"))

# ## Ocean component (hydrostatic free-surface + CATKE mixing, CLOSED)
#
# `ocean_simulation` returns a `HydrostaticFreeSurfaceModel` with (T, S), TEOS-10, a
# split-explicit free surface, and CATKE vertical mixing — all by default. Its top BCs
# are the coupling fields. We keep the ocean CLOSED (no PerturbationAdvection / open
# lateral BCs): the lateral walls reflect, isolating the surface-coupling difference
# from Phase 1. Idealized init: a warm mixed layer (surface 10 °C) over a stratified
# interior, S = 35, at rest. Temperature in **°C** (TEOS-10), not Kelvin.

ocean = ocean_simulation(ocean_grid)
checkpoint("ocean built")

T_surface = T_sea_celsius; h_ml = 15.0; N²_o = 1e-4; S₀ = 35.0
α_T = 2e-4; g_oce = 9.81
dTdz = N²_o / (g_oce * α_T)
Tᵢ(λ, φ, z) = T_surface + (z > -h_ml ? 0.0 : dTdz * (z + h_ml))
set!(ocean.model, T = Tᵢ, S = S₀)
checkpoint("ocean initialized")

# ## Couple atmosphere + ocean (single surface interface, no land)

model = AtmosphereOceanModel(atmosphere, ocean;
                             clock = Oceananigans.TimeSteppers.Clock{FT}(time = 0))
checkpoint("coupled (atmosphere + ocean) model built")

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
add_callback!(simulation, rotate_geostrophic_wind!, IterationInterval(10))

# ## Outputs (same variable names/structure as case 10's ocean/atmos/flux/statics)

u_a, v_a, w_a = atmosphere.model.velocities
u_o, v_o, w_o = ocean.model.velocities
T_o, S_o = ocean.model.tracers.T, ocean.model.tracers.S
e_o = hasproperty(ocean.model.tracers, :e) ? ocean.model.tracers.e : nothing

jmid = Nφ ÷ 2 + 1
k_a_surface = 2
k_o_surface = Nz_o

## Statics for the viz (terrain, bathymetry, water mask + lon/lat) as a plain JLD2 file.
jldsave("fjord_phase2_statics.jld2"; lon = λc, lat = φc,
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
    filename = "fjord_phase2_atmos.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = "fjord_phase2_ocean.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "fjord_phase2_fluxes.jld2", schedule = out_schedule, overwrite_existing = true)

checkpoint("starting run!")

# ## Go time
run!(simulation)

@info "Phase 2 (atmosphere + prognostic ocean) complete"
nothing #hide
