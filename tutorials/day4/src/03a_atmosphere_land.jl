# # Phase 1: atmosphere + "land-as-ocean" over Sunnmøre
#
# *The first half of a controlled comparison. The water surface is represented not by a
# prognostic ocean but by a wet, warm `SlabLand` hacked to mimic the sea — a fixed
# 10 °C skin that evaporates freely over the fjords and stays dry over the mountains.*
#
# This phase couples the shared terrain-following atmosphere (with its rotating
# geostrophic wind and open lateral boundaries) to a 2D `SlabLand` via
# `AtmosphereLandModel`. There is **no ocean** and **no `land_fraction` blending** —
# the single land interface IS the surface. Compare against Phase 2
# (`10b_atmosphere_ocean.jl`), which swaps the land-as-ocean slab for a prognostic ocean
# while keeping the atmosphere, terrain, and wind identical.
#
# The "wet warm slab = ocean surface" hack: soil water is set near saturation over water
# cells (free evaporation) and near zero over land (throttled evaporation), and the slab
# is anchored to a warm deep reservoir at the sea temperature (10 °C / 283.15 K) so it
# behaves like a fixed-SST sea surface.

include(joinpath(@__DIR__, "10_fjord_setup.jl"))

# ## Land-as-ocean slab
#
# `VariablySaturatedHydrology` + `WaterCoupledEnergy` with a WARM deep reservoir held at
# the sea temperature (283.15 K, matching Phase 2's 10 °C ocean surface) and a 12-hour
# `deep_time_scale` so the skin stays pinned near the SST. The slab itself is identical
# to case 03's mountain-soil slab except for the warm reservoir and the wet/dry init.

hydrology = VariablySaturatedHydrology(eltype(land_grid);
    slab_depth = 1.0, porosity = 0.4, residual_liquid_fraction = 0.05,
    storage_height = 1000, critical_saturation = 0.5,
    retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

energy = WaterCoupledEnergy(eltype(land_grid);
    dry_heat_capacity = 1480 * 1500 * 0.10, liquid_heat_capacity = 4186,
    reference_temperature = 273.15, deep_temperature = T_sea_kelvin, deep_time_scale = 12hours,
    advect_deep_liquid_energy = false, advect_surface_liquid_energy = false)

land = SlabLand(land_grid; hydrology, energy)

# ## Wet over water, dry over land
#
# Water storage Mˡᵃ⁺ = ρˡ ν D (porosity × slab depth × 1000). We set it near saturation
# over water cells (`land_fraction → 0`) and near zero over land (`land_fraction → 1`),
# using a bilinear interpolation of the `land_frac_cpu` mask as a function of (λ, φ).
# Skin temperature is initialized to the sea temperature everywhere.
M_sat = hydrology.porosity * hydrology.slab_depth * 1000
M_wet = 0.95 * M_sat
M_dry = 0.02 * M_sat

land_frac_fun = bilinear(land_frac_cpu, λc, φc)   # ≈1 over land, ≈0 over water
M_init(λ, φ) = M_dry + (M_wet - M_dry) * (1 - clamp(land_frac_fun(λ, φ), 0, 1))

set!(land; T = T_sea_kelvin, M = M_init)
Oceananigans.TimeSteppers.update_state!(land)
checkpoint("land-as-ocean built")

# ## Couple atmosphere + land (single surface interface, no ocean)

model = AtmosphereLandModel(atmosphere, land;
                            clock = Oceananigans.TimeSteppers.Clock{FT}(time = 0))
checkpoint("coupled (atmosphere + land) model built")

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
    a = cm.atmosphere.model
    𝒮 = cm.land.saturation
    T = cm.land.temperature
    φ = rad2deg(wind_angle(cm.clock.time, wind_params))
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, wind∠ %.0f°, max|w_atm| %.2e, 𝒮∈[%.2f,%.2f], T_land∈[%.2f,%.2f] K",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed), φ,
                   maximum(abs, a.velocities.w), minimum(𝒮), maximum(𝒮),
                   minimum(interior(T)), maximum(interior(T)))
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))
add_callback!(simulation, rotate_geostrophic_wind!, IterationInterval(10))

# ## Outputs (names mirror Phase 2 / case 10 where the fields are shared)

u_a, v_a, w_a = atmosphere.model.velocities

jmid = Nφ ÷ 2 + 1
k_a_surface = 2

## Statics for the viz (terrain, bathymetry, water mask + lon/lat) as a plain JLD2 file.
jldsave("fjord_phase1_statics.jld2"; lon = λc, lat = φc,
        h = land_elev, depth = -bottom_cpu, water = FT.(land_frac_cpu .== 0))

atmos_outputs = (u_xy = view(u_a, :, :, k_a_surface), v_xy = view(v_a, :, :, k_a_surface),
                 w_xy = view(w_a, :, :, k_a_surface), w_xz = view(w_a, :, jmid, :))

## Land surface state: skin temperature and diagnostic surface saturation.
land_outputs = (; T = land.temperature, 𝒮 = land.saturation)

## Air–land interface turbulent fluxes (the single surface interface in this phase).
al = model.interfaces.atmosphere_land_interface.fluxes
flux_outputs = (tau_x = al.x_momentum, tau_y = al.y_momentum,
                Q_sensible = al.sensible_heat, Q_latent = al.latent_heat)

out_schedule = TimeInterval(PROD ? 5minutes : 2minutes)
simulation.output_writers[:atmos] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = "fjord_phase1_atmos.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:land] = JLD2Writer(model, land_outputs;
    filename = "fjord_phase1_land.jld2", schedule = out_schedule, overwrite_existing = true)
simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "fjord_phase1_fluxes.jld2", schedule = out_schedule, overwrite_existing = true)

checkpoint("starting run!")

# ## Go time
run!(simulation)

@info "Phase 1 (atmosphere + land-as-ocean) complete"
nothing #hide
