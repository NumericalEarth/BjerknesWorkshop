# # A first taste of the atmosphere: convection over a flat surface — and over a mountain
#
# *One surface forcing, two boundary shapes. Heat the air from below over a flat sea,
# then do exactly the same thing with a mountain in the way, and watch what the terrain
# does to the turbulence.*
#
# This is the **introductory** example for the Thursday atmosphere cases, and it
# introduces the two ideas every later case builds on:
#
# 1. **Free convection.** When the surface is warmer than the air above it, the lowest
#    layer warms, becomes buoyant, and rises in coherent **thermals** — plumes that
#    punch upward through the boundary layer while cooler air sinks between them. This
#    buoyancy-driven overturning is what fills the sky with fair-weather cumulus.
# 2. **Flow over terrain.** Put a mountain under the *same* heated, windy boundary
#    layer and the flow must respond: air accelerates over the crest, launches
#    vertically propagating **mountain waves** into the stable air aloft, and sheds a
#    turbulent, convecting wake in its lee.
#
# We run the **same simulation twice** — identical surface forcing, wind, and
# stratification — once with a flat lower boundary and once with a
# **Witch of Agnesi** hill,
#
# ```math
# h(x) = \frac{h_0}{1 + x^2 / a^2},
# ```
#
# the classic bell-shaped profile of mountain-wave theory. The pair makes the role of
# the boundary unmistakable: the *forcing* is identical, so every difference between
# the two movies is the mountain's doing.
#
# ## Surface fluxes from a prescribed sea surface temperature
#
# In both runs the heating is not prescribed: we prescribe a warm **surface
# temperature** `T₀` and let Breeze compute the turbulent fluxes with **bulk
# aerodynamic formulae** (the same approach as Breeze's
# `prescribed_sea_surface_temperature` example, and how real atmosphere models exchange
# heat and momentum with the ocean):
#
# ```math
# Jᶿ = -ρ₀\, Cᵀ\, |ΔU|\, (θ - T_0), \qquad τˣ = -ρ₀\, Cᴰ\, |ΔU|\, (u - u_0),
# ```
#
# with wind- and stability-dependent exchange coefficients (Large & Yeager 2009) and a
# temporally filtered surface wind. Because the surface is warmer than the air, the
# exchange is unstable-enhanced and heat flows upward — driving the convection.
#
# Both runs are **2D** (`Flat` in `y`): cheap enough to run in minutes, rich enough to
# show thermals, waves, and the lee-side wake. Real turbulence is 3D — treat these as
# conceptual cartoons that the later 3D cases make rigorous.

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, PolynomialCoefficient, FilteredSurfaceVelocities
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using CUDA   # provides Oceananigans' no-argument `GPU()` architecture
using Printf
using Random

arch = GPU()
Oceananigans.defaults.FloatType = Float64
nothing #hide

# ## The shared setup
#
# A 24 km × 8 km vertical slice. The top 2.5 km is a sponge that absorbs the mountain
# waves before they reflect off the model top. Both runs use a terrain-following grid —
# for the flat run the "terrain" is simply `h(x) = 0`, so the grids (and everything
# else) are identical except for the hill.

Lx = 24kilometers
Lz = 8kilometers

Nx = 384
Nz = 160

# Atmosphere: a stably stratified profile with constant buoyancy frequency
# `N = 0.01 s⁻¹` — the resistance the surface heating must erode (it sets the
# boundary-layer depth) and the medium the mountain waves propagate in. A mean wind
# `U₀` advects the thermals and forces flow over the hill.

p₀ = 1e5     # Pa, surface pressure
θ₀ = 290     # K, surface potential temperature
N² = 1e-4    # s⁻², stratification (N = 0.01 s⁻¹)
g  = 9.81
U₀ = 8       # m s⁻¹, mean wind

potential_temperature_profile(z) = θ₀ * exp(N² * z / g)

# The Witch of Agnesi hill. With `h₀ = 600 m`, `N = 0.01 s⁻¹`, and `U₀ = 8 m s⁻¹` the
# nondimensional mountain height is `M = N h₀ / U₀ = 0.75` — high enough for vigorous
# waves and a disturbed lee, low enough that the flow still makes it over the crest.

h₀ = 600     # m, hill height
a  = 1kilometer   # hill half-width

agnesi(x) = h₀ / (1 + x^2 / a^2)
flat(x) = zero(x)
nothing #hide

# The warm surface: `T₀ = θ₀ + 5 K`, an unstable air–sea contrast that drives an
# upward sensible heat flux of order 10² W m⁻² once the wind is accounted for.

T₀ = θ₀ + 5   # K, prescribed surface temperature
Uᵍ = 1e-2     # m s⁻¹, gustiness floor for the bulk formulae

# ## One function, two boundaries
#
# Everything for a single run: terrain-following grid, compressible dynamics with
# acoustic substepping (the terrain-capable solver), bulk surface fluxes, initial
# conditions, time stepping, and output. The *only* input that differs between the
# two runs is the hill function.

function run_convection(name, hill)
    @info "=== Convection over $(name) topography ==="

    z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz + 1));
                  formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                              small_scale_height = Lz / 8))

    grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                           x = (-Lx/2, Lx/2), z = z_faces,
                           topology = (Periodic, Flat, Bounded))

    materialize_terrain!(grid, hill)

    time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                              sponge = UpperSponge(damping_rate = 0.1, depth = 2.5kilometers))

    dynamics = CompressibleDynamics(time_discretization;
                                    slope_stencil = SlopeInsideInterpolation(),
                                    surface_pressure = p₀,
                                    reference_potential_temperature = potential_temperature_profile)

    ## The same bulk surface fluxes for both runs: sensible heat from the warm surface,
    ## and a bulk drag. The exchange coefficients respond to wind and stability.
    coef = PolynomialCoefficient(roughness_length = 1.5e-4)
    filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale = 1hour)

    ρθ_bcs = FieldBoundaryConditions(bottom =
        BulkSensibleHeatFlux(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))
    ρu_bcs = FieldBoundaryConditions(bottom =
        BulkDrag(coefficient = coef; gustiness = Uᵍ, surface_temperature = T₀, filtered_velocities))

    model = AtmosphereModel(grid; dynamics, advection = WENO(order = 9),
                            timestepper = :AcousticRungeKutta3,
                            boundary_conditions = (; ρθ = ρθ_bcs, ρu = ρu_bcs))

    ## Hydrostatically balanced initial state (essential on terrain-following grids),
    ## the stable θ profile, the mean wind, and small near-surface θ perturbations to
    ## seed the convection.
    Random.seed!(1994)
    δθ = 0.1     # K, perturbation amplitude
    zδ = 400     # m, perturbation depth
    θᵢ(x, z) = potential_temperature_profile(z) + δθ * (rand() - 0.5) * (z < zδ)

    set!(model, ρ = model.dynamics.terrain_reference_density,
         θ = θᵢ, u = U₀, w = 0, enforce_mass_conservation = false)
    Oceananigans.TimeSteppers.update_state!(model)

    ## The acoustic substepper handles the sound waves, so the outer step follows the
    ## advective CFL — the wizard targets cfl = 1.
    simulation = Simulation(model; Δt = 1.0, stop_time = 2hours)
    conjure_time_step_wizard!(simulation, cfl = 1.0)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    wall_clock = Ref(time_ns())
    function progress(sim)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("[%s] Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                       name, iteration(sim), prettytime(sim), prettytime(sim.Δt),
                       prettytime(elapsed), maximum(abs, sim.model.velocities.w))
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(200))

    ## 2D already (`Flat` in `y`): save w and θ every 30 s for the animation.
    w = model.velocities.w
    θ = liquid_ice_potential_temperature(model)
    simulation.output_writers[:slices] = JLD2Writer(model, (; w, θ);
        filename = "$(name)_convection.jld2", schedule = TimeInterval(30seconds),
        overwrite_existing = true)

    run!(simulation)
    @info "$(name) run complete."
    return nothing
end

# ## Run both
#
# The flat control first, then the mountain. These are the only expensive steps and the
# only ones that do **not** run during the documentation build — they run ahead of time
# on a GPU and cache their output; the
# [visualization page](05_intro_atmosphere_convection_viz.md) renders the comparison
# live from that cached output.

run_convection("flat", flat)
run_convection("agnesi", agnesi)

@info "Intro atmosphere case complete (flat + agnesi)."
nothing #hide

# ## References
#
# - **Deardorff, J. W. (1970).** Convective velocity and temperature scales for the
#   unstable planetary boundary layer. *J. Atmos. Sci.*, 27, 1211–1213 — the convective
#   velocity scale `w★` for the thermals.
# - **Large, W. G., & Yeager, S. G. (2009).** The global climatology of an
#   interannually varying air–sea flux data set. *Climate Dynamics*, 33, 341–364 —
#   the wind- and stability-dependent bulk exchange coefficients.
# - **Smith, R. B. (1989).** Hydrostatic airflow over mountains. *Adv. Geophys.*, 31,
#   1–41 — `M = N h / U` and the mountain-wave regimes the Agnesi run explores.
#
# !!! note "Why 2D?"
#     Both runs are two-dimensional for speed and clarity. Real turbulence is 3D — 2D
#     suppresses vortex stretching, so treat the structures as schematic. The sea-ice
#     lead (case 1) and coastal Norway (case 3) are the genuine 3D LES.
