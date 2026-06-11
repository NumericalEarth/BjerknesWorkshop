# # A first taste of convection: 2D atmospheric free convection
#
# *The gentlest possible Breeze LES — heat the ground and watch the air rise.*
#
# This is the **introductory** example for the Thursday atmosphere cases. Before
# we tackle sea-ice leads (case 1) or 100 m flow over coastal Norway (case 3), we
# strip the physics down to its essentials: a flat surface, a warm patch of ground
# everywhere, a little wind, and *dry* air. No microphysics, no terrain, no
# heterogeneous surface mask. Just **free convection** — the single most important
# process in the daytime atmospheric boundary layer.
#
# ## What is free convection?
#
# When the ground is warmer than the air above it, the surface transfers heat
# upward as a **sensible heat flux**. The lowest layer of air warms, becomes
# buoyant relative to its surroundings, and rises in coherent **thermals** —
# plumes of warm air that punch upward through the boundary layer. Cooler air sinks
# between them to conserve mass. This overturning is *free* convection: it is driven
# by buoyancy from below, not by mechanical shear. It is what fills the sky with
# fair-weather cumulus on a sunny afternoon.
#
# ## The convective velocity scale w⋆
#
# How fast do the thermals rise? [Deardorff (1970)](https://doi.org/10.1175/1520-0469(1970)027%3C1211:CVATSF%3E2.0.CO;2)
# showed that the natural velocity scale for buoyancy-driven turbulence in a
# boundary layer of depth `zᵢ` is the **convective velocity scale**
#
# ```math
# w_⋆ = \left( \frac{g}{θ_0}\, \overline{w'θ'}_\mathrm{sfc}\, z_i \right)^{1/3}
# ```
#
# where `(w′θ′)_sfc` is the surface kinematic heat flux. For our `Qʰ = 300 W m⁻²`
# (kinematic flux ≈ 0.25 K m s⁻¹ for `θ₀ = 290 K`), and a boundary layer that
# deepens to `zᵢ ≈ 1 km`, `w⋆ ≈ 2 m s⁻¹`. Updrafts of order `w⋆` and turnover
# times `zᵢ/w⋆ ≈ 8 min` are what you should expect to see in the movie.
#
# ## A little wind shears the thermals
#
# With no wind, free-convection thermals are upright and roughly symmetric. Add a
# mean wind and the rising thermals are tilted and stretched downwind; with enough
# shear they organize into **convective rolls** aligned with the wind. We add a
# light `U₀ = 2 m s⁻¹` wind and a simple surface drag so you can see the thermals
# lean — the seed of the organized, wind-sheared convection that dominates the
# lead and Norway cases.
#
# This is a **2D** simulation: one horizontal direction `x`, the vertical `z`, and
# `Flat` in `y`. Two dimensions cannot capture real turbulence (vortex stretching
# is a 3D process), so treat this as a *conceptual cartoon* of convection — cheap
# enough to run in a couple of minutes, rich enough to show thermals, plumes, and
# the deepening boundary layer. The lead and Norway cases are the real 3D physics.

using Breeze
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(1994)

arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float32
nothing #hide

# ## Domain and grid
#
# A 6 km × 2 km vertical slice at uniform ≈ 23 m × 16 m resolution: 256 × 128 cells.
# `Periodic` in `x` (an infinite repeating slice), `Flat` in `y` (no `y`-dependence
# — this is what makes it 2D), and `Bounded` in `z`.

Lx = 6kilometers
Lz = 2kilometers

Nx = 256
Nz = 128

memory_report(Nx, 1, Nz; nfields = 5)

grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                       x = (0, Lx), z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## Reference state and anelastic dynamics
#
# The anelastic approximation filters sound waves while retaining buoyancy and
# stratification — the standard choice for boundary-layer LES. We anchor it on a
# surface pressure of 1000 hPa and a reference potential temperature of 290 K.

p₀ = 1e5   # Pa
θ₀ = 290   # K

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)
dynamics = AnelasticDynamics(reference_state)

cₚ = constants.dry_air.heat_capacity
nothing #hide

# ## Surface heating: the engine of convection
#
# We drive convection with a **uniform positive sensible heat flux** at the ground.
# Breeze's prognostic thermodynamic variable is the potential-temperature density
# `ρθ`, so a heat flux `Qʰ` (W m⁻²) becomes a `ρθ` flux of `Qʰ / cₚ`. A *positive*
# bottom flux warms the lowest air — exactly what a sun-warmed surface does.
#
# `Qʰ = 300 W m⁻²` is a strong but realistic midday land-surface sensible heat
# flux, vigorous enough to spin up convection within minutes.

Qʰ = 300   # W m⁻², surface sensible heat flux
ρθ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(Qʰ / cₚ))

# ## A bit of wind stress
#
# We give the air a light mean wind `U₀` and let it feel the ground through a
# simple quadratic bottom drag with friction velocity `u★` — the same wall model
# the neutral-ABL and lead examples use. In 2D (`Flat` in `y`) only the `ρu`
# momentum component exists, so we only need a drag on `ρu`.

U₀ = 2     # m s⁻¹, light mean wind
u★ = 0.2   # m s⁻¹, friction velocity

# A small CONSTANT surface stress — the simplest, GPU-trivial "bit of wind stress."
# (A velocity-dependent bulk drag, as in the lead case 01, is the richer option, but
# a continuous-function momentum BC tripped a GPU codegen path here; a constant stress
# is the clean choice for the intro.) τˣ < 0 removes +x momentum, i.e. drags the wind.
τˣ = -0.02   # N m⁻² (Pa), surface drag on ρu
ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(τˣ))

# ## Model
#
# 9th-order WENO advection (low numerical dissipation, so the thermals stay crisp)
# and a Smagorinsky–Lilly subgrid closure. Dry: no microphysics, no moisture — the
# cleanest possible convection demo.

advection = WENO(order = 9)
closure = SmagorinskyLilly()

model = AtmosphereModel(grid; dynamics, advection, closure,
                        boundary_conditions = (; ρθ = ρθ_bcs, ρu = ρu_bcs))

# ## Initial conditions
#
# We start from a weakly **stably stratified** atmosphere, `θ(z) = θ₀ + Γ z` with
# `Γ = 0.003 K m⁻¹`. Stable stratification is the resistance the surface heating
# must overcome: thermals rise until they reach air as warm as themselves, which
# is what sets the boundary-layer depth and ultimately `w⋆`. We seed turbulence
# with small random `θ` perturbations in the lowest ≈ 300 m, and set the wind to
# the uniform `U₀`.

Γ = 0.003   # K m⁻¹, background lapse rate (stable)
δθ = 0.1    # K, perturbation amplitude
zδ = 300    # m, perturbation depth

θᵣ(z) = θ₀ + Γ * z

ϵ() = rand() - 0.5
θᵢ(x, z) = θᵣ(z) + δθ * ϵ() * (z < zδ)

set!(model, θ = θᵢ, u = U₀)

# ## Simulation
#
# Adaptive time stepping at CFL 0.7, integrated for 2 hours — long enough for the
# convective boundary layer to spin up and deepen through several turnovers
# (`zᵢ/w⋆ ≈ 8 min`). A NaN check guards against blow-up, and a progress callback
# prints the maximum vertical velocity, which should grow toward ≈ `w⋆`.

simulation = Simulation(model; Δt = 0.5, stop_time = 2hours)
conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 5.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_clock = Ref(time_ns())
function progress(sim)
    wmax = maximum(abs, sim.model.velocities.w)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

# ## Outputs
#
# Because the grid is `Flat` in `y`, the vertical velocity `w` and the
# liquid-ice potential temperature `θ` are already 2D `x–z` fields — no slicing
# needed. We save both every 30 seconds for the animation.

w = model.velocities.w
θ = liquid_ice_potential_temperature(model)

outputs = (; w, θ)

simulation.output_writers[:slices] = JLD2Writer(model, outputs;
    filename = "free_convection.jld2", schedule = TimeInterval(30seconds), overwrite_existing = true)

# ## Go time
#
# This is the one expensive step, and the only one that does **not** run during the
# documentation build — it runs ahead of time on a GPU and caches its output. The
# [visualization page](05_intro_atmosphere_convection_viz.md) then loads that cached
# output and renders the figures and animation live when the docs are built, so what
# you see there is the genuine production-resolution result.

run!(simulation)

@info "Intro convection complete"
nothing #hide

# ## References
#
# - **Deardorff, J. W. (1970).** Convective velocity and temperature scales for
#   the unstable planetary boundary layer. *J. Atmos. Sci.*, 27, 1211–1213.
#   <https://doi.org/10.1175/1520-0469(1970)027%3C1211:CVATSF%3E2.0.CO;2> —
#   defines `w⋆`, the convective velocity scale used here and throughout the
#   atmosphere cases.
#
# !!! note "Why 2D?"
#     This example is two-dimensional for speed and clarity: it runs in a couple of
#     minutes and makes individual thermals easy to see. Real turbulence is
#     three-dimensional — 2D suppresses vortex stretching and the energy cascade, so
#     the thermal *structure* here is schematic. The sea-ice lead (case 1) and the
#     coastal Norway run (case 3) are the genuine 3D large-eddy simulations.
