# # A first taste of the atmosphere: from a thermal bubble to mountain drizzle
#
# *One atmosphere, five acts. We start with the simplest experiment in atmospheric
# fluid dynamics — a warm bubble rising through stratification — and end with
# drizzle falling out of orographic clouds on a three-dimensional mountain. Each
# act adds exactly one idea; everything else carries over unchanged.*
#
# | Act | What's new                                              | Dynamics       |
# |:---:|---------------------------------------------------------|----------------|
# | I   | Buoyancy: a dry thermal bubble                           | anelastic      |
# | II  | A warm surface drives free convection (bulk fluxes, wind)| anelastic      |
# | III | The *same* convection, fully compressible                | split-explicit |
# | IV  | A mountain: terrain-following coordinates and lee waves  | split-explicit |
# | V   | Moisture: clouds and drizzle on a 3D mountain            | split-explicit |
#
# The point of the progression is reuse: acts I–III share a single grid, all five
# acts share one background atmosphere, one advection scheme, one surface-flux
# recipe, and one time-stepping driver. When only one ingredient changes at a
# time, every difference you see in the movies has exactly one cause.
#
# This page is the **simulation** half — it runs ahead of time on a GPU and caches
# its output. The [visualization half](05_intro_atmosphere_convection_viz.md)
# renders the movies from that cached output at docs-build time.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random
using CairoMakie

arch = CPU()
Oceananigans.defaults.FloatType = Float32

Lx, Lz = 24kilometers, 8kilometers
Nx, Nz = 384, 160

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       halo = (5, 5),
                       x = (-Lx/2, Lx/2),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## The background atmosphere
#
# We build a stably stratified "background"
# atmosphere with constant buoyancy frequency `N = 0.01 s⁻¹`
# over a surface at `θ₀ = 290 K`,
#
# ```math
# \bar θ (z) = θ_0 \, e^{N^2 z / g} .
# ```

θ₀ = 290     # K, surface potential temperature
N² = 1e-4    # s⁻², stratification (N = 0.01 s⁻¹)
g  = 9.81    # m s⁻², gravitational acceleration

# Background stratified state
θ̄(z) = θ₀ * exp(N² * z / g)

reference_state = ReferenceState(grid; potential_temperature = θ̄)
dynamics = AnelasticDynamics(reference_state)
advection = WENO(order = 9)

model = AtmosphereModel(grid; dynamics, advection)

Δθ = 10             # K, bubble amplitude
r₀ = 1.5kilometers  # bubble radius
z₀ = 2kilometers    # release height

θ_bubble(x, z) = θ̄(z) + Δθ * max(0, 1 - sqrt(x^2 + (z - z₀)^2) / r₀)
set!(model, θ=θ_bubble)

simulation = Simulation(model; Δt=1, stop_time=25minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    @info @sprintf("thermal bubble | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ̄_field = Field{Nothing, Nothing, Center}(grid)
set!(θ̄_field, θ̄)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

bubble_ow = JLD2Writer(model, outputs; 
                       filename = "thermal_bubble.jld2",
                       schedule = TimeInterval(1minute),
                       overwrite_existing = true)

simulation.output_writers[:fields] = bubble_ow

run!(simulation)

θ′_ts = FieldTimeSeries("thermal_bubble.jld2", "θ′")
Nt = length(θ′_ts)

fig = Figure(size=(1200, 400), aspect=3)
ax = Axis(fig[1, 1])

n = Observable(1)
θ′n = @lift θ′_ts[$n]
heatmap!(ax, θ′n)

record(fig, "thermal_bubble.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

# ## Free convection

coefficient = PolynomialCoefficient(roughness_length = 1.5e-4)
filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale = 10minutes)

coefficient = 1e-3
gustiness = 1e-1
q_bottom_bc = BulkVaporFlux(; coefficient, gustiness, surface_temperature=θ₀)
θ_bottom_bc = BulkSensibleHeatFlux(; coefficient, gustiness, surface_temperature=θ₀)
u_bottom_bc = BulkDrag(; coefficient, gustiness)

ρq_bcs = FieldBoundaryConditions(bottom=q_bottom_bc)
ρθ_bcs = FieldBoundaryConditions(bottom=θ_bottom_bc)
ρu_bcs = FieldBoundaryConditions(bottom=u_bottom_bc)

boundary_conditions = (; ρq=ρq_bcs, ρθ=ρθ_bcs, ρu=ρu_bcs)
model = AtmosphereModel(grid; dynamics, advection, boundary_conditions)

θᵢ(x, z) = θ̄(z) + 1e-2 * randn()
set!(model, θ=θᵢ, u=5)

simulation = Simulation(model; Δt=1, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

convection_ow = JLD2Writer(model, outputs; 
                           filename = "free_convection.jld2",
                           schedule = TimeInterval(1minute),
                           overwrite_existing = true)


simulation.output_writers[:fields] = convection_ow

run!(simulation)

# ## Terrain following

h₀ = 600          # m, hill height
a = 1kilometer   # hill half-width

agnesi_hill(x) = h₀ / (1 + (x / a)^2)

r = collect(range(0, Lz, length = Nz + 1))
level_formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                  small_scale_height = Lz / 8)

z = TerrainFollowingVerticalDiscretization(r, formulation=level_formulation)

agnesi_grid = RectilinearGrid(arch; z,
                              size = (Nx, Nz),
                              halo = (5, 5),
                              x = (-Lx/2, Lx/2),
                              topology = (Periodic, Flat, Bounded))

materialize_terrain!(agnesi_grid, agnesi_hill)
sponge = UpperSponge(damping_rate = 0.1, depth = 2.5kilometers)
split_explicit_discretization = SplitExplicitTimeDiscretization(acoustic_cfl=0.5; sponge)

dynamics = CompressibleDynamics(split_explicit_discretization;
                                reference_potential_temperature = θ̄)

model = AtmosphereModel(agnesi_grid; dynamics, advection, boundary_conditions)

θᵢ(x, z) = θ̄(z) + 1e-2 * randn()
set!(model, θ=θᵢ, u=10)

simulation = Simulation(model; Δt=1, stop_time=2hour)
conjure_time_step_wizard!(simulation, cfl=1)

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

hilly_ow = JLD2Writer(model, outputs; 
                      filename = "hilly_free_convection.jld2",
                      schedule = TimeInterval(1minute),
                      overwrite_existing = true)

simulation.output_writers[:fields] = hilly_ow

run!(simulation)

θ′_ts = FieldTimeSeries("hilly_free_convection.jld2", "θ′")
Nt = length(θ′_ts)

fig = Figure(size=(1200, 400), aspect=3)
ax = Axis(fig[1, 1])

n = Observable(1)
θ′n = @lift θ′_ts[$n]
heatmap!(ax, θ′n)

record(fig, "hilly_free_convection.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

#=
# And one vertical slice for acts I–III: 24 km wide, 8 km tall, `Flat` in `y`.
# Two-dimensional dynamics are a conceptual cartoon — 2D turbulence has no vortex
# stretching — but they are cheap enough to run in minutes and rich enough to show
# everything we want to point at. Act V goes 3D.

# ## The shared surface: bulk fluxes from a warm sea
#
# From act II onward the forcing is always the same: we prescribe a **surface
# temperature** `T₀ = θ₀ + 5 K` — a warm sea under cooler air — and let Breeze
# compute the turbulent exchange with **bulk aerodynamic formulae**,
#
# ```math
# Jᶿ = -ρ₀ \, Cᵀ \, |ΔU| \, (θ - T_0), \qquad
# τˣ = -ρ₀ \, Cᴰ \, |ΔU| \, (u - u_0),
# ```
#
# with wind- and stability-dependent exchange coefficients (Large & Yeager 2009)
# and a temporally filtered surface wind (feeding the *instantaneous* LES wind
# into a quadratic bulk formula aliases resolved turbulence into the mean flux).
# Because the surface is warmer than the air, the exchange is unstable-enhanced
# and heat flows upward — that upward flux is what drives the convection.
#
# `surface_fluxes` builds a fresh set of boundary conditions for any grid; act V
# passes `moisture = true` to add evaporation (a bulk vapor flux) and a drag on
# the second velocity component, and warms the sea a little further.

T₀ = θ₀ + 5   # K, prescribed sea surface temperature
Uᵍ = 1e-2     # m s⁻¹, gustiness floor for the bulk formulae

function surface_fluxes(grid; moisture = false, surface_temperature = T₀)
   end
nothing #hide

# ## The shared initial state and driver
#
# Convecting acts start from the background profile plus centimeter-scale
# thermals' worth of noise in the lowest 400 m. We re-seed the generator before
# every run, so acts II and III start from *bit-identical* initial conditions.

δθ = 0.1     # K, perturbation amplitude
zδ = 400     # m, perturbation depth

θᵢ(x, z) = θ̄(z) + δθ * (rand() - 0.5) * (z < zδ)
θᵢ(x, y, z) = θᵢ(x, z)
nothing #hide

# One driver runs every act: build a `Simulation`, attach a CFL wizard (each act
# chooses its own target — that choice is the entire point of act III), a NaN
# checker, a progress log, and a JLD2 writer. It returns the wall-clock time so
# we can compare the formulations' cost.

function evolve!(model, name; stop_time, cfl, outputs, save_interval = 1minute)
    wall = simulation.run_wall_time
    @info @sprintf("[%s] done: %s wall over %d iterations", name, prettytime(wall), iteration(simulation))
    return wall
end
nothing #hide

# ## Act I — a dry thermal bubble
#
# The "hello, world" of atmospheric dynamics: a blob of air 5 K warmer than its
# surroundings, released at rest. It is buoyant, so it rises; as it rises it
# rolls up into the classic mushroom vortex pair, overshoots the height where
# the stratification matches its excess warmth, and rings the surrounding
# atmosphere with gravity waves.
#
# The dynamics are **anelastic**: sound waves are filtered analytically by the
# constraint `∇·(ρ̄ 𝐮) = 0`, and pressure comes from an elliptic solve at every
# substep. The reference state carries the background density profile `ρ̄(z)`
# that lets thin warm bubbles accelerate correctly through a deep atmosphere.

evolve!(model, "thermal_bubble"; stop_time = 25minutes, cfl = 0.7,
        save_interval = 15seconds,
        outputs = (; w = model.velocities.w, θ = liquid_ice_potential_temperature(model)))
nothing #hide

# ## Act II — free convection off a warm surface
#
# Same grid, same anelastic dynamics — but now the warmth comes through the
# *boundary* instead of being painted on the initial condition. The bulk fluxes
# deliver an upward sensible heat flux of order 10² W m⁻²; the lowest layer
# warms, goes unstable, and organizes into **thermals** that punch upward and
# erode the stratification from below, growing a convective boundary layer. A
# light mean wind `Uᶜ = 5 m s⁻¹` works the bulk formulae and leans the plumes.
#
# The expected updraft scale is Deardorff's convective velocity
# `w★ = (g/θ₀ · Q · h)^{1/3} ≈ 1 m s⁻¹` for a kinematic flux `Q ≈ 0.05 K m s⁻¹`
# and a boundary layer `h ≈ 1 km` — thermals turn over in 15–20 minutes, so two
# hours buys several generations.

Uᶜ = 5   # m s⁻¹, mean wind for the convection acts

model = AtmosphereModel(flat_grid; dynamics = AnelasticDynamics(reference_state),
                        advection, boundary_conditions = surface_fluxes(flat_grid))

Random.seed!(1994)
set!(model, θ = θᵢ, u = Uᶜ)

wall_anelastic =
    evolve!(model, "free_convection_anelastic"; stop_time = 1hour, cfl = 0.7,
            outputs = (; w = model.velocities.w, θ = liquid_ice_potential_temperature(model)))
nothing #hide

# ## Act III — the same convection, fully compressible
#
# Now we run the *identical* case — same grid, same fluxes, same wind, same
# noise (we re-seed the generator) — with **compressible dynamics**. Density and
# `ρθ` become prognostic, pressure comes from the equation of state, and sound
# waves are part of the solution.
#
# Sound is the fastest thing in the room (`c ≈ 340 m s⁻¹`), and an explicit step
# limited by it would be `Δt ≲ Δz/c ≈ 0.15 s` — a hundred times smaller than
# advection requires. The **split-explicit** scheme breaks the deadlock: the
# acoustic and buoyancy terms are integrated with cheap small substeps
# (`acoustic_cfl = 0.5` sets the substep from the sound speed), while advection
# and physics take the long outer step.
#
# ### Why CFL = 1 instead of 0.7
#
# The two formulations want different advective CFL targets:
#
# - **Anelastic (act II):** with standard Runge–Kutta time stepping and WENO
#   advection the conventional stability target is `cfl = 0.7`.
# - **Split-explicit (this act):** the outer step uses the Wicker & Skamarock
#   (2002) three-stage Runge–Kutta that mesoscale models like WRF are built on,
#   and it remains stable up to an advective Courant number of **one** — so the
#   wizard targets `cfl = 1.0`, a ~40 % longer time step that claws back part of
#   the cost of the acoustic substepping.
#
# Watch the logs: the two acts march through the same two simulated hours with
# different `Δt`, and `evolve!` returns each one's wall-clock time so we can
# compare at the end.

acoustic = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5)
dynamics = CompressibleDynamics(acoustic; surface_pressure = p₀,
                                reference_potential_temperature = θ̄)

model = AtmosphereModel(flat_grid; dynamics, advection,
                        timestepper = :AcousticRungeKutta3,
                        boundary_conditions = surface_fluxes(flat_grid))

# A compressible model must also be told its density. We initialize from the
# dynamics' hydrostatically balanced reference profile, so the only initial
# imbalance is the 0.1 K seed noise.

Random.seed!(1994)
set!(model, ρ = model.dynamics.reference_state.density, θ = θᵢ, u = Uᶜ,
     enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(model)

wall_acoustic =
    evolve!(model, "free_convection_acoustic"; stop_time = 1hour, cfl = 1.0,
            outputs = (; w = model.velocities.w, θ = liquid_ice_potential_temperature(model)))

@info @sprintf("Formulation shoot-out over identical physics: anelastic %s vs split-explicit compressible %s wall time.",
               prettytime(wall_anelastic), prettytime(wall_acoustic))
nothing #hide

# ## Act IV — a mountain in the wind: lee waves
#
# Why bother with the compressible core, if act III just reproduces act II at a
# different price? Because in Breeze the compressible core is the one that
# speaks **terrain**. The anelastic pressure solve needs a separable, regular
# geometry; the split-explicit substepper handles the terrain-following metric
# terms — contravariant vertical velocity, corrected pressure gradients,
# terrain-aware divergence — at acoustic cost.
#
# So we put a hill in the way: the **Witch of Agnesi**,
#
# ```math
# h(x) = \frac{h_0}{1 + x^2 / a^2},
# ```
#
# the classic bell of mountain-wave theory (Queney 1948). The grid's vertical
# coordinate follows the terrain, decaying back to flat aloft (`TwoLevelDecay`).
# Everything else is act III — the same warm surface, the same bulk fluxes, the
# same stratification — except the wind, which we raise to `Uᵐ = 10 m s⁻¹` so
# the flow clears the crest: the nondimensional mountain height is
# `M = N h₀ / Uᵐ = 0.6`, squarely in the vigorous-but-not-blocked wave regime
# (Smith 1989), and the vertical wavelength `2π Uᵐ / N ≈ 6.3 km` fits the domain.
#
# One new ingredient: a **sponge** in the top 2.5 km, which absorbs the mountain
# waves before they reflect off the model lid back into the physics.

h₀ = 600          # m, hill height
a  = 1kilometer   # hill half-width
Uᵐ = 10           # m s⁻¹, mean wind for the mountain acts

agnesi(x) = h₀ / (1 + (x / a)^2)

z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz + 1));
              formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                          small_scale_height = Lz / 8))

agnesi_grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                              x = (-Lx/2, Lx/2), z = z_faces,
                              topology = (Periodic, Flat, Bounded))

materialize_terrain!(agnesi_grid, agnesi)

sponge = UpperSponge(damping_rate = 0.1, depth = 2.5kilometers)
acoustic = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5, sponge = sponge)

dynamics = CompressibleDynamics(acoustic;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = θ̄)

model = AtmosphereModel(agnesi_grid; dynamics, advection,
                        timestepper = :AcousticRungeKutta3,
                        boundary_conditions = surface_fluxes(agnesi_grid))

# On a terrain-following grid the initial state must be in *discrete* hydrostatic
# balance — `CompressibleDynamics` has already computed that reference for us in
# `terrain_reference_density`. We set `w = 0`; the kinematic bottom boundary
# condition makes the flow follow the terrain from the first step.

Random.seed!(1994)
set!(model, ρ = model.dynamics.terrain_reference_density, θ = θᵢ, u = Uᵐ, w = 0,
     enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(model)

evolve!(model, "agnesi_lee_waves"; stop_time = 1hour, cfl = 1.0,
        outputs = (; w = model.velocities.w, θ = liquid_ice_potential_temperature(model)))
nothing #hide

# ## Act V — clouds and drizzle on a 3D mountain
#
# The finale adds the missing ingredient of real weather: **water**. Three
# changes, all in the direction of realism:
#
# 1. **Moisture.** The air starts at ≈ 80 % relative humidity near the surface,
#    and the sea now also *evaporates* (a bulk vapor flux joins the bulk heat
#    flux). We warm the sea a touch further (`θ₀ + 8` instead of `θ₀ + 5`):
#    saturation humidity grows exponentially with temperature
#    (Clausius–Clapeyron), so those three kelvin nearly double the evaporation,
#    to a trade-wind-like ≈ 400 W m⁻² — the moisture supply the clouds live on.
# 2. **Microphysics.** A standard one-moment bulk scheme from
#    [CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/dev/):
#    cloud formation by warm-phase **saturation adjustment** (the same moist
#    thermodynamics the later coupled cases use), plus prognostic rain with
#    autoconversion, accretion, rain evaporation, and sedimentation. Where the
#    cloud water in updraft cores exceeds the autoconversion threshold, the
#    clouds **drizzle**.
# 3. **Three dimensions.** Real turbulence stretches vortices, and a real
#    mountain has flanks the flow can go *around* as well as over. The bell
#    becomes a Gaussian **mountain** `h(x, y)`, and the slice becomes a volume
#    (at twice the grid spacing, to keep the run quick).
#
# Everything else — stratification, wind `Uᵐ`, sponge, terrain machinery, the
# bulk-flux recipe — is act IV verbatim. Moist scalars get bounds-preserving
# WENO so condensate can never go negative.

Ly = 12kilometers
Nx₃, Ny₃, Nz₃ = 192, 96, 96

σᵐ = 1.5kilometers   # mountain width (a touch wider than the 2D hill, for the coarser grid)
mountain(x, y) = h₀ * exp(-(x^2 + y^2) / (2σᵐ^2))

z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz₃ + 1));
              formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                          small_scale_height = Lz / 8))

mountain_grid = RectilinearGrid(arch; size = (Nx₃, Ny₃, Nz₃), halo = (5, 5, 5),
                                x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
                                topology = (Periodic, Periodic, Bounded))

materialize_terrain!(mountain_grid, mountain)

acoustic = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5, sponge = sponge)
dynamics = CompressibleDynamics(acoustic;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = θ̄)

# The one-moment scheme lives in CloudMicrophysics.jl; loading that package
# activates Breeze's bridge extension.

using CloudMicrophysics
BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

bounded = WENO(order = 5, bounds = (0, 1))
scalar_advection = (; ρθ = advection, ρqᵉ = bounded, ρqᶜˡ = bounded, ρqʳ = bounded)

model = AtmosphereModel(mountain_grid; dynamics, microphysics,
                        momentum_advection = advection, scalar_advection,
                        timestepper = :AcousticRungeKutta3,
                        boundary_conditions = surface_fluxes(mountain_grid; moisture = true,
                                                             surface_temperature = θ₀ + 8))

# Initial moisture: ≈ 80 % relative humidity at the surface — which puts the
# lifting condensation level just *below* the summit, so air forced over the
# crest must condense — decaying over 2 km, fast enough that the cold air aloft
# stays subsaturated. Every cloud in the movie is therefore made by the
# boundary layer or the mountain, not by the sounding.

q₀ = 9.5e-3  # kg kg⁻¹ surface specific humidity
hq = 2kilometers

qᵢ(x, y, z) = q₀ * exp(-z / hq)

Random.seed!(1994)
set!(model, ρ = model.dynamics.terrain_reference_density, θ = θᵢ, qᵗ = qᵢ,
     u = Uᵐ, w = 0, enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(model)

# In 3D we save slices instead of volumes: a vertical slice along the wind
# through the summit, a horizontal slice at cloud level (≈ 1.5 km, in the
# terrain-following coordinate), and the rain field at the lowest model level —
# the drizzle that reaches the ground.

qᶜˡ = model.microphysical_fields.qᶜˡ   # cloud liquid
qʳ  = model.microphysical_fields.qʳ    # rain
w   = model.velocities.w

j_axis  = Ny₃ ÷ 2                                # y row through the summit
k_cloud = round(Int, 1.5kilometers / (Lz / Nz₃)) # ≈ 1.5 km level

outputs = (wxz   = view(w,   :, j_axis, :),
           qᶜˡxz = view(qᶜˡ, :, j_axis, :),
           qʳxz  = view(qʳ,  :, j_axis, :),
           qᶜˡxy = view(qᶜˡ, :, :, k_cloud),
           qʳxy  = view(qʳ,  :, :, 1))

evolve!(model, "mountain_clouds"; stop_time = 1hour, cfl = 1.0, outputs)

@info "Five acts complete: bubble → convection → compressible → lee waves → mountain drizzle."
nothing #hide

# ## Curtain call
#
# Looking back at what carried through: one background atmosphere `θ̄(z)`, one
# advection scheme, one bulk-flux recipe, one driver. What changed, act by act:
# a bubble became a heated boundary; the anelastic core became a split-explicit
# compressible core (and the CFL target rose from 0.7 to 1.0); a hill bent the
# flow into lee waves; and water turned the same circulation into clouds and
# drizzle. The later Thursday cases are these same ingredients at full strength:
# the sea-ice lead (case 1) is act II with a heterogeneous surface, and coastal
# Norway (case 3) is acts IV–V with real topography.
#
# ## References
#
# - **Deardorff, J. W. (1970).** Convective velocity and temperature scales for
#   the unstable planetary boundary layer. *J. Atmos. Sci.*, 27, 1211–1213 —
#   the convective velocity scale `w★` behind act II.
# - **Large, W. G., & Yeager, S. G. (2009).** The global climatology of an
#   interannually varying air–sea flux data set. *Climate Dynamics*, 33,
#   341–364 — the bulk exchange coefficients.
# - **Wicker, L. J., & Skamarock, W. C. (2002).** Time-splitting methods for
#   elastic models using forward time schemes. *Mon. Wea. Rev.*, 130,
#   2088–2097 — the split-explicit Runge–Kutta of acts III–V and its CFL ≈ 1
#   stability.
# - **Queney, P. (1948).** The problem of airflow over mountains. *Bull. Amer.
#   Meteor. Soc.*, 29, 16–26 — the Witch of Agnesi lee wave.
# - **Smith, R. B. (1989).** Hydrostatic airflow over mountains. *Adv.
#   Geophys.*, 31, 1–41 — the `M = N h / U` mountain-wave regimes.
# - **vanZanten, M. C., et al. (2011).** Controls on precipitation and
#   cloudiness in simulations of trade-wind cumulus as observed during RICO.
#   *J. Adv. Model. Earth Syst.*, 3, M06001 — the drizzling shallow-cumulus
#   regime act V's one-moment warm-rain microphysics is built for.
#
# !!! note "Why 2D for acts I–IV?"
#     Two-dimensional dynamics suppress vortex stretching, so treat those
#     structures as schematic. Act V and the later case studies are genuine 3D
#     LES.
#     =#
