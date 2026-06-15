# # A Breeze tutorial: thermal bubbles to cloudy hills
#
# This tutorial introduces Breeze using four experiments:
#
# | Part | What's new                              | Dynamics                    |
# |:----:|-----------------------------------------|-----------------------------|
# | I    | Dry thermal bubble                      | anelastic                   |
# | II   | Forced convection over a warm surface   | anelastic                   |
# | III  | Forced convection over a hill           | split-explicit compressible |
# | IV   | Cloudy, hilly convection                | split-explicit compressible |
#
# ## Environment management
#
# We begin by instantiating the environment:

using Pkg
Pkg.instantiate()

# and then we can get onto to building the environment,

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random
using CairoMakie
using Base64

arch = CPU()
Oceananigans.defaults.FloatType = Float32

# ## The shared grid and background atmosphere
#
# A single vertical slice serves every part: 24 km wide, 8 km tall, periodic in
# `x` and `Flat` in `y`. Two-dimensional dynamics are a cartoon — 2D turbulence
# has no vortex stretching — but they are cheap enough to run in minutes and rich
# enough to show everything we want to point at. WENO advection of order 9 needs a
# halo of five cells. (Crank `Nx, Nz` back up for a crisper movie.)

Lx, Lz = 24kilometers, 8kilometers
#Nx, Nz = 384, 160
Nx, Nz = 128, 64

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       halo = (5, 5),
                       x = (-Lx/2, Lx/2),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## The anelastic approximation and the reference state
#
# The first three parts use **anelastic** dynamics. The atmosphere is compressible,
# but sound waves are energetically irrelevant to convection and would force a tiny
# time step, so the anelastic approximation filters them: each field splits into a
# static, horizontally-uniform **reference profile** (overbar) plus a small
# **perturbation** (prime),
#
# ```math
# ρ = \bar ρ(z) + ρ', \qquad p = \bar p(z) + p', \qquad |ρ'| \ll \bar ρ ,
# ```
#
# with the reference state in hydrostatic balance, `d\bar p/dz = -\bar ρ g`. Mass
# continuity becomes the **anelastic constraint** `∇·(\bar ρ \, 𝐮) = 0` — dropping
# `∂ρ'/∂t` is what removes the acoustic modes — and `p'` is diagnostic, from an
# elliptic solve enforcing that constraint.
#
# The reference column is fixed by a single function, the **reference potential
# temperature** `\bar θ(z)`: with a surface pressure, hydrostatic balance and the
# ideal-gas law then close `\bar p(z)`, `\bar T(z)`, and the background density
# `\bar ρ(z)`. So handing `ReferenceState` a `\bar θ(z)` is what produces `\bar ρ(z)`.
# We take a constant buoyancy frequency `N`, which (since `N^2 = (g/\bar θ)\,d\bar θ/dz`)
# makes `\bar θ` exponential,
#
# ```math
# \bar θ(z) = θ_0 \, e^{N^2 z / g} ,
# ```
#
# anchored at `θ₀ = 290 K`. WENO order-9 advection serves all four parts.

θ₀ = 290     # K, surface potential temperature
N² = 1e-6    # s⁻², stratification (N = 0.01 s⁻¹)
g  = 9.81    # m s⁻², gravitational acceleration

θ̄(z) = θ₀ * exp(N² * z / g)

reference_state = ReferenceState(grid; potential_temperature = θ̄)
dynamics = AnelasticDynamics(reference_state)
advection = WENO(order = 9)

model = AtmosphereModel(grid; dynamics, advection)

# ## Part I — a dry thermal bubble
#
# A warm perturbation `θ' = θ - \bar θ(z)` feels a buoyancy
#
# ```math
# b = -g \, \frac{ρ'}{\bar ρ} = g \, \frac{θ'}{\bar θ}
# ```
#
# (exact in the anelastic system, where buoyancy is evaluated at the reference
# pressure so `p'` drops out): `θ' > 0` rises, which is why we plot `θ'` throughout.
#
# The "hello, world" of atmospheric dynamics: a blob of air 10 K warmer than its
# surroundings, released at rest. It is buoyant, so it rises; as it rises it rolls
# up into the classic mushroom vortex pair, overshoots the height where the
# stratification matches its excess warmth, and rings the surrounding atmosphere
# with gravity waves. We paint the warmth onto the initial condition as a smooth
# cone of radius `r₀` centered at height `z₀`.

Δθ = 2              # K, bubble amplitude
r₀ = 1.5kilometers  # bubble radius
z₀ = 2kilometers    # release height

θ_bubble(x, z) = θ̄(z) + Δθ * max(0, 1 - sqrt(x^2 + (z - z₀)^2) / r₀)
set!(model, θ=θ_bubble)

# Let's visualize the initial condition:

heatmap(liquid_ice_potential_temperature(model))
display(current_figure())

# A CFL wizard adapts the time step as the bubble accelerates; with Runge–Kutta
# time stepping and WENO advection the conventional anelastic stability target is
# `cfl = 0.7`. A progress callback logs the wall-clock march periodically.

simulation = Simulation(model; Δt=1, stop_time=25minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    @info @sprintf("thermal bubble | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# We save the potential-temperature *perturbation* `θ′ = θ - θ̄(z)` — the signal
# the bubble carries above the background — together with the velocities, once a
# minute.

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

# Now we are ready to run the simulation,

run!(simulation)

# ### The movie
#
# Replay every saved frame as a heatmap of `θ′`.
# movie so it plays inline.

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

# ## Part II — free convection off a warm surface
#
# Same grid, same anelastic dynamics — but now the warmth comes through the
# *boundary* instead of being painted on the initial condition. Breeze computes
# the turbulent surface exchange with bulk aerodynamic formulae: a sensible heat
# flux and a vapor flux out of a sea held at `θ₀`, plus a drag on the wind. The
# `coefficient` sets the exchange strength; feeding the instantaneous LES wind
# into a quadratic bulk formula would alias resolved turbulence into the mean
# flux, so a `gustiness` floor keeps the exchange finite in calm spots.

coefficient = 2e-3
gustiness = 1e-1
surface_temperature = θ₀ + 10
q_bottom_bc = BulkVaporFlux(; coefficient, gustiness, surface_temperature)
θ_bottom_bc = BulkSensibleHeatFlux(; coefficient, gustiness, surface_temperature)
u_bottom_bc = BulkDrag(; coefficient, gustiness)

ρq_bcs = FieldBoundaryConditions(bottom=q_bottom_bc)
ρθ_bcs = FieldBoundaryConditions(bottom=θ_bottom_bc)
ρu_bcs = FieldBoundaryConditions(bottom=u_bottom_bc)

boundary_conditions = (; ρq=ρq_bcs, ρθ=ρθ_bcs, ρu=ρu_bcs)
model = AtmosphereModel(grid; dynamics, advection, boundary_conditions)

# The lowest layer warms, goes unstable, and organizes into thermals that punch
# upward and erode the stratification from below, growing a convective boundary
# layer. We start from the background profile plus a whisper of noise and a light
# mean wind (`u = 5 m s⁻¹`) to lean the plumes and work the bulk formulae.

θᵢ(x, z) = θ̄(z) + 1e-2 * randn()
set!(model, θ=θᵢ, u=5)

simulation = Simulation(model; Δt=1, stop_time=10minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    @info @sprintf("free convection | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

convection_ow = JLD2Writer(model, outputs;
                           filename = "free_convection.jld2",
                           schedule = TimeInterval(1minute),
                           overwrite_existing = true)

simulation.output_writers[:fields] = convection_ow

# and then we run the simulation,

run!(simulation)

# ## Part III — the same convection, now over a hill
#
# Why introduce a second dynamical core? Because in Breeze the *compressible* core
# is the one that speaks terrain. The anelastic pressure solve needs a separable,
# regular geometry; the split-explicit compressible substepper handles the
# terrain-following metric terms at acoustic cost. So we put a hill in the way —
# the Witch of Agnesi,
#
# ```math
# h(x) = \frac{h_0}{1 + x^2 / a^2} ,
# ```
#
# the classic bell of mountain-wave theory. The grid's vertical coordinate follows
# the terrain near the ground and decays back to flat aloft (`TwoLevelDecay`).

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

# A sponge in the top 2.5 km absorbs the waves before they reflect off the model
# lid back into the physics. The split-explicit discretization integrates the fast
# acoustic and buoyancy terms with cheap small substeps (`acoustic_cfl = 0.5` sets
# the substep from the sound speed) while advection takes the long outer step; it
# stays stable up to an advective Courant number of one, so the wizard targets
# `cfl = 1`. A compressible model carries density as a prognostic field, so we
# initialize it from the terrain-following hydrostatic reference. The warm surface
# and bulk fluxes are reused verbatim from Part II; the wind rises to `u = 10
# m s⁻¹` so the flow clears the crest.

sponge = UpperSponge(damping_rate = 0.1, depth = 2.5kilometers)
split_explicit_discretization = SplitExplicitTimeDiscretization(acoustic_cfl=0.5; sponge)

dynamics = CompressibleDynamics(split_explicit_discretization;
                                reference_potential_temperature = θ̄)

model = AtmosphereModel(agnesi_grid; dynamics, advection, boundary_conditions)

θᵢ(x, z) = θ̄(z) + 1e-2 * randn()
ρᵢ = model.dynamics.terrain_reference_density

set!(model, ρ=ρᵢ, θ=θᵢ, u=10)

simulation = Simulation(model; Δt=1, stop_time=2hour)
conjure_time_step_wizard!(simulation, cfl=1)

function progress(sim)
    @info @sprintf("hilly free convection | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

hilly_ow = JLD2Writer(model, outputs;
                      filename = "hilly_free_convection.jld2",
                      schedule = TimeInterval(1minute),
                      overwrite_existing = true)

simulation.output_writers[:fields] = hilly_ow

run!(simulation)

# ### The movie
#
# Same recipe as Part I: a `θ′` heatmap of every saved frame, embedded inline.

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

# ## Part IV — cloud microphysics on the hilly flow
#
# The same hilly, terrain-following, compressible setup as Part III — same grid,
# same dynamics, same surface fluxes — now handed a one-moment warm-rain bulk
# scheme from [CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/dev/).
# Loading that package activates Breeze's bridge extension. Cloud water forms by
# warm-phase **saturation adjustment**, and where it exceeds the autoconversion
# threshold the clouds rain.

using CloudMicrophysics

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

model = AtmosphereModel(agnesi_grid; microphysics, dynamics, advection, boundary_conditions)

θᵢ(x, z) = θ̄(z) + 1e-2 * randn()
ρᵢ = model.dynamics.terrain_reference_density

set!(model, ρ=ρᵢ, θ=θᵢ, u=10)

simulation = Simulation(model; Δt=1, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=1)

function progress(sim)
    @info @sprintf("hilly cloud physics | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ̄_field
outputs = (; θ′, model.velocities...)

cloudy_ow = JLD2Writer(model, outputs;
                       filename = "hilly_cloud_physics.jld2",
                       schedule = TimeInterval(1minute),
                       overwrite_existing = true)

simulation.output_writers[:fields] = cloudy_ow

run!(simulation)

# ### The movie

θ′_ts = FieldTimeSeries("hilly_cloud_physics.jld2", "θ′")
Nt = length(θ′_ts)

fig = Figure(size=(1200, 400), aspect=3)
ax = Axis(fig[1, 1])

n = Observable(1)
θ′n = @lift θ′_ts[$n]
heatmap!(ax, θ′n)

record(fig, "hilly_cloud_physics.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end


