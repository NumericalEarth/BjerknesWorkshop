# # Upwelling: how resolution effects biogeochemical models
# *Friday - physical insights from Biogeochemistry
#
# Biogeochemical modelling is inherently resolution dependant because 
# the parametrisations are non-linear in concentration. We think that, for example
# the nutrients evolves following an equation like:
# ```math
# \frac{\partial N}{\partial t} + \vec{u}\cdot\nabla N = \kappa \nabla^2 N + T(N, P, ...),
# ```
# but when we solve this numerically were actually solving for the average over a control volume, 
# so the values on a discrete grid don't represent the full range of values.
# Biogeochemical parametrisations therefore deal with the spatial average.
#
# Consider the classical phytoplankton growth rate:
# ```math
# \mu \sim \mu_0f(\text{PAR}, T, ...)\frac{N}{k+N}P,
# ```
# where $k$ is the "half saturation" (this is a Holling type II or Mondo model), $N$ is 
# the nutrient concentration, $PAR$ is the light, $T$ the temperature, etc.
# As this is non-linear in $N$ when we solve this numerically the resolution is important 
# because,
# ```math
# \int_\Omega \frac{N}{N + k} dV \neq \frac{\int_\Omega N dV}{k + \int_\Omega N dV}.
# ```
#
# Before we get into the full example we can construct think about $\mu$ and consider a
# unit "box" with some amount of nutrients. If the box is well mixed then we get:
# ```math
# \mu \sim \frac{\bar{N}}{\bar{N} + k} ∀ x,
# ```
# in the whole box, but if the box has all the nutrients in one half then:
# ```math
# \mu(x<0.5) \sim \frac{2\bar{N}}{2\bar{N} + k},
# ```
# and
# ```math
# \mu(x>0.5) \sim 0.
# ```
# So the average growth rate is:
# ```math
# \bar{\mu} \sim \frac{\bar{N}}{2\bar{N} + k} < \frac{\bar{N}}{\bar{N} + k}.
# ```

# This means that if nutrients are segregated in some small volume we would expect
# to "resolve" a lower growth rate in a more coarse model where the volume is averaged.
# Since there is also phytoplankton concentration dependency, if the nutrients and 
# phytoplankton are colocated in the volume, we resolve lower growth, but if they are
# segregated in reality but mixed in a coarse model then we may resolve more overall 
# growth (consider that $P(x<0) = 0$ and $P(x>0)>0$ in the case above).
# 
# There are lots different configurations which can produce different results, you can 
# play with this example to try and get different things to happen:
using CairoMakie

# We setup a simple timestepping:
function step!(Pt, Nt, n, Δt = 0.1)
    N = Nt[n-1]
    P = Pt[n-1]

    μ =  P * N / (N + 1)

    Nt[n] = N - Δt * μ
    Pt[n] = P + Δt * μ
    
    return nothing
end

nt = 100
Δt = 0.1

# Then we setup the "average" and two compartments
P̄,  N̄  = zeros(nt), zeros(nt)
P₁, N₁ = zeros(nt), zeros(nt)
P₂, N₂ = zeros(nt), zeros(nt)

P₁[1], N₁[1] = 0.1, 0.9
P₂[1], N₂[1] = 0.1, 0.1

P̄[1],  N̄[1]  = (P₁[1] + P₂[1])/2, (N₁[1] + N₂[1])/2 

for n in 2:nt
    step!(P̄, N̄, n, Δt)
    step!(P₁, N₁, n, Δt)
    step!(P₂, N₂, n, Δt)
end

fig = Figure()

ax = Axis(fig[1, 1])

lines!(ax, 0:Δt:(nt-1)*Δt, P̄)
lines!(ax, 0:Δt:(nt-1)*Δt, @. (P₁ + P₂)/2)

fig
# 
# We could consider taking this to the extreme where we might want to consider every
# cell (~$10^6$ per litre!) but that is obviously infeasible. 
# Instead we parameterise, for example using the Holling relation above, so must find
# a balance that captures sufficient detail for a chosen problem, but we also have to 
# remember that a model calibrated for some resolution, is not necessarily valid at 
# a different resolution. 
# Another example of this is averaged light, which models historically were 
# calibrated using, vs diurnally varying since 
# $\int_0^T\frac{sin(2\pi t/T)}{sin(2\pi t/T)+k}dt 
# \neq \frac{\overline{sin(2\pi t/T)}}{\overline{sin(2\pi t/T)} + k})$
#
# ## A more realistic situation
# To investing this effect in a more realistic setting we are going to take the baroclinic
# adjustment case from earlier in the week and add a simple NPZD model.
# For details of the physical case, see the notebook from earlier in the week, but 
# simply put we have an unstable buoyancy configuration which generates eddies. 
# If we imagine a case of a stratified ocean which is depleted of nutrients in the 
# surface where there is sufficient light for the phytoplankton to grow. 
# The upwelling produced by the eddies brings nutrients to the surface inducing a bloom.
# Lets set up the case as before:
using Oceananigans
using Oceananigans.Units
using OceanBioME
using Printf
using Random

Random.seed!(1234)

Lx = 1000kilometers
Ly = 1000kilometers
Lz = 1kilometers

Nx, Ny, Nz = 96, 96, 16

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (-Ly/2, Ly/2),
                       z = (-Lz, 0),
                       topology = (Periodic, Bounded, Bounded))

Cd = 3e-3 
@inline u_drag(x, y, t, u, v, p) = -p.Cd * sqrt(u^2 + v^2) * u
@inline v_drag(x, y, t, u, v, p) = -p.Cd * sqrt(u^2 + v^2) * v

u_bottom_bc = FluxBoundaryCondition(u_drag, field_dependencies = (:u, :v), parameters = (; Cd))
v_bottom_bc = FluxBoundaryCondition(v_drag, field_dependencies = (:u, :v), parameters = (; Cd))

u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)
nothing #hide

# But before we setup the model we need to define the biogeochemistry.
# This is similar to the box example, but we need a few more elements.
# First we're going to manually set `plankton` to `PhytoZoo`, which is the 
# plankton from the LOBSTER model (because its nicer behaved than the NPZD
# default). As we've not got a non-`Flat` z dimension we just need to set the
# surface PAR, initially to a constant.
#
# In this (and most) cases we also need to handel tracers going negative from 
# numerical errors (from the stiffness of the loss terms like $-νP$ or from 
# non-positivity preserving transport errors). Our preferred method is 
# `ScaleNegativeTracers`, which sets negatives values to zero but removes 
# mass from other positive tracers to maintain mass conservation. An Alternative
# common method is to just clip negative values which is cheaper, but does 
# lead to non-conservation.

include("00_tools.jl")

particles = BiogeochemicalParticles(1; grid,
                                    biogeochemistry = Whaleish(; grazing_rate = 0.0,#200000/days,
                                                                 grazing_half_saturation = 0.1,
                                                                 excretion_rate = 700000.0/days),
                                    advection = SwimmingUpAndDown(; cycle_time = 90minutes,
                                                                    dive_depth = 900.0,
                                                                    horizontal_radius = 250kilometers))

set!(particles, x = Lx/2, biomass = 4e8)

biogeochemistry = NPZD(grid;
                       modifiers = ScaleNegativeTracers((:N, :P, :Z, :D); invalid_fill_value = 0),
                       nutrients = Nutrients(nitrogen = OceanBioME.N, iron = OceanBioME.Fe),
                       plankton = PhytoZoo(),
                       surface_PAR = 100,
                       particles)

# We put the `biogeochemistry` into the model as before, and make an extra change
# by changing the `tracer_advection` to be bounded:
model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = BetaPlane(latitude = 70),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(bounds = (0, 1000)),
                                    biogeochemistry,
                                    boundary_conditions = (u = u_bcs, v = v_bcs))

# Building on the previous initial conditions, we set the nutrients to the inverse
# of the vertical buoyancy profile to represent nutrients in the depth, and set the
# phytoplankton to an arbitrary low value.

N² = 1e-5  # s⁻², vertical buoyancy gradient
M² = 1e-7  # s⁻², horizontal buoyancy gradient across the front

Δy = 100kilometers
Δb = Δy * M²

ramp(y, Δy) = min(max(0, y / Δy + 1/2), 1)

bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + 1e-2 * Δb * randn()
Nᵢ(x, y, z) = - 40 *  z / Lz
Zᵢ(x, y, z) = - 1 *  z / Lz

set!(model, b = bᵢ, 
            N = Nᵢ, 
            #Fe = 0.0002,
            P = 0.1, 
)#Z = Zᵢ)

# ## Simulation with adaptive time stepping
#
# Eddying flows accelerate as the instability grows, and a time step chosen for the
# quiet beginning would be wasteful (or unstable) later. The `TimeStepWizard` adapts
# ``\Delta t`` to track a target advective CFL number; `conjure_time_step_wizard!`
# attaches it to the simulation as a callback:

simulation = Simulation(model; Δt = 20minutes, stop_time = 30days)

conjure_time_step_wizard!(simulation, IterationInterval(20), cfl = 0.2, max_Δt = 20minutes)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("[%05.2f%%] iteration: %d, time: %s, wall time: %s, max|u|: %.2f m s⁻¹, next Δt: %s",
                   100 * time(sim) / sim.stop_time, iteration(sim), prettytime(sim),
                   prettytime(elapsed), maximum(abs, sim.model.velocities.u),
                   prettytime(sim.Δt))
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

N  = model.tracers.N
Fe = model.tracers.Fe
P  = model.tracers.P

kinetic_energy = Average((u^2 + v^2) / 2)

filename = "baroclinic_instability"

simulation.output_writers[:surface] = JLD2Writer(model, (; N, P, Fe);
                                                 filename = filename * "_surface.jld2",
                                                 indices = (:, :, grid.Nz),
                                                 schedule = TimeInterval(12hours),
                                                 overwrite_existing = true)

run!(simulation)

# now to plot

N_timeseries = FieldTimeSeries(filename * "_surface.jld2", "N")
P_timeseries = FieldTimeSeries(filename * "_surface.jld2", "P")

times = N_timeseries.times

x, y, _ = nodes(N_timeseries[1])

n = Observable(1)

title = @lift @sprintf("baroclinic instability after t = %.1f days", times[$n] / days)

n = Observable(1)

Nₙ = @lift interior(N_timeseries[$n], :, :, 1)
Pₙ = @lift interior(P_timeseries[$n], :, :, 1)

fig = Figure(size = (1000, 520))
fig[1, :] = Label(fig, title, fontsize = 20, tellwidth = false)

ax_N = Axis(fig[2, 1], xlabel = "x [km]", ylabel = "y [km]",
            title = "Nutrients", aspect = 1)
hm_N = heatmap!(ax_N, x ./ 1e3, y ./ 1e3, Nₙ, colorrange = (0, 10), colormap = Reverse(:bamako))
Colorbar(fig[2, 2], hm_N, label = "mmolN/m³")

ax_P = Axis(fig[2, 3], xlabel = "x [km]", ylabel = "y [km]",
            title = "Phytoplankton", aspect = 1)
hm_P = heatmap!(ax_P, x ./ 1e3, y ./ 1e3, Pₙ, colormap = :lapaz)
Colorbar(fig[2, 4], hm_P, label = "mmolN/m³")

CairoMakie.record(fig, "baroclinic_instability_bgc.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](baroclinic_instability_bgc.mp4)
#
# It might be interesting to look at how zooplankton responds to the bloom (just set it to some low value)
# at the start, or to plot the detritus and see how it sinks after the bloom passes
