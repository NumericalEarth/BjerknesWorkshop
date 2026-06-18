using Pkg; Pkg.activate("..")
using Base64

mp4_html(path) = HTML(string("<video autoplay loop muted playsinline controls ",
                             "src=\"data:video/mp4;base64,", base64encode(read(path)),
                             "\" style=\"max-width:100%\"></video>"))

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

include("../src/00_tools.jl")

particles = BiogeochemicalParticles(1; grid,
                                    biogeochemistry = ReleaseIron(1.0))# 1mmol Fe / s

set!(particles, x = Lx)

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

set!(model, b = bᵢ, N = Nᵢ, P = 0.1)

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
mp4_html("baroclinic_instability_bgc.mp4")

#
# It might be interesting to look at how zooplankton responds to the bloom (just set it to some low value)
# at the start, or to plot the detritus and see how it sinks after the bloom passes
