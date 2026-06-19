# # Some whales?
# In this example we will build on the previous example adding iron limitation 
# to the bgc and adding an "active" Lagrangian particle that represent maybe 
# a whale swimming around, diving to eat and surfacing to defecate, fertilising
# surface water.
#
# The "whale" is not realistic and maybe represents ~100,000 going along together,
# also if I haven't managed to fix it, the whales don't actually eat and their weight
# goes negative.

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
using CairoMakie

Random.seed!(1234)

Lx = 1000kilometers
Ly = 1000kilometers
Lz = 1kilometers

Nx, Ny, Nz = (96, 96, 16).*2

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

# we fetch the whale model from this script:
include("../src/00_tools.jl")

# then build `BiogeochemicalParticles`,
particles = BiogeochemicalParticles(1; grid,
                                    biogeochemistry = Whaleish(; grazing_rate = 0.0,#200000/days,
                                                                 grazing_half_saturation = 0.1,
                                                                 excretion_rate = 100000*700000.0/days),
                                    advection = SwimmingUpAndDown(; cycle_time = 90minutes,
                                                                    dive_depth = 900.0,
                                                                    horizontal_radius = 250kilometers))

# set the initial positions,
set!(particles, x = Lx/2, biomass = 4e8)

# and build the biogeochemical model, now with the particles and iron limiting growth:
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
nothing
#

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

filename = "baroclinic_instability_whaleish"

simulation.output_writers[:surface] = JLD2Writer(model, model.tracers;
                                                 filename = filename * "_surface.jld2",
                                                 indices = (:, :, grid.Nz),
                                                 schedule = TimeInterval(12hours),
                                                 overwrite_existing = true)

run!(simulation)

# now to plot

N_timeseries = FieldTimeSeries(filename * "_surface.jld2", "N")
Fe_timeseries = FieldTimeSeries(filename * "_surface.jld2", "Fe")
P_timeseries = FieldTimeSeries(filename * "_surface.jld2", "P")
nothing
#
N_timeseries = FieldTimeSeries(filename * "_surface.jld2", "N")
Fe_timeseries = FieldTimeSeries(filename * "_surface.jld2", "Fe")
P_timeseries = FieldTimeSeries(filename * "_surface.jld2", "P")
Z_timeseries = FieldTimeSeries(filename * "_surface.jld2", "Z")

times = N_timeseries.times

x, y, _ = nodes(N_timeseries[1])

n = Observable(1)

title = @lift @sprintf("baroclinic instability after t = %.1f days", times[$n] / days)

n = Observable(1)

Nₙ = @lift interior(N_timeseries[$n], :, :, 1)
Feₙ = @lift interior(Fe_timeseries[$n], :, :, 1)
Pₙ = @lift interior(P_timeseries[$n], :, :, 1)

fig = Figure(size = (1000, 1000))
fig[1, :] = Label(fig, title, fontsize = 20, tellwidth = false)

ax_N = Axis(fig[2, 1], xlabel = "x [km]", ylabel = "y [km]",
            title = "Nitrate", aspect = 1)
hm_N = heatmap!(ax_N, x ./ 1e3, y ./ 1e3, Nₙ, colorrange = (0, 10), colormap = Reverse(:bamako))
Colorbar(fig[2, 2], hm_N, label = "mmolN/m³")

ax_Fe = Axis(fig[2, 3], xlabel = "x [km]", ylabel = "y [km]",
            title = "Iron", aspect = 1)
hm_Fe = heatmap!(ax_Fe, x ./ 1e3, y ./ 1e3, Feₙ, colormap = :lapaz, colorrange = (0, 1e-5))
Colorbar(fig[2, 4], hm_Fe, label = "mmolFe/m³")

ax_P = Axis(fig[3, 1], xlabel = "x [km]", ylabel = "y [km]",
            title = "Phytoplankton", aspect = 1)
hm_P = heatmap!(ax_P, x ./ 1e3, y ./ 1e3, Pₙ, colormap = :lapaz)
Colorbar(fig[3, 2], hm_P, label = "mmolN/m³")

ax_Z = Axis(fig[3, 3], xlabel = "x [km]", ylabel = "y [km]",
            title = "Zooplankton", aspect = 1)
hm_Z = heatmap!(ax_Z, x ./ 1e3, y ./ 1e3, Pₙ, colormap = :lapaz)
Colorbar(fig[3, 4], hm_Z, label = "mmolN/m³")

CairoMakie.record(fig, "baroclinic_instability_bgc.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
mp4_html("baroclinic_instability_bgc.mp4")
mp4_html("baroclinic_instability_bgc.mp4")

#
# It might be interesting to look at how zooplankton responds to the bloom (just set it to some low value)
# at the start, or to plot the detritus and see how it sinks after the bloom passes
