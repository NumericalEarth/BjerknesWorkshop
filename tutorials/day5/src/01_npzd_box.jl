# # NPZD box model: the basics of biogeochemical modelling
# * Friday - physical insights from Biogeochemistry
# 
using Pkg; Pkg.activate(".."); Pkg.instantiate()

using Oceananigans
using OceanBioME
using CairoMakie

using Oceananigans.Units
using Oceananigans.Fields: FunctionField

z = -10

grid = RectilinearGrid(; topology = (Flat, Flat, Flat),
                         size = tuple(),
                         z)

clock = Clock(time = zero(grid))

@inline function normalised_light(t; φ = 60)
    d = floor(Int, mod(t, 365days)/24hours)

    δ = -23.45*cos(2π*(d+10)/365)

    return max(0, sind(φ) * sind(δ) + cosd(φ) * cosd(δ))
end

surface_light = FunctionField{Nothing, Nothing, Nothing}(t->60*normalised_light(t), grid; clock)

kᵈ = 0.3

light_attenuation = PrescribedPhotosyntheticallyActiveRadiation(surface_light * exp(kᵈ * z))

biogeochemistry = NPZD(grid; 
                       light_attenuation)

# This creates the simplest form of a biogeochemical model with the tracers 
# `N`utrients, `P`hytoplankton, `Z`ooplankton, and `D`etritus,
# which get automatically added.

model = NonhydrostaticModel(grid;
                            advection = nothing,
                            biogeochemistry,
                            clock)

set!(model, P = 0.1, Z = 0.01, N = 10, T = 5)

simulation = Simulation(model, Δt = 10minutes, stop_time = 2*365days)

simulation.output_writers[:tracers] = JLD2Writer(model, merge(model.tracers, light_attenuation.fields),
                                                 filename = "npzd_box.jld2",
                                                 schedule = TimeInterval(1day),
                                                 overwrite_existing = true)

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time)

add_callback!(simulation, prog, IterationInterval(10000))

run!(simulation)

# and plot

fds = FieldDataset("npzd_box.jld2")

fig = Figure()

ax1 = Axis(fig[1, 1], title = "Nutrients (mmol N/m³)")
ax2 = Axis(fig[1, 2], title = "Phytoplankton (mmol N/m³)")
ax3 = Axis(fig[2, 1], title = "Zooplankton (mmol N/m³)")
ax4 = Axis(fig[2, 2], title = "Detritus (mmol N/m³)")

times = fds["N"].times

lines!(ax1, times./days, interior(fds["N"], 1, 1, 1, :))
lines!(ax2, times./days, interior(fds["P"], 1, 1, 1, :))
lines!(ax3, times./days, interior(fds["Z"], 1, 1, 1, :))
lines!(ax4, times./days, interior(fds["D"], 1, 1, 1, :))

fig

# and we can plot in phase space:

fig = Figure()

ax1 = Axis(fig[1, 1], xlabel = "N", ylabel = "P")
ax2 = Axis(fig[1, 2], xlabel = "PAR", ylabel = "P")
ax3 = Axis(fig[2, 1], xlabel = "P", ylabel = "Z")
ax4 = Axis(fig[2, 2], xlabel = "P", ylabel = "D")

lines!(ax1, interior(fds["N"], 1, 1, 1, :), interior(fds["P"], 1, 1, 1, :))
lines!(ax2, interior(fds["PAR"], 1, 1, 1, :), interior(fds["P"], 1, 1, 1, :))
lines!(ax3, interior(fds["P"], 1, 1, 1, :), interior(fds["Z"], 1, 1, 1, :))
lines!(ax4, interior(fds["P"], 1, 1, 1, :), interior(fds["D"], 1, 1, 1, :))

fig
