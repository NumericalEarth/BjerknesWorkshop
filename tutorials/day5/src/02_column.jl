
using Pkg; Pkg.activate("..")

using Oceananigans
using OceanBioME
using CairoMakie

using Oceananigans.Units

Nz = 32
Lz = 250

grid = RectilinearGrid(topology = (Flat, Flat, Bounded),
                       size = (Nz, ),
                       z = ExponentialDiscretization(Nz, -Lz, 0; scale = Lz/2))

surface_PAR(t) = 60 * (1 - cos(2π * (t + 23days) / (365days)))

# Here we're going to setup a biogeochemical model with NO₃, NH₄, P, Z, DOM, sPOM, and bPOM
# (small and big POM)
biogeochemistry = LOBSTER(grid;
                          surface_PAR)

nutrient_restoring = Forcing((z, t, NO₃, p) -> (p.NO₃ - NO₃) / p.τ * (z < p.z₀),
                             parameters = (NO₃ = 40, τ = 30days, z₀ = -100),
                             field_dependencies = :NO₃)

model = HydrostaticFreeSurfaceModel(grid;
                                    biogeochemistry,
                                    free_surface = nothing,
                                    momentum_advection = nothing,
                                    tracer_advection = UpwindBiased(),
                                    forcing = (; NO₃ = nutrient_restoring),
                                    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ = 1e-4))

set!(model, P = 0.1,  Z = 0.1, NO₃ = 10)

Δt = 0.5 * minimum_zspacing(grid) / (200/day) # limited by the fast sinking tracers because of the explicit sinking

simulation = Simulation(model; Δt, stop_time = 365*3days)

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time) * " max(P) = $(round(maximum(sim.model.tracers.P), digits = 2))"

add_callback!(simulation, prog, IterationInterval(10000))

simulation.output_writers[:tracers] = 
    JLD2Writer(model, merge(model.tracers, (; PAR = biogeochemistry.light_attenuation.field)), 
               filename = "02_column.jld2",
               schedule = TimeInterval(1day), 
               overwrite_existing=true)

# and run

run!(simulation)

# now plot

fds = FieldDataset("02_column.jld2")
fds["NH₄"] .*= 100 # for visulisation

n = Observable(1)
z = znodes(fds["P"])

title = @lift prettytime(fds["P"].times[$n])

fig = Figure(; title)

ax0 = Axis(fig[0, 1:3], xlabel = "Surface P (mmolN/m³)", ylabel = "Time (years)")
ax1 = Axis(fig[1:2, 1], title = "Nutrients")
ax2 = Axis(fig[1:2, 2], title = "Plankton")
ax3 = Axis(fig[1:2, 3], title = "Detritus")

lines!(ax0, fds["P"].times./365days, interior(fds["P"], 1, 1, Nz, :))
scatter!(ax0, (@lift [fds["P"].times[$n]./365days]), (@lift [interior(fds["P"], 1, 1, Nz, $n)[1, 1, 1]]))

lines!(ax1, (@lift fds["NO₃"][$n]), z, label = "NO₃")
lines!(ax1, (@lift fds["NH₄"][$n]), z, label = "100 × NH₄")
lines!(ax2, (@lift fds["P"][$n]), z, label = "P")
lines!(ax2, (@lift fds["Z"][$n]), z, label = "Z")
lines!(ax3, (@lift fds["DOM"][$n]), z, label = "DOM")
lines!(ax3, (@lift fds["sPOM"][$n]), z, label = "sPOM")
lines!(ax3, (@lift fds["bPOM"][$n]), z, label = "bPOM")

xlims!(ax1, 0, 40)
xlims!(ax2, 0, 5)
xlims!(ax3, 0, 1.5)

axislegend(ax1, position = :rb)
axislegend(ax2, position = :rb)
axislegend(ax3, position = :rb)

record(fig, "02_column.mp4", 1:length(fds["P"]), framerate=8) do i
    n[] = i
end

# Things to try:
# - Prescribed seasonal mixed layer - Replace constant diffusivity with a time-varying κ
# - Change the restoring
# - How about diurnally varying light
# - Or different nutrient limitations/other parameters
