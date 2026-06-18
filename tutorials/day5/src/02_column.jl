
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

simulation = Simulation(model; Δt, stop_time = 365*10days)

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time) * " max(P) = $(round(maximum(sim.model.tracers.P), digits = 2))"

add_callback!(simulation, prog, IterationInterval(1000))

simulation.output_writers[:tracers] = 
    JLD2Writer(model, model.tracers, 
               filename = "02_column.jld2",
               schedule = TimeInterval(1day), 
               overwrite_existing=true)

# and run

run!(simulation)

# now plot

fds = FieldDataset("02_column.jld2")

n = Observable(1)
z = znodes(fds["P"])

title = @lift prettytime(fds["P"].times[$n])

fig = Figure(; title)

ax = Axis(fig[1, 1])

lines!(ax, (@lift fds["NO₃"][$n]), z)
lines!(ax, (@lift fds["NH₄"][$n]), z)
lines!(ax, (@lift fds["P"][$n]), z)
lines!(ax, (@lift fds["Z"][$n]), z)
lines!(ax, (@lift fds["DOM"][$n]), z)
lines!(ax, (@lift fds["sPOM"][$n]), z)
lines!(ax, (@lift fds["bPOM"][$n]), z)

record(fig, "02_column.mp4", 1:Nz, framerate=8) do i
    n[] = i
end

# Things to try:
# - Prescribed seasonal mixed layer - Replace constant diffusivity with a time-varying κ
# - Change the restoring
# - How about diurnally varying light
# - Or different nutrient limitations/other parameters
