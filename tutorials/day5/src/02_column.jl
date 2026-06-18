using Pkg; Pkg.activate("..")

using Base64

mp4_html(path) = HTML(string("<video autoplay loop muted playsinline controls ",
                             "src=\"data:video/mp4;base64,", base64encode(read(path)),
                             "\" style=\"max-width:100%\"></video>"))

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

# We can make this remarkably realistic with data, here for the Faeroe islands
include("../src/faeroe_data.jl") 

# This returns `faeroe_data` which contains `surface_PAR`, `NO₃_itp`, `DIC_itp`, `Alk_itp`,
# `T_itp`, `wind_itp`, and `mld_itp`. Next we construct the grid and biogeochemistry as before:
Nz = 32
grid = RectilinearGrid(topology = (Flat, Flat, Bounded), 
                       size = (32, ),
                       z = ExponentialDiscretization(Nz, -60, 0; scale = 60/2))

biogeochemistry = LOBSTER(grid;
                          surface_PAR = faeroe_data.surface_PAR, 
                          inorganic_carbon = CarbonateSystem(),
                          scale_negatives = true)

# except now we have added `inorganic_carbon` so have `DIC` and `Alk`alinity tracers. Notice 
# how the model is now automatically trying to conserve both nitrogen and carbon (although this 
# is a little flawed). For the DIC we also define define the surface flux which solves the carbon 
# chemistry equilibrium in the water and computes the exchange with the air:
surface_CO₂_flux = 
    CarbonDioxideGasExchangeBoundaryCondition(; wind_speed = (x, y, t) -> faeroe_data.wind_itp(t),
                                                air_concentration = 409)

# We want the mixed layer to move seasonally to bring nutrients to the surface, so we can make a 
# function to compute it based on a mixed layer depth:

function κₘ(i, j, k, grid, lx, ly, lz, clock, fields, p)
    z = Oceananigans.Grids.znode(i, j, k, grid, lx, ly, lz)
    t = clock.time

    return p.κB + p.κM * (1 + tanh((z - p.mld(t))/p.δ))/2
end

closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); 
                            κ = κₘ, 
                            parameters = (κB = 1e-5, 
                                          κM = 1e-1,
                                          mld = mld_itp,
                                          δ = 5),
                            discrete_form = true) 

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Day", ylabel = "Mixed layer depth (m)")
lines!(ax, DateTime(2022, 6, 1) .+ Second.(data_times), mld_data)
fig

# And since we're in a column we aren't considering the lateral input of nutrients,
# DIC, and alkalinity so we restore them:
τ = 10days

@inline restoring(z, t, X, p) = @inbounds X * (log(p.X(t)) - log(X))/p.τ * (z < max(p.mld(t),  -30))

nitrate_restoring = Forcing(restoring, 
                            parameters = (; X = faeroe_data.NO₃_itp, τ, mld = faeroe_data.mld_itp),
                            field_dependencies = :NO₃) 

DIC_restoring = Forcing(restoring, 
                        parameters = (; X = faeroe_data.DIC_itp, τ, mld = faeroe_data.mld_itp),
                        field_dependencies = :DIC)

Alk_restoring = Forcing(restoring, 
                        parameters = (; X = faeroe_data.Alk_itp, τ, mld = faeroe_data.mld_itp),
                        field_dependencies = :Alk)

# The carbon chemistry needs the temperature and salinity, and the most straightforward way to
# handel this here is to make them auxiliary fields.
using Oceananigans.Fields: FunctionField, ConstantField
clock = Clock(grid)
T = FunctionField{Center, Center, Center}((z, t) -> faeroe_data.T_itp(t), grid; clock)
S = ConstantField(34.9)

# Then we can put the model together:
Oceananigans.TimeSteppers.reset!(clock)
model = NonhydrostaticModel(grid; 
                            advection = WENO(bounds = (0, 99999), order = 3),
                            biogeochemistry, 
                            closure,
                            clock,
                            auxiliary_fields = (; T, S),
                            forcing = (; NO₃ = nitrate_restoring,
                                         DIC = DIC_restoring,
                                         Alk = Alk_restoring),
                            boundary_conditions = (; DIC = FieldBoundaryConditions(top = surface_CO₂_flux)))

# Setting the initial conditions and build the simulation:
set!(model, NO₃ = 10, P = 0.75, Z = 0.4, DIC = 2147, Alk = 2375)

simulation = Simulation(model; Δt = 20minutes, stop_time = 365*3days)

fname = "faeroes"

simulation.output_writers[:tracers] = JLD2Writer(model, model.tracers, 
                                                 filename = fname*"_tracers.jld2",
                                                 schedule = TimeInterval(1days),
                                                 overwrite_existing = true)

CO₂_flux = BoundaryConditionOperation(model.tracers.DIC, :top, model)

simulation.output_writers[:fCO₂] = JLD2Writer(model, (; qCO₂ = CO₂_flux), indices = (1, 1, grid.Nz),
                                              filename = fname*"_co2_flux.jld2",
                                              schedule = TimeInterval(1days),
                                              overwrite_existing = true)

prog(sim) = @info prettytime(sim) * " in " * prettytime(sim.run_wall_time) * ", DIC∈$(extrema(sim.model.tracers.DIC))" 

add_callback!(simulation, prog, IterationInterval(100))

# and run:
run!(simulation)

# Now we can plot alongside data:
P = FieldTimeSeries(fname*"_tracers.jld2", "P")
NO₃ = FieldTimeSeries(fname*"_tracers.jld2", "NO₃")
qCO₂ = FieldTimeSeries(fname*"_co2_flux.jld2", "qCO₂") # mmol C / m² / s 

fig = Figure(size=(512, 750))

ax = Axis(fig[1, 1], ylabel = "P (mmol N / m³)")
ax2 = Axis(fig[2, 1], ylabel = "NO₃ (mmol N / m³)")
ax3 = Axis(fig[3, 1], ylabel = "Carbon flux (mol C / m² / year)", xlabel = "Date") 

lines!(ax, faeroe_data.plotting.P_obs_dt, mean(faeroe_data.plotting.P_obs, dims = 1)[1, :], color = :black)
lines!(ax, DateTime(2022, 6, 1) .+ Second.(P.times), map(n->mean(P[n]), 1:length(P.times)))

lines!(ax2, DateTime(2022, 6, 1) .+ Second.(faeroe_data.plotting.data_times), mean(faeroe_data.plotting.NO₃_data, dims = 1)[1, :], color = :black)
lines!(ax2, DateTime(2022, 6, 1) .+ Second.(NO₃.times), map(n->mean(NO₃[n]), 1:length(NO₃.times)))

lines!(ax3, faeroe_data.plotting.qCO₂_obs_dt, faeroe_data.plotting.qCO₂_obs, color = :black)
lines!(ax3, DateTime(2022, 6, 1) .+ Second.(qCO₂.times[1:end-30]), map(n->-mean(qCO₂[n:n+30]) / 1000 * 365days , 1:length(qCO₂.times)-30))

fig
#
fig = Figure()

ax = Axis(fig[1, 1], ylabel = "Depth (m)", xticks = ([0:365days:3*365days;], "06/".*string.(year.([DateTime(2022, 6, 1):Year(1):DateTime(2025, 6, 1);]))))
ax2 = Axis(fig[2, 1], ylabel = "Depth (m)")

hm = heatmap!(ax, P)
hm2 = heatmap!(ax2, NO₃)

lines!(ax, P.times, map(t->max(-60, faeroe_data.mld_itp(t)), P.times), color = :black)
lines!(ax2, P.times, map(t->max(-60, faeroe_data.mld_itp(t)), P.times), color = :black)

Colorbar(fig[1, 2], hm, label = "Phytoplankton (mmolN/m³)")
Colorbar(fig[2, 2], hm2, label = "Nitrate (mmolN/m³)")

fig