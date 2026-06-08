# # Sea ice I: thermodynamics, from a freezing bucket to the Arctic seasonal cycle
#
# *Tuesday — one day in the high-latitude ocean, part 3: the freezing surface.*
#
# [ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) is the sea-ice component of
# the model family you are meeting this week: built on Oceananigans grids and fields, it
# runs on CPUs and GPUs, and couples to the ocean in the Barents Sea simulation of part 6. A
# sea-ice model consists in two reasonably independent halves — *thermodynamics* (how
# thick the ice grows) and *dynamics* (where the wind and ocean push it). This tutorial
# covers the first half; the next one the second.
#
# We proceed the way the field did historically: first the Stefan problem — ice growing
# into a bucket of freshwater, solvable by hand since 1891 — then the celebrated
# "zero-layer" Arctic seasonal cycle of
# [Semtner (1976)](https://doi.org/10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2),
# which is, give or take a few decades of refinement, the thermodynamic heart of the sea
# ice in every CMIP model.
#
# Along the way we implement **custom interface fluxes** with `FluxFunction` — the
# mechanism for teaching the model new surface physics, from a crude frazil
# parameterization to a temperature-dependent albedo.
#
# ## The slab model
#
# The prognostic variables are the ice thickness ``h``, the ice concentration ``\aleph``
# (the fraction of each cell that is ice-covered), and the surface temperature ``T_u``.
# The slab ("zero-layer") thermodynamics assumes a linear temperature profile through the
# ice, so conduction carries a flux ``Q_c = k (T_b - T_u)/h`` from the bottom interface
# (pinned at the freezing temperature ``T_b = T_m``) to the surface. The surface
# temperature adjusts to balance ``Q_c`` against the external fluxes; whatever imbalance
# remains at the *bottom* freezes new ice or melts old one at the rate set by the latent
# heat:
#
# ```math
# \rho_i \mathcal{L} \frac{dh}{dt} = Q_{conduction} - Q_{ocean} .
# ```
#
# No heat storage, no brine pockets, no snow — and still, as Semtner showed, the Arctic
# equilibrium thickness comes out within tens of centimeters of observations.
#
# ## Act I: the freezing bucket
#
# An insulated, infinitely deep bucket of freshwater at its freezing point, capped by a
# lid held at −10 °C. Perhaps surprisingly, this consists in a *zero-dimensional* grid —
# all the action is in the thickness:

using Oceananigans
using Oceananigans.Units
using ClimaSeaIce
using CairoMakie

grid = RectilinearGrid(size = (), topology = (Flat, Flat, Flat))

# The internal conductive flux and the phase-transition parameters (the heat capacity
# enters a Stefan-number correction to the latent heat — see Act II's reference):

conductivity = 2 # W m⁻¹ K⁻¹
internal_heat_flux = ConductiveFlux(; conductivity)

heat_capacity = 2100 # J kg⁻¹ K⁻¹
phase_transitions = PhaseTransitions(; heat_capacity)

top_heat_boundary_condition = PrescribedTemperature(-10) # °C, the cold lid

ice_thermodynamics = SlabThermodynamics(grid;
                                        internal_heat_flux,
                                        top_heat_boundary_condition)

# ### A first custom flux: frazil ice
#
# The slab equation above cannot start from ``h = 0`` (the conductive flux diverges).
# Real oceans do not care: supercooled open water produces *frazil* crystals that
# consolidate into a first ice cover. We implement this as our first **`FluxFunction`** —
# a plain Julia function of the interface state, here a bottom heat flux that extracts
# 1 W m⁻² from the ocean until the concentration reaches one:

@inline frazil_ice_formation(i, j, grid, Tuᵢ, clock, fields) = -(1 - fields.ℵ[i, j, 1]) # W m⁻²

bottom_heat_flux = FluxFunction(frazil_ice_formation)

# The signature `(i, j, grid, surface_temperature, clock, fields)` gives access to
# anything the model knows, and the function is fused into the GPU kernel that steps the
# thermodynamics — the same zero-overhead extension pattern as the `Forcing` and the
# `FluxBoundaryCondition` of the ocean tutorials. This is precisely how the
# ocean–ice and atmosphere–ice fluxes are wired up in the coupled simulations of this
# afternoon.
#
# Now the model — note the familiar shapes: `SeaIceModel` takes a grid and physics
# choices, exactly like the ocean models:

model = SeaIceModel(grid;
                    ice_thermodynamics,
                    phase_transitions,
                    sea_ice_density = 900,
                    bottom_heat_flux)

simulation = Simulation(model, Δt = 10minutes, stop_time = 10days)

# A callback accumulates the thickness time series at every iteration (with a
# zero-dimensional model we can afford the extravagance):

bucket_series = []

accumulate_bucket!(sim) = push!(bucket_series,
                                (time(sim), first(sim.model.ice_thickness),
                                 first(sim.model.ice_concentration)))

simulation.callbacks[:save] = Callback(accumulate_bucket!)

run!(simulation)

# The Stefan solution predicts ``h \propto \sqrt{t}``: the thicker the ice, the weaker
# the conductive flux through it, the slower the growth — the basic negative feedback of
# sea-ice thermodynamics. We overlay a ``\sqrt{t}`` reference anchored at the endpoint:

t = [datum[1] for datum in bucket_series]
h = [datum[2] for datum in bucket_series]

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "time [days]", ylabel = "ice thickness [cm]")
lines!(ax, t ./ day, 1e2 .* h, linewidth = 4, label = "ClimaSeaIce")
lines!(ax, t ./ day, 1e2 .* h[end] .* sqrt.(t ./ t[end]),
       linewidth = 3, linestyle = :dash, color = :gray, label = "√t reference")
axislegend(ax, position = :rb)
save("freezing_bucket.png", fig)
nothing #hide

# ![](freezing_bucket.png)
#
# The early departure from the dashed line is the frazil phase, when the concentration
# is still building up; afterwards the slab follows Stefan faithfully.
#
# ## Act II: the Arctic seasonal cycle of Semtner (1976)
#
# We now replace the cold lid by the real thing: the seasonal cycle of radiative and
# turbulent fluxes over the Arctic basin, using the monthly climatology of Fletcher
# (1965) exactly as tabulated in Semtner's table 1 (in kcal cm⁻² month⁻¹, an energy unit
# deprecated since 1948 — converting it is part of the experience):

using Oceananigans.Units: Time
using Oceananigans.OutputReaders: Cyclical

#          Month:        Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
tabulated_shortwave = -[   0,     0,   1.9,   9.9,  17.7,  19.2,  13.6,   9.0,   3.7,   0.4,     0,     0] .* 1e4
tabulated_longwave  = -[10.4,  10.3,  10.3,  11.6,  15.1,  18.0,  19.1,  18.7,  16.5,  13.9,  11.2,  10.9] .* 1e4
tabulated_sensible  = -[1.18,  0.76,  0.72,  0.29, -0.45, -0.39, -0.30, -0.40, -0.17,   0.1,  0.56,  0.79] .* 1e4
tabulated_latent    = -[   0, -0.02, -0.03, -0.09, -0.46, -0.70, -0.64, -0.66, -0.39, -0.19, -0.01, -0.01] .* 1e4

month_days = 30
times = (15:30:360 - 15) .* day  # mid-month times, in an idealized 360-day year

kcal_to_joules = 4184
flux_conversion = kcal_to_joules / (month_days * days)

tabulated_shortwave .*= flux_conversion
tabulated_longwave  .*= flux_conversion
tabulated_sensible  .*= flux_conversion
tabulated_latent    .*= flux_conversion
nothing #hide

# The negative signs follow the model's convention: fluxes are positive *upward*, so a
# negative top flux heats the ice. Let's look at the forcing:

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1], xlabel = "time [days]", ylabel = "heat flux [W m⁻²]")
for (flux, label) in ((tabulated_shortwave, "shortwave"), (tabulated_longwave, "longwave"),
                      (tabulated_sensible, "sensible"), (tabulated_latent, "latent"))
    scatterlines!(ax, times ./ day, flux; label, linewidth = 3)
end
axislegend(ax, position = :lb)
save("semtner_forcing.png", fig)
nothing #hide

# ![](semtner_forcing.png)
#
# ### Time-dependent fluxes via `FieldTimeSeries`
#
# To hand the monthly climatology to the model we store each flux in a
# `FieldTimeSeries` with *cyclical* time indexing, so that asking for day 380 wraps
# around to day 20 — the forcing repeats year after year:

Rs = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical())
Rl = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical())
Qs = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical())
Ql = FieldTimeSeries{Nothing, Nothing, Nothing}(grid, times; time_indexing = Cyclical())

for (i, time) in enumerate(times)
    set!(Rs[i], tabulated_shortwave[i:i])
    set!(Rl[i], tabulated_longwave[i:i])
    set!(Qs[i], tabulated_sensible[i:i])
    set!(Ql[i], tabulated_latent[i:i])
end
nothing #hide

# A `FluxFunction` interpolates the series to the current clock time. Indexing a
# `FieldTimeSeries` with `Time(t)` performs the linear interpolation in time for us:

@inline function climatological_flux(i, j, grid, Tu, clock, fields, flux)
    t = Time(clock.time)
    return flux[i, j, 1, t]
end
nothing #hide

# ### A second custom flux: the albedo feedback
#
# Shortwave radiation needs more care: most of it is reflected, and *how much* depends
# on the surface state — bare cold ice reflects ~75%, melting ice ~64%. Encoding this
# temperature dependence takes three lines, and those three lines contain the famous
# ice–albedo feedback (this crude on/off switch is Semtner's original; making it smooth,
# or snow-aware, is a tempting variation — see below):

@inline function climatological_solar_flux(i, j, grid, Tu, clock, fields, flux)
    Q = climatological_flux(i, j, grid, Tu, clock, fields, flux)
    α = ifelse(Tu < -0.1, 0.75, 0.64)
    return Q * (1 - α)
end

Q_shortwave = FluxFunction(climatological_solar_flux, parameters = Rs)
Q_longwave  = FluxFunction(climatological_flux,       parameters = Rl)
Q_sensible  = FluxFunction(climatological_flux,       parameters = Qs)
Q_latent    = FluxFunction(climatological_flux,       parameters = Ql)
nothing #hide

# The ice also radiates according to its own surface temperature, with a built-in flux:

σ = 5.67e-8 * 1.02 # Semtner's (slightly wrong) Stefan–Boltzmann constant, kept for fidelity
Q_emission = RadiativeEmission(emissivity = 1, stefan_boltzmann_constant = σ)

# Fluxes compose by putting them in a tuple — the model sums them at the interface:

top_heat_flux = (Q_shortwave, Q_longwave, Q_sensible, Q_latent, Q_emission)

model = SeaIceModel(grid; top_heat_flux)
set!(model, h = 0.3, ℵ = 1)

# Thirty years at an 8-hour step, to let the thickness forget the initial condition and
# settle onto the equilibrium seasonal cycle:

simulation = Simulation(model, Δt = 8hours, stop_time = 30 * 360days)

arctic_series = []

function accumulate_arctic!(sim)
    T = sim.model.ice_thermodynamics.top_surface_temperature
    h = sim.model.ice_thickness
    ℵ = sim.model.ice_concentration
    push!(arctic_series, (time(sim), first(h), first(T), first(ℵ)))
end

simulation.callbacks[:save] = Callback(accumulate_arctic!)

run!(simulation)

# ### The equilibrium seasonal cycle

t = [datum[1] for datum in arctic_series]
h = [datum[2] for datum in arctic_series]
T = [datum[3] for datum in arctic_series]

fig = Figure(size = (900, 600))

axh = Axis(fig[1, 1], xlabel = "time [years]", ylabel = "thickness [m]")
lines!(axh, t ./ (360days), h, linewidth = 2)

axT = Axis(fig[2, 1], xlabel = "time [years]", ylabel = "surface temperature [°C]")
lines!(axT, t ./ (360days), T, linewidth = 2)

save("arctic_seasonal_cycle.png", fig)
nothing #hide

# ![](arctic_seasonal_cycle.png)
#
# Starting from 30 cm, the ice thickens winter after winter — each year a little less,
# the Stefan feedback again — and converges to an equilibrium cycle with ~3 m of ice,
# a summer melt of several tens of centimeters, and a surface temperature that sits at
# the melting point through the melt season and plunges below −30 °C in the polar night.
# Semtner obtained 2.88 m with the same numbers in 1976, on hardware considerably less
# convenient than yours.
#
# ## Things to try
#
# !!! tip "The greenhouse, crudely"
#     Add a constant `4` W m⁻² downward (i.e. `-4` in the model's convention) to the
#     longwave column and rerun. How much equilibrium thickness is lost? Compare the
#     transient with the multi-decadal Arctic thinning. Then try `8`: does a perennial
#     ice cover survive?
#
# !!! tip "An ocean heat flux"
#     Semtner included a constant 2 W m⁻² heat flux from the ocean below. Add it as a
#     `bottom_heat_flux` (mind the sign convention!) and quantify the effect on the
#     equilibrium thickness.
#
# !!! tip "A better albedo"
#     Replace the on/off albedo with a smooth ramp between the cold and warm values over
#     the last degree below melting, ``\alpha(T_u)``. Does the equilibrium cycle change?
#     What does this say about the sensitivity of the system to the albedo formulation?
#
# ## Further reading
#
# - [Semtner (1976)](https://doi.org/10.1175/1520-0485(1976)006<0379:AMFTTG>2.0.CO;2) —
#   the zero-layer model, still readable, still relevant
# - [Maykut and Untersteiner (1971)](https://doi.org/10.1029/JC076i006p01550) — the
#   multi-layer benchmark Semtner was simplifying
# - The [ClimaSeaIce.jl](https://github.com/CliMA/ClimaSeaIce.jl) documentation and
#   examples, from which this tutorial is adapted
