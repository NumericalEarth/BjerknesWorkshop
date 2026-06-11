# # Two fluids, one interface: 2D coupled air–sea convection
#
# *The gentlest possible coupled example — a warm ocean and a cool atmosphere,
# talking to each other through the sea surface.*
#
# So far the Thursday cases have run **one fluid at a time**. The atmosphere
# examples (free convection, the sea-ice lead) prescribed what happens at the
# surface — a heat flux, a surface temperature — and let the air respond. The
# ocean example did the mirror image. But the real boundary layer is *two* fluids
# coupled across one interface: the ocean warms the air, the air cools the ocean,
# and neither side's surface forcing is known ahead of time — it is *computed* from
# the instantaneous state of both fluids.
#
# This example introduces that coupling with the **`EarthSystemModel`** interface
# from NumericalEarth. We stack a 2D atmosphere directly above a 2D ocean column,
# share the same horizontal extent, and let the coupler do the bookkeeping:
#
# 1. Each step, **similarity theory** (Monin–Obukhov bulk formulae) reads the
#    near-surface air and the sea-surface temperature and computes the turbulent
#    **sensible heat, latent heat, and momentum fluxes** at the interface.
# 2. Those fluxes are handed *up* into the atmosphere as its bottom boundary
#    condition and *down* into the ocean as its top boundary condition — with
#    opposite signs, because what leaves the ocean enters the air.
# 3. Both fluids step forward; the surface state changes; the fluxes are
#    recomputed. That feedback loop is **two-way air–sea coupling**.
#
# ## The physics: convection above *and* below
#
# We start the ocean ≈ 3 K warmer than the air (a warm sea surface under a cool
# atmosphere — think a marine cold-air outbreak, or warm water under a polar
# airmass). Heat flows upward across the interface, so:
#
# - **Above the surface**, the warmed near-surface air becomes buoyant and rises
#   in thermals — atmospheric free convection, exactly as in the intro atmosphere
#   case, but now the heating is *earned* from the ocean rather than prescribed.
# - **Below the surface**, the ocean *loses* heat from its top. Cooled surface
#   water becomes denser than the water beneath it and sinks — **ocean convection**,
#   the upside-down mirror of the atmospheric plumes. The same interface drives
#   overturning in both directions.
#
# The coupling is what makes this self-consistent: the air–sea temperature
# difference that drives the flux is itself eroded by the flux (the air warms, the
# ocean cools), so the exchange is largest at the start and relaxes as the two
# fluids approach each other — a behavior you cannot get from a prescribed flux.
#
# This is a **2D** demo (`Flat` in `y`) sized to run quickly: a conceptual cartoon
# of coupled convection, the foundation for the warm-filament case. Real
# air–sea turbulence is 3D.
#
# !!! warning "Untested coupling path"
#     `AtmosphereOceanModel` with a **nonhydrostatic** ocean (LES) is not yet
#     exercised by the upstream NumericalEarth test suite — the tested path uses a
#     `SlabOcean`. The construction below follows the patterns in
#     `NumericalEarth.jl/test/test_breeze_coupling.jl` as closely as possible; treat
#     the nonhydrostatic-ocean coupling as experimental and verify on GPU.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(1994)

config = RunConfig("07_intro_coupled_convection")
arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float64  # coupled: ESM clock is Float64, component grids must match
FT = Float64
nothing #hide

# ## Two grids that share a sea surface
#
# The atmosphere lives in `z ∈ [0, Lz_a]` above the interface; the ocean lives in
# `z ∈ [-Lz_o, 0]` below it. Both share the *same* horizontal extent `Lx` and the
# *same* number of horizontal cells `Nx`, so the coupler can map surface columns
# one-to-one across the interface. Both are `Periodic` in `x`, `Flat` in `y`
# (this is what makes the demo 2D), and `Bounded` in `z`.

const Lx   = 4kilometers    # shared horizontal extent
const Lz_a = 3kilometers    # atmosphere depth
const Lz_o = 100meters      # ocean depth

const Nx   = 128
const Nz_a = 96             # atmosphere vertical cells
const Nz_o = 64             # ocean vertical cells

memory_report(Nx, 1, Nz_a; FT, nfields = 6)

atmos_grid = RectilinearGrid(arch; size = (Nx, Nz_a), halo = (5, 5),
                             x = (0, Lx), z = (0, Lz_a),
                             topology = (Periodic, Flat, Bounded))

ocean_grid = RectilinearGrid(arch; size = (Nx, Nz_o), halo = (5, 5),
                             x = (0, Lx), z = (-Lz_o, 0),
                             topology = (Periodic, Flat, Bounded))

# ## The atmosphere component
#
# `atmosphere_simulation` builds a Breeze `AtmosphereModel` wrapped in a
# `Simulation`, **pre-wired for coupling**: its bottom boundary conditions on
# momentum, energy, and moisture are blank 2D fields that the coupler fills each
# step. We must *not* add our own surface flux BCs here — the coupler owns them.
#
# The atmosphere is moist by default (warm-phase saturation-adjustment
# microphysics), so if the rising air saturates it can form cloud. We give it a
# reference potential temperature of 290 K and a Smagorinsky–Lilly LES closure, and
# initialize it isothermal at the reference state with a light 1 m s⁻¹ wind (so the
# bulk formulae have a nonzero wind to work with, and the thermals lean).

atmosphere = atmosphere_simulation(atmos_grid;
                                   potential_temperature = FT(290),
                                   closure = SmagorinskyLilly())

## Initialize the atmosphere at its reference potential-temperature profile with a
## light mean wind. `set!` acts on the wrapped `atmosphere.model`.
set!(atmosphere.model,
     θ = atmosphere.model.dynamics.reference_state.potential_temperature,
     u = 1)

# ## The ocean component
#
# `ocean_simulation(grid; model = :nonhydrostatic)` builds an Oceananigans
# `NonhydrostaticModel` (full 3D pressure, LES-ready) wrapped in a `Simulation`,
# with `(T, S)` tracers and a TEOS-10 seawater equation of state. Like the
# atmosphere, its *top* boundary conditions are blank coupling fields the coupler
# fills — we do not set surface fluxes by hand.
#
# We start the ocean **warm and uniform**: `T = 20 °C`, about 3 K warmer than the
# overlying air, with a uniform salinity of 35 g kg⁻¹. That ≈ 3 K air–sea contrast is
# the engine: it drives an upward heat flux, convection in the air above, and
# cooling-driven convection in the ocean below.
#
# !!! warning "Units: the ocean is in °C, the atmosphere is in K"
#     The ocean's TEOS-10 equation of state and the air–sea coupler both expect the
#     ocean temperature in **degrees Celsius** (`ocean_temperature_units =
#     DegreesCelsius()`), so `T = 20 °C` here — *not* Kelvin. The Breeze atmosphere is
#     in Kelvin (`potential_temperature = 290 K`). Mixing the two up silently breaks
#     the flux calculation: a Kelvin SST is read as 293 °C, the saturation humidity is
#     evaluated at ~566 K (above the ambient pressure), and the interface humidity goes
#     negative — driving a runaway spurious-condensation instability. Keep the ocean in °C.

ocean = ocean_simulation(ocean_grid; model = :nonhydrostatic)

## A warm, salty, initially quiescent ocean. `ocean_simulation` returns a
## `Simulation`, so `set!` targets `ocean.model`.
set!(ocean.model, T = FT(20), S = FT(35))

# ## Couple them
#
# `AtmosphereOceanModel(atmosphere, ocean)` returns an `EarthSystemModel`: it builds
# the interface bookkeeping (the `atmosphere_ocean_interface`) and wires the
# similarity-theory flux computation between the two components. The *coupled*
# `Simulation` wrapped around it owns the time step `Δt` — the inner atmosphere and
# ocean simulations defer to it.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Simulation
#
# The coupled model is stepped by a single outer `Simulation`. A fixed 10 s step
# is comfortable for this modest grid; we integrate for 2 hours of simulated time,
# long enough to spin up convection on both sides of the interface and to watch the
# air–sea temperature contrast relax as heat is exchanged. A progress callback
# prints the peak vertical velocity in each fluid.

simulation = Simulation(model; Δt = 10, stop_time = 2hours)

wall_clock = Ref(time_ns())
function progress(sim)
    cm = sim.model
    wa = maximum(abs, cm.atmosphere.model.velocities.w)
    wo = maximum(abs, cm.ocean.model.velocities.w)
    Q  = cm.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
    Qmax = maximum(abs, Q)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, wall %s, max|w_atm| %.2e, max|w_oce| %.2e m/s, max|Qsens| %.1f W/m²",
                   cm.clock.iteration, prettytime(cm.clock.time),
                   prettytime(elapsed), wa, wo, Qmax)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

# ## Outputs
#
# Because both grids are `Flat` in `y`, the vertical velocities and tracers are
# already 2D `x–z` fields — no slicing needed. We save, from each fluid, the
# vertical velocity and a thermodynamic field, plus the air–sea sensible heat flux
# at the interface, every 30 seconds for the animation.
#
# Output writers attach to the *coupled* `simulation.output_writers`, but each
# `JLD2Writer` is built around the relevant **component model** (`atmosphere.model`
# or `ocean.model`), whose fields it samples.

## Atmosphere-side diagnostics.
w_a = atmosphere.model.velocities.w
θ_a = liquid_ice_potential_temperature(atmosphere.model)

atmos_outputs = (; w_a, θ_a)

## Cloud liquid (fog) if the moist microphysics exposes it — saved when present.
if hasproperty(atmosphere.model, :microphysical_fields) &&
   hasproperty(atmosphere.model.microphysical_fields, :qˡ)
    qˡ_a = atmosphere.model.microphysical_fields.qˡ
    atmos_outputs = merge(atmos_outputs, (; qˡ_a))
end

## Ocean-side diagnostics.
w_o = ocean.model.velocities.w
T_o = ocean.model.tracers.T

ocean_outputs = (; w_o, T_o)

## The interface sensible heat flux — a 2D (x) line along the sea surface, computed
## by similarity theory. (Latent heat lives alongside it as `.latent_heat`.)
Q_sensible = model.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
flux_outputs = (; Q_sensible)

simulation.output_writers[:atmosphere] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = output_name(config, "atmosphere"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = output_name(config, "ocean"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = output_name(config, "fluxes"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

# ## Go time
run!(simulation)

@info "Intro coupled convection complete" run_stamp(config)...
nothing #hide

# ## References
#
# - **Monin, A. S., Obukhov, A. M. (1954).** Basic laws of turbulent mixing in the
#   surface layer of the atmosphere. *Tr. Akad. Nauk SSSR Geofiz. Inst.*, 24, 163–187.
#   — the similarity theory underlying the bulk air–sea flux formulae the coupler uses.
# - **Large, W. G., Yeager, S. G. (2009).** The global climatology of an
#   interannually varying air–sea flux data set. *Climate Dynamics*, 33, 341–364.
#   <https://doi.org/10.1007/s00382-008-0441-3> — bulk transfer coefficients for the
#   air–sea turbulent fluxes.
# - **Deardorff, J. W. (1970).** Convective velocity and temperature scales for the
#   unstable planetary boundary layer. *J. Atmos. Sci.*, 27, 1211–1213.
#   <https://doi.org/10.1175/1520-0469(1970)027%3C1211:CVATSF%3E2.0.CO;2> — the
#   convective velocity scale `w⋆` for the buoyancy-driven thermals on both sides.
#
# !!! note "Why 2D, and why a thin ocean?"
#     This example is two-dimensional (`Flat` in `y`) and uses a shallow 100 m ocean
#     so it spins up convection on both sides quickly and stays cheap. Real air–sea
#     turbulence is three-dimensional, and a real oceanic mixed layer is deeper; treat
#     the structure here as schematic. The point is the *coupling* — fluxes computed
#     at the interface, not prescribed — which carries over unchanged to the 3D
#     warm-filament case.

