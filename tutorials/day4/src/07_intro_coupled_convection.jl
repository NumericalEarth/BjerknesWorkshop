# # Two fluids, one interface: a warm ocean filament drives 2D coupled convection
#
# *The gentlest possible coupled example — a warm filament in the ocean lights up
# convection in the air above and the water below, with the air–sea fluxes computed
# at the interface rather than prescribed.*
#
# So far the Thursday cases have run **one fluid at a time**. The atmosphere examples
# (free convection, the sea-ice lead) prescribed what happens at the surface — a heat
# flux, a sea surface temperature — and let the air respond. The ocean example did the
# mirror image. But the real boundary layer is *two* fluids coupled across one
# interface: the ocean warms the air, the air cools the ocean, and neither side's
# surface forcing is known ahead of time — it is *computed* from the instantaneous
# state of both fluids.
#
# This example introduces that coupling with the **`EarthSystemModel`** interface from
# NumericalEarth. We stack a 2D atmosphere directly above a 2D ocean, share the same
# horizontal extent, and let the coupler do the bookkeeping:
#
# 1. Each step, **similarity theory** (Monin–Obukhov bulk formulae) reads the
#    near-surface air and the sea-surface temperature and computes the turbulent
#    **sensible heat, latent heat, and momentum fluxes** at the interface.
# 2. Those fluxes are handed *up* into the atmosphere as its bottom boundary condition
#    and *down* into the ocean as its top boundary condition — with opposite signs,
#    because what leaves the ocean enters the air.
# 3. Both fluids step forward; the surface state changes; the fluxes are recomputed.
#    That feedback loop is **two-way air–sea coupling**.
#
# ## The physics: a warm filament writes convection above and below
#
# The ocean carries a warm **filament** — a localized ribbon of water a few hundred
# metres wide, several K warmer than its surroundings (the 2D cousin of the 3D
# warm-filament case). The background sea surface sits near the air temperature, so
# away from the filament the air–sea contrast is small and little happens. Over the
# filament the water is several K warmer than the air, so heat flows upward and:
#
# - **Above the surface**, the warmed near-surface air becomes buoyant and rises in
#   thermals — atmospheric free convection, but now the heating is *earned* from the
#   warm water rather than prescribed, and it is *localized* over the filament.
# - **Below the surface**, the filament *loses* heat from its top; the cooled surface
#   water becomes denser and sinks — **ocean convection**, the upside-down mirror of
#   the atmospheric plumes.
#
# The air–sea fluxes that drive all of this are computed every step by similarity
# theory, so the filament's imprint on the surface fluxes is an *emergent* result —
# which we analyze at the end.
#
# This is a **2D** demo (`Flat` in `y`) sized to run quickly: a conceptual cartoon of
# coupled convection, the foundation for the 3D warm-filament case.
#
# !!! warning "Untested coupling path"
#     `AtmosphereOceanModel` with a **nonhydrostatic** ocean (LES) is not yet exercised
#     by the upstream NumericalEarth test suite — the tested path uses a `SlabOcean`.
#     The construction below follows the patterns in
#     `NumericalEarth.jl/test/test_breeze_coupling.jl` as closely as possible; treat the
#     nonhydrostatic-ocean coupling as experimental.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using CUDA   # provides Oceananigans' no-argument `GPU()` architecture
using Printf
using Random

Random.seed!(1994)

arch = GPU()
Oceananigans.defaults.FloatType = Float64  # coupled: the ESM clock is Float64, so the component grids must match
nothing #hide

# ## Two grids that share a sea surface
#
# The atmosphere lives in `z ∈ [0, Lz_a]` above the interface; the ocean lives in
# `z ∈ [-Lz_o, 0]` below it. Both share the *same* horizontal extent `Lx` and the
# *same* number of horizontal cells `Nx`, so the coupler maps surface columns
# one-to-one across the interface. Both are `Periodic` in `x`, `Flat` in `y` (this is
# what makes the demo 2D), and `Bounded` in `z`.

Lx   = 4kilometers    # shared horizontal extent
Lz_a = 3kilometers    # atmosphere depth
Lz_o = 100meters      # ocean depth

Nx   = 128
Nz_a = 96             # atmosphere vertical cells
Nz_o = 64             # ocean vertical cells

atmos_grid = RectilinearGrid(arch; size = (Nx, Nz_a), halo = (5, 5),
                             x = (0, Lx), z = (0, Lz_a),
                             topology = (Periodic, Flat, Bounded))

ocean_grid = RectilinearGrid(arch; size = (Nx, Nz_o), halo = (5, 5),
                             x = (0, Lx), z = (-Lz_o, 0),
                             topology = (Periodic, Flat, Bounded))

# ## The atmosphere component
#
# `atmosphere_simulation` builds a Breeze `AtmosphereModel` wrapped in a `Simulation`,
# **pre-wired for coupling**: its bottom boundary conditions on momentum, energy, and
# moisture are blank 2D fields that the coupler fills each step. We must *not* add our
# own surface flux BCs — the coupler owns them.
#
# The atmosphere is moist by default (warm-phase saturation-adjustment microphysics),
# so if the rising air saturates it can form cloud. We give it a reference potential
# temperature of 290 K (≈ 16.9 °C) and a Smagorinsky–Lilly LES closure, and initialize
# it isothermal at the reference state with a light 1 m s⁻¹ wind (so the bulk formulae
# have a nonzero wind to work with, and the thermals lean).

atmosphere = atmosphere_simulation(atmos_grid;
                                   potential_temperature = 290,
                                   closure = SmagorinskyLilly())

## Initialize the atmosphere at its reference potential-temperature profile with a
## light mean wind. `set!` acts on the wrapped `atmosphere.model`.
set!(atmosphere.model,
     θ = atmosphere.model.dynamics.reference_state.potential_temperature,
     u = 1)

# ## The ocean component, with a warm filament
#
# `ocean_simulation(grid; model = :nonhydrostatic)` builds an Oceananigans
# `NonhydrostaticModel` (full 3D pressure, LES-ready) wrapped in a `Simulation`, with
# `(T, S)` tracers and a TEOS-10 seawater equation of state. Like the atmosphere, its
# *top* boundary conditions are blank coupling fields the coupler fills.
#
# The initial ocean temperature is a **warm Gaussian filament**: a near-neutral
# background sea surface (`T_ambient ≈` the 16.9 °C airmass) with a warm ribbon of
# water centered in the domain, peaking `ΔT` warmer. That warm band is the engine —
# it is where the air–sea contrast is large, so it is where convection lights up on
# both sides of the interface.
#
# !!! warning "Units: the ocean is in °C, the atmosphere is in K"
#     The ocean's TEOS-10 equation of state and the air–sea coupler both expect the
#     ocean temperature in **degrees Celsius**, so the filament is built in °C — *not*
#     Kelvin. The Breeze atmosphere is in Kelvin (`potential_temperature = 290 K`).
#     Mixing the two up silently breaks the flux calculation: a Kelvin SST is read as
#     hundreds of °C, the saturation humidity blows up, and the interface humidity goes
#     negative — driving a runaway spurious-condensation instability.

ocean = ocean_simulation(ocean_grid; model = :nonhydrostatic)

T_ambient = 17     # °C, background sea surface ≈ the 290 K (≈ 16.9 °C) airmass → near-neutral
ΔT        = 3      # °C, filament warm anomaly (peak ≈ 20 °C, ≈ 3 K warmer than the air)
x₀        = Lx / 2 # filament center
σ         = 700    # m, filament Gaussian half-width scale (≈ 1.6 km full width)

## A warm Gaussian filament, uniform with depth; uniform salinity. `ocean_simulation`
## returns a `Simulation`, so `set!` targets `ocean.model`.
Tᵢ(x, z) = T_ambient + ΔT * exp(-(x - x₀)^2 / (2σ^2))
set!(ocean.model, T = Tᵢ, S = 35)

# ## Couple them
#
# `AtmosphereOceanModel(atmosphere, ocean)` returns an `EarthSystemModel`: it builds the
# interface bookkeeping (the `atmosphere_ocean_interface`) and wires the
# similarity-theory flux computation between the two components. The *coupled*
# `Simulation` wrapped around it owns the time step `Δt` — the inner atmosphere and
# ocean simulations defer to it.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Simulation
#
# The coupled model is stepped by a single outer `Simulation`. A fixed 2 s step keeps
# the vigorous, localized ocean convection over the warm filament within CFL on the
# thin 100 m ocean grid; we integrate for 2 hours of simulated time, long enough to
# spin up convection over the filament and to watch the surface fluxes develop. A
# progress callback prints the peak vertical velocity in each fluid and the peak
# sensible heat flux.

simulation = Simulation(model; Δt = 2, stop_time = 2hours)

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
# Because both grids are `Flat` in `y`, the vertical velocities and tracers are already
# 2D `x–z` fields — no slicing needed. From each fluid we save the vertical velocity and
# a thermodynamic field; from the interface we save the **air–sea fluxes** computed by
# similarity theory (a 1D line along the surface), every 30 seconds.
#
# Output writers attach to the *coupled* `simulation.output_writers`, but each
# `JLD2Writer` is built around the relevant **component model** (`atmosphere.model` or
# `ocean.model`), whose fields it samples.

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

## Ocean-side diagnostics — `T_o` at the first output captures the initial filament.
w_o = ocean.model.velocities.w
T_o = ocean.model.tracers.T
ocean_outputs = (; w_o, T_o)

## The air–sea fluxes along the surface, computed by similarity theory: sensible and
## latent heat (W m⁻²) and the x-momentum flux (kg m⁻¹ s⁻²). These are the analysis target.
fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
Q_sensible = fluxes.sensible_heat
Q_latent   = fluxes.latent_heat
τx         = fluxes.x_momentum
flux_outputs = (; Q_sensible, Q_latent, τx)

simulation.output_writers[:atmosphere] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = "coupled_convection_atmosphere.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = "coupled_convection_ocean.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "coupled_convection_fluxes.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

# ## Go time

run!(simulation)

@info "Intro coupled convection complete"
nothing #hide

# ## References
#
# - **Monin, A. S., Obukhov, A. M. (1954).** Basic laws of turbulent mixing in the
#   surface layer of the atmosphere. *Tr. Akad. Nauk SSSR Geofiz. Inst.*, 24, 163–187.
#   — the similarity theory underlying the bulk air–sea flux formulae the coupler uses.
# - **Large, W. G., Yeager, S. G. (2009).** The global climatology of an interannually
#   varying air–sea flux data set. *Climate Dynamics*, 33, 341–364.
#   <https://doi.org/10.1007/s00382-008-0441-3> — bulk transfer coefficients for the
#   air–sea turbulent fluxes.
#
# !!! note "Why 2D, and why a thin ocean?"
#     This example is two-dimensional (`Flat` in `y`) and uses a shallow 100 m ocean so
#     it spins up convection on both sides quickly and stays cheap. Real air–sea
#     turbulence is three-dimensional, and a real oceanic mixed layer is deeper; treat
#     the structure here as schematic. The point is the *coupling* — fluxes computed at
#     the interface, not prescribed — which carries over unchanged to the 3D
#     warm-filament case.
