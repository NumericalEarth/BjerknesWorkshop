# # Atmospheric convection over four ocean models
#
# This example couples a Breeze atmospheric large eddy simulation (LES) to four
# different ocean models, one after another, using NumericalEarth's
# `EarthSystemModel` framework. The coupling computes turbulent surface fluxes
# (sensible heat, latent heat, and momentum) using Monin--Obukhov similarity
# theory. These fluxes cool the ocean and warm the atmosphere, creating a two-way
# feedback.
#
# We walk through four oceans in order of increasing physical fidelity:
#
# 1. **Prescribed ocean** — constant SST that does not respond to surface fluxes
# 2. **Slab ocean** (10 m) — a well-mixed layer that responds uniformly to fluxes
# 3. **Hydrostatic ocean** (50 m) — CATKE vertical mixing and stratification
# 4. **Nonhydrostatic ocean** (50 m) — resolved LES turbulence with WENO advection
#
# Rather than build four atmospheres, we build *one* and reuse it. Between cases we
# `reset!` the simulation (zeroing its clock and time stepper) and `set!` the
# atmosphere back to its initial state, so every ocean meets the same atmosphere.
# For the first three oceans we run the simulation and plot only the final state —
# there is no need to pay for an animation. For the nonhydrostatic ocean we record
# an animation of the turbulence that develops on both sides of the interface.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: reset!
using Printf
using Random
using Statistics: mean
using CairoMakie
using CUDA
using Base64

# Every case runs on the GPU.

arch = GPU()

# A small helper that base64-embeds a finished `.mp4` in an HTML5 `<video>` tag, so
# the animation plays inline in the notebook (no external file serving needed).

mp4_html(path) = HTML(string("<video autoplay loop muted playsinline controls ",
                             "src=\"data:video/mp4;base64,", base64encode(read(path)),
                             "\" style=\"max-width:100%\"></video>"))

# ## Grid setup
#
# We use a 2D domain in the x--z plane: 20 km wide and 10 km tall with 64 × 64 grid
# points. The `Periodic` x-topology lets convective cells wrap around, and the
# `Flat` y-topology makes this a 2D simulation.

Nx = 64    # atmosphere horizontal resolution (shared with every ocean)
Nz = 64    # atmosphere vertical resolution
Nzᵒᶜ = 20  # hydrostatic ocean vertical resolution (2.5 m spacing)
Nzⁿʰ = 50  # nonhydrostatic ocean vertical resolution (1 m spacing)

grid = RectilinearGrid(arch, size = (Nx, Nz), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## One atmosphere, reused across every case
#
# `atmosphere_simulation` returns an Oceananigans `Simulation` wrapping a Breeze
# `AtmosphereModel`. We keep a handle on both: the `Simulation` is what we `reset!`
# and couple, while `atmos` (the underlying model) is what we `set!` and plot.

Tᵒᶜ = 290 # K — ocean surface temperature
θᵃᵗ = 250 # K — initial atmospheric potential temperature
U₀ = 10   # m/s — background zonal wind
coriolis = FPlane(latitude=33)

atmosphere = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)
atmos = atmosphere.model

# We initialize the atmosphere with the reference potential-temperature profile
# plus small random perturbations below 500 m. These perturbations seed the
# convective instability that develops into turbulence driven by the surface heat
# flux. The background zonal wind `U₀` provides a nonzero surface wind speed for the
# similarity-theory flux computation.

reference_state = atmos.dynamics.reference_state
θᵢ(x, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)

# `initialize!` returns the shared atmosphere to its starting point: `reset!` zeroes
# the clock, time stepper, and stop criteria, and `set!` restores the initial
# fields. We call it before each coupled run.

function initialize!(atmosphere)
    reset!(atmosphere)
    set!(atmosphere.model, θ=θᵢ, u=U₀)
    return atmosphere
end

# ## Coupling
#
# We disable gustiness in the similarity-theory flux computation, so the surface
# wind speed is determined entirely by the resolved velocity field.

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(gustiness_parameter=0, minimum_gustiness=0)

# The deep oceans (hydrostatic and nonhydrostatic) carry temperature in °C, since
# TEOS-10 expects Celsius; the coupling framework converts to Kelvin for the flux
# computation. We initialize them with a fixed surface temperature and a linear
# stratification below 10 m.

celsius_to_kelvin = 273.15
T₀ = Tᵒᶜ - celsius_to_kelvin                # ocean surface temperature in °C
Tᵢ(x, z) = T₀ + (z + 10) / 50 * (z < -10)  # linear stratification below 10 m

Δt = 5seconds
stop_time = 4hours

# `couple` re-initializes the shared atmosphere, wires it to `ocean` through the
# similarity-theory interface, and returns a `Simulation` around the coupled
# `AtmosphereOceanModel`. A progress logger prints atmospheric statistics every 400
# iterations.

function progress(sim)
    u, v, w = atmos.velocities
    msg = @sprintf("iter %d, t = %s, max|u| = %.2e m/s, max|w| = %.2e m/s",
                   iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w))
    @info msg
    return nothing
end

function couple(ocean)
    initialize!(atmosphere)
    interfaces = ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes)
    model = AtmosphereOceanModel(atmosphere, ocean; interfaces)
    simulation = Simulation(model; Δt, stop_time)
    add_callback!(simulation, progress, IterationInterval(400))
    return simulation
end

# The prescribed and slab oceans live on a single surface (`Flat` in z); the
# hydrostatic and nonhydrostatic oceans are 50 m deep columns. All inherit the
# atmosphere's architecture and horizontal grid.

sst_grid = RectilinearGrid(grid.architecture, size = grid.Nx, halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

# ## 1. Prescribed ocean (constant SST)
#
# The prescribed ocean holds a fixed temperature that does not evolve. Surface
# fluxes are still computed — the atmosphere feels the ocean — but the ocean
# temperature is pinned. Its temperature is in Kelvin.

prescribed_ocean = PrescribedOcean(sst_grid)
set!(prescribed_ocean, T=Tᵒᶜ)

prescribed_simulation = couple(prescribed_ocean)
run!(prescribed_simulation)

# Over the warm surface, the atmosphere has organized into convective plumes. We
# plot the final liquid--ice potential temperature and zonal velocity.

fig = Figure(size=(800, 600))
axθ = Axis(fig[1, 1], title="θₗᵢ (K)",   ylabel="z (m)")
axu = Axis(fig[2, 1], title="u (m s⁻¹)", xlabel="x (m)", ylabel="z (m)")
heatmap!(axθ, liquid_ice_potential_temperature(atmos), colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(axu, atmos.velocities.u, colormap=:balance, colorrange=(-30, 30))
Label(fig[0, 1], "Prescribed ocean — final state (t = $(prettytime(prescribed_simulation)))", fontsize=16)
save("four_oceans_prescribed.png", fig)
display(fig)

# ## 2. Slab ocean (10 m depth)
#
# The slab ocean represents a well-mixed ocean layer of fixed depth that responds
# uniformly to surface fluxes. Its temperature is in Kelvin.

slab_ocean = SlabOcean(sst_grid, depth=10)
set!(slab_ocean, T=Tᵒᶜ)

slab_simulation = couple(slab_ocean)
run!(slab_simulation)

# Now the SST is free to respond. The cooling is strongest beneath the most
# vigorous convection. We plot the atmosphere together with the final SST.

fig = Figure(size=(800, 850))
axθ = Axis(fig[1, 1], title="θₗᵢ (K)",   ylabel="z (m)")
axu = Axis(fig[2, 1], title="u (m s⁻¹)", ylabel="z (m)")
axT = Axis(fig[3, 1], title="SST (K)",   xlabel="x (m)", ylabel="SST (K)")
heatmap!(axθ, liquid_ice_potential_temperature(atmos), colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(axu, atmos.velocities.u, colormap=:balance, colorrange=(-30, 30))
lines!(axT, slab_ocean.temperature, color=:red, linewidth=2)
Label(fig[0, 1], "Slab ocean — final state (t = $(prettytime(slab_simulation)))", fontsize=16)
save("four_oceans_slab.png", fig)
display(fig)

# ## 3. Hydrostatic ocean (50 m depth with CATKE mixing)
#
# The full ocean uses a `HydrostaticFreeSurfaceModel` with the default TEOS-10
# equation of state and CATKE vertical mixing. The grid has 20 vertical levels
# (2.5 m resolution). We disable advection since this is primarily a 1D vertical
# mixing problem.

ocean_grid = RectilinearGrid(grid.architecture, size = (grid.Nx, Nzᵒᶜ), halo = (grid.Hx, 5),
                             x = (-10kilometers, 10kilometers), z = (-50, 0),
                             topology = (Periodic, Flat, Bounded))

ocean = ocean_simulation(ocean_grid; coriolis,
                         closure = CATKEVerticalDiffusivity(),
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         Δt = 2,
                         warn = false)

set!(ocean.model, T=Tᵢ, S=35)

hydrostatic_simulation = couple(ocean)
run!(hydrostatic_simulation)

# CATKE mixes the surface cooling downward, deepening a weakly stratified boundary
# layer. We plot the atmospheric cloud water and vertical velocity together with
# the ocean temperature cross-section.

fig = Figure(size=(800, 850))
axq = Axis(fig[1, 1], title="cloud water qˡ", ylabel="z (m)")
axw = Axis(fig[2, 1], title="w (m s⁻¹)",      ylabel="z (m)")
axT = Axis(fig[3, 1], title="ocean T (°C)",   xlabel="x (m)", ylabel="z (m)")
heatmap!(axq, atmos.microphysical_fields.qˡ, colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(axw, atmos.velocities.w, colormap=:balance, colorrange=(-25, 25))
heatmap!(axT, ocean.model.tracers.T, colormap=:thermal, colorrange=(T₀ - 1.5, T₀ + 0.5))
Label(fig[0, 1], "Hydrostatic ocean — final state (t = $(prettytime(hydrostatic_simulation)))", fontsize=16)
save("four_oceans_hydrostatic.png", fig)
display(fig)

# ## 4. Nonhydrostatic ocean LES (50 m depth)
#
# The nonhydrostatic ocean uses a `NonhydrostaticModel` that resolves the full 3D
# pressure field. With 1 m vertical resolution and WENO advection it performs
# implicit LES — no turbulence closure is needed, and the convective turbulence
# below the surface is resolved rather than parameterized. This is the case worth
# animating.

nh_ocean_grid = RectilinearGrid(grid.architecture, size = (grid.Nx, Nzⁿʰ), halo = (grid.Hx, 5),
                                x = (-10kilometers, 10kilometers), z = (-50, 0),
                                topology = (Periodic, Flat, Bounded))

nh_ocean = ocean_simulation(nh_ocean_grid; model=:nonhydrostatic, coriolis, Δt=2)
set!(nh_ocean.model, T=Tᵢ, S=35)

nh_simulation = couple(nh_ocean)

# This time we attach output writers so we can animate the evolution. We save the
# atmospheric cloud water and vertical velocity, and the ocean temperature, once a
# minute of simulated time.

nh_simulation.output_writers[:atmos] = JLD2Writer(nh_simulation.model,
                                                  (; qˡ=atmos.microphysical_fields.qˡ, w=atmos.velocities.w),
                                                  filename = "four_oceans_nh_atmos",
                                                  schedule = TimeInterval(1minute),
                                                  overwrite_existing = true)

nh_simulation.output_writers[:ocean] = JLD2Writer(nh_simulation.model,
                                                  (; T=nh_ocean.model.tracers.T),
                                                  filename = "four_oceans_nh_ocean",
                                                  schedule = TimeInterval(1minute),
                                                  overwrite_existing = true)

run!(nh_simulation)

# ## Animation
#
# We load the saved fields back as `FieldTimeSeries` and animate the resolved
# turbulence: cloud water and vertical velocity in the atmosphere, and the ocean
# temperature below.

# We load the series without passing a grid, so they are reconstructed on the CPU
# from the file — Makie iterates over each frame to render it, which is a scalar
# operation that must not touch GPU arrays.

qˡ_ts = FieldTimeSeries("four_oceans_nh_atmos.jld2", "qˡ")
w_ts  = FieldTimeSeries("four_oceans_nh_atmos.jld2", "w")
T_ts  = FieldTimeSeries("four_oceans_nh_ocean.jld2", "T")

times = w_ts.times
Nt = length(times)

n = Observable(1)
qˡn = @lift qˡ_ts[$n]
wn  = @lift w_ts[$n]
Tn  = @lift T_ts[$n]

fig = Figure(size=(1000, 850))
axq = Axis(fig[1, 1], title="cloud water qˡ", ylabel="z (m)")
axw = Axis(fig[2, 1], title="w (m s⁻¹)",      ylabel="z (m)")
axT = Axis(fig[3, 1], title="ocean T (°C)",   xlabel="x (m)", ylabel="z (m)")
heatmap!(axq, qˡn, colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(axw, wn,  colormap=:balance,          colorrange=(-25, 25))
heatmap!(axT, Tn,  colormap=:thermal,          colorrange=(T₀ - 1.5, T₀ + 0.5))

title = @lift "Nonhydrostatic ocean — t = " * prettytime(times[$n])
Label(fig[0, 1], title, fontsize=16)

@info "Rendering animation..."
CairoMakie.record(fig, "four_oceans_nonhydrostatic.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end
@info "Animation saved."

mp4_html("four_oceans_nonhydrostatic.mp4")
