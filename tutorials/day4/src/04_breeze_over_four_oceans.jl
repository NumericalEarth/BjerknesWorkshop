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

#nb # First, activate the tutorial environment and install its dependencies.
#nb using Pkg
#nb Pkg.activate(@__DIR__)
#nb Pkg.instantiate()
#nb #-

using NumericalEarth
using Breeze
using Oceananigans, Oceananigans.Units

using Oceananigans.TimeSteppers: reset!
using Printf, Random, Statistics
using CUDA

using CairoMakie

# Every case runs on the GPU.

arch = GPU()

# A small helper that base64-embeds a finished `.mp4` in an HTML5 `<video>` tag, so
# the animation plays inline in the notebook (no external file serving needed).

using Base64

mp4_html(path) = HTML(string("<video autoplay loop muted playsinline controls ",
                             "src=\"data:video/mp4;base64,", base64encode(read(path)),
                             "\" style=\"max-width:100%\"></video>"))

# ## Grid setup
#
# We use a 2D domain in the x--z plane. The horizontal extent `Lx` is shared by the
# atmosphere and every ocean, so the horizontal spacing `Lx / Nx` is common to both:
# at 10 km and `Nx = 128` it is ≈ 78 m. (For a true ocean LES we would refine toward
# `Δx = 4 m`; the coarse grid here keeps this version cheap for testing.) The vertical extent `Lz` is
# the depth of the atmospheric column, while the deep oceans occupy a separate column of
# depth `Lzᵒᶜ` resolved much more finely in the vertical. The `Periodic` x-topology lets
# convective cells wrap around, and the `Flat` y-topology makes this a 2D simulation.

Lx  = 10kilometers  # horizontal extent (shared by the atmosphere and every ocean)
Lz  = 10kilometers  # atmosphere vertical extent
Lzᵒᶜ = 50meters     # ocean depth (hydrostatic and nonhydrostatic columns)

Nx = 128   # horizontal resolution → Δx = Lx / Nx = 78 m (shared by atmosphere and ocean)
Nz = 128   # atmosphere vertical resolution → Δz = Lz / Nz = 78 m
Nzᵒᶜ = 20  # hydrostatic ocean vertical resolution → Δz = Lzᵒᶜ / Nzᵒᶜ = 2.5 m
Nzⁿʰ = 50  # nonhydrostatic ocean vertical resolution → Δz = Lzᵒᶜ / Nzⁿʰ = 1 m

grid = RectilinearGrid(arch, size = (Nx, Nz), halo = (5, 5),
                       x = (-Lx/2, Lx/2),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## One atmosphere, reused across every case
#
# `atmosphere_simulation` returns an Oceananigans `Simulation` wrapping a Breeze
# `AtmosphereModel`. We keep a handle on both: the `Simulation` is what we `reset!`
# and couple, while `atmosphere.model` (the underlying model) is what we `set!` and plot.

Tᵒᶜ = 290 # K — ocean surface temperature
θᵃᵗ = 250 # K — initial atmospheric potential temperature
U₀ = 10   # m/s — background zonal wind
coriolis = FPlane(latitude=33)

atmosphere = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)

# We initialize the atmosphere with the reference potential-temperature profile
# plus small random perturbations below 500 m. These perturbations seed the
# convective instability that develops into turbulence driven by the surface heat
# flux. The background zonal wind `U₀` provides a nonzero surface wind speed for the
# similarity-theory flux computation.

reference_state = atmosphere.model.dynamics.reference_state
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

# The atmospheric CFL is set by the resolved winds on the fine horizontal grid, so we
# let a `TimeStepWizard` adapt `Δt` to a fixed CFL every iteration rather than guessing
# a stable fixed step. `Δt` below is only the (deliberately small) initial step; the
# wizard grows it toward the CFL limit, capped by `max_Δt`.

Δt = 1seconds  # initial step; the wizard adapts it from here
stop_time = 1hours

# `build_coupled_simulation` re-initializes the shared atmosphere, wires it to `ocean`
# through the similarity-theory interface, and returns a `Simulation` around the coupled
# `AtmosphereOceanModel`. `conjure_time_step_wizard!` adapts `Δt` to track a CFL of 0.7
# (the coupled model reports the atmosphere's advection timescale), and a progress logger
# prints atmospheric statistics every 400 iterations.

function progress(sim)
    u, v, w = atmosphere.model.velocities
    msg = @sprintf("iter %d, t = %s, Δt = %.2f s, max|u| = %.2e m/s, max|w| = %.2e m/s",
                   iteration(sim), prettytime(sim), sim.Δt, maximum(abs, u), maximum(abs, w))
    @info msg
    return nothing
end

function build_coupled_simulation(ocean)
    initialize!(atmosphere)
    interfaces = ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes)
    model = AtmosphereOceanModel(atmosphere, ocean; interfaces)
    simulation = Simulation(model; Δt, stop_time)
    conjure_time_step_wizard!(simulation; cfl=0.7)
    add_callback!(simulation, progress, IterationInterval(400))
    return simulation
end

# The prescribed and slab oceans live on a single surface (`Flat` in z); the
# hydrostatic and nonhydrostatic oceans are 50 m deep columns. All inherit the
# atmosphere's architecture and horizontal grid.

# ## 1. Prescribed ocean (constant SST)
#
# The prescribed ocean holds a fixed temperature that does not evolve. Surface
# fluxes are still computed — the atmosphere feels the ocean — but the ocean
# temperature is pinned. Its temperature is in Kelvin.

sst_grid = RectilinearGrid(grid.architecture, size = grid.Nx, halo = grid.Hx,
                           x = (-Lx/2, Lx/2),
                           topology = (Periodic, Flat, Flat))

prescribed_ocean = PrescribedOcean(sst_grid)
set!(prescribed_ocean, T=Tᵒᶜ)

prescribed_simulation = build_coupled_simulation(prescribed_ocean)
run!(prescribed_simulation)

# Over the warm surface, the atmosphere has organized into convective plumes. We
# plot the final liquid--ice potential temperature and zonal velocity.

fig = Figure(size=(1200, 600))
axθ = Axis(fig[1, 1], title="θₗᵢ (K)",   ylabel="z (m)", aspect=Lx/Lz)
axu = Axis(fig[1, 2], title="u (m s⁻¹)", xlabel="x (m)", ylabel="z (m)", aspect=Lx/Lz)
heatmap!(axθ, liquid_ice_potential_temperature(atmosphere.model), colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(axu, atmosphere.model.velocities.u, colormap=:balance, colorrange=(-30, 30))
Label(fig[0, 1:2], "Prescribed ocean — final state (t = $(prettytime(prescribed_simulation)))", fontsize=16)
save("four_oceans_prescribed.png", fig)
display(fig)

# ### Air–sea fluxes from similarity theory
#
# Even though the prescribed ocean never changes, the coupler still computes the
# turbulent air–sea fluxes every step — they are exactly what the atmosphere feels at
# its lower boundary. The fluxes come from the bulk aerodynamic (Monin–Obukhov
# similarity) formulae, which write each flux as a transfer coefficient times the
# near-surface wind speed $|\mathbf{U}|$ times the air–sea contrast of the transported
# quantity:
#
# $$
# \begin{aligned}
# \rho\,\tau &= \rho_a\, C_D\, |\mathbf{U}|\,\mathbf{U}, &&\text{(momentum / drag, N m}^{-2}) \\
# Q_s &= \rho_a\, c_p\, C_T\, |\mathbf{U}|\,(\theta_o - \theta_a), &&\text{(sensible heat, W m}^{-2}) \\
# Q_\ell &= \rho_a\, \mathcal{L}_v\, C_q\, |\mathbf{U}|\,(q_o - q_a). &&\text{(latent heat, W m}^{-2})
# \end{aligned}
# $$
#
# The transfer coefficients $C_D, C_T, C_q$ are *not* constants: similarity theory makes
# them functions of the surface-layer stability and roughness, so the same wind drives
# different fluxes over a stable versus an unstable surface. Here the warm surface
# ($T_o = 290\,\mathrm{K}$) under cool air ($\theta_a = 250\,\mathrm{K}$) makes the
# surface layer convectively unstable, so the sensible and latent heat fluxes are
# directed upward, into the atmosphere.
#
# The interface stores each flux as a field along the sea surface; with a `Flat`
# y-direction they are 1D profiles in `x`. The variations along `x` are the surface
# imprint of the convective plumes — the wind speed and near-surface air properties the
# plumes carry to the surface modulate the local exchange.

fluxes = prescribed_simulation.model.interfaces.atmosphere_ocean_interface.fluxes
x = xnodes(fluxes.sensible_heat)

fig = Figure(size=(900, 700))
axτ = Axis(fig[1, 1], title="momentum flux ρτˣ (drag)", ylabel="N m⁻²")
axs = Axis(fig[2, 1], title="sensible heat flux Qˢ",    ylabel="W m⁻²")
axℓ = Axis(fig[3, 1], title="latent heat flux Qˡ",      xlabel="x (m)", ylabel="W m⁻²")
lines!(axτ, x, view(fluxes.x_momentum,    :, 1, 1))
lines!(axs, x, view(fluxes.sensible_heat, :, 1, 1))
lines!(axℓ, x, view(fluxes.latent_heat,   :, 1, 1))
Label(fig[0, 1], "Prescribed ocean — air–sea fluxes (t = $(prettytime(prescribed_simulation)))", fontsize=16)
save("four_oceans_prescribed_fluxes.png", fig)
display(fig)

# ## 2. Slab ocean (10 m depth)
#
# The slab ocean is the simplest *responding* ocean: a single well-mixed layer of fixed
# depth `H` whose temperature evolves under the net surface heat flux,
#
# $$
# \frac{\partial T}{\partial t} = \frac{Q}{\rho\, c_p\, H},
# $$
#
# where $Q$ is the net downward surface heat flux (W m⁻²), $\rho$ the seawater density,
# $c_p$ the heat capacity, and $H$ the slab depth. There is no vertical structure — the
# whole layer warms or cools together — so the slab simply integrates the air–sea flux
# in time. A thinner layer (smaller `H`) has less heat capacity and responds faster: at
# `H = 10 m` the surface cools visibly within the run, unlike the pinned prescribed
# ocean. Its temperature is in Kelvin.

slab_ocean = SlabOcean(sst_grid, depth=10)
set!(slab_ocean, T=Tᵒᶜ)

slab_simulation = build_coupled_simulation(slab_ocean)
run!(slab_simulation)

# Now the SST is free to respond. The cooling is strongest beneath the most
# vigorous convection. We plot the atmosphere together with the final SST.

fig = Figure(size=(1200, 400))
axθ = Axis(fig[1, 1], title="θₗᵢ (K)",   ylabel="z (m)", aspect=Lx/Lz)
axu = Axis(fig[1, 2], title="u (m s⁻¹)", ylabel="z (m)", aspect=Lx/Lz)
axT = Axis(fig[1, 3], title="SST (K)",   xlabel="x (m)", ylabel="SST (K)", aspect=Lx/Lz)
heatmap!(axθ, liquid_ice_potential_temperature(atmosphere.model), colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(axu, atmosphere.model.velocities.u, colormap=:balance, colorrange=(-30, 30))
lines!(axT, slab_ocean.temperature, color=:red, linewidth=2)
Label(fig[0, 1:3], "Slab ocean — final state (t = $(prettytime(slab_simulation)))", fontsize=16)
save("four_oceans_slab.png", fig)
display(fig)

# ## 3. Hydrostatic ocean (50 m depth with CATKE mixing)
#
# The full ocean uses a `HydrostaticFreeSurfaceModel` with the default TEOS-10
# equation of state and CATKE vertical mixing. The grid has 20 vertical levels
# (2.5 m resolution). We disable advection since this is primarily a 1D vertical
# mixing problem.

ocean_grid = RectilinearGrid(grid.architecture, size = (grid.Nx, Nzᵒᶜ), halo = (grid.Hx, 5),
                             x = (-Lx/2, Lx/2), z = (-Lzᵒᶜ, 0),
                             topology = (Periodic, Flat, Bounded))

ocean = ocean_simulation(ocean_grid; coriolis,
                         closure = CATKEVerticalDiffusivity(),
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         Δt = 2,
                         warn = false)

set!(ocean.model, T=Tᵢ, S=35)

hydrostatic_simulation = build_coupled_simulation(ocean)
run!(hydrostatic_simulation)

# CATKE mixes the surface cooling downward, deepening a weakly stratified boundary
# layer. We plot the atmospheric cloud water and vertical velocity together with
# the ocean temperature cross-section.

fig = Figure(size=(800, 850))
axq = Axis(fig[1, 1], title="cloud water qˡ", ylabel="z (m)")
axw = Axis(fig[2, 1], title="w (m s⁻¹)",      ylabel="z (m)")
axT = Axis(fig[3, 1], title="ocean T (°C)",   xlabel="x (m)", ylabel="z (m)")
heatmap!(axq, atmosphere.model.microphysical_fields.qˡ, colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(axw, atmosphere.model.velocities.w, colormap=:balance, colorrange=(-25, 25))
heatmap!(axT, ocean.model.tracers.T, colormap=:thermal, colorrange=(T₀ - 1.5, T₀ + 0.5))
Label(fig[0, 1], "Hydrostatic ocean — final state (t = $(prettytime(hydrostatic_simulation)))", fontsize=16)
save("four_oceans_hydrostatic.png", fig)
display(fig)

# ### The developing mixed layer
#
# The cross-section shows horizontal structure, but the clearest signature of the
# surface cooling is in the *horizontal average*. We average the ocean temperature over
# `x` and compare the final profile to the initial stratification. CATKE has carried the
# surface buoyancy loss downward and homogenized the upper ocean into a **mixed layer** —
# a near-surface slab of nearly uniform temperature — that sits above the still-stratified
# water below. The base of the mixed layer (where the profile rejoins the dashed initial
# line) marks how deep the cooling has penetrated.

zc = znodes(ocean.model.tracers.T)
T̄ = Field(Average(ocean.model.tracers.T, dims=1))
compute!(T̄)

fig = Figure(size=(450, 600))
ax = Axis(fig[1, 1], title="Hydrostatic ocean — horizontally averaged T",
          xlabel="T (°C)", ylabel="z (m)")
lines!(ax, Tᵢ.(0, zc), zc, color=(:gray, 0.8), linestyle=:dash, label="initial")
lines!(ax, view(T̄, 1, 1, :), zc, color=:black, linewidth=2, label="final")
axislegend(ax, position=:rb)
save("four_oceans_hydrostatic_profile.png", fig)
display(fig)

# ## 4. Nonhydrostatic ocean LES (50 m depth)
#
# The nonhydrostatic ocean uses a `NonhydrostaticModel` that resolves the full 3D
# pressure field. With 1 m vertical resolution and WENO advection it performs
# implicit LES — no turbulence closure is needed, and the convective turbulence
# below the surface is resolved rather than parameterized. This is the case worth
# animating.

nh_ocean_grid = RectilinearGrid(grid.architecture, size = (grid.Nx, Nzⁿʰ), halo = (grid.Hx, 5),
                                x = (-Lx/2, Lx/2), z = (-Lzᵒᶜ, 0),
                                topology = (Periodic, Flat, Bounded))

nh_ocean = ocean_simulation(nh_ocean_grid; model=:nonhydrostatic, coriolis, Δt=2)

# Unlike the hydrostatic ocean (whose CATKE closure mixes from a smooth profile), the
# nonhydrostatic LES must *resolve* convection, so we seed the initial temperature with
# small noise — without it the symmetric initial state has nothing for the instability
# to grow from.
Tᵢⁿʰ(x, z) = Tᵢ(x, z) + 1e-2 * rand()
set!(nh_ocean.model, T=Tᵢⁿʰ, S=35)

nh_simulation = build_coupled_simulation(nh_ocean)

# This time we attach output writers so we can animate the evolution. We save the
# atmospheric cloud water and vertical velocity, and the ocean temperature, once a
# minute of simulated time.

nh_simulation.output_writers[:atmos] = JLD2Writer(nh_simulation.model,
                                                  (; qˡ=atmosphere.model.microphysical_fields.qˡ, w=atmosphere.model.velocities.w),
                                                  filename = "four_oceans_nh_atmos",
                                                  schedule = TimeInterval(1minute),
                                                  overwrite_existing = true)

nh_simulation.output_writers[:ocean] = JLD2Writer(nh_simulation.model,
                                                  (; T=nh_ocean.model.tracers.T, w=nh_ocean.model.velocities.w),
                                                  filename = "four_oceans_nh_ocean",
                                                  schedule = TimeInterval(1minute),
                                                  overwrite_existing = true)

run!(nh_simulation)

# ## Animation
#
# We load the saved fields back as `FieldTimeSeries` and animate the resolved
# turbulence: cloud water and vertical velocity in the atmosphere, and the ocean
# temperature below.

qˡ_ts = FieldTimeSeries("four_oceans_nh_atmos.jld2", "qˡ")
w_ts  = FieldTimeSeries("four_oceans_nh_atmos.jld2", "w")
T_ts  = FieldTimeSeries("four_oceans_nh_ocean.jld2", "T")
wᵒ_ts = FieldTimeSeries("four_oceans_nh_ocean.jld2", "w")

times = w_ts.times
Nt = length(times)

n = Observable(1)
qˡn = @lift qˡ_ts[$n]
wn  = @lift w_ts[$n]
Tn  = @lift T_ts[$n]
wᵒn = @lift wᵒ_ts[$n]

# Four panels: the atmosphere (cloud water, vertical velocity) on top, the ocean
# (temperature, vertical velocity) below — the turbulence on both sides of the interface.
fig = Figure(size=(1200, 700))
axqᵃ = Axis(fig[1, 1], title="atmosphere cloud water qˡ", ylabel="z (m)")
axwᵃ = Axis(fig[1, 2], title="atmosphere w (m s⁻¹)",      ylabel="z (m)")
axTᵒ = Axis(fig[2, 1], title="ocean T (°C)",    xlabel="x (m)", ylabel="z (m)")
axwᵒ = Axis(fig[2, 2], title="ocean w (m s⁻¹)", xlabel="x (m)", ylabel="z (m)")
heatmap!(axqᵃ, qˡn, colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(axwᵃ, wn,  colormap=:balance,          colorrange=(-25, 25))
heatmap!(axTᵒ, Tn,  colormap=:thermal,          colorrange=(T₀ - 1.5, T₀ + 0.5))
heatmap!(axwᵒ, wᵒn, colormap=:balance,          colorrange=(-0.02, 0.02))

title = @lift "Nonhydrostatic ocean — t = " * prettytime(times[$n])
Label(fig[0, 1:2], title, fontsize=16)

@info "Rendering animation..."
CairoMakie.record(fig, "four_oceans_nonhydrostatic.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end
@info "Animation saved."

mp4_html("four_oceans_nonhydrostatic.mp4")
