# # A Breeze tutorial: thermal bubbles to cloudy hills
#
# Documentation for Breeze is available at
# [numericalearth.github.io/BreezeDocumentation](https://numericalearth.github.io/BreezeDocumentation/dev/).
#
# This tutorial introduces Breeze using four experiments:
#
# | Part | What's new                              | Dynamics                    |
# |:----:|-----------------------------------------|-----------------------------|
# | I    | Dry thermal bubble                      | anelastic                   |
# | II   | Forced convection over a warm surface   | anelastic                   |
# | III  | Forced convection over a hill           | split-explicit compressible |
# | IV   | Cloudy, hilly convection                | split-explicit compressible |
#
# ## Environment management
#
# We begin by instantiating the environment:

using Pkg
Pkg.instantiate()

# and then we can get onto to building the environment,

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random
using CairoMakie
using CUDA

# Print the environment status *after* the `using` statements: only then can
# `Pkg.status` flag a package whose loaded version differs from the resolved one
# (shown as `[loaded: …]`). If you see that, restart the kernel — Julia can't
# hot-swap an already-loaded package.

Pkg.status()

arch = GPU()
Oceananigans.defaults.FloatType = Float32

# A small helper that base64-embeds a finished `.mp4` in an HTML5 `<video>` tag,
# so the animation plays inline in the notebook (no external file serving needed).

using Base64

mp4_html(path) = HTML(string("<video autoplay loop muted playsinline controls ",
                             "src=\"data:video/mp4;base64,", base64encode(read(path)),
                             "\" style=\"max-width:100%\"></video>"))

# ## The shared grid and background atmosphere
#
# A single vertical slice serves every part: periodic in `x`, `Flat` in `y`. Two-
# dimensional dynamics are a cartoon — no vortex stretching — but cheap to run and
# rich enough to show what we are after.

Lx, Lz = 24kilometers, 8kilometers
Nx, Nz = 384, 160

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       halo = (5, 5),
                       x = (-Lx/2, Lx/2),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

# ## The anelastic approximation and the reference state
#
# The first three parts use **anelastic** dynamics. The atmosphere is compressible,
# but sound waves are energetically irrelevant to convection and would force a tiny
# time step, so the anelastic approximation filters them: each field splits into a
# static, horizontally-uniform **reference profile** plus a small
# **perturbation** (prime),
#
# ```math
# ρ = ρᵣ(z) + ρ', \qquad p = pᵣ(z) + p', \qquad |ρ'| \ll ρᵣ ,
# ```
#
# with the reference state in hydrostatic balance, ``dpᵣ/dz = -ρᵣ g``. Mass
# conservation reads
#
# ```math
# \frac{∂ρ}{∂t} + ∇·(ρ \, 𝐮) = 0 ,
# ```
#
# and dropping ``∂ρ'/∂t`` — which is what removes the acoustic modes — leaves the
# **anelastic constraint**
#
# ```math
# ∇·(ρᵣ \, 𝐮) = 0 ,
# ```
#
# while ``p'`` is then diagnostic, from an elliptic solve enforcing that constraint.
#
# The reference column is fixed by the **reference potential temperature**: with a
# surface pressure, hydrostatic balance and the ideal-gas law then close ``pᵣ(z)``,
# ``Tᵣ(z)``, and the density ``ρᵣ(z)``. For the first two parts we take a *neutral*
# atmosphere — a constant reference potential temperature — and introduce
# stratification only at the hill in Part III. Advection is WENO with no closure.

θ₀ = 290     # K, reference potential temperature

reference_state = ReferenceState(grid; potential_temperature = θ₀)
dynamics = AnelasticDynamics(reference_state)
advection = WENO(order = 9)

model = AtmosphereModel(grid; dynamics, advection)

# Notice that `AtmosphereModel` solves the governing equations in **conservative
# (flux) form**: the prognostic variables are the *densities* of momentum, heat, and
# moisture — ``ρ𝐮``, ``ρθ``, ``ρqᵛ``, ``ρe`` — rather than the velocities and
# potential temperature themselves. "Flux form" means a tracer is advanced as the
# divergence of a flux, not by an advective (material) derivative. The two are equal:
# expand the flux-form tendency with the product rule and collect terms,
#
# ```math
# \frac{∂(ρθ)}{∂t} + ∇·(ρ \, 𝐮 \, θ)
#   = ρ \left( \frac{∂θ}{∂t} + 𝐮·∇θ \right)
#   + θ \underbrace{\left( \frac{∂ρ}{∂t} + ∇·(ρ \, 𝐮) \right)}_{=\,0} ,
# ```
#
# and the second group vanishes by **mass conservation**. So the conservative tendency
# ``∂(ρθ)/∂t + ∇·(ρ𝐮θ)`` is *identical* to the advective tendency
# ``ρ(∂θ/∂t + 𝐮·∇θ)`` — but advancing ``ρθ`` through the flux ``∇·(ρ𝐮θ)`` conserves
# total heat exactly (up to boundary fluxes), which is why the model carries ``ρθ`` and
# its siblings. This is visible in the printed summary above, where the advected
# quantities are listed as `ρθ` and `ρqᵛ` and the forcing entries act on `ρu`, `ρv`,
# `ρw`, `ρθ`, `ρqᵛ`, and `ρe`. Diagnostics such as
# `liquid_ice_potential_temperature(model)` recover the familiar primitive variables
# (e.g. ``θ = ρθ / ρ``) on demand.
#
# For the full conservative-form governing equations and the details of the anelastic
# approximation, see the
# [anelastic dynamics](https://numericalearth.github.io/BreezeDocumentation/dev/anelastic_dynamics/)
# page of the Breeze documentation.

# ## Part I — a dry thermal bubble
#
# A warm perturbation ``θ' = θ - θ₀`` feels a buoyancy
#
# ```math
# b = -g \, \frac{ρ'}{ρᵣ} = g \, \frac{θ'}{θ₀}
# ```
#
# (exact in the anelastic system, where buoyancy is evaluated at the reference
# pressure so ``p'`` drops out): warm air rises, which is why we plot ``θ'``.
#
# The "hello, world" of atmospheric dynamics: a warm blob released at rest in a
# neutral atmosphere. Being buoyant it rises, rolling up into the classic mushroom
# vortex pair. We paint the warmth on as a smooth cone.

Δθ = 2              # K, bubble amplitude
r₀ = 1.5kilometers  # bubble radius
z₀ = 2kilometers    # release height

θ_bubble(x, z) = θ₀ + Δθ * max(0, 1 - sqrt(x^2 + (z - z₀)^2) / r₀)
set!(model, θ=θ_bubble)

# Let's visualize the initial condition:

fig = Figure(size=(1200, 400), aspect=3)
ax = Axis(fig[1, 1])
heatmap!(ax, liquid_ice_potential_temperature(model))
display(fig)

# A CFL wizard adapts the time step as the bubble accelerates, and a progress
# callback logs the march periodically.

simulation = Simulation(model; Δt=1, stop_time=25minutes)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    @info @sprintf("thermal bubble | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# We save the potential-temperature perturbation ``θ' = θ - θ₀`` and the velocities.

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ₀
outputs = (; θ′, model.velocities...)

bubble_ow = JLD2Writer(model, outputs;
                       filename = "thermal_bubble.jld2",
                       schedule = TimeInterval(1minute),
                       overwrite_existing = true)

simulation.output_writers[:fields] = bubble_ow

# Now we are ready to run the simulation,

run!(simulation)

# ### The movie
#
# Replay every saved frame as a heatmap of ``θ'``.

θ′_ts = FieldTimeSeries("thermal_bubble.jld2", "θ′")
Nt = length(θ′_ts)

fig = Figure(size=(1200, 400), aspect=3)
ax = Axis(fig[1, 1])

n = Observable(1)
θ′n = @lift θ′_ts[$n]
heatmap!(ax, θ′n)

CairoMakie.record(fig, "thermal_bubble.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

mp4_html("thermal_bubble.mp4")

# ## Part II — free convection off a warm surface
#
# Same grid, same anelastic dynamics — but now the warmth enters through the
# *boundary* rather than the initial condition. Breeze computes the turbulent
# surface exchange with bulk aerodynamic formulae: a sensible heat flux and a vapor
# flux from a warm surface, plus a drag on the wind. A `gustiness` floor keeps the
# exchange finite where the resolved wind goes slack.

coefficient = 2e-3
gustiness = 1e-1
surface_temperature = θ₀ + 10
q_bottom_bc = BulkVaporFlux(; coefficient, gustiness, surface_temperature)
θ_bottom_bc = BulkSensibleHeatFlux(; coefficient, gustiness, surface_temperature)
u_bottom_bc = BulkDrag(; coefficient, gustiness)

ρq_bcs = FieldBoundaryConditions(bottom=q_bottom_bc)
ρθ_bcs = FieldBoundaryConditions(bottom=θ_bottom_bc)
ρu_bcs = FieldBoundaryConditions(bottom=u_bottom_bc)

boundary_conditions = (; ρq=ρq_bcs, ρθ=ρθ_bcs, ρu=ρu_bcs)
model = AtmosphereModel(grid; dynamics, advection, boundary_conditions)

# The lowest layer warms, goes unstable, and organizes into thermals that grow a
# convective boundary layer. We start from the neutral background plus a whisper of
# noise and a light mean wind to work the bulk formulae.

θᵢ(x, z) = θ₀ + 1e-2 * randn()
set!(model, θ=θᵢ, u=2)

simulation = Simulation(model; Δt=1, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    @info @sprintf("free convection | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θ₀
outputs = (; θ′, model.velocities...)

convection_ow = JLD2Writer(model, outputs;
                           filename = "free_convection.jld2",
                           schedule = TimeInterval(1minute),
                           overwrite_existing = true)

simulation.output_writers[:fields] = convection_ow

# and then we run the simulation,

run!(simulation)

# ### The movie
#
# We plot two fields side by side: the potential-temperature perturbation `θ′` and the
# vertical velocity `w`. Both are signed, so both use the diverging `:balance` colormap
# over a symmetric, frame-fixed range. The `w` panel makes the convective structure
# explicit — narrow, vigorous updrafts punching up between broader, gentler downdrafts —
# which is exactly what we will contrast against the terrain-forced flow in Part III.

θ′_ts = FieldTimeSeries("free_convection.jld2", "θ′")
w_ts  = FieldTimeSeries("free_convection.jld2", "w")
Nt = length(θ′_ts)
θ′max = maximum(abs, θ′_ts[Nt])
wmax  = maximum(abs, w_ts)

fig = Figure(size=(1200, 700))
axθ = Axis(fig[1, 1], title="potential temperature perturbation θ′ (K)")
axw = Axis(fig[2, 1], title="vertical velocity w (m s⁻¹)")

n = Observable(1)
θ′n = @lift θ′_ts[$n]
wn  = @lift w_ts[$n]
hmθ = heatmap!(axθ, θ′n, colormap = :balance, colorrange = (-θ′max, θ′max))
hmw = heatmap!(axw, wn,  colormap = :balance, colorrange = (-wmax, wmax))
Colorbar(fig[1, 2], hmθ)
Colorbar(fig[2, 2], hmw)

CairoMakie.record(fig, "free_convection.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

mp4_html("free_convection.mp4")

# ## Part III — the same convection, now over a hill
#
# Next we illustrate how to simulate flows over terrain using **terrain-following
# coordinates**, which warp the grid so that it conforms to the underlying topography.
# This warping renders the box-grid anelastic pressure solver invalid: we would either
# need a new anelastic solver built for the warped geometry, or we have to solve the
# equations a different way. We take the latter route and switch to a **fully
# compressible** formulation, which substeps the fast acoustic dynamics to accelerate
# the solve. So we put a hill in the way — the Witch of Agnesi,
#
# ```math
# h(x) = \frac{h_0}{1 + x^2 / a^2} ,
# ```
#
# the classic bell of mountain-wave theory. The grid's vertical coordinate follows
# the terrain near the ground and decays back to flat aloft (`TwoLevelDecay`).

Lx = 100kilometers
Lz = 20kilometers
h₀ = 300          # m, hill height
a = 5kilometers   # hill half-width

agnesi_hill(x) = h₀ / (1 + (x / a)^2)

r = collect(range(0, Lz, length = Nz + 1))
level_formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                                  small_scale_height = Lz / 8)

z = TerrainFollowingVerticalDiscretization(r, formulation=level_formulation)

agnesi_grid = RectilinearGrid(arch; z,
                              size = (Nx, Nz),
                              halo = (5, 5),
                              x = (-Lx/2, Lx/2),
                              topology = (Periodic, Flat, Bounded))

materialize_terrain!(agnesi_grid, agnesi_hill)

# Unlike the neutral first two parts, the hill needs a **stratified** background so
# that flow over the terrain can radiate gravity (mountain) waves. We give the
# reference state a constant buoyancy frequency, so the reference potential
# temperature grows exponentially with height:
#
# ```math
# θᵣ(z) = θ₀ \, e^{N^2 z / g} .
# ```

N² = 1e-4   # s⁻², background stratification
g  = 9.81   # m s⁻², gravitational acceleration
θᵣ(z) = θ₀ * exp(N² * z / g)

# A sponge near the lid absorbs the waves before they reflect. The split-explicit
# discretization substeps the fast acoustic and buoyancy terms while advection takes
# the long outer step, so the wizard can target a larger Courant number than in the
# anelastic parts. With uniform stratification, flow over the hill launches a
# vertically propagating mountain wave whose phase lines tilt upstream with height;
# the hill is kept low enough to stay in the clean, linear (non-breaking) regime.

sponge = UpperSponge(damping_rate = 0.1, depth = 2.5kilometers)
split_explicit_discretization = SplitExplicitTimeDiscretization(acoustic_cfl=0.5; sponge)

dynamics = CompressibleDynamics(split_explicit_discretization;
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(agnesi_grid; dynamics, advection, boundary_conditions)

# The printed summary now reports `CompressibleDynamics` on the terrain-following
# grid, with the same conservative prognostic variables. A compressible model carries
# density as a prognostic field, so we initialize it from the hydrostatic reference
# density, add a mean wind, and run in the next block.

θᵣ_field = Field{Nothing, Nothing, Center}(agnesi_grid)
set!(θᵣ_field, θᵣ)
θᵢ(x, z) = θᵣ(z) + 1e-2 * randn()
ρᵢ = model.dynamics.terrain_reference_density

set!(model, ρ=ρᵢ, θ=θᵢ, u=10)

simulation = Simulation(model; Δt=1, stop_time=2hour)
conjure_time_step_wizard!(simulation, cfl=1)

function progress(sim)
    @info @sprintf("hilly free convection | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θᵣ_field
outputs = (; θ′, model.velocities...)

hilly_ow = JLD2Writer(model, outputs;
                      filename = "hilly_free_convection.jld2",
                      schedule = TimeInterval(1minute),
                      overwrite_existing = true)

simulation.output_writers[:fields] = hilly_ow

run!(simulation)

# ### The movie
#
# The same two-panel layout as Part II — `θ′` above, `w` below, both `:balance` over a
# fixed symmetric range. Compare the `w` panel here against the flat case: the terrain
# organizes the vertical velocity, anchoring ascent over the hill rather than letting
# the plumes wander freely.

θ′_ts = FieldTimeSeries("hilly_free_convection.jld2", "θ′")
w_ts  = FieldTimeSeries("hilly_free_convection.jld2", "w")
Nt = length(θ′_ts)
θ′max = maximum(abs, θ′_ts[Nt])
wmax  = maximum(abs, w_ts)

fig = Figure(size=(1200, 700))
axθ = Axis(fig[1, 1], title="potential temperature perturbation θ′ (K)")
axw = Axis(fig[2, 1], title="vertical velocity w (m s⁻¹)")

n = Observable(1)
θ′n = @lift θ′_ts[$n]
wn  = @lift w_ts[$n]
hmθ = heatmap!(axθ, θ′n, colormap = :balance, colorrange = (-θ′max, θ′max))
hmw = heatmap!(axw, wn,  colormap = :balance, colorrange = (-wmax, wmax))
Colorbar(fig[1, 2], hmθ)
Colorbar(fig[2, 2], hmw)

CairoMakie.record(fig, "hilly_free_convection.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

mp4_html("hilly_free_convection.mp4")

# ## Part IV — cloud microphysics on the hilly flow
#
# The same hilly, terrain-following, compressible setup as Part III — same grid,
# same dynamics, same surface fluxes — now handed a one-moment warm-rain bulk
# scheme from [CloudMicrophysics.jl](https://clima.github.io/CloudMicrophysics.jl/dev/).
# Loading that package activates Breeze's bridge extension. Cloud water forms by
# warm-phase **saturation adjustment**, and where it exceeds the autoconversion
# threshold the clouds rain.

using CloudMicrophysics

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

cloud_formation = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
microphysics = OneMomentCloudMicrophysics(; cloud_formation)

model = AtmosphereModel(agnesi_grid; microphysics, dynamics, advection, boundary_conditions)

θᵢ(x, z) = θᵣ(z) + 1e-2 * randn()
ρᵢ = model.dynamics.terrain_reference_density

# A humid initial state primes the flow for condensation. We set the total water
# mixing ratio `qᵗ` directly — moist near the surface, drying aloft — rather than a
# relative humidity (the `ℋ` path does not yet compile on the GPU in this Breeze
# release; prescribing `qᵗ` is what Breeze's GPU cloud examples do). Orographic lifting
# and the surface vapor flux push parcels past saturation, forming cloud that rains
# once it exceeds the autoconversion threshold.

qᵗᵢ(x, z) = 0.012 * exp(-z / 3kilometers)   # kg/kg, moist near the surface, drying aloft
set!(model, ρ=ρᵢ, θ=θᵢ, u=10, qᵗ=qᵗᵢ)

simulation = Simulation(model; Δt=1, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=1)

# `qᶜˡ` is the specific cloud-liquid mixing ratio and `qʳ` the specific rain mixing
# ratio, read straight off the model's microphysical fields.

qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ  = model.microphysical_fields.qʳ

function progress(sim)
    @info @sprintf("hilly cloud physics | iter: %d, t: %s, Δt: %s, max|w|: %.2e m s⁻¹, max qᶜˡ: %.2e, max qʳ: %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w), maximum(qᶜˡ), maximum(qʳ))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

θ = liquid_ice_potential_temperature(model)
θ′ = θ - θᵣ_field
outputs = (; θ′, model.velocities..., qᶜˡ, qʳ)

cloudy_ow = JLD2Writer(model, outputs;
                       filename = "hilly_cloud_physics.jld2",
                       schedule = TimeInterval(1minute),
                       overwrite_existing = true)

simulation.output_writers[:fields] = cloudy_ow

run!(simulation)

# ### The movie
#
# Now there is something to *see*: the vertical velocity `w` lifting air over the hill,
# the cloud liquid `qᶜˡ` that condenses where parcels pass saturation, and the rain
# `qʳ` that falls out below. Clouds use the sequential `:dense` colormap and rain the
# `:amp` colormap, each over a fixed range; `w` keeps the diverging `:balance` map.

w_ts   = FieldTimeSeries("hilly_cloud_physics.jld2", "w")
qᶜˡ_ts = FieldTimeSeries("hilly_cloud_physics.jld2", "qᶜˡ")
qʳ_ts  = FieldTimeSeries("hilly_cloud_physics.jld2", "qʳ")
Nt = length(w_ts)

wmax   = maximum(abs, w_ts)
qᶜˡmax = max(1e-6, maximum(qᶜˡ_ts))
qʳmax  = max(1e-6, maximum(qʳ_ts))

fig = Figure(size=(1000, 900))
axw = Axis(fig[1, 1], title="vertical velocity w (m s⁻¹)")
axc = Axis(fig[2, 1], title="cloud liquid qᶜˡ (kg kg⁻¹)")
axr = Axis(fig[3, 1], title="rain qʳ (kg kg⁻¹)")

n = Observable(1)
wn   = @lift w_ts[$n]
qᶜˡn = @lift qᶜˡ_ts[$n]
qʳn  = @lift qʳ_ts[$n]

hmw = heatmap!(axw, wn,   colormap = :balance, colorrange = (-wmax, wmax))
hmc = heatmap!(axc, qᶜˡn, colormap = :dense,   colorrange = (0, qᶜˡmax))
hmr = heatmap!(axr, qʳn,  colormap = :amp,     colorrange = (0, qʳmax))

Colorbar(fig[1, 2], hmw)
Colorbar(fig[2, 2], hmc)
Colorbar(fig[3, 2], hmr)

CairoMakie.record(fig, "hilly_cloud_physics.mp4", 1:Nt; framerate = 24, compression = 28) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

mp4_html("hilly_cloud_physics.mp4")


