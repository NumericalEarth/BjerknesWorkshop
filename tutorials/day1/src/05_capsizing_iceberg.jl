# # Implementing new physics: a capsizing iceberg
#
# *Teaching the model new physics from user space.*
#
# Most simulations use physics the models already know. This one does not: we
# teach Oceananigans about an object it has never heard of — a rigid, buoyant,
# *rotating* iceberg — using nothing but the public extension points
# (`Forcing`, `Callback`, abstract operations). The point here is less
# the iceberg itself than the demonstration that a new piece of coupled physics —
# a moving boundary with its own dynamics — consists in ~100 lines of ordinary Julia,
# without touching the model source. And because we write those lines with the GPU
# rules in mind, the same script runs unmodified on a laptop CPU and on a device.
#
# The phenomenon is worth the trouble. Icebergs calving from Greenland's outlet glaciers
# are frequently taller than they are wide, and a floating slab with width-to-height
# ratio below ``\epsilon \approx 0.75`` is gravitationally *unstable*: it capsizes,
# rotating by 90° and releasing a potential energy comparable to a small earthquake —
# the source of the teleseismic "glacial earthquakes"
# ([Tsai and Ekström, 2007](https://doi.org/10.1029/2006JF000596)) and of local
# tsunamis, studied in the laboratory by
# [Burton et al. (2012)](https://doi.org/10.1029/2011JF002055). Capsize events also
# stir and mix the fjord water column, with a possible role in bringing warm Atlantic
# water into contact with glacier fronts.
#
# ## The method: volume penalization
#
# We represent the berg cross-section (this is a 2D, ``x``–``z`` experiment) by a smooth
# mask ``\chi(x, z) \in [0, 1]`` and add to the momentum equations a *penalization*
# forcing that relaxes the fluid velocity, inside the mask, toward the local rigid-body
# velocity of the berg on a fast timescale ``\tau``
# ([Angot et al., 1999](https://doi.org/10.1007/s002110050401)):
#
# ```math
# \mathbf{F} = -\frac{\chi(x, z)}{\tau}
#     \left( \mathbf{u} - \mathbf{u}_{berg} \right), \qquad
# \mathbf{u}_{berg} = (U - \Omega \, \tilde z, \; W + \Omega \, \tilde x),
# ```
#
# with ``(\tilde x, \tilde z)`` the position relative to the berg center and
# ``(U, W, \Omega)`` its translational and angular velocities. The berg, in turn, obeys
# Newton: it feels gravity, the Archimedean buoyancy of its submerged area (computed by
# quadrature over material points, so the torque that drives the capsize comes out of
# the geometry automatically), and the *reaction* to the penalization force — the fluid
# pushing back. This two-way coupling is the same idea behind fluid–structure
# interaction methods in engineering, in its simplest possible clothing.
#
# The choice of ``\tau`` deserves a word, because it is a genuine trade-off and not a
# numerical afterthought. A very small ``\tau`` holds the masked fluid rigidly but acts
# as a powerful artificial damper: every bit of relative motion in the smoothed rim of
# the mask is destroyed within ``\tau``, and with it the rotational kinetic energy that
# should let the berg *overshoot and rock* around its new equilibrium. A large ``\tau``
# returns that energy to the physics — livelier wake, visible ring-down — at the price
# of a mushier body. A few times the time step is rigid; a few tens of time steps is
# lively; beyond that the berg turns to jelly.
#
# Honesty box: this is a pedagogical model, not a validated calving simulator. The
# nonhydrostatic model has a rigid lid (no tsunami wave — the fjord seiche is left as a
# natural extension with `SplitExplicitFreeSurface`), we neglect the added-mass correction
# to the berg inertia, and the penalized region drags its tracer content around. None of
# this changes the character of the capsize.
#
# ## Setup
#
# A 1.2 km × 300 m slice of quiescent, stratified fjord water. The model is a
# `NonhydrostaticModel` — at these scales the hydrostatic
# approximation would be exactly wrong. The architecture is chosen once, here; the
# float type comes from the grid, and every device-side constant below is converted
# through it (Apple GPUs, for instance, only speak `Float32`):

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Printf

arch = CPU()
## arch = GPU()                                    # NVIDIA
## using Metal; arch = GPU(Metal.MetalBackend())   # Apple (expect a long first-time
##                                                 # kernel compilation; FT must then
##                                                 # be Float32)

FT = Float64                              # Float32 on Metal, and faster everywhere

Lx = 1200
Lz = 300
Nx, Nz = 512, 128

grid = RectilinearGrid(arch, FT,
                       size = (Nx, Nz),
                       x = (-Lx/2, Lx/2),
                       z = (-Lz, 0),
                       halo = (6, 6),
                       topology = (Bounded, Flat, Bounded))

# ## The iceberg: host truth, device mirror
#
# The berg state lives in a `mutable struct` that the rigid-body integrator updates —
# on the *host*. Here is the GPU subtlety this tutorial exists to teach: a mutable
# struct is a pointer into CPU heap memory, and a GPU kernel cannot chase it — kernel
# arguments must be `isbits` values (plain numbers, immutable structs) or device
# arrays. So the forcing cannot simply close over `iceberg`. Instead we keep **two**
# representations:
#
# - the mutable struct: the truth, owned and evolved by the host;
# - a 6-element **device array** `state` holding `(x_c, z_c, θ, U, W, Ω)`: a snapshot
#   the kernels can read, refreshed by the callback with one `copyto!` per time step —
#   24 bytes of traffic, irrelevant next to anything else the model does.
#
# This host-truth/device-mirror pattern is not a workaround but *the* standard idiom
# for coupling scalar dynamics (a rigid body, a controller, an ice-sheet point model)
# to a GPU-resident fluid:

mutable struct Iceberg
    xc :: FT        # center position [m]
    zc :: FT
    θ  :: FT        # tilt angle [rad]
    U  :: FT        # translational velocity [m s⁻¹]
    W  :: FT
    Ω  :: FT        # angular velocity [s⁻¹]
end

const ρₒ = 1025  # water density [kg m⁻³]
const ρᵢ = 917   # ice density [kg m⁻³]
const g  = 9.81

width  = 70.0
height = 200.0
τ      = 2.0     # penalization timescale [s] — see the trade-off discussion above

# A width-to-height ratio of 0.35 — well below the 0.75 stability threshold. Floating at
# hydrostatic equilibrium the berg's center sits at ``z_c = (1/2 - \rho_i/\rho_o) H``
# below the waterline. We start from an *almost vertical* position, misaligned by only
# 2°: the instability needs nothing more than a seed, and watching it grow from nearly
# nothing is half the lesson:

zc_equilibrium = (1/2 - ρᵢ/ρₒ) * height

iceberg = Iceberg(0, zc_equilibrium, deg2rad(2), 0, 0, 0)

state = on_architecture(arch, FT[iceberg.xc, iceberg.zc, iceberg.θ,
                                 iceberg.U,  iceberg.W,  iceberg.Ω])

# The mask: rotate into the berg frame, then take a smoothed box indicator. The
# function depends only on its scalar arguments — no global state, no structs — which
# makes it equally at home in a GPU kernel, a broadcast, or a REPL test:

λ = 2 * (Lx / Nx)   # mask smoothing width [m]

@inline step_up(s, λ) = (1 + tanh(s / λ)) / 2

@inline function berg_mask(x, z, xc, zc, θ, width, height, λ)
    s, c = sincos(θ)
    ξ =  c * (x - xc) + s * (z - zc)   # berg-frame coordinates
    η = -s * (x - xc) + c * (z - zc)
    return step_up(width/2 - abs(ξ), λ) * step_up(height/2 - abs(η), λ)
end
nothing #hide

# ## The penalization forcing
#
# Two `Forcing`s relax `u` and `w` toward the rigid-body motion. The `parameters` are
# an immutable named tuple — scalars converted to the grid's float type, plus the
# device `state` array. When the model launches its kernels, Oceananigans `adapt`s the
# parameters for the device, and the `state` array arrives as a device pointer the
# kernel can legally read:

forcing_parameters = (; state, width = FT(width), height = FT(height), τ = FT(τ), λ = FT(λ))

@inline function u_penalization(x, z, t, u, p)
    @inbounds xc, zc, θ, U, Ω = p.state[1], p.state[2], p.state[3], p.state[4], p.state[6]
    χ = berg_mask(x, z, xc, zc, θ, p.width, p.height, p.λ)
    return -χ * (u - (U - Ω * (z - zc))) / p.τ
end

@inline function w_penalization(x, z, t, w, p)
    @inbounds xc, zc, θ, W, Ω = p.state[1], p.state[2], p.state[3], p.state[5], p.state[6]
    χ = berg_mask(x, z, xc, zc, θ, p.width, p.height, p.λ)
    return -χ * (w - (W + Ω * (x - xc))) / p.τ
end

u_forcing = Forcing(u_penalization, field_dependencies = :u, parameters = forcing_parameters)
w_forcing = Forcing(w_penalization, field_dependencies = :w, parameters = forcing_parameters)
nothing #hide

# The model: a strongly stratified fjord (``N^2 = 10^{-4}`` s⁻², a buoyancy period of
# ~10 minutes), so the capsize-stirred water mass slumps back as gravity currents and,
# on longer runs than ours, radiates internal waves:

# (Note `WENO(FT, ...)`: numerics objects carry their own float type, independently
# from the grid — forget it and a `Float64` sneaks into your `Float32` GPU kernels.)

model = NonhydrostaticModel(grid;
                            advection = WENO(FT, order = 9),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            forcing = (u = u_forcing, w = w_forcing))

N² = FT(1e-4)
set!(model, b = (x, z) -> N² * z)

# ## The rigid-body dynamics
#
# Now the berg's side of the bargain, advanced by a `Callback` every iteration:
#
# 1. **Fluid reaction.** The penalization exerts ``-\rho_o \chi (\mathbf{u} -
#    \mathbf{u}_{berg})/\tau`` per unit volume on the fluid; the berg receives the
#    opposite. We integrate it (and its torque) over the grid.
# 2. **Buoyancy and weight.** Material points tile the berg; those below the waterline
#    contribute ``\rho_o g \, dA`` of lift. The first moment of the submerged area gives
#    the buoyancy torque — the term whose sign makes tall bergs capsize.
# 3. **Symplectic Euler.** Velocities first, then positions with the new velocities —
#    the cheapest integrator that does not spuriously pump energy into the bobbing and
#    rolling oscillations. Then the new state is mirrored to the device.
#
# A subtlety worth making explicit, because it is the kind that silently ruins coupled
# models: *does the berg feel the hydrostatic pressure, and from where?* A Boussinesq
# solver subtracts the background hydrostatic pressure ``\rho_o g z`` once and for all —
# the pressure it computes is only the dynamic, perturbation part. But the dominant
# force on a floating body **is** the background hydrostatic pressure: its surface
# integral over the wetted area is Archimedes' force ``\rho_o g A_{sub}``, and its first
# moment is the righting (or, for our tall berg, capsizing) torque. Since the fluid
# solver cannot deliver it, step 2 restores it analytically. The *perturbation*
# pressure — the flow pushing back, including the rigid-lid pressure that stands in for
# the missing surface elevation — reaches the berg implicitly through the penalization
# reaction of step 1, which in the ``\tau \to 0`` limit converges to the surface
# integral of the perturbation stresses plus the inertia of the enclosed fluid. What
# remains neglected is only the hydrostatic pressure of the density *anomalies*
# (``\sim N^2 H``, four orders of magnitude below the effective gravity
# ``g(1 - \rho_i/\rho_o)``) and the change of the waterline geometry by the real free
# surface.
#
# The reaction integrals of step 1 must also obey the GPU rules: a scalar `for` loop
# over a device array would be somewhere between catastrophically slow and illegal.
# Broadcasts and reductions, on the other hand, compile to device kernels on any
# backend — so we phrase the integrals as exactly that, over coordinate arrays built
# once, on the right architecture, before the run:

u, v, w = model.velocities

xu, _, zu = nodes(u)
xw, _, zw = nodes(w)

Xᵘ = on_architecture(arch, repeat(collect(FT, xu), 1, length(zu)))
Zᵘ = on_architecture(arch, repeat(collect(FT, zu)', length(xu), 1))
Xʷ = on_architecture(arch, repeat(collect(FT, xw), 1, length(zw)))
Zʷ = on_architecture(arch, repeat(collect(FT, zw)', length(xw), 1))
nothing #hide

# The mass, inertia, and material points of the berg (host-side, like all the
# rigid-body bookkeeping):

m_berg = ρᵢ * width * height                      # mass per unit length [kg m⁻¹]
I_berg = m_berg * (width^2 + height^2) / 12       # moment of inertia [kg m]

material_ξ = range(-width/2,  width/2,  length = 20)
material_η = range(-height/2, height/2, length = 64)
dA = width * height / (length(material_ξ) * length(material_η))

iceberg_history = []

function advance_iceberg!(sim)
    berg = iceberg
    Δt   = sim.Δt
    dV   = (Lx / Nx) * (Lz / Nz)
    u, v, w = sim.model.velocities
    
    ## snapshot of the state in the grid's float type, for the device-side broadcasts
    xc, zc, θ      = FT(berg.xc), FT(berg.zc), FT(berg.θ)
    U, W, Ω        = FT(berg.U),  FT(berg.W),  FT(berg.Ω)
    Wᶠ, Hᶠ, λᶠ, τᶠ = FT(width),   FT(height),  FT(λ),     FT(τ)

    ## 1. reaction to the penalization: broadcast + reduce (GPU-legal, CPU-fast)
    uᵢ = interior(u, :, 1, :)
    Δu = @. berg_mask(Xᵘ, Zᵘ, xc, zc, θ, Wᶠ, Hᶠ, λᶠ) * (uᵢ - (U - Ω * (Zᵘ - zc)))
    Fx = ρₒ * dV / τ * sum(Δu)
    torque = -ρₒ * dV / τ * sum(Δu .* (Zᵘ .- zc))

    wᵢ = interior(w, :, 1, :)
    Δw = @. berg_mask(Xʷ, Zʷ, xc, zc, θ, Wᶠ, Hᶠ, λᶠ) * (wᵢ - (W + Ω * (Xʷ - xc)))
    Fz = ρₒ * dV / τ * sum(Δw)
    torque += ρₒ * dV / τ * sum(Δw .* (Xʷ .- xc))

    ## 2. buoyancy (quadrature over submerged material points) and weight
    s, c = sincos(berg.θ)
    submerged_area = 0.0
    submerged_moment = 0.0
    for ξ in material_ξ, η in material_η
        z = berg.zc + s * ξ + c * η
        if z < 0
            submerged_area += dA
            submerged_moment += (c * ξ - s * η) * dA   # x-offset from center
        end
    end

    Fz += g * (ρₒ * submerged_area - ρᵢ * width * height)
    torque += ρₒ * g * submerged_moment

    ## 3. symplectic Euler, then mirror the new state to the device
    berg.U  += Δt * Fx / m_berg
    berg.W  += Δt * Fz / m_berg
    berg.Ω  += Δt * torque / I_berg
    berg.xc += Δt * berg.U
    berg.zc += Δt * berg.W
    berg.θ  += Δt * berg.Ω

    copyto!(state, FT[berg.xc, berg.zc, berg.θ, berg.U, berg.W, berg.Ω])

    push!(iceberg_history, (time(sim), berg.xc, berg.zc, berg.θ))
    return nothing
end
nothing #hide

# That is the entire implementation. It is worth pausing on what we did *not* do: no
# subclassing, no registration with an interface, no recompilation of the package —
# and no CPU-only shortcuts: every line that touches model data is a kernel, a
# broadcast, or a reduction, so the architecture at the top of the script is a free
# choice.
#
# ## Run
#
# Three simulated minutes — from a 2° seed the berg leans imperceptibly for a while,
# then rolls over in a few tens of seconds:

simulation = Simulation(model, Δt = 0.1, stop_time = 3minutes)

add_callback!(simulation, advance_iceberg!, IterationInterval(1))

function progress(sim)
    @info @sprintf("t = %5.1f s, tilt = %6.1f°, Ω = %+.4f s⁻¹, max|u| = %.2f m s⁻¹",
                   time(sim), rad2deg(iceberg.θ), iceberg.Ω,
                   maximum(abs, sim.model.velocities.u))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(300))

ζ = ∂z(u) - ∂x(w)
b = model.tracers.b

simulation.output_writers[:fields] = JLD2Writer(model, (; ζ, b);
                                                filename = "capsizing_iceberg.jld2",
                                                schedule = TimeInterval(2),
                                                overwrite_existing = true)

run!(simulation)

# ## The capsize, frame by frame
#
# We animate the vorticity with the berg outline drawn on top from the recorded rigid
# body trajectory:

using CairoMakie

ζ_timeseries = FieldTimeSeries("capsizing_iceberg.jld2", "ζ")
times = ζ_timeseries.times

history_times = [datum[1] for datum in iceberg_history]

function berg_corners(xc, zc, θ)
    s, c = sincos(θ)
    corners = [(-width/2, -height/2), (width/2, -height/2),
               (width/2, height/2), (-width/2, height/2), (-width/2, -height/2)]
    return [Point2f(xc + c * ξ - s * η, zc + s * ξ + c * η) for (ξ, η) in corners]
end

xζ, _, zζ = nodes(ζ_timeseries[1])

n = Observable(1)

title = @lift begin
    i = searchsortedfirst(history_times, times[$n])
    i = min(i, length(iceberg_history))
    @sprintf("capsizing iceberg — t = %.0f s, tilt = %.0f°",
             times[$n], rad2deg(iceberg_history[i][4]))
end

ζₙ = @lift interior(ζ_timeseries[$n], :, 1, :)

outline = @lift begin
    i = searchsortedfirst(history_times, times[$n])
    i = min(i, length(iceberg_history))
    _, xc, zc, θ = iceberg_history[i]
    berg_corners(xc, zc, θ)
end

fig = Figure(size = (1000, 400))
ax = Axis(fig[1, 1]; title, xlabel = "x [m]", ylabel = "z [m]", aspect = DataAspect())
hm = heatmap!(ax, xζ, zζ, ζₙ, colormap = :balance, colorrange = (-0.2, 0.2))
lines!(ax, outline, color = :black, linewidth = 2)
Colorbar(fig[1, 2], hm, label = "vorticity [s⁻¹]")
ylims!(ax, -Lz, 50)

CairoMakie.record(fig, "capsizing_iceberg.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](capsizing_iceberg.mp4)
#
# The tilt grows slowly at first — the instability is exponential from a small seed —
# then the berg rolls over in a few tens of seconds, shedding a vortex dipole from each
# corner, overshoots past horizontal, and rocks back and forth around its new
# equilibrium with the *long* side at the waterline, now a stable ``\epsilon > 1``
# configuration. It is possible to notice that the capsize also *propels* the berg
# horizontally — momentum conservation working on the asymmetric roll, an effect well
# documented in the laboratory — while the stirred, displaced water slumps back through
# the stratification (look at the buoyancy field in the saved output).
#
# And the tilt history, the quantitative summary of the event:

θ_history = [rad2deg(datum[4]) for datum in iceberg_history]

fig = Figure(size = (700, 350))
ax = Axis(fig[1, 1], xlabel = "time [s]", ylabel = "tilt [°]")
lines!(ax, history_times, θ_history, linewidth = 3)
hlines!(ax, [90], linestyle = :dash, color = :gray)
save("iceberg_tilt.png", fig)
nothing #hide

# ![](iceberg_tilt.png)
#
