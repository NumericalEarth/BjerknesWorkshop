# # Hydrostatic ocean modeling: internal tides over a sill
#
# *Tuesday — one day in the high-latitude ocean, part 1: tidal energy at the sill.*
#
# Today's tutorials are stations along a single transect, from a Norwegian fjord sill
# out to the Barents Sea. We start where the coastal ocean meets the tide (part 1),
# follow the Atlantic water north through the eddy field that carries it (part 2),
# arrive at the freezing surface (part 3) and watch the pack it forms move and tear
# (part 4), pause at a calving glacier front to teach the model some physics it does
# not yet know (part 5), and finally assemble everything into a regional coupled
# ocean–sea ice simulation of the Barents Sea itself (part 6) — with an epilogue on
# what changes when one GPU stops being enough (part 7).
#
# Yesterday we built a nonhydrostatic solver from scratch; today we move to the model
# class that powers basin-scale and global ocean simulations: the
# `HydrostaticFreeSurfaceModel`. As a first experiment we simulate the generation of an
# **internal tide**: a barotropic tidal current sloshing back and forth over a sill
# radiates internal gravity waves into the stratified interior. This is a process close
# to home — the sills at the mouths of Norwegian fjords and the ridges of the
# Greenland–Scotland system convert barotropic tidal energy into internal waves whose
# breaking sustains a good part of the abyssal mixing
# ([Garrett and Kunze, 2007](https://doi.org/10.1146/annurev.fluid.39.050905.110227)).
#
# Along the way this tutorial introduces, one by one, the Oceananigans objects that every
# script of this week is composed of: grids, immersed boundaries, forcings, models,
# simulations, callbacks, output writers, and `FieldTimeSeries` for the analysis.
#
# ## The hydrostatic approximation, in two sentences
#
# When the aspect ratio of the motion is small (horizontal scales ≫ vertical scales), the
# vertical momentum equation collapses onto the hydrostatic balance
# ``\partial_z p = -\rho g``: the pressure becomes a *diagnostic* of the density field
# plus the free-surface elevation, and the expensive three-dimensional Poisson solve of
# yesterday is replaced by a two-dimensional problem for the free surface ``\eta``.
# Vertical velocity is no longer prognostic — it is diagnosed from continuity,
# ``w = -\int_{-H}^z \nabla_h \cdot \mathbf{u}_h \, dz'`` — which is the reason why the
# model carries no `w` equation and no nonhydrostatic pressure.
#
# Contrarily to the nonhydrostatic model, which marches acoustic-filtered dynamics with a
# single time step, the hydrostatic model must deal with surface gravity waves moving at
# ``\sqrt{gH} \approx 140`` m s⁻¹: a *split-explicit* scheme advances the fast barotropic
# mode with many cheap two-dimensional substeps inside each baroclinic step
# ([Shchepetkin and McWilliams, 2005](https://doi.org/10.1016/j.ocemod.2004.08.002)).
# We will set this up explicitly below.
#
# ## Grid and bathymetry
#
# We work in an ``x``–``z`` slice — 2000 km wide, 2 km deep — by declaring the
# ``y``-direction `Flat`, exactly like the 2D turbulence of yesterday but in the vertical
# plane. `Oceananigans.Units` allows to write dimensional numbers the way we would say
# them aloud:

using Oceananigans
using Oceananigans.Units
using Printf

Nx, Nz = 256, 128
H = 2kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (-1000kilometers, 1000kilometers),
                                  z = (-H, 0),
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))

# The sill is a Gaussian ridge 250 m tall and 20 km wide. Topography enters Oceananigans
# through the *immersed boundary* method: we keep the regular grid and mask the cells
# below the bottom height, which is the same approach the Barents Sea simulation of
# part 6 uses for coasts and submarine ridges:

h₀ = 250meters
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = -H + hill(x)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

# Let's look at the domain we built:

using CairoMakie

x = xnodes(grid, Center())
bottom_height = interior(grid.immersed_boundary.bottom_height, :, 1, 1)

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "z [m]",
          limits = ((-1000, 1000), (-H, 0)))
band!(ax, x / 1e3, bottom_height, 0 * x, color = :lightsteelblue)
save("internal_tide_domain.png", fig)
nothing #hide

# ![](internal_tide_domain.png)
#
# (The ridge looks dramatic only because the plot is stretched ~100× in the vertical —
# a permanent occupational hazard of oceanography.)
#
# ## Tidal forcing
#
# We force the zonal momentum equation with an oscillating body force at the M₂ frequency,
# mimicking the pressure gradient that the barotropic tide imposes. On an `FPlane` at
# 60°N — the latitude of the Norwegian Sea — the inertial frequency is close below the
# M₂ frequency:

coriolis = FPlane(latitude = 60)

T₂ = 12.421hours
ω₂ = 2π / T₂

@printf "ω₂ = %.4e s⁻¹, f = %.4e s⁻¹\n" ω₂ coriolis.f

# It is possible to notice that ``\omega_2 > f`` — barely. Internal waves propagate
# freely only for ``f < \omega < N``, so M₂ internal tides exist at 60°N but become
# evanescent poleward of the *critical latitude* ≈ 74.5°N: in the Arctic proper, the M₂
# internal tide cannot radiate, with well-known consequences for the weak abyssal mixing
# there — something worth watching happen, below.
#
# We choose a tidal excursion parameter ``\epsilon = U_2 / (\sigma_2 \, l)= 0.1``, in the regime of
# linear wave radiation, and from it the forcing amplitude that sustains the oscillating
# barotropic flow:

ϵ = 0.1
U₂ = ϵ * ω₂ * width
forcing_amplitude = U₂ * (ω₂^2 - coriolis.f^2) / ω₂

@inline tidal_forcing(x, z, t, p) = p.forcing_amplitude * sin(p.ω₂ * t)

u_forcing = Forcing(tidal_forcing, parameters = (; forcing_amplitude, ω₂))
nothing #hide

# `Forcing` wraps a plain Julia function; the `parameters` named tuple travels with it
# into the GPU kernel that evaluates the right-hand side — the same mechanism you would
# use for a relaxation term, a wave-maker, or a crude parameterization.
#
# ## The model
#
# Now the model itself. Buoyancy is treated as a single tracer ``b`` (`BuoyancyTracer`) —
# the Barents Sea simulation of part 6 will replace it with temperature, salinity and
# `SeawaterBuoyancy` with a full equation of state. We use WENO advection for momentum
# and tracers, and the split-explicit free surface discussed above, with 30 barotropic
# substeps per baroclinic step:

model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                    forcing = (; u = u_forcing))

# The initial state consists in a uniform stratification ``N^2 = 10^{-4}`` s⁻² (a
# buoyancy period of ~10 minutes, on the strong side of oceanic, which speeds up the
# wave dynamics for the tutorial) and a barotropic flow already at the tidal velocity, so
# the spin-up transient is mild:

N² = 1e-4
bᵢ(x, z) = N² * z
uᵢ(x, z) = U₂

set!(model, u = uᵢ, b = bᵢ)

# ## The simulation
#
# A `Simulation` manages the time-stepping loop: stop criteria, callbacks, output. With
# ``\Delta t = 5`` minutes we resolve the M₂ period with ~150 steps:

Δt = 5minutes
stop_time = 4days

simulation = Simulation(model; Δt, stop_time)

# A callback prints a progress message every 200 iterations. The maximum vertical
# velocity is our physical heartbeat — it grows as internal waves are radiated:

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %.2e m s⁻¹",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# ## Diagnostics and output
#
# Oceananigans' *abstract operations* allow to define diagnostics with near-mathematical
# notation, lazily — they are computed on the fly (on the GPU, when there is one) every
# time the output writer fires. We save the internal-tide velocity
# ``u' = u - \bar{u}`` (the deviation from the instantaneous domain mean, i.e. from the
# barotropic tide), the vertical velocity, and the stratification ``N^2 = \partial_z b``:

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))
u′ = u - U
N²_field = ∂z(b)

filename = "internal_tide.jld2"

simulation.output_writers[:fields] = JLD2Writer(model, (; u′, w, N² = N²_field);
                                                filename,
                                                schedule = TimeInterval(30minutes),
                                                overwrite_existing = true)

# And off we go:

run!(simulation)

# ## Analysis: watching the wave beams
#
# `FieldTimeSeries` loads the saved output lazily, with the grid and times attached.
# This decoupling of simulation and analysis — run once, explore the output as long as
# you like — is the workflow we suggest for everything bigger than a toy:

u′_timeseries = FieldTimeSeries(filename, "u′")
w_timeseries  = FieldTimeSeries(filename, "w")

times = u′_timeseries.times
nothing #hide

# We animate the internal-tide velocity and the vertical velocity side by side, using
# Makie's `Observable` machinery: the figure is built once around the observable frame
# index, and `record` advances it.

xu, _, zu = nodes(u′_timeseries[1])
xw, _, zw = nodes(w_timeseries[1])

n = Observable(1)

title = @lift @sprintf("internal tide at t = %1.2f days (%.1f M₂ periods)",
                       times[$n] / day, times[$n] / T₂)

u′ₙ = @lift u′_timeseries[$n]
wₙ  = @lift w_timeseries[$n]

u′max = maximum(abs, u′_timeseries[end])
wmax  = maximum(abs, w_timeseries[end])

axis_kwargs = (xlabel = "x [km]", ylabel = "z [m]",
               limits = ((-1000, 1000), (-H, 0)))

fig = Figure(size = (700, 500))
fig[1, :] = Label(fig, title, fontsize = 18, tellwidth = false)

ax_u = Axis(fig[2, 1]; title = "u′ — internal-tide velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, xu ./ 1e3, zu, u′ₙ, colorrange = (-u′max, u′max), colormap = :balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w — vertical velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, xw ./ 1e3, zw, wₙ, colorrange = (-wmax, wmax), colormap = :balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

CairoMakie.record(fig, "internal_tide.mp4", 1:length(times), framerate = 16) do i
    n[] = i
end
nothing #hide

# ![](internal_tide.mp4)
#
# The wave energy leaves the sill along straight *beams*. Their slope is no accident:
# internal waves at frequency ``\omega`` propagate at the angle ``\theta`` to the
# horizontal fixed by the dispersion relation,
#
# ```math
# \tan^2 \theta = \frac{\omega^2 - f^2}{N^2 - \omega^2},
# ```
#
# which for our parameters gives a beam slope of about 6 m per km — check it against the
# movie. Where the beams reflect from the surface and the bottom they interfere, and in
# the real ocean their shear is a preferred site for breaking and mixing.
#
# ## Things to try
#
# !!! tip "Supercritical topography"
#     The wave response changes regime with the ratio of the topographic slope to the
#     beam slope. Make the sill four times taller and twice as narrow, and look for waves
#     trapped on the flanks and stronger beams launched from the sill crest.
#
# !!! tip "The critical latitude"
#     Move the `FPlane` to `latitude = 76` and rerun. The forcing still oscillates, but
#     ``\omega_2 < f``: watch the radiated beams disappear, leaving a trapped, evanescent
#     response around the sill. This is the reason why tidal mixing maps of the Arctic
#     look so different from the rest of the world ocean.
#
# !!! tip "Nonlinearity"
#     Raise the excursion parameter to ``\epsilon = 1``. The flow now sweeps fluid
#     parcels over the entire sill within a tidal period: look for lee waves, hydraulic
#     jumps, and higher tidal harmonics in the ``w`` field (the beams at twice the M₂
#     beam slope).
#
# !!! tip "Onto the GPU"
#     Pass `GPU()` as the first argument of `RectilinearGrid` (everything else unchanged)
#     and double the resolution in both directions. On the cluster, compare wall times.
#     This one-argument switch is not a slogan — it is the actual porting effort.
