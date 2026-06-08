# # Sea ice II: dynamics, rheology, and the making of leads
#
# *Tuesday — one day in the high-latitude ocean, part 4: the pack in motion.*
#
# The previous tutorial grew ice in place; real pack ice *moves*. Wind and ocean
# currents push it around at a few percent of the wind speed, and where the drift
# converges the ice rafts and ridges, where it diverges it tears open into **leads** —
# the narrow ribbons of open water whose enormous winter heat fluxes you will meet again
# in Thursday's large-eddy simulations. The sea-ice momentum balance, per unit area,
# reads
#
# ```math
# m \frac{D\mathbf{u}}{Dt} =
#     - m f \hat{\mathbf{z}} \times \mathbf{u}
#     + \boldsymbol{\tau}_a + \boldsymbol{\tau}_o
#     + \nabla \cdot \boldsymbol{\sigma},
# ```
#
# with ``m = \rho_i h \aleph`` the ice mass per area, ``\boldsymbol{\tau}_a`` and
# ``\boldsymbol{\tau}_o`` the atmosphere and ocean stresses, and
# ``\nabla \cdot \boldsymbol{\sigma}`` the divergence of the *internal stress* — the
# term that distinguishes a continuum of colliding floes from a passive tracer. Without
# it the ice is in **free drift**; with it, the pack resists compression and shear, and
# the velocity field develops the quasi-discontinuous shear lines — *linear kinematic
# features* — that satellites observe all over the Arctic.
#
# ## Rheology: from VP to EVP, in three paragraphs
#
# What is ``\boldsymbol{\sigma}``? The standard answer since
# [Hibler (1979)](https://doi.org/10.1175/1520-0485(1979)009<0815:ADTSIM>2.0.CO;2) is
# the **viscous–plastic** (VP) law: stresses lie on or inside an elliptic yield curve
# whose size is set by the ice strength ``P^\star h \, e^{-C(1-\aleph)}`` — thick,
# compact ice is strong; thin or fragmented ice is weak. When the deformation tries to
# push the stress outside the ellipse the ice *yields* and deforms plastically at
# constant stress; inside the ellipse it creeps viscously. The exponential dependence on
# concentration means a few percent of open water is enough to soften the pack
# substantially — which is the reason why leads, once opened, like to stay open.
#
# Solving the VP law implicitly is expensive, so
# [Hunke and Dukowicz (1997)](https://doi.org/10.1175/1520-0485(1997)027<1849:AEVPMF>2.0.CO;2)
# added an artificial *elastic* term that turns the stress equation into a prognostic
# one, relaxed toward the VP solution through ~a hundred cheap explicit *substeps* per
# ice time step. This **elasto–visco–plastic** (EVP) scheme is embarrassingly parallel —
# the elasticity is a pseudo-time iteration device, not physics — and for this reason it
# is the workhorse of GPU-resident sea-ice models, ClimaSeaIce included. You will
# recognize the architecture: it is the same split-explicit trick the ocean's free
# surface uses, applied to a different stiff term.
#
# In this tutorial we reproduce a classic benchmark from
# [Mehlmann et al. (2021)](https://doi.org/10.1029/2021MS002523): pack ice in a
# 512 km box, sheared by an atmospheric anticyclone that travels diagonally across the
# domain over a cyclonic ocean eddy. The moving wind systematically deforms the ice and
# a web of leads and ridges emerges within two simulated days.
#
# ## Domain
#
# A bounded 512 km square at 4 km resolution — and notice, for the third time today,
# that the grid is the only place where the architecture appears; `GPU()` here is all it
# takes to move the whole experiment to a device:

using ClimaSeaIce
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CairoMakie
using Printf

architecture = CPU()

L = 512kilometers

grid = RectilinearGrid(architecture;
                       size = (128, 128),
                       x = (0, L),
                       y = (0, L),
                       halo = (7, 7),
                       topology = (Bounded, Bounded, Flat))

# No-slip walls for the ice velocity:

u_bcs = FieldBoundaryConditions(north = ValueBoundaryCondition(0),
                                south = ValueBoundaryCondition(0))

v_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(0),
                                east = ValueBoundaryCondition(0))
nothing #hide

# ## Ocean stress
#
# The ocean below carries a steady, gentle cyclonic eddy (1 cm s⁻¹). The ice–ocean
# stress is quadratic in the velocity *difference* between ice and ocean, which couples
# the ice momentum to itself; `SemiImplicitStress` treats that feedback stably without
# iterating:

ocean_speed = 0.01 # m s⁻¹

Uₒ = XFaceField(grid)
Vₒ = YFaceField(grid)

set!(Uₒ, (x, y) -> ocean_speed * (2y - L) / L)
set!(Vₒ, (x, y) -> ocean_speed * (L - 2x) / L)
fill_halo_regions!((Uₒ, Vₒ))

τₒ = SemiImplicitStress(uₑ = Uₒ, vₑ = Vₒ)

# ## A moving atmospheric anticyclone
#
# The wind field is an anticyclonic vortex whose center crosses the domain diagonally
# in 10 days, with velocities spiraling 18° across the isobars — the boundary-layer
# turning angle (in this idealization, the lone survivor of the whole atmospheric
# boundary layer physics):

atmosphere_speed = 30 # m s⁻¹, modifier of the maximum wind

@inline center(t) = 256kilometers + 51.2kilometers * t / day
@inline radius(x, y, t) = sqrt((x - center(t))^2 + (y - center(t))^2)
@inline shape(x, y, t) = exp(-radius(x, y, t) / 100kilometers) / 100

@inline ua(x, y, t) = -atmosphere_speed * shape(x, y, t) * ( cosd(72) * (x - center(t)) + sind(72) * (y - center(t))) / 1000
@inline va(x, y, t) = -atmosphere_speed * shape(x, y, t) * (-sind(72) * (x - center(t)) + cosd(72) * (y - center(t))) / 1000

Uₐ = XFaceField(grid)
Vₐ = YFaceField(grid)

set!(Uₐ, (x, y) -> ua(x, y, 0))
set!(Vₐ, (x, y) -> va(x, y, 0))
fill_halo_regions!((Uₐ, Vₐ))

# The wind enters the ice momentum equation as a bulk-formula stress,
# ``\boldsymbol{\tau}_a = \rho_a C_d |\mathbf{u}_a| \mathbf{u}_a`` — assembled here as a
# lazy abstract operation wrapped in a `Field`, recomputed whenever we update the wind:

τₐu = Field(-Uₐ * sqrt(Uₐ^2 + Vₐ^2) * 1.3 * 1.2e-3)
τₐv = Field(-Vₐ * sqrt(Uₐ^2 + Vₐ^2) * 1.3 * 1.2e-3)
compute!(τₐu)
compute!(τₐv)
nothing #hide

# ## The momentum equation and the model
#
# Now the dynamics: EVP rheology, 120 substeps per ice step, Coriolis, and the two
# boundary stresses. This object is the dynamical half of the sea-ice model, in the same
# way `SlabThermodynamics` was the thermodynamic half in the previous tutorial:

dynamics = SeaIceMomentumEquation(grid;
                                  top_momentum_stress = (u = τₐu, v = τₐv),
                                  bottom_momentum_stress = τₒ,
                                  coriolis = FPlane(f = 1e-4),
                                  rheology = ElastoViscoPlasticRheology(),
                                  solver = SplitExplicitSolver(substeps = 120))

model = SeaIceModel(grid;
                    dynamics,
                    ice_thermodynamics = nothing, # pure dynamics: no freezing or melting today
                    advection = WENO(order = 7),
                    boundary_conditions = (u = u_bcs, v = v_bcs))

# Initial conditions from the benchmark: full concentration, 30 cm mean thickness with
# small sinusoidal corrugations that seed the deformation features:

hᵢ(x, y) = 0.3 + 0.005 * (sin(60 * x / 1000kilometers) + sin(30 * y / 1000kilometers))

set!(model, h = hᵢ, ℵ = 1)

# ## Simulation
#
# Two days with a 2-minute time step. A callback re-evaluates the traveling wind at
# every iteration — the by-now-familiar pattern of *small mutable pieces around a fast
# static core*:

simulation = Simulation(model, Δt = 2minutes, stop_time = 2days)

function update_wind!(sim)
    t = sim.model.clock.time
    set!(Uₐ, (x, y) -> ua(x, y, t))
    set!(Vₐ, (x, y) -> va(x, y, t))
    fill_halo_regions!((Uₐ, Vₐ))
    compute!(τₐu)
    compute!(τₐv)
    return nothing
end

simulation.callbacks[:wind] = Callback(update_wind!, IterationInterval(1))

# For the output we add the ice-velocity divergence — positive divergence is a lead
# opening, convergence is rafting and ridging:

h = model.ice_thickness
ℵ = model.ice_concentration
u, v = model.velocities

δ = ∂x(u) + ∂y(v)

simulation.output_writers[:fields] = JLD2Writer(model, (; h, ℵ, δ);
                                                filename = "sea_ice_dynamics.jld2",
                                                schedule = TimeInterval(30minutes),
                                                overwrite_existing = true)

run!(simulation)

# ## Watching the pack deform

h_timeseries = FieldTimeSeries("sea_ice_dynamics.jld2", "h")
ℵ_timeseries = FieldTimeSeries("sea_ice_dynamics.jld2", "ℵ")
δ_timeseries = FieldTimeSeries("sea_ice_dynamics.jld2", "δ")

times = h_timeseries.times

n = Observable(1)

title = @lift @sprintf("pack ice under a traveling anticyclone — t = %.1f days", times[$n] / day)

hₙ = @lift interior(h_timeseries[$n], :, :, 1)
ℵₙ = @lift interior(ℵ_timeseries[$n], :, :, 1)
δₙ = @lift interior(δ_timeseries[$n], :, :, 1) .* day  # per day, more readable

fig = Figure(size = (1300, 480))
fig[1, :] = Label(fig, title, fontsize = 20, tellwidth = false)

ax_h = Axis(fig[2, 1], title = "thickness [m]", aspect = 1)
hm_h = heatmap!(ax_h, hₙ, colormap = :magma, colorrange = (0.23, 0.37))
Colorbar(fig[2, 2], hm_h)

ax_ℵ = Axis(fig[2, 3], title = "concentration", aspect = 1)
hm_ℵ = heatmap!(ax_ℵ, ℵₙ, colormap = Reverse(:deep), colorrange = (0.9, 1))
Colorbar(fig[2, 4], hm_ℵ)

ax_δ = Axis(fig[2, 5], title = "divergence [day⁻¹]", aspect = 1)
hm_δ = heatmap!(ax_δ, δₙ, colormap = :balance, colorrange = (-0.1, 0.1))
Colorbar(fig[2, 6], hm_δ)

CairoMakie.record(fig, "sea_ice_dynamics.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](sea_ice_dynamics.mp4)
#
# As the anticyclone sweeps through, the divergence field organizes into elongated,
# quasi-one-dimensional structures — the linear kinematic features. It is possible to
# notice that they are much sharper than anything in the smooth forcing: they are
# spontaneous localizations of the plastic flow, the continuum cousin of the fracture
# patterns in the real pack, and their statistics (intersection angles, spacing,
# scaling) are an active research subject and a discriminating test between rheologies.
#
# ## Things to try
#
# !!! tip "Free drift"
#     Replace the dynamics with
#     `SeaIceMomentumEquation(grid; top_momentum_stress = (u = τₐu, v = τₐv),
#     bottom_momentum_stress = τₒ, coriolis = FPlane(f = 1e-4),
#     rheology = ViscousRheology(ν = 1000))` — a plain viscous fluid, no plastic yield —
#     and rerun. The leads disappear and the deformation follows the smooth forcing:
#     the localization *is* the rheology.
#
# !!! tip "Substeps and stability"
#     Lower the EVP substeps to 30. Look for elastic noise in the divergence field.
#     The EVP substepping trades cheap iterations for accuracy of the VP limit — too
#     few, and the artificial elasticity survives into the solution.
#
# !!! tip "Weak ice"
#     Initialize with `ℵ = 0.9` instead of 1. With 10% open water the pack strength
#     drops by the exponential factor ``e^{-C(1-\aleph)}`` (``C = 20`` by default —
#     i.e. ~87% weaker): watch the drift speed up and the deformation features change
#     character.
#
# ## Further reading
#
# - [Hibler (1979)](https://doi.org/10.1175/1520-0485(1979)009<0815:ADTSIM>2.0.CO;2) — the VP rheology
# - [Hunke and Dukowicz (1997)](https://doi.org/10.1175/1520-0485(1997)027<1849:AEVPMF>2.0.CO;2) — EVP
# - [Mehlmann et al. (2021)](https://doi.org/10.1029/2021MS002523) — this benchmark, across models and grids
