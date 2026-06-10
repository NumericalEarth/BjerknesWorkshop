# # Ocean free convection: a 2D introduction
#
# *The ocean analogue of the atmospheric intro — convection driven from the top.*
#
# This is the simplest possible large-eddy simulation of **oceanic free
# convection**: a two-dimensional (x–z) slab of ocean that is **cooled at the
# surface**. Heat loss to a cold atmosphere makes the near-surface water denser
# than the water just below it. That top-heavy density profile is unstable, so it
# overturns: dense **plumes sink**, drag the surface water down with them, and stir
# the upper ocean into a deepening **mixed layer**. This is open-ocean deep
# convection in miniature (Marshall & Schott 1999).
#
# It is the ocean twin of the day-4 atmospheric intro, and the two together make a
# clean pedagogical contrast in the *sign* of the forcing:
#
# ```text
# atmosphere:  heated from BELOW (warm surface)  →  buoyant plumes RISE
# ocean:       cooled from ABOVE (cold surface)  →  dense  plumes SINK
# ```
#
# Both systems convect; they simply turn the buoyancy source upside-down relative
# to each other. The atmosphere's unstable layer grows upward from a warm ground;
# the ocean's grows downward from a cold sky.
#
# This case is also the *uniform-surface* baseline for the lead-ocean case
# (`02_...`): there, the cooling, brine rejection, and waves are all confined to a
# narrow open-water lead, which localizes the plumes. Here the cooling is uniform
# across the whole surface, so the convection is statistically homogeneous in `x`
# and the only structure is the random plume field itself. Strip the heterogeneity
# away and you are left with this — the elementary convecting boundary layer.
#
# Coordinate orientation:
#
# ```text
# x = horizontal
# z = vertical, z = 0 at the surface, negative downward
# ```
#
# We solve the nonhydrostatic Boussinesq equations with Oceananigans, the full
# TEOS-10 nonlinear equation of state for seawater buoyancy, and prognostic
# temperature and salinity. The grid is kept deliberately small so the case runs
# quickly; refine `Nx`, `Nz` (and shrink `Δt`) for a production rendering.

using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Printf
using Random
using CairoMakie

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(2025)

config = RunConfig("06_intro_ocean_convection")
arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float32
const FT = Float32
nothing #hide

# ## Domain and grid
#
# A 512 m wide, 128 m deep two-dimensional slab at 2 m resolution: 256 × 128 cells.
# The third dimension is `Flat`, so this is a genuine 2D simulation — cheap enough
# to watch the plumes form interactively. The `x` topology is `Periodic` (an
# infinite homogeneous surface) and `z` is `Bounded`. A uniform vertical grid is
# perfectly adequate for the intro; the lead-ocean case shows how to stretch it.

const Lx = 512meters     # horizontal extent
const Lz = 128meters     # depth

const Nx = 256
const Nz = 128

grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                       x = (0, Lx), z = (-Lz, 0),
                       topology = (Periodic, Flat, Bounded))

@info "Grid" Δx = Lx / Nx Δz = Lz / Nz
memory_report(Nx, 1, Nz; FT, nfields = 5)

# ## Buoyancy from temperature and salinity
#
# Seawater buoyancy uses the full TEOS-10 nonlinear equation of state, with a
# reference density `ρₒ` typical of the surface ocean. Both temperature `T` and
# salinity `S` are carried as prognostic tracers; here only `T` is forced, and `S`
# is uniform, so buoyancy is set by the temperature field.

const ρₒ = FT(1026)   # kg m⁻³, reference surface density
equation_of_state = TEOS10EquationOfState(reference_density = ρₒ)
buoyancy = SeawaterBuoyancy(FT; equation_of_state)

# ## Surface cooling: the engine of the convection
#
# We remove `Q = 200 W m⁻²` of heat from the ocean surface — a moderate wintertime
# cooling. The model is forced in terms of a kinematic *temperature flux*
# `Jᵀ = Q / (ρₒ cᴾ)` in K m s⁻¹.
#
# !!! note "Sign convention (verified)"
#     A **positive** top temperature flux fluxes heat *upward*, out of the ocean —
#     i.e. it **cools** the surface. (This is the same convention noted in
#     Oceananigans' `ocean_wind_mixing_and_convection.jl`.) So a positive `Jᵀ` is
#     surface cooling, which is exactly the unstable forcing we want.
#
# At the bottom we hold a weak stable temperature gradient `dTdz`, which both seeds
# the initial stratification (below) and lets the mixed layer entrain into a
# realistic stable interior as it deepens.

const Q    = FT(200)    # W m⁻², surface heat loss (cooling)
const cᴾ   = FT(3991)   # J K⁻¹ kg⁻¹, seawater heat capacity
const Jᵀ   = Q / (ρₒ * cᴾ)   # K m s⁻¹, surface temperature flux (positive ⇒ cooling)
const dTdz = FT(0.01)   # K m⁻¹, stable interior temperature gradient

T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# ### A bit of wind stress
#
# Real ocean convection rarely happens in still air, so we add a light along-`x`
# wind stress `τx` (a kinematic momentum flux, m² s⁻²). It is weak compared to the
# convective forcing, so the plumes still dominate the picture, but it tilts them
# and adds a sheared near-surface current — a small step toward the wind+convection
# competition of the full lead-ocean case. Set `τx = 0` for pure free convection.

const τx = FT(-2e-5)   # m² s⁻², surface wind stress on u
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx))

# ## Model
#
# Nonhydrostatic Boussinesq with 9th-order WENO advection (which supplies the
# grid-scale dissipation), the `AnisotropicMinimumDissipation` LES closure, and an
# `FPlane` Coriolis force.

model = NonhydrostaticModel(grid;
                            advection = WENO(order = 9),
                            tracers = (:T, :S),
                            buoyancy,
                            coriolis = FPlane(f = 1e-4),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (; T = T_bcs, u = u_bcs))

# ## Initial conditions
#
# A warm, weakly stratified mixed layer: `T(z) = 20 + dTdz·z` (so it warms toward
# the surface) plus tiny random noise to seed the convective instability. Salinity
# is uniform at 35 g kg⁻¹. The flow starts at rest.

const T₀ = FT(20)     # °C, reference surface temperature
const S₀ = FT(35)     # g kg⁻¹, uniform salinity
const δT = FT(1e-4)   # K, initial noise amplitude

ϵ() = rand(FT) - FT(0.5)
Tᵢ(x, z) = T₀ + dTdz * z + δT * ϵ()

set!(model, T = Tᵢ, S = S₀)

# ## Simulation
#
# Adaptive time stepping at CFL 0.7, integrated for 4 hours of simulated time — long
# enough for the surface to cool, plumes to organize, and the mixed layer to deepen
# by tens of meters.

simulation = Simulation(model; Δt = 2.0, stop_time = 4hours)
conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 30.0)

wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), maximum(abs, sim.model.velocities.w))
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

# ## Outputs
#
# The 2D `w` and `T` fields for animation, plus a horizontally averaged `⟨w²⟩(z)`
# profile that diagnoses how deep the convective turbulence reaches.

u, v, w = model.velocities
T = model.tracers.T

slice_outputs = (; w, T)

profiles = (; w² = Average(w^2, dims = 1))

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = slice_name(config), schedule = TimeInterval(1minute), overwrite_existing = true)

simulation.output_writers[:profiles] = JLD2Writer(model, profiles;
    filename = output_name(config, "profiles"),
    schedule = TimeInterval(1minute), overwrite_existing = true)

# ## Go time
run!(simulation)

# ## Visualization
#
# A vertical velocity transect `w(x, z, t)` shows the cold dense plumes plunging
# from the surface; the temperature transect `T(x, z, t)` shows the cold anomalies
# they carry down and the mixed layer deepening over time. We build a movie and a
# final-frame figure.

if isfile(slice_name(config))
    w_ts = FieldTimeSeries(slice_name(config), "w")
    T_ts = FieldTimeSeries(slice_name(config), "T")
    times = w_ts.times
    Nt = length(times)

    xw, _, zw = nodes(w_ts)

    n = Observable(Nt)
    wn = @lift interior(w_ts[$n], :, 1, :)
    Tn = @lift interior(T_ts[$n], :, 1, :)
    title = @lift "Ocean free convection — t = " * prettytime(times[$n])

    fig = Figure(size = (1000, 700))
    Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
    axw = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "w (m s⁻¹)")
    axT = Axis(fig[2, 1], xlabel = "x (m)", ylabel = "z (m)", title = "T (°C)")

    wlim = max(1e-5, maximum(abs, interior(w_ts[Nt])))
    Tn_last = interior(T_ts[Nt], :, 1, :)
    Tlims = (minimum(Tn_last), maximum(Tn_last))

    hmw = heatmap!(axw, xw, zw, wn, colormap = :balance, colorrange = (-wlim, wlim))
    hmT = heatmap!(axT, xw, zw, Tn, colormap = :thermal, colorrange = Tlims)
    Colorbar(fig[1, 2], hmw)
    Colorbar(fig[2, 2], hmT)

    save(figure_name(config, "intro_ocean_convection_final"), fig)

    if Nt > 1
        record(fig, movie_name(config, "intro_ocean_convection"), 1:Nt; framerate = 12) do i
            n[] = i
        end
        @info "Wrote movie" movie_name(config, "intro_ocean_convection")
    end
end

@info "Intro ocean convection complete" run_stamp(config)...
nothing #hide

# ## References
#
# - **Marshall, J. & Schott, F. (1999).** Open-ocean convection: Observations,
#   theory, and models. *Rev. Geophys.* **37**, 1–64.
#   <https://doi.org/10.1029/98RG02739> — the standard review of oceanic deep
#   convection: surface buoyancy loss, plume dynamics, and mixed-layer deepening.
