# # Nærøyfjord ocean: hydrostatic + CATKE, wind-driven mixing
#
# *Coupled air–sea LES of a fjord — the ocean stability gate & the core science test.*
#
# The science question: a **stratified fjord** sits quiescent while the wind blows
# *cross-fjord* (blocked by the walls, little surface stress on the water); when the
# wind swings **down-fjord**, the along-axis stress drives surface currents and
# **wind-driven mixing** that erodes the stratification. This script tests exactly that
# with the ocean alone — a **hydrostatic** `HydrostaticFreeSurfaceModel` with **CATKE**
# vertical mixing (no 3-D LES; CATKE parameterizes the turbulence), on an
# **immersed-boundary** grid carved to the real fjord bathymetry (from `10a`), forced by
# a **prescribed, time-rotating wind stress**. Coupling the actual atmosphere is the
# next step (`10_naeroyfjord_coupled...`); here the wind is imposed so we can isolate and
# verify the ocean's stagnant→mixing response and its CFL/stability.
#
# Env knobs: FJORD_ARTIFACT (depth source), FJORD_NX/NY, FJORD_NZ, FJORD_STOPMIN,
# FJORD_TAUMAX (peak wind stress N/m²), RUN_TAG.

using Oceananigans
using NumericalEarth          # ocean_simulation (hydrostatic + CATKE)
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
using CUDA
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(2606)
arch = GPU()
gpu_report()
Oceananigans.defaults.FloatType = Float64   # ocean runs in Float64
FT = Float64

# ## Load the fjord bathymetry (depth ≥ 0) from 10a

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
topo_name = get(ENV, "FJORD_ARTIFACT", "naeroyfjord_topography_real.jld2")
topo_path = joinpath(repo_root, "thursday", "data", topo_name)
isfile(topo_path) || error("Missing $topo_path — run 10a (TOPO_SOURCE=real) first.")
topo = load(topo_path)
xt, yt = topo["x"], topo["y"]
depth_data = topo["depth"]          # ≥ 0 water depth, 0 over land
meta = topo["source_metadata"]
maxdepth = maximum(depth_data)
@info "Loaded fjord depth" topo_path size = size(depth_data) maxdepth source = meta.source
depth_fun = bilinear(depth_data, xt, yt)

# ## Grid: metric box + stretched z, carved to the bathymetry (immersed bottom)

Lx = meta.Lx
Ly = meta.Ly
Lz = ceil(maxdepth / 10) * 10 + 20    # a bit deeper than the deepest point

Nx = parse(Int, get(ENV, "FJORD_NX", "160"))
Ny = parse(Int, get(ENV, "FJORD_NY", "320"))
Nz = parse(Int, get(ENV, "FJORD_NZ", "48"))

## Surface-refined stretched vertical coordinate (fine ~1–2 m near the surface where the
## fresh lens and wind mixing live, coarsening with depth).
refinement = 12; stretching = 5
_h(k) = (k - 1) / Nz
_ζ(k) = 1 + (_h(k) - 1) / refinement
_Σ(k) = (1 - exp(-stretching * _h(k))) / (1 - exp(-stretching))
z_faces(k) = Lz * (_ζ(k) * _Σ(k) - 1)

underlying_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (7, 7, 7),
                                  x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
                                  topology = (Periodic, Periodic, Bounded))

## Immersed bottom at z = −depth(x, y); land columns (depth 0) become fully dry.
@inline bottom_height(x, y) = -depth_fun(x, y)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
memory_report(Nx, Ny, Nz; FT, nfields = 8)
@info "Fjord ocean grid" Lz Nz wet_fraction = round(sum(depth_data .> 0) / length(depth_data), digits = 3)

# ## Rotating wind stress (cross-fjord → down-fjord)
#
# Box axes: `+x` cross-fjord, `+y` along-fjord (toward the mouth; Gudvangen head at −y).
# Hold cross-fjord (`+x`), then rotate the stress onto the fjord axis (`−y`, down-fjord).
# Imposed as a kinematic momentum flux `τ/ρ₀` at the surface (only wet columns feel it).

const ρ₀     = 1020.0
const τmax   = parse(Float64, get(ENV, "FJORD_TAUMAX", "0.2"))   # N/m² (≈ 10–12 m/s wind)
const t_hold = 1hour
const t_rot  = 2hours
@inline function wind_angle(t)
    s = clamp((t - t_hold) / t_rot, 0, 1)
    return (-π/2) * (3s^2 - 2s^3)          # 0 (cross +x) → −π/2 (down-fjord −y)
end
@inline τx_top(x, y, t) = -τmax * cos(wind_angle(t)) / ρ₀   # flux of u-momentum
@inline τy_top(x, y, t) = -τmax * sin(wind_angle(t)) / ρ₀

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τx_top))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τy_top))

# ## Build the hydrostatic + CATKE ocean

## Pass an explicit Δt: the default `estimate_maximum_Δt(grid)` assumes a spherical
## grid (`grid.radius`) and errors on an immersed RectilinearGrid. We set our own Δt on
## the Simulation below anyway.
ocean = ocean_simulation(grid; model = :hydrostatic, Δt = 1.0,
                         coriolis = FPlane(latitude = 60.9),
                         boundary_conditions = (u = u_bcs, v = v_bcs))
model = ocean.model

# ## Initial condition: strongly stratified fjord (fresh surface lens over salty deep)
#
# Fjords are estuarine: a thin brackish lens from river runoff (here the Nærøydalselvi
# at Gudvangen) caps salty marine water. Buoyancy is salinity-dominated; the sharp
# near-surface pycnocline is what keeps the fjord "stagnant" until the down-fjord wind is
# strong enough to mix it.

S_surf = 18.0     # g/kg brackish lens
S_deep = 33.5     # g/kg marine deep water
h_lens = 6.0      # m, lens (halocline) depth
dS     = 4.0      # m, halocline sharpness
T₀     = 8.0      # °C, ~uniform (salinity dominates stratification)
@inline Sᵢ(x, y, z) = S_surf + (S_deep - S_surf) * (1 + tanh(-(z + h_lens) / dS)) / 2
@inline Tᵢ(x, y, z) = T₀ + 0.5 * (z + Lz) / Lz   # very weak warm-surface gradient
set!(model, S = Sᵢ, T = Tᵢ)

# ## Simulation + stability instrumentation

stop_minutes = parse(Float64, get(ENV, "FJORD_STOPMIN", "60"))
simulation = Simulation(model; Δt = 1.0, stop_time = stop_minutes * 60)
conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 30.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall[])
    u, v, w = sim.model.velocities
    S = sim.model.tracers.S
    Ssurf = view(S, :, :, sim.model.grid.Nz)
    θw = rad2deg(wind_angle(sim.model.clock.time))
    @info @sprintf("Iter %d  t %s  Δt %s  wall %s  max|u| %.3f  max|w| %.2e  Ssurf∈[%.1f,%.1f]  wind∠ %.0f°",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(abs, w), minimum(Ssurf), maximum(Ssurf), θw)
    wall[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# ## Outputs: surface fields + along-fjord transect (track the mixing as wind aligns)

u, v, w = model.velocities
S, T = model.tracers.S, model.tracers.T
imid = Nx ÷ 2 + 1
ksurf = Nz
run_tag = get(ENV, "RUN_TAG", "")
oname = "naeroyfjord_ocean" * (isempty(run_tag) ? "" : "_" * run_tag) * ".jld2"
outputs = (S_xy = view(S, :, :, ksurf), u_xy = view(u, :, :, ksurf), w_xy = view(w, :, :, ksurf),
           S_yz = view(S, imid, :, :), u_yz = view(u, imid, :, :), w_yz = view(w, imid, :, :))
simulation.output_writers[:slices] = JLD2Writer(model, outputs;
    filename = joinpath(repo_root, "thursday", "data", oname),
    schedule = TimeInterval(2minutes), overwrite_existing = true)

@info "Starting ocean run" stop_minutes τmax Lz
run!(simulation)
@info "Ocean run complete" iterations = iteration(simulation)
nothing #hide
