# # Nærøyfjord atmosphere stability harness (terrain-following LES)
#
# *Coupled air–sea LES of a fjord — the atmosphere stability gate.*
#
# Before coupling an ocean, we must confirm the **terrain-following compressible LES is
# stable over the Nærøyfjord's near-vertical ~1200 m walls** — the steepest terrain in
# the workshop. This harness runs the atmosphere alone over the cached fjord terrain
# (from `10a`), driven by a **geostrophic wind** (Coriolis on a background wind that the
# terrain blocks/funnels), with a NaN checker and a short stop time. We start with a
# *static cross-fjord* geostrophic wind (the hardest case for the solver: flow into the
# wall) and, once stable, switch on the rotating-wind schedule (`WIND_ROTATE=1`) that
# sweeps the wind onto the fjord axis (the science run).
#
# Env knobs (for the autonomous stability sweep):
#   FJORD_NX, FJORD_NY, FJORD_NZTOP   grid (defaults: a fast smoke grid)
#   FJORD_STOPMIN                     stop time in minutes (default 20)
#   FJORD_UGEO                        geostrophic wind speed m/s (default 8)
#   FJORD_SMOOTH_INFO                 (artifact already smoothed in 10a)
#   WIND_ROTATE=1                     enable the rotating-wind schedule

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode
using CUDA
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(2606)

_t0 = Ref(time_ns())
checkpoint(msg) = (@info @sprintf("⏱ %-26s %8.1f s", msg, 1e-9 * (time_ns() - _t0[])); flush(stderr))

arch = GPU()
gpu_report()
Oceananigans.defaults.FloatType = Float32   # compressible terrain solver's native precision
FT = Float32

# ## Load the cached fjord terrain (from 10a)

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
topo_name = get(ENV, "FJORD_ARTIFACT", "naeroyfjord_topography.jld2")
topo_path = joinpath(repo_root, "thursday", "data", topo_name)
isfile(topo_path) || error("Missing $topo_path — run 10a_prepare_naeroyfjord_topography.jl first.")
topo = load(topo_path)
xt, yt = topo["x"], topo["y"]
h_data = topo["h_terrain"]
meta = topo["source_metadata"]
@info "Loaded fjord terrain" topo_path size_h = size(h_data) source = meta.source max_wall = maximum(h_data)
h_fun = bilinear(h_data, xt, yt)

# ## Grid and terrain-following vertical coordinate
#
# Box matches the artifact extent (x cross-fjord, y along-fjord; Gudvangen head at −y).
# Fine near-surface spacing through the fjord-wall depth, coarsening aloft under a sponge.

Lx = meta.Lx
Ly = meta.Ly
Lz = 8kilometers

Nx = parse(Int, get(ENV, "FJORD_NX", "80"))
Ny = parse(Int, get(ENV, "FJORD_NY", "160"))

z_faces = PiecewiseStretchedDiscretization(z = [0, 2000, 4000, Int(Lz)], Δz = [100, 150, 350, 700])
Nz = length(z_faces) - 1

z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz / 2, small_scale_height = Lz / 8))

memory_report(Nx, Ny, Nz; FT, nfields = 6)

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_coord,
                       topology = (Periodic, Periodic, Bounded))

checkpoint("start")
materialize_terrain!(grid, (x, y) -> h_fun(x, y))
checkpoint("terrain materialized")

# ## Compressible dynamics with acoustic substepping + upper sponge

θ₀ = 283          # K, summer marine airmass (non-polar)
p₀ = 1e5
N² = 1.0e-4       # s⁻², stable stratification (N ≈ 0.01 s⁻¹)
g  = 9.81
potential_temperature_profile(z) = θ₀ * exp(N² * z / g)

sponge_depth = 3kilometers
time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = sponge_depth))

dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)
checkpoint("dynamics built")

# ## Geostrophic wind forcing
#
# `geostrophic_forcings(uᵍ, vᵍ)` builds the Coriolis-balanced large-scale wind; the
# terrain then blocks (cross-fjord) or funnels (along-fjord) it. Box axes: `+x`
# cross-fjord, `+y` along-fjord (toward the mouth; Gudvangen head at −y). A static
# **cross-fjord** wind `(U, 0)` is the hardest stability case (flow straight into the
# wall). The rotating schedule (below) sweeps it onto `−y` (down-fjord).

U_geo = parse(Float64, get(ENV, "FJORD_UGEO", "8"))
const ROTATE = get(ENV, "WIND_ROTATE", "0") == "1"

uᵍ(z) = U_geo     # cross-fjord component
vᵍ(z) = 0.0       # along-fjord component (rotated in by the callback when ROTATE)
geo = geostrophic_forcings(uᵍ, vᵍ)

atmosphere = atmosphere_simulation(grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = 60.9),
                                   forcing = geo)
checkpoint("atmosphere built")

# ## Initial condition: stratified, balanced with the cross-fjord geostrophic wind

δθ = 0.2; zδ = 400; qᵗ₀ = 3e-3
ϵ() = rand() - 0.5
uᵢ(x, y, z) = U_geo
θᵢ(x, y, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀

let N = sqrt(N²)
    @info @sprintf("Nondimensional mountain height M = N h / U = %.2f (N=%.4f, h=%.0f, U=%.1f)",
                   N * maximum(h_data) / U_geo, N, maximum(h_data), U_geo)
end

set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = uᵢ, v = 0, w = 0, qᵗ = qᵢ, enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)
checkpoint("set! done")

# ## Rotating-wind schedule (the science forcing)
#
# Hold cross-fjord, then smoothly rotate the geostrophic wind onto the fjord axis
# (`+x → −y`, i.e. down-fjord toward Gudvangen). We mutate the geostrophic-velocity
# fields the forcing holds. Enabled by `WIND_ROTATE=1`.

const t_hold = 30minutes
const t_rot  = 60minutes
@inline function wind_angle(t)
    s = clamp((t - t_hold) / t_rot, 0, 1)
    smooth = 3s^2 - 2s^3            # smoothstep
    return (-π/2) * smooth          # 0 (cross, +x) → −π/2 (along, −y)
end

## After model construction the forcing is re-keyed to prognostic names and each entry
## is a `SpecificForcing` wrapping the `GeostrophicForcing`. So the live geostrophic
## field is `model.forcing.ρu.forcing.geostrophic_velocity` (which holds vᵍ; ρv holds uᵍ).
## We `set!` those uniform fields each step to rotate the large-scale wind.
function rotate_wind!(sim)
    θ = wind_angle(sim.model.clock.time)
    ug = U_geo * cos(θ); vg = U_geo * sin(θ)
    set!(sim.model.forcing.ρu.forcing.geostrophic_velocity, vg)
    set!(sim.model.forcing.ρv.forcing.geostrophic_velocity, ug)
    return nothing
end

# ## Simulation + stability instrumentation

stop_minutes = parse(Float64, get(ENV, "FJORD_STOPMIN", "20"))
cfl = parse(Float64, get(ENV, "FJORD_CFL", "0.7"))
simulation = Simulation(atmosphere.model; Δt = 0.5, stop_time = stop_minutes * 60)
conjure_time_step_wizard!(simulation, cfl = cfl, max_Δt = 3.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
ROTATE && add_callback!(simulation, rotate_wind!, IterationInterval(5))

wall = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall[])
    u, v, w = sim.model.velocities
    θw = ROTATE ? rad2deg(wind_angle(sim.model.clock.time)) : 0.0
    @info @sprintf("Iter %d  t %s  Δt %s  wall %s  max|u| %.2f  max|v| %.2f  max|w| %.2e  wind∠ %.0f°",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), θw)
    wall[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(20))

# ## Light outputs (near-surface wind + an along-fjord transect) for later inspection

u, v, w = atmosphere.model.velocities
imid = Nx ÷ 2 + 1
k_surf = 2
outputs = (u_xy = view(u, :, :, k_surf), v_xy = view(v, :, :, k_surf), w_xy = view(w, :, :, k_surf),
           w_xz = view(w, imid, :, :))
run_tag = get(ENV, "RUN_TAG", "")
slices_name = "naeroyfjord_atmos_slices" * (isempty(run_tag) ? "" : "_" * run_tag) * ".jld2"
simulation.output_writers[:slices] = JLD2Writer(atmosphere.model, outputs;
    filename = joinpath(repo_root, "thursday", "data", slices_name),
    schedule = TimeInterval(30seconds), overwrite_existing = true)

checkpoint("starting run!")
run!(simulation)
@info "Atmosphere stability run complete" iterations = iteration(simulation)
checkpoint("done")
nothing #hide
