# # The ocean across many GPUs: distributed hydrostatic modeling
#
# *Tuesday — one day in the high-latitude ocean, part 7 (epilogue): scaling up.*
#
# The eddying channel of this morning lived happily on one device, and even the global
# one-degree configuration fits on a single modern GPU. But push the resolution to the
# kilometer scale — where the Nordic Seas eddies, the overflows, and the shelf exchange
# actually live — and a single device runs out of memory long before it runs out of
# ambition. A global simulation at 1 km needs ``\mathcal{O}(10^{10})`` grid points and
# terabytes of state: this is the regime of the
# [GB-25 project](https://github.com/NumericalEarth/GB-25), which ran
# global Oceananigans configurations on thousands of GPUs — and it all starts from the
# two objects of this tutorial, `Distributed` and `Partition`.
#
# The good news, anticipated in the Monday distributed tutorial
# (`day1/02_distributed_nonhydrostatic.jl`, where `Distributed`, halos, and launching
# are introduced from scratch): hydrostatic models are the *best possible* candidates
# for distribution. Contrarily to the nonhydrostatic solver with its all-to-all FFT
# transposes, the hydrostatic time step is dominated by pointwise and stencil work plus
# the split-explicit barotropic substeps, whose communication is nearest-neighbor halo
# exchange only. This is the deep reason why global ocean models have scaled on
# distributed machines since the 1990s, and why GB-25's weak scaling stays close to
# ideal out to thousands of devices.
#
# ## The driver: this morning's channel, sixteen times bigger
#
# We reuse the baroclinic-instability channel verbatim — physics, drag, initial front —
# at four times the horizontal resolution in each direction: 2.6 km grid spacing, well
# into the eddy-*resolving* regime, ~25× the points of the morning run. The only
# distributed ingredients are the architecture (one GPU per rank, slabs in ``y``) and
# rank-suffixed output filenames. Everything else is untouched — compare line by line
# with `02_baroclinic_instability.jl`:

channel_driver = raw"""
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units
using Printf
using Random
using MPI
MPI.Init()

Nh = parse(Int, get(ENV, "CHANNEL_NH", "384"))      # global horizontal points
Nz = parse(Int, get(ENV, "CHANNEL_NZ", "32"))
stop_time = parse(Float64, get(ENV, "CHANNEL_DAYS", "60")) * days

child = get(ENV, "CHANNEL_ARCH", "GPU") == "GPU" ? GPU() : CPU()
arch = Distributed(child, partition = Partition(y = Equal()))
rank = arch.local_rank

Random.seed!(1234 + rank)   # different noise per rank is fine — it is noise

grid = RectilinearGrid(arch,
                       size = (Nh, Nh, Nz),
                       x = (0, 1000kilometers),
                       y = (-500kilometers, 500kilometers),
                       z = (-1kilometer, 0),
                       topology = (Periodic, Bounded, Bounded))

Cd = 3e-3
@inline u_drag(x, y, t, u, v, p) = -p.Cd * sqrt(u^2 + v^2) * u
@inline v_drag(x, y, t, u, v, p) = -p.Cd * sqrt(u^2 + v^2) * v
u_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(u_drag, field_dependencies = (:u, :v), parameters = (; Cd)))
v_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(v_drag, field_dependencies = (:u, :v), parameters = (; Cd)))

# The split-explicit free surface is mandatory here: its barotropic substeps
# communicate only with nearest neighbors, while the default implicit (FFT) solver
# would need global transposes that distributed grids do not support.
model = HydrostaticFreeSurfaceModel(grid;
                                    coriolis = BetaPlane(latitude = 70),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps = 30),
                                    boundary_conditions = (u = u_bcs, v = v_bcs))

N² = 1e-5
M² = 1e-7
Δy = 100kilometers
Δb = Δy * M²
ramp(y, Δy) = min(max(0, y / Δy + 1/2), 1)
bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy) + 1e-2 * Δb * randn()
set!(model, b = bᵢ)

simulation = Simulation(model; Δt = 5minutes, stop_time)
conjure_time_step_wizard!(simulation, IterationInterval(20), cfl = 0.2, max_Δt = 10minutes)

wall_clock = Ref(time_ns())
function progress(sim)
    msg = @sprintf("[%05.2f%%] iter: %d, t: %s, wall: %s, max|u|: %.2f m s⁻¹",
                   100 * time(sim) / sim.stop_time, iteration(sim), prettytime(sim),
                   prettytime(1e-9 * (time_ns() - wall_clock[])),
                   maximum(abs, sim.model.velocities.u))
    @root @info msg
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

u, v, w = model.velocities
ζ = ∂x(v) - ∂y(u)

# Oceananigans suffixes the filename per rank (channel_rank0.jld2, ...) and knows how
# to reassemble the global field at load time.
simulation.output_writers[:surface] =
    JLD2Writer(model, (; ζ);
               filename = "channel.jld2",
               indices = (:, :, Nz),
               schedule = TimeInterval(12hours),
               overwrite_existing = true)

run!(simulation)
@root @info "Done."
"""

write("distributed_channel.jl", channel_driver)
nothing #hide

# Launch on four GPUs,
#
# ```bash
# mpiexec -n 4 julia --project distributed_channel.jl
# ```
#
# or smoke-test the identical script right now, on two laptop CPU ranks at the morning's
# resolution:
#
# ```bash
# CHANNEL_ARCH=CPU CHANNEL_NH=96 CHANNEL_DAYS=5 mpiexec -n 2 julia --project distributed_channel.jl
# ```
#
# Reassembling the global field: for full-3D output `FieldTimeSeries("channel.jld2",
# "ζ")` combines the rank files automatically; for surface slices like ours we glue the
# ``y``-slabs by hand — one `hcat`:
#
# ```julia
# using Oceananigans, CairoMakie
#
# nranks = 4
# ζ_ranks = [FieldTimeSeries("channel_rank$(r).jld2", "ζ"; combine = false)
#            for r in 0:nranks-1]
# times = ζ_ranks[1].times
#
# ζ_global(n) = hcat((interior(ζ_ranks[r][n], :, :, 1) for r in 1:nranks)...)
#
# n = Observable(length(times))
# fig, ax, hm = heatmap(@lift(ζ_global($n) ./ 1.37e-4),
#                       colormap = :balance, colorrange = (-0.5, 0.5),
#                       axis = (aspect = 1, title = "ζ/f, eddy-resolving channel"))
# record(fig, "distributed_channel.mp4", 1:length(times)) do i
#     n[] = i
# end
# ```
#
# Count the rank-aware lines in the whole workflow: the architecture and one `hcat`.
# The model neither knows nor cares that its neighbor cells live on another GPU — the
# `FullyConnected` halo machinery delivers them on schedule.
#
# ## Distributing the realistic global ocean
#
# The same two ingredients carry directly to the realistic configurations of part 6 —
# regional or global. Conceptually, a quarter-degree global setup reads:
#
# ```julia
# arch = Distributed(GPU(), partition = Partition(y = Equal()))
#
# grid = TripolarGrid(arch; size = (1440, 600, 50), halo = (5, 5, 4), z)
# bottom_height = regrid_bathymetry(grid; minimum_depth = 10, major_basins = 2)
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
#
# ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface)
# set!(ocean.model, MetadataSet((:temperature, :salinity); dataset = ECCO4Monthly(), date))
# ```
#
# — `regrid_bathymetry` and the dataset machinery partition their products onto the
# local grids for you. Practical advice for this route, in rough order of pain saved:
#
# 1. **Pre-stage the data.** Run a one-rank CPU script once to download and cache
#    ECCO/JRA55/ETOPO before the parallel job — N ranks discovering an empty cache
#    simultaneously is a classic first-day-on-the-cluster experience.
# 2. **Partition in `y` only** for tripolar global grids (the default): the tripolar
#    north seam couples ranks along `x`, and `y`-slabs keep its communication local.
# 3. **Match substeps to halo width.** The split-explicit barotropic solver exchanges
#    halos once per substep batch; the default halo of 5 with ~70 substeps is tuned —
#    if you change one, revisit the other.
# 4. **Measure before scaling.** The progress callback's wall-time-per-interval *is*
#    your profiler; double the ranks, halve the points per rank, and watch what the
#    time per model day does. When it stops halving, you have found your machine's
#    surface-to-volume limit.
#
# For the full-dress version — sharding, parallel IO, fault tolerance at thousands of
# ranks — read the GB-25 repository; it is the same `Distributed` object all the way
# up.
#
# ## Things to try
#
# !!! tip "Strong scaling on the workshop cluster"
#     Run the channel at `CHANNEL_NH=384` on 1, 2, and 4 GPUs and tabulate the wall
#     time per model day. Compute the parallel efficiency, and explain (with rule 1 of
#     the Monday tutorial) why 4 GPUs at this size disappoint.
#
# !!! tip "The eddy-resolving dividend"
#     Compare the equilibrated surface vorticity of the distributed `CHANNEL_NH=384`
#     run against this morning's 96² single-device run. Where does the extra resolution
#     spend itself? (Look at the filament widths and the cyclone–anticyclone
#     asymmetry.)
#
# !!! tip "Partition geometry"
#     The channel is periodic in `x` and walled in `y`. Re-partition with
#     `Partition(x = Equal())` and measure. Does communicating across the periodic
#     direction cost more, less, or the same as across the walls?
