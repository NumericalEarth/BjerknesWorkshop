# # Scaling up: polar deep convection on many GPUs
#
# *Monday afternoon — Introduction to GPU-based modelling, part 2.*
#
# The turbulence solver of the previous tutorial fit comfortably on one GPU. Production
# science often does not: a large-eddy simulation with ``10^9``–``10^{10}`` grid points
# needs more memory than any single device owns, and more throughput than any single
# device delivers. The answer is the same one ocean modeling has used for thirty years —
# **domain decomposition over MPI** — except that each rank now drives a GPU instead of
# a CPU core.
#
# In this tutorial we distribute Oceananigans' `NonhydrostaticModel` over several GPUs
# and simulate **wintertime open-ocean deep convection** at a resolution worthy of the
# phenomenon: a rotating, stratified water column losing ~800 W m⁻² to a polar
# atmosphere, the process that ventilates the deep Nordic Seas and, through them,
# feeds the overflows across the Greenland–Scotland ridge.
#
# ## Domain decomposition in one figure
#
# Each MPI rank owns a rectangular slab of the global grid, plus a rind of *halo*
# points mirroring its neighbors' edge values:
#
# ```text
#         global grid (Nx × Ny)              rank 0        rank 1        rank 2
#   ┌──────────────────────────────┐      ┌─────────┐╔═╗┌─────────┐╔═╗┌─────────┐
#   │                              │      │         │║h║│         │║h║│         │
#   │     partitioned in y    ───▶ │      │ local   │║a║│ local   │║a║│ local   │
#   │                              │      │ grid    │║l║│ grid    │║l║│ grid    │
#   └──────────────────────────────┘      └─────────┘╚═╝└─────────┘╚═╝└─────────┘
# ```
#
# After every operation that needs neighbor values (advection stencils, the implicit
# vertical solve, the FFT pressure solver), the ranks exchange halos over MPI — ideally
# through GPU-aware transports (NVLink within a node, RDMA between nodes) so the data
# never visits host memory. All of this bookkeeping lives inside the `Distributed`
# architecture; the model code — *your* code — is identical to the single-GPU version.
# The nonhydrostatic pressure solve deserves a special mention: a distributed FFT
# requires transposing the whole field between ranks twice per solve, and it is the
# communication-heaviest part of the time step — one of the reasons why hydrostatic
# models (tomorrow) scale even more graciously than nonhydrostatic ones.
#
# ## Hello, `Distributed`
#
# An MPI program is one program executed simultaneously by N ranks; what each rank does
# consists in the same script, parameterized by its rank. The snippet below builds a
# distributed grid; we write it to a file and launch it on two CPU ranks right here, in
# the tutorial — distributed computing demystifies quickly once you can poke it on a
# laptop:

architecture_demo = """
using Oceananigans
using Oceananigans.DistributedComputations
using MPI
MPI.Init()

arch = Distributed()   # defaults: CPU child architecture, ranks partitioned along x

grid = RectilinearGrid(arch,
                       size = (64, 64, 32),
                       x = (0, 1), y = (0, 1), z = (0, 1),
                       topology = (Periodic, Periodic, Bounded))

@onrank 0 @info "rank 0 sees the local grid:" grid
@onrank 1 @info "rank 1 sees the local grid:" grid
"""

write("distributed_demo.jl", architecture_demo)

using MPI

run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) distributed_demo.jl`)

# Two things to notice in the output. First, each rank reports a *local* grid with
# **half** the points we asked for: `size` always refers to the global grid, and the
# partition divides it. Second, the partitioned direction is no longer `Periodic` but
# `FullyConnected` — its boundary condition is "ask the neighboring rank", and
# periodicity re-emerges globally from the ring of connections.
#
# Useful controls, all keyword arguments of `Distributed`:
#
# - `Distributed(GPU())` — each rank drives a GPU (assigned round-robin within a node)
# - `Distributed(CPU(); partition = Partition(x = 2, y = 2))` — a 2 × 2 rank layout
# - `Partition(y = Equal())` — split whatever the total rank count is, along ``y``
#
# and the communication-aware macros `@root`, `@onrank`, `@handshake` for the moments
# when ranks must not talk over each other (printing, writing shared files, ...).
#
# ## The science driver: deep convection at 10-meter resolution
#
# Now the real thing. The driver below is a complete, launch-ready script: a
# 5 km × 5 km × 1 km column of Nordic-Seas winter water (``N^2 = 10^{-5}`` s⁻²,
# ``f = 1.37 \times 10^{-4}`` s⁻¹) cooled by a destabilizing buoyancy flux equivalent
# to ≈ 800 W m⁻², on a 512 × 512 × 128 grid distributed across however many GPUs
# `mpiexec` provides. Rotation makes this flavor of convection special: the plumes
# feel ``f`` within a few hours and organize into coherent, cyclonic rim-current
# structures — watch for them in the movie.
#
# The grid size and architecture are parameterized by environment variables with
# sensible defaults, a cheap trick that allows to smoke-test the identical script on
# two laptop CPU ranks before burning GPU hours:

convection_driver = raw"""
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.Units
using Printf
using MPI
MPI.Init()

Nx = parse(Int, get(ENV, "CONVECTION_NX", "512"))
Ny = parse(Int, get(ENV, "CONVECTION_NY", "512"))
Nz = parse(Int, get(ENV, "CONVECTION_NZ", "128"))
stop_time = parse(Float64, get(ENV, "CONVECTION_HOURS", "24")) * hours

child = get(ENV, "CONVECTION_ARCH", "GPU") == "GPU" ? GPU() : CPU()
arch = Distributed(child, partition = Partition(y = Equal()))

@root @info "Distributed convection on $(MPI.Comm_size(MPI.COMM_WORLD)) ranks, child = $child"

grid = RectilinearGrid(arch,
                       size = (Nx, Ny, Nz),
                       x = (0, 5kilometers),
                       y = (0, 5kilometers),
                       z = (-1kilometer, 0),
                       topology = (Periodic, Periodic, Bounded))

# Destabilizing surface buoyancy flux: Q = 800 W m⁻² of heat loss,
# Qᵇ = α g Q / (ρ cₚ) ≈ 3.7e-7 m² s⁻³, shut off after 18 hours.
Qᵇ = 3.7e-7
@inline cooling(x, y, t, p) = ifelse(t < 18hours, p.Qᵇ, zero(p.Qᵇ))
b_top_bc = FluxBoundaryCondition(cooling, parameters = (; Qᵇ))
b_bcs = FieldBoundaryConditions(top = b_top_bc)

model = NonhydrostaticModel(grid;
                            advection = WENO(order = 9),
                            coriolis = FPlane(f = 1.37e-4),
                            buoyancy = BuoyancyTracer(),
                            tracers = :b,
                            boundary_conditions = (; b = b_bcs))

N² = 1e-5
ϵ(x, y, z) = 1e-3 * N² * grid.Lz * (2rand() - 1)  # seed noise
set!(model, b = (x, y, z) -> N² * z + ϵ(x, y, z))

simulation = Simulation(model; Δt = 10, stop_time)
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl = 0.7, max_Δt = 30)

wall_clock = Ref(time_ns())
function progress(sim)
    msg = @sprintf("iter: %d, t: %s, Δt: %s, max|w|: %.3f m s⁻¹, wall: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, sim.model.velocities.w),
                   prettytime(1e-9 * (time_ns() - wall_clock[])))
    @root @info msg
    wall_clock[] = time_ns()
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# Each rank writes its own slab: Oceananigans suffixes the filename with the rank
# automatically (deep_convection_rank0.jld2, _rank1.jld2, ...).
u, v, w = model.velocities
b = model.tracers.b

simulation.output_writers[:slices] =
    JLD2Writer(model, (; w, b);
               filename = "deep_convection.jld2",
               indices = (:, :, Nz - 4),
               schedule = TimeInterval(5minutes),
               overwrite_existing = true)

run!(simulation)

@root @info "Done."
"""

write("distributed_convection.jl", convection_driver)
nothing #hide

# ### Launching
#
# On a machine with 4 GPUs (or via Slurm on the workshop cluster):
#
# ```bash
# mpiexec -n 4 julia --project distributed_convection.jl
# ```
#
# ```bash
# #!/bin/bash
# #SBATCH --job-name=convection
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=4     # one rank per GPU
# #SBATCH --gpus-per-node=4
# #SBATCH --time=01:00:00
#
# srun julia --project distributed_convection.jl
# ```
#
# Two practical notes for the cluster. First, configure MPI.jl to use the *system* MPI
# (`MPIPreferences.use_system_binary()`) so that GPU-aware transport is available —
# the Julia-shipped MPICH works everywhere but routes GPU halos through host memory.
# Second, set `JULIA_CUDA_MEMORY_POOL=none` when oversubscribing memory-hungry runs;
# the workshop cluster documentation has the blessed incantations.
#
# And for the impatient without a GPU in reach — the same script, blessedly unchanged,
# on two laptop CPU ranks at toy resolution:
#
# ```bash
# CONVECTION_ARCH=CPU CONVECTION_NX=64 CONVECTION_NY=64 CONVECTION_NZ=32 \
# CONVECTION_HOURS=2 mpiexec -n 2 julia --project distributed_convection.jl
# ```
#
# ### Putting the ranks back together
#
# Each rank wrote the slice of its own slab. For full-3D output, `FieldTimeSeries`
# pointed at the stem filename reassembles the global field automatically from the rank
# files; for *sliced* output like ours we do the (one-line) stitching ourselves: we
# partitioned in ``y``, so the global slice is the rank slabs glued along the second
# dimension:
#
# ```julia
# using Oceananigans, CairoMakie
#
# nranks = 4
# w_ranks = [FieldTimeSeries("deep_convection_rank$(r).jld2", "w"; combine = false)
#            for r in 0:nranks-1]
# times = w_ranks[1].times
#
# w_global(n) = hcat((interior(w_ranks[r][n], :, :, 1) for r in 1:nranks)...)
#
# n = Observable(length(times))
# fig, ax, hm = heatmap(@lift(w_global($n)), colormap = :balance, colorrange = (-0.05, 0.05),
#                       axis = (aspect = 1, title = "vertical velocity at z = -35 m"))
# record(fig, "convection.mp4", 1:length(times)) do i
#     n[] = i
# end
# ```
#
# (At the thousand-rank scale one graduates to parallel IO — NetCDF over MPI-IO, or a
# shared Zarr store — but rank-suffixed JLD2 files carry remarkably far, and never
# produce a corrupted global file at hour 47 of 48.)
#
# ## How well does it scale?
#
# Three rules of thumb, in decreasing order of importance:
#
# 1. **Saturate the device before adding devices.** A GPU wants ≥ ~10⁷ grid points per
#    rank to amortize kernel launches; below that, strong scaling is a money bonfire.
#    Distribute for *memory* first, for speed second.
# 2. **Surface-to-volume is destiny.** Halo traffic scales with the slab surface,
#    compute with its volume: thicker slabs (fewer ranks per direction, or 2D
#    partitions) communicate proportionally less.
# 3. **The pressure solver is the tax** (nonhydrostatic only). The distributed FFT
#    transposes the field across all ranks twice per solve; past a few dozen ranks it
#    dominates. The hydrostatic models of tomorrow replace it with a 2D problem — which
#    is the deep reason global hydrostatic oceans scale to thousands of GPUs (see the
#    [GB-25 project](https://github.com/NumericalEarth/GB-25) for where that road
#    leads).
#
# ## Things to try
#
# !!! tip "Rotation on, rotation off"
#     Run twice at toy resolution with `f = 1.37e-4` and `f = 0`. Compare the `w`
#     slices after 12 hours: rotation tames the plumes into smaller, more numerous
#     columns of width ``\ell \sim \sqrt{Q^b / (N^2 f)}``-ish — measure it.
#
# !!! tip "Weak scaling, measured"
#     Time the script (the progress callback prints wall time per 50 iterations) at
#     256² on 1 rank, 512×256 on 2, 512² on 4, keeping points-per-rank constant. The
#     ratio of the timings *is* your weak-scaling efficiency. Where does the FFT tax
#     start to show?
#
# !!! tip "Partition shapes"
#     Repeat the 4-rank run with `Partition(x = 2, y = 2)` instead of `Partition(y =
#     Equal())`. Squarer slabs, less halo surface per rank — measurably faster, or lost
#     in the noise?
