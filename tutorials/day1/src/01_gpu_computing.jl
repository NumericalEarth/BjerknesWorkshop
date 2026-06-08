# # GPU computing in Julia: from arrays to a turbulence solver
#
# *Monday afternoon — Introduction to GPU-based modelling.*
#
# Modern climate models live or die by their throughput: a century-long projection at
# eddy-resolving resolution consists in advancing O(10⁹) grid points over O(10⁷) time steps,
# and the hardware that today offers the highest memory bandwidth per watt is the GPU.
# All the models you will meet this week — Oceananigans, ClimaSeaIce, Breeze, the
# NumericalEarth stack — run natively on GPUs, and they do so through the same handful of
# ideas that we develop in this tutorial.
#
# In this session we will:
#
# 1. understand why GPUs are fast (and when they are not),
# 2. move arrays to the GPU and operate on them without writing any GPU code,
# 3. write portable compute *kernels* with KernelAbstractions.jl,
# 4. use those kernels to discretize a PDE, and
# 5. assemble a complete two-dimensional Navier–Stokes solver — a miniature Oceananigans —
#    that runs unmodified on your laptop's CPU and on an HPC GPU.
#
# Nothing here requires a GPU: every code block falls back to the CPU when no device is
# found, so you can follow along on your laptop and rerun the same script later on Betzy
# or LUMI.
#
# ## The professor and the army
#
# A CPU core is a brilliant professor: deep caches, branch prediction, out-of-order
# execution — machinery designed to make *one* instruction stream as fast as possible
# (low latency). A GPU is an army of thousands of modest workers: most of the silicon is
# spent on arithmetic units, and performance comes from doing the *same* operation on
# many data elements at once (high throughput).
#
# | | CPU | GPU |
# |--|-----|-----|
# | Cores | 4–64 | thousands |
# | Clock | ~4 GHz | ~1.5 GHz |
# | Memory bandwidth | ~100 GB/s | 1–8 TB/s |
# | Strength | complex logic | data parallelism |
#
# The last row of the table is the one that matters for us. Finite-volume fluid dynamics
# consists in applying identical stencil operations at every grid cell — exactly the
# pattern the army executes well. It is worth noticing that the bottleneck of a stencil
# code is almost never arithmetic but *memory bandwidth*: each tendency evaluation reads a
# handful of neighboring values and writes one, so the model advances only as fast as the
# memory can stream fields through the chip. The ~10–40× bandwidth advantage of a GPU is,
# to first order, the speedup you should expect for an ocean model.
#
# One more thing to keep in mind: the GPU never works alone. The CPU (the *host*)
# allocates memory, copies data, launches the compute tasks (called *kernels*) and
# collects results; the GPU (the *device*) executes them. Every simulation you will run
# this week follows this choreography:
#
# 1. the host sets up the problem (grid, fields, parameters),
# 2. the host moves the data to device memory,
# 3. the host launches kernels — functions executed by thousands of device threads,
# 4. the device crunches numbers in parallel,
# 5. the host pulls back only what it needs (diagnostics, output).
#
# Step 5 deserves respect: host–device transfers cross the PCIe bus at a bandwidth far
# lower than device memory, and a code that copies fields back and forth every time step
# throws the GPU advantage away. Good GPU codes keep data resident on the device.
#
# ## GPU backends in Julia
#
# Julia exposes every major vendor through the same generic interface, built on
# GPUArrays.jl and GPUCompiler.jl. Switching vendor consists in loading a different
# package:
#
# - **CUDA.jl** (NVIDIA) provides the `CuArray`
# - **AMDGPU.jl** (AMD) provides the `ROCArray`
# - **oneAPI.jl** (Intel) provides the `oneArray`
# - **Metal.jl** (Apple) provides the `MtlArray`
#
# In this tutorial we target CUDA, which is what you will find on most HPC systems, but
# everything below works identically with the other three. We detect the hardware once,
# at the top of the script, and write the rest of the code independently from it:

using CUDA
using KernelAbstractions
using Printf

if CUDA.functional()
    GPUArray = CuArray
    backend = CUDA.CUDABackend()
    CUDA.versioninfo()
else
    GPUArray = Array
    backend = KernelAbstractions.CPU()
    @info "No CUDA device found — running every 'GPU' section on the CPU instead."
end

# `backend` identifies *where* kernels execute, `GPUArray` *where* memory lives. The rest
# of the script never mentions CUDA again.
#
# ## GPU arrays: parallelism for free
#
# The simplest way to use a GPU is to never write GPU code at all. Julia's broadcasting
# (the dot syntax) and linear algebra dispatch to device implementations automatically:

host_matrix = rand(Float32, 1000, 1000)

device_matrix = GPUArray(host_matrix)    # host → device copy
typeof(device_matrix)

# Standard operations now run on the device:

second_matrix = GPUArray(rand(Float32, 1000, 1000))

sum_matrix     = device_matrix .+ second_matrix     # element-wise, on the device
product_matrix = device_matrix * second_matrix      # cuBLAS matrix multiply
sine_matrix    = sin.(device_matrix)                # broadcasting a function
nothing #hide

# and `Array(device_matrix)` copies the result back to the host when needed.
#
# Two practical notes. First, climate models on GPUs usually adopt `Float32`: it halves
# the memory traffic (remember, bandwidth is the currency) and modern devices execute it
# at twice the `Float64` rate or better. Second, GPU operations are *asynchronous* —
# the host queues work and continues — so timing measurements must synchronize first.
# Let us measure the one operation where the army shines, a large matrix multiply:

using BenchmarkTools
using LinearAlgebra

N = 2048
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = zeros(Float32, N, N)

cpu_time = @belapsed mul!($C, $A, $B)
@printf "CPU matrix multiply: %.3f s → %.1f GFLOP/s\n" cpu_time 2N^3 / cpu_time / 1e9

if CUDA.functional()
    dA, dB, dC = GPUArray(A), GPUArray(B), GPUArray(C)
    gpu_time = @belapsed CUDA.@sync mul!($dC, $dA, $dB)
    @printf "GPU matrix multiply: %.4f s → %.1f GFLOP/s (%.0f× speedup)\n" gpu_time 2N^3 / gpu_time / 1e9 cpu_time / gpu_time
end

# !!! tip "The bandwidth ceiling, measured"
#     Repeat the comparison for an element-wise operation, `sin.(A)`, at sizes
#     `N = 100`, `1000`, `5000`. At which size does the GPU start to win? Why is the
#     crossover so much larger than for the matrix multiply? (count the floating
#     point operations *per byte of memory traffic* in the two cases.)
#
# ## Writing kernels with KernelAbstractions.jl
#
# Broadcasting covers element-wise work, but a PDE solver needs *stencils*: the update at
# cell ``(i, j)`` reads the neighbors at ``(i \pm 1, j \pm 1)``. For this we write kernels
# ourselves. [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
# (KA) allows to write a kernel once and execute it on every backend — NVIDIA, AMD, Intel,
# Apple, or the CPU. It is the layer on which Oceananigans is built, so what follows is
# precisely what happens behind the scenes when you call `time_step!` tomorrow.
#
# A kernel describes the work of *one thread*; the `@index` macro tells each thread which
# element it owns:

@kernel function _add!(c, a, b)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

# Instantiating the kernel on a backend and launching it over `n` threads consists in:

a = GPUArray(ones(Float32, 1024))
b = GPUArray(2 .* ones(Float32, 1024))
c = similar(a)

add! = _add!(backend)              # compile for this backend
add!(c, a, b, ndrange = length(c)) # launch one thread per element
KernelAbstractions.synchronize(backend)

all(Array(c) .== 3)

# On the GPU the threads are organized hierarchically: groups of threads (a *workgroup*,
# or "block" in CUDA jargon) execute together on the same multiprocessor and can share
# fast memory; the collection of all workgroups (the *grid*) covers the `ndrange`. KA
# picks a sensible workgroup size automatically, and accepts an explicit one — e.g.
# `_add!(backend, 256)` — when you want control.
#
# !!! note "A naming convention you will see all week"
#     Throughout the NumericalEarth stack, a leading underscore marks the `@kernel`
#     function (`_add!`) launched by a same-named host-side function (`add!`). We adopt
#     the same convention here.
#
# ### A first stencil: the Laplacian
#
# Stencils need multi-dimensional indices, which `@index(Global, NTuple)` provides. Here
# is the five-point Laplacian ``\nabla^2 f`` on a doubly-periodic domain. The `left` and
# `right` helpers wrap the indices around the boundary — written with `ifelse` rather than
# `if`/`else` because all the threads of a workgroup march in lockstep, and uniform
# (branchless) code keeps them from diverging:

@inline left(i, N)  = ifelse(i == 1, N, i - 1)
@inline right(i, N) = ifelse(i == N, 1, i + 1)

@inline function ∇²(i, j, f, Δx, Δy, Nx, Ny)
    @inbounds begin
        ∂²fx = (f[right(i, Nx), j] - 2f[i, j] + f[left(i, Nx), j]) / Δx^2
        ∂²fy = (f[i, right(j, Ny)] - 2f[i, j] + f[i, left(j, Ny)]) / Δy^2
    end
    return ∂²fx + ∂²fy
end

@kernel function _laplacian!(∇²f, f, Δx, Δy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds ∇²f[i, j] = ∇²(i, j, f, Δx, Δy, Nx, Ny)
end

# Note the division of labor, another pattern taken straight from Oceananigans: the
# *operator* `∇²` is an `@inline` function of the indices, and the *kernel* merely maps it
# over the grid. Operators compose — we will reuse `∇²` inside the Navier–Stokes tendency
# kernel below without rewriting anything.
#
# We can verify the kernel on a function with a known Laplacian, ``f = \sin x \sin y``
# for which ``\nabla^2 f = -2f``:

Nx, Ny = 128, 128
Δx, Δy = 2π / Nx, 2π / Ny

f   = GPUArray([sin((i - 0.5) * Δx) * sin((j - 0.5) * Δy) for i in 1:Nx, j in 1:Ny])
∇²f = similar(f)

laplacian! = _laplacian!(backend)
laplacian!(∇²f, f, Δx, Δy, Nx, Ny, ndrange = size(f))
KernelAbstractions.synchronize(backend)

maximum(abs, Array(∇²f) .+ 2 .* Array(f))  # discretization error, O(Δx²) ≈ 2e-3

# The same kernel object runs on the CPU backend with the same results — this is the
# property that allows to develop and debug on a laptop and deploy on a supercomputer
# without touching the code.
#
# !!! tip "A stencil of your own"
#     A good way to make this stick: a kernel `_gradient_magnitude!` computing
#     ``|\nabla f| = \sqrt{(\partial_x f)^2 + (\partial_y f)^2}``
#     with centered differences, checked against ``f(x, y) = x + 2y`` (where
#     ``|\nabla f| = \sqrt{5}``) away from the boundaries.
#
# ## Interlude: multiple dispatch, or how to make kernels extensible
#
# Before assembling the solver we need one Julia-specific idea. A numerical model offers
# choices — advection schemes, equations of state, parameterizations — and the classical
# implementations select among them with strings and `if`/`else` ladders inside the inner
# loop. Julia instead dispatches on *types*: we define one empty `struct` per scheme and
# one method per (function, scheme) pair, and the compiler selects the method at compile
# time, also inside a GPU kernel, at zero runtime cost.
#
# Consider the reconstruction of a cell-centered quantity at the face ``i + 1/2``,
# the central ingredient of finite-volume advection:

abstract type AbstractScheme end
struct SecondOrderCentered <: AbstractScheme end
struct FourthOrderCentered <: AbstractScheme end

@inline reconstruct(f, i, ::SecondOrderCentered) = (f[i] + f[i+1]) / 2
@inline reconstruct(f, i, ::FourthOrderCentered) = (-f[i-1] + 7f[i] + 7f[i+1] - f[i+2]) / 12

cell_values = [0.0, 1.0, 4.0, 9.0, 16.0]      # f = x² at x = 0, 1, 2, 3, 4
exact_face_value = 2.5^2

(second_order = reconstruct(cell_values, 3, SecondOrderCentered()) - exact_face_value,
 fourth_order = reconstruct(cell_values, 3, FourthOrderCentered()) - exact_face_value)

# Adding a fifth scheme does not touch the existing four — we just define a new type and
# a new method. Contrarily to an `if`/`else` ladder, the dispatch costs nothing inside a
# kernel: the scheme type is known at compile time, so the compiler emits straight-line
# code for the chosen stencil. This is precisely how `advection = WENO()` works in
# Oceananigans, and we will use the same mechanism for the solver's advection scheme.
#
# ## The target: two-dimensional turbulence
#
# Time to put everything together. We solve the two-dimensional incompressible
# Navier–Stokes equations,
#
# ```math
# \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} =
#     -\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0,
# ```
#
# in a doubly-periodic square — the simplest system that produces a genuinely turbulent
# flow, and (not by chance) the system behind the mesoscale eddy fields you will simulate
# tomorrow. Two dimensional turbulence transfers energy *upscale*: small vortices merge
# into larger ones, which is the reason why the ocean is full of coherent eddies.
#
# For the time discretization we use forward Euler — the simplest possible choice, stable
# enough at small time step and free of implicit complications:
#
# ```math
# \frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta t} = \mathrm{RHS}^n .
# ```
#
# ### The projection method
#
# The system gives explicit update formulas for ``u`` and ``v``, but no evolution
# equation for the pressure: ``p`` is whatever it must be for the velocity to remain
# divergence-free. The classical solution is the *projection method* of
# [Chorin (1968)](https://doi.org/10.1090/S0025-5718-1968-0242392-2), which splits each
# time step into a prediction that ignores the pressure and a correction that restores
# incompressibility:
#
# 1. **Tendencies**: ``G^n = -(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n``
# 2. **Prediction**: ``\mathbf{u}^\star = \mathbf{u}^n + \Delta t \, G^n`` (generally not divergence-free)
# 3. **Pressure solve**: requiring ``\nabla \cdot \mathbf{u}^{n+1} = 0`` in the correction
#    below yields the Poisson equation
#    ``\nabla^2 p^{n+1} = \nabla \cdot \mathbf{u}^\star / \Delta t``
# 4. **Correction**: ``\mathbf{u}^{n+1} = \mathbf{u}^\star - \Delta t \, \nabla p^{n+1}``
#
# The pressure acts as a Lagrange multiplier, and the correction is an orthogonal
# projection of ``\mathbf{u}^\star`` onto the divergence-free subspace — hence the name.
# Oceananigans' `NonhydrostaticModel` advances the same algorithm (with a fancier time
# stepping scheme); the hydrostatic models of tomorrow replace the 3D pressure solve with
# a free-surface solve, but the architecture is unchanged.
#
# ### The staggered C-grid
#
# Where do `u`, `v` and `p` live? Following
# [Arakawa and Lamb (1977)](https://doi.org/10.1016/B978-0-12-460817-7.50009-4), on a
# *staggered* C-grid:
#
# ```text
#     +-------v[i,j+1]-------+
#     |                      |
#   u[i,j]      p[i,j]    u[i+1,j]
#     |                      |
#     +--------v[i,j]--------+
# ```
#
# pressure at cell centers, ``u`` on the west/east faces, ``v`` on the south/north faces.
# The staggering buys two properties that collocated grids lose: the discrete divergence
# is *exactly* zero after the correction (to machine precision — we will check), and the
# pressure cannot develop the spurious checkerboard mode that haunts collocated
# discretizations. Every model in this workshop, and almost every ocean model in
# existence, lives on a C-grid.
#
# ## Building the solver
#
# ### The grid

struct Grid{T, A}
    Nx :: Int
    Ny :: Int
    Lx :: T
    Ly :: T
    Δx :: T
    Δy :: T
    x  :: A     # cell-center coordinates
    y  :: A
end

function Grid(ArrayType, T, Nx, Ny, Lx, Ly)
    Δx, Δy = T(Lx / Nx), T(Ly / Ny)
    x = ArrayType(T[(i - 0.5) * Δx for i in 1:Nx])
    y = ArrayType(T[(j - 0.5) * Δy for j in 1:Ny])
    return Grid(Nx, Ny, T(Lx), T(Ly), Δx, Δy, x, y)
end

Base.size(grid::Grid) = (grid.Nx, grid.Ny)
nothing #hide

# ### Difference operators
#
# Forward and backward differences on the periodic grid; together with `∇²` defined
# above, they are all the calculus we need. On the C-grid the *forward* difference of a
# face quantity lands at the center and the *backward* difference of a center quantity
# lands at the face — keeping track of this bookkeeping is most of the work of writing a
# staggered-grid model:

@inline δx⁺(i, j, f, Δx, Nx) = @inbounds (f[right(i, Nx), j] - f[i, j]) / Δx
@inline δy⁺(i, j, f, Δy, Ny) = @inbounds (f[i, right(j, Ny)] - f[i, j]) / Δy
@inline δx⁻(i, j, f, Δx, Nx) = @inbounds (f[i, j] - f[left(i, Nx), j]) / Δx
@inline δy⁻(i, j, f, Δy, Ny) = @inbounds (f[i, j] - f[i, left(j, Ny)]) / Δy
nothing #hide

# ### Advection
#
# In flux form, the advection terms of the two momentum equations read
#
# ```math
# \partial_x(U u) + \partial_y(V u) \quad \text{and} \quad
# \partial_x(U v) + \partial_y(V v),
# ```
#
# so for each velocity component we need the momentum fluxes through its cell faces. To
# build a flux ``F = U \phi`` at a face we need two ingredients at that face: the
# *advecting* velocity ``U``, which we obtain by symmetric interpolation, and the
# *advected* quantity ``\phi``, which we **reconstruct** with the chosen advection scheme.
# This distinction — interpolate the carrier, reconstruct the cargo — is the key pattern,
# and the reconstruction is where the scheme types of the interlude enter:

@inline symmetric_x(i, j, f, N) = @inbounds (f[i, j] + f[right(i, N), j]) / 2
@inline symmetric_y(i, j, f, N) = @inbounds (f[i, j] + f[i, right(j, N)]) / 2

abstract type AbstractAdvectionScheme end
struct Centered <: AbstractAdvectionScheme end

@inline reconstruct_in_x(i, j, f, N, U, ::Centered) = symmetric_x(i, j, f, N)
@inline reconstruct_in_y(i, j, f, N, V, ::Centered) = symmetric_y(i, j, f, N)
nothing #hide

# The centered scheme ignores the advecting velocity `U`; an upwind scheme (something to try
# at the end) selects a biased stencil according to its sign.
#
# Each momentum equation needs fluxes in both directions, which makes four combinations.
# The same-direction fluxes (``Uu``, ``Vv``) live at cell centers, between two like
# velocity points; the cross-direction fluxes (``Vu``, ``Uv``) live at cell corners,
# where the advecting velocity must additionally be interpolated across the grid. Each
# function returns the flux pair through the right and left faces of the cell owned by
# ``(i, j)``:

@inline function x_flux_of_u(i, j, u, Nx, scheme)
    im = left(i, Nx)
    Uᴿ = symmetric_x(i,  j, u, Nx)
    Uᴸ = symmetric_x(im, j, u, Nx)
    return Uᴿ * reconstruct_in_x(i,  j, u, Nx, Uᴿ, scheme),
           Uᴸ * reconstruct_in_x(im, j, u, Nx, Uᴸ, scheme)
end

@inline function y_flux_of_v(i, j, v, Ny, scheme)
    jm = left(j, Ny)
    Vᴿ = symmetric_y(i, j,  v, Ny)
    Vᴸ = symmetric_y(i, jm, v, Ny)
    return Vᴿ * reconstruct_in_y(i, j,  v, Ny, Vᴿ, scheme),
           Vᴸ * reconstruct_in_y(i, jm, v, Ny, Vᴸ, scheme)
end

@inline function y_flux_of_u(i, j, u, v, Nx, Ny, scheme)
    im, jm = left(i, Nx), left(j, Ny)
    Vᴿ = symmetric_x(im, right(j, Ny), v, Nx)   # v interpolated to the u corner above
    Vᴸ = symmetric_x(im, j,            v, Nx)   # ... and below
    return Vᴿ * reconstruct_in_y(i, j,  u, Ny, Vᴿ, scheme),
           Vᴸ * reconstruct_in_y(i, jm, u, Ny, Vᴸ, scheme)
end

@inline function x_flux_of_v(i, j, u, v, Nx, Ny, scheme)
    im, jm = left(i, Nx), left(j, Ny)
    Uᴿ = symmetric_y(right(i, Nx), jm, u, Ny)   # u interpolated to the v corner right
    Uᴸ = symmetric_y(i,            jm, u, Ny)   # ... and left
    return Uᴿ * reconstruct_in_x(i,  j, v, Nx, Uᴿ, scheme),
           Uᴸ * reconstruct_in_x(im, j, v, Nx, Uᴸ, scheme)
end
nothing #hide

# ### The pressure solver
#
# On a doubly-periodic domain the Poisson equation ``\nabla^2 p = f`` diagonalizes under
# the Fourier transform, so the solve consists in one FFT, one division by the
# eigenvalues, and one inverse FFT — ``O(N^2 \log N)`` work, executed by cuFFT on the
# device when the storage is a `CuArray`.
#
# One subtlety with important consequences: we must divide by the eigenvalues of the
# *discrete* Laplacian,
#
# ```math
# \lambda_{mn} = -\frac{4 \sin^2(\pi m / N_x)}{\Delta x^2}
#                -\frac{4 \sin^2(\pi n / N_y)}{\Delta y^2},
# ```
#
# and not by the continuous ``-(k_x^2 + k_y^2)``. The reason is consistency: the
# correction step removes the divergence *as measured by our finite differences*, and
# only the discrete eigenvalues invert exactly the same operator that ``δx⁺``/``δx⁻``
# build. Using the continuous symbols would leave a residual divergence at every step.
# With the discrete ones, ``\nabla \cdot \mathbf{u} = 0`` holds to machine precision —
# a check worth performing on every solver you ever write:

using FFTW
using AbstractFFTs

struct FFTPoissonSolver{E, S, P, IP}
    eigenvalues :: E
    storage     :: S
    plan        :: P
    iplan       :: IP
end

function FFTPoissonSolver(grid::Grid{T}, ArrayType) where T
    (; Nx, Ny, Δx, Δy) = grid

    eigenvalues = T[4sin(π * (i - 1) / Nx)^2 / Δx^2 + 4sin(π * (j - 1) / Ny)^2 / Δy^2
                    for i in 1:Nx÷2+1, j in 1:Ny]
    eigenvalues[1, 1] = T(Inf)    # mean mode: pressure is defined up to a constant

    storage = ArrayType(zeros(Complex{T}, Nx÷2+1, Ny))
    example = ArrayType(zeros(T, Nx, Ny))
    plan    = AbstractFFTs.plan_rfft(example)
    iplan   = AbstractFFTs.plan_irfft(storage, Nx)

    return FFTPoissonSolver(ArrayType(eigenvalues), storage, plan, iplan)
end

function solve!(p, solver::FFTPoissonSolver, rhs)
    mul!(solver.storage, solver.plan, rhs)
    solver.storage .= .- solver.storage ./ solver.eigenvalues
    mul!(p, solver.iplan, solver.storage)
    return p
end
nothing #hide

# ### The kernels
#
# Four kernels implement the four stages of the projection algorithm. It is possible to
# notice that they contain no reference to CUDA, to the backend, or to the array type:
# they are pure index-space descriptions of the numerics.

@kernel function _compute_tendencies!(Gu, Gv, u, v, ν, Δx, Δy, Nx, Ny, scheme)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Fᵘ⁺, Fᵘ⁻ = x_flux_of_u(i, j, u, Nx, scheme)
        Gᵘ⁺, Gᵘ⁻ = y_flux_of_u(i, j, u, v, Nx, Ny, scheme)
        Gu[i, j] = -(Fᵘ⁺ - Fᵘ⁻) / Δx - (Gᵘ⁺ - Gᵘ⁻) / Δy + ν * ∇²(i, j, u, Δx, Δy, Nx, Ny)

        Fᵛ⁺, Fᵛ⁻ = x_flux_of_v(i, j, u, v, Nx, Ny, scheme)
        Gᵛ⁺, Gᵛ⁻ = y_flux_of_v(i, j, v, Ny, scheme)
        Gv[i, j] = -(Fᵛ⁺ - Fᵛ⁻) / Δx - (Gᵛ⁺ - Gᵛ⁻) / Δy + ν * ∇²(i, j, v, Δx, Δy, Nx, Ny)
    end
end

@kernel function _predict!(u, v, Gu, Gv, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u[i, j] += Δt * Gu[i, j]
        v[i, j] += Δt * Gv[i, j]
    end
end

@kernel function _divergence!(d, u, v, Δx, Δy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds d[i, j] = δx⁺(i, j, u, Δx, Nx) + δy⁺(i, j, v, Δy, Ny)
end

@kernel function _pressure_correct!(u, v, p, Δt, Δx, Δy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u[i, j] -= Δt * δx⁻(i, j, p, Δx, Nx)
        v[i, j] -= Δt * δy⁻(i, j, p, Δy, Ny)
    end
end

@kernel function _vorticity!(ω, u, v, Δx, Δy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds ω[i, j] = δx⁺(i, j, v, Δx, Nx) - δy⁺(i, j, u, Δy, Ny)
end
nothing #hide

# ### The model
#
# A `struct` collects grid, fields, parameters and solver — the moral equivalent of
# Oceananigans' `NonhydrostaticModel`:

struct NavierStokesModel{T, G, F, S, A, B}
    grid           :: G
    u              :: F
    v              :: F
    p              :: F
    Gu             :: F
    Gv             :: F
    divergence     :: F
    ω              :: F
    ν              :: T
    advection      :: A
    poisson_solver :: S
    backend        :: B
end

function NavierStokesModel(grid::Grid{T}, backend, ArrayType; ν, advection = Centered()) where T
    fields = Tuple(ArrayType(zeros(T, size(grid))) for _ in 1:7)
    poisson_solver = FFTPoissonSolver(grid, ArrayType)
    return NavierStokesModel(grid, fields..., T(ν), advection, poisson_solver, backend)
end
nothing #hide

# and the time step launches the kernels in sequence — compare it line by line with the
# algorithm summary above:

function time_step!(model::NavierStokesModel, Δt)
    (; grid, u, v, p, Gu, Gv, divergence, ν, advection, poisson_solver, backend) = model
    (; Nx, Ny, Δx, Δy) = grid
    worksize = size(grid)

    _compute_tendencies!(backend)(Gu, Gv, u, v, ν, Δx, Δy, Nx, Ny, advection, ndrange = worksize)
    _predict!(backend)(u, v, Gu, Gv, Δt, ndrange = worksize)
    _divergence!(backend)(divergence, u, v, Δx, Δy, Nx, Ny, ndrange = worksize)
    divergence ./= Δt
    solve!(p, poisson_solver, divergence)
    _pressure_correct!(backend)(u, v, p, Δt, Δx, Δy, Nx, Ny, ndrange = worksize)

    return nothing
end

function compute_vorticity!(model::NavierStokesModel)
    (; grid, u, v, ω, backend) = model
    (; Nx, Ny, Δx, Δy) = grid
    _vorticity!(backend)(ω, u, v, Δx, Δy, Nx, Ny, ndrange = size(grid))
    return ω
end
nothing #hide

# Notice that the host code launches kernels and orchestrates, while every floating-point
# operation happens on the device, and no field ever leaves it during time stepping —
# exactly the choreography described at the beginning.
#
# ## An initial condition: a sea of vortices
#
# We seed the flow with a few dozen Lamb–Oseen vortices — Gaussian blobs of vorticity
# with random positions and circulations. Rather than differentiating an analytical
# velocity (whose ``1/r`` tails break periodicity), we set the *vorticity* and recover
# the velocity from the streamfunction: solve ``\nabla^2 \psi = -\omega`` — conveniently,
# with the Poisson solver we already own — then ``u = \partial_y \psi``,
# ``v = -\partial_x \psi``. The resulting field is smooth, periodic, and divergence-free
# by construction.

using Random

@kernel function _gaussian_vortices!(ω, xᵥ, yᵥ, Γᵥ, σ, Δx, Δy, Lx, Ly)
    i, j = @index(Global, NTuple)
    x = (i - 0.5) * Δx
    y = (j - 0.5) * Δy

    ωᵢⱼ = zero(eltype(ω))
    for n in eachindex(xᵥ)
        δx = x - xᵥ[n]
        δy = y - yᵥ[n]
        δx -= Lx * round(δx / Lx)   # periodic minimum-image distance
        δy -= Ly * round(δy / Ly)
        r² = δx^2 + δy^2
        ωᵢⱼ += Γᵥ[n] / (π * σ^2) * exp(-r² / σ^2)
    end
    @inbounds ω[i, j] = ωᵢⱼ
end

@kernel function _velocity_from_streamfunction!(u, v, ψ, Δx, Δy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u[i, j] =  δy⁻(i, j, ψ, Δy, Ny)
        v[i, j] = -δx⁻(i, j, ψ, Δx, Nx)
    end
end

function seed_vortices!(model; number_of_vortices = 50, σ = 0.15, maximum_circulation = 2, seed = 1234)
    (; grid, u, v, p, ω, poisson_solver, backend) = model
    (; Nx, Ny, Lx, Ly, Δx, Δy) = grid
    T = eltype(u)
    ArrayType = typeof(parent(u)).name.wrapper

    Random.seed!(seed)
    xᵥ = ArrayType(rand(T, number_of_vortices) .* Lx)
    yᵥ = ArrayType(rand(T, number_of_vortices) .* Ly)
    Γᵥ = ArrayType((2 .* rand(T, number_of_vortices) .- 1) .* T(maximum_circulation))

    _gaussian_vortices!(backend)(ω, xᵥ, yᵥ, Γᵥ, T(σ), Δx, Δy, Lx, Ly, ndrange = size(grid))

    ω .*= -1
    solve!(p, poisson_solver, ω)   # p temporarily holds ψ
    ω .*= -1

    _velocity_from_streamfunction!(backend)(u, v, p, Δx, Δy, Nx, Ny, ndrange = size(grid))
    return nothing
end
nothing #hide

# ## Running on the CPU
#
# A driver advances the model and stores vorticity snapshots for the animation:

function run!(model, Δt, stop_time; snapshot_interval = 0.1)
    iterations_per_snapshot = ceil(Int, snapshot_interval / Δt)
    total_iterations = ceil(Int, stop_time / Δt)

    compute_vorticity!(model)
    snapshots = [Array(model.ω)]
    times = [0.0]

    elapsed = @elapsed for iteration in 1:total_iterations
        time_step!(model, Δt)

        if iteration % iterations_per_snapshot == 0
            compute_vorticity!(model)
            push!(snapshots, Array(model.ω))
            push!(times, iteration * Δt)
        end
    end

    cell_updates_per_second = prod(size(model.grid)) * total_iterations / elapsed
    @printf "%d iterations in %.1f s → %.2e cell-updates/s\n" total_iterations elapsed cell_updates_per_second

    return snapshots, times
end
nothing #hide

# Now we build a model at laptop-friendly resolution and let the vortices stir:

T = Float64
grid  = Grid(GPUArray, T, 256, 256, 2π, 2π)
model = NavierStokesModel(grid, backend, GPUArray; ν = 5e-4)

seed_vortices!(model)
snapshots, times = run!(model, 5e-4, 5.0)
nothing #hide

# Before admiring the result, the sanity check promised above — the discrete divergence
# after a step must vanish to machine precision (this single number validates the grid
# staggering, the operators, and the consistency of the FFT eigenvalues all at once):

(; u, v, divergence) = model
(; Nx, Ny, Δx, Δy) = grid
_divergence!(backend)(divergence, u, v, Δx, Δy, Nx, Ny, ndrange = size(grid))
KernelAbstractions.synchronize(backend)
@printf "maximum |∇·u| = %.2e\n" maximum(abs, Array(divergence))

# And the movie:

using CairoMakie

function animate_vorticity(snapshots, times, grid, filename)
    n = Observable(1)
    ωₙ = @lift snapshots[$n]
    label = @lift @sprintf "t = %.2f" times[$n]

    ωmax = maximum(abs, snapshots[1]) * 0.8

    fig = Figure(size = (600, 600))
    ax = Axis(fig[1, 1]; title = label, xlabel = "x", ylabel = "y", aspect = DataAspect())
    heatmap!(ax, Array(grid.x), Array(grid.y), ωₙ,
             colormap = :balance, colorrange = (-ωmax, ωmax))

    CairoMakie.record(fig, filename, eachindex(snapshots), framerate = 12) do frame
        n[] = frame
    end
end

animate_vorticity(snapshots, times, grid, "two_dimensional_turbulence.mp4")
nothing #hide

# ![](two_dimensional_turbulence.mp4)
#
# Opposite-signed vortices pair up and propagate, like-signed vortices merge, and the
# population coarsens — the inverse energy cascade in action.
#
# ## Unleashing the GPU
#
# Everything above ran through the portable backend, so "switching on" the GPU consists
# in nothing more than what we already did at the top of the script: when a device is
# available, `GPUArray === CuArray` and the same constructors produce a device-resident
# model. We celebrate the bandwidth with 16× more grid points and a sharper viscosity:

if CUDA.functional()
    grid  = Grid(GPUArray, Float32, 1024, 1024, 2π, 2π)
    model = NavierStokesModel(grid, backend, GPUArray; ν = 5e-5)

    seed_vortices!(model; number_of_vortices = 100, σ = 0.1)
    snapshots, times = run!(model, 1e-4, 5.0)
    animate_vorticity(snapshots, times, grid, "two_dimensional_turbulence_gpu.mp4")
else
    @info "No GPU found — skipping the high-resolution run (try this section on the cluster!)."
end

# On an H100 the 1024² model sustains roughly 10⁹ cell-updates per second — compare with
# the number your laptop printed above, and note that the *script did not change*. This
# is the workflow we recommend for everything this week: develop and debug at small size
# on the CPU, then move to the GPU only to turn the resolution dial.
#
# ## Things to try
#
# !!! tip "Third-order upwind advection"
#     The centered scheme is energy-conserving but dispersive: at marginal resolution it
#     decorates sharp gradients with grid-scale wiggles. Implement
#     `struct UpwindThirdOrder <: AbstractAdvectionScheme end` with the biased stencils
#     ```math
#     f_{i+1/2} = \begin{cases}
#     (-f_{i-1} + 5f_i + 2f_{i+1})/6 & U \ge 0 \\
#     (2f_i + 5f_{i+1} - f_{i+2})/6 & U < 0
#     \end{cases}
#     ```
#     by adding two methods `reconstruct_in_x` / `reconstruct_in_y` that select the
#     stencil with `ifelse(U ≥ 0, …, …)`. Use `left`/`right` for the index arithmetic
#     (e.g. ``i+2`` is `right(right(i, N), N)`). Then rerun the simulation with
#     `advection = UpwindThirdOrder()` — no other line changes — and compare the
#     vorticity fields.
#
# !!! tip "Finding the CPU–GPU crossover"
#     Benchmark `time_step!` at `Nx = Ny = 64, 128, 256, 512, 1024` on the CPU and (on
#     the cluster) on the GPU. Plot the two curves of cell-updates per second. Where do
#     they cross? What does the GPU curve do below 128², and why? (every kernel
#     launch carries a fixed ~10 μs overhead, independently from the work it contains.)
#
# !!! tip "Fifth-order upwind, for the brave"
#     Same as the third-order scheme, with the five-point stencils
#     ``(2f_{i-2} - 13f_{i-1} + 47f_i + 27f_{i+1} - 3f_{i+2})/60`` for ``U \ge 0`` and
#     ``(-3f_{i-1} + 27f_i + 47f_{i+1} - 13f_{i+2} + 2f_{i+3})/60`` for ``U < 0``.
#
# ## Key takeaways
#
# - GPUs win through memory bandwidth and data parallelism; stencil codes are
#   bandwidth-bound and inherit the full advantage.
# - Julia + KernelAbstractions allows to write a kernel once and run it on every vendor's
#   hardware, and on the CPU for development.
# - Multiple dispatch turns numerical choices (advection schemes, closures, ...) into
#   types, extensible without touching — or paying anything inside — the kernels.
# - The C-grid + projection method + FFT pressure solver triad is the blueprint of
#   Oceananigans' nonhydrostatic model; you have now written the essential parts of it
#   yourself.
#
# ## Further reading
#
# - [Chorin (1968), *Numerical solution of the Navier–Stokes equations*](https://doi.org/10.1090/S0025-5718-1968-0242392-2)
# - [Arakawa and Lamb (1977), on staggered grids](https://doi.org/10.1016/B978-0-12-460817-7.50009-4)
# - [KernelAbstractions.jl documentation](https://juliagpu.github.io/KernelAbstractions.jl/stable/)
# - [CUDA.jl documentation](https://cuda.juliagpu.org/stable/)
# - [Oceananigans.jl](https://github.com/NumericalEarth/Oceananigans.jl) — the production
#   version of what we built today, and the protagonist of tomorrow
# - The [PolarPlunge.jl](https://github.com/NumericalEarth/PolarPlunge.jl) notebooks, from
#   which this tutorial is distilled, including a Lagrangian-particle extra where
#   turbulence shreds a text banner
