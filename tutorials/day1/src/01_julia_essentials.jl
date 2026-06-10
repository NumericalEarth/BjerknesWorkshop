# # Julia essentials: enough to drive an ocean model
#
# *Arrays, broadcasting, multiple dispatch, and a portable GPU kernel.*
#
# You just watched three simulations that are, underneath, short Julia scripts. This notebook gives you
# exactly the Julia you need to read and write those scripts — no more: enough to make the configuration
# files of the rest of the week *readable*, and to change them with confidence. Everything runs on your
# laptop.
#
# We lean on a handful of ideas, in order: **arrays and broadcasting** (how fields are stored and updated),
# **functions** and **multiple dispatch** (why independent pieces compose), and a closing **diffusion step**
# that ties them together — which we then watch drop down to its machine code and back up to a single
# kernel that runs unchanged on the GPU.
#
# ## Getting started: where to run Julia, and the package manager
#
# Julia installs in one line through [`juliaup`](https://github.com/JuliaLang/juliaup), which also manages
# versions. You can drive it from several places: the **REPL** (the interactive prompt), a **Jupyter**
# notebook like this one (via `IJulia`), **Pluto** reactive notebooks, or an editor such as **VS Code** with
# the Julia extension. Everything below runs the same way in any of them.
#
# Code lives in *packages*, and every project carries its own *environment* — a `Project.toml` listing its
# direct dependencies and a `Manifest.toml` pinning exact versions, so a setup is reproducible to the digit.
# This notebook's environment is `tutorials/day1`; it is already instantiated, and `using` brings names into
# scope. We keep the imports modest:

using Statistics          # mean, std — small conveniences from the standard library
using Printf              # @sprintf, for clean numeric output
using InteractiveUtils    # @code_llvm, @code_native — to peek at the compiled code

# The package manager `Pkg` is itself a Julia package. You do not need it now, but the commands you will
# reach for are `Pkg.activate("tutorials/day1")` to enter an environment, `Pkg.instantiate()` to install what
# its `Manifest.toml` pins, `Pkg.add("Oceananigans")` to add a dependency, and `Pkg.status()` to see what is
# active. The documentation is excellent: the [Julia manual](https://docs.julialang.org),
# [Discourse](https://discourse.julialang.org) for questions, and each package's own docs carry runnable examples.
#
# ## Variables, types, and a word on speed
#
# Variables are dynamically typed, but every value has a concrete type, and that type is *why* Julia is
# fast: the compiler specializes each function on the types it is called with, emitting machine code as
# tight as C for that particular combination of arguments. You rarely write a type annotation — values
# carry their type, and the compiler does the rest.

temperature = 4.0         # a Float64 — the default, and what Oceananigans uses throughout
salinity    = 35.0
typeof(temperature), typeof(salinity)

# ## Under the hood: high-level, yet you can read the machine code
#
# That specialization is not a black box. `@code_llvm` prints the LLVM intermediate representation the
# compiler emits — and because it specializes on the argument types, the *same* source becomes *different*
# instructions for a `Float64` and an `Int`:

square(x) = x * x
@code_llvm debuginfo=:none square(2.0)     # a Float64 → floating-point multiply: `fmul double`

# Call it on an integer and the compiler emits integer arithmetic instead — one function, two machine codes:

@code_llvm debuginfo=:none square(2)       # an Int → `mul i64`

# `@code_native` drops one level further, to the CPU's own assembly: a single hardware multiply, exactly
# what a C compiler would emit for the same operation.

@code_native debuginfo=:none square(2.0)

# So Julia is high-level when you want it — you wrote no types — and low-level when you need it: you can
# read exactly what runs. It is the same compiler path that re-targets this code to the GPU further down.
#
# ## Arrays — how a field is stored
#
# An ocean field is an array. Build them explicitly, or from a function over a range:

depths = [0.0, -10.0, -50.0, -200.0, -1000.0]      # a 1D array, a column of interfaces
field  = zeros(4, 4)                               # a 4×4 array of zeros, ready to fill
size(field), length(depths)

# Indexing is 1-based and the first index is fastest-varying (column-major), exactly like Fortran — so
# the loops you may know from NEMO or MITgcm keep the same memory-friendly order here:

depths[1], depths[end]                             # first and last; `end` is the last index

# Ranges are lazy and cheap; `collect` materializes one if you ever need to:

x = range(0, 2π, length = 128)                     # 128 points around a periodic domain
Δx = step(x)

# ## Tuples and named tuples — how settings travel together
#
# A *tuple* is a fixed, ordered group of values, written with parentheses. Grid sizes and domain extents
# ride around the model as tuples — `size = (128, 128, 64)`, `extent = (Lx, Ly, Lz)` — and you index or
# destructure them like a tiny, immutable array:

grid_size = (128, 128, 64)                          # (Nx, Ny, Nz)
Nx, Ny, Nz = grid_size                              # destructuring, in one line
grid_size[3], Nx * Ny * Nz                          # index by position; total number of cells

# A *named tuple* labels each field, so a bundle of settings reads like the keyword arguments it usually
# becomes. Access is by name, and the names are part of the type — immutable and allocation-light:

surface = (T = 4.0, S = 35.0, depth = 0.0)          # a labelled record of surface conditions
surface.T, surface.S                                # reach in by name

# ## Control flow, loops, and comprehensions
#
# The usual `if`/`elseif`/`else` and `for`/`while` are all here. A short conditional classifies a water
# column by the sign of its buoyancy frequency `N²`:

stratification(N²) = N² > 0 ? "stable" : N² < 0 ? "unstable" : "neutral"   # the ternary `cond ? a : b`
stratification(1e-5), stratification(-1e-5)

# `for` iterates anything iterable; mutating an array inside the loop needs no `global`, so building a
# column interface-by-interface is natural:

column = Float64[]
for d in (10.0, 50.0, 200.0, 1000.0)
    push!(column, -d)                               # grow the vector in place
end
column

# A *comprehension* does the same in one expression — the compact way to lay down a coordinate or a
# stretched vertical grid, finer near the surface:

z_faces = [-1000.0 * (1 - i / 8)^2 for i in 0:8]    # 9 interfaces spanning −1000 m to the surface
length(z_faces), z_faces[begin], z_faces[end]

# ## Broadcasting — the central idiom
#
# A trailing dot applies any scalar operation element-wise, and *fuses* adjacent dotted operations into a
# single pass with no temporaries. This is how you will write every field update for the rest of the week:

c = @. sin(x)                                      # @. dots the whole expression: c[i] = sin(x[i])
c² = c .^ 2                                         # element-wise square
rms = sqrt(mean(c²))                               # reductions read like maths
@sprintf("rms = %.4f", rms)

# The fused form `@. a + b * c` allocates *one* array, not three — the same code, the same performance
# story, whether `a, b, c` live on the CPU or, as we will see, on a GPU.
#
# ## Functions
#
# Short functions get a one-line form; longer ones a block. A trailing `!` is a convention — not syntax —
# marking a function that *mutates* its first argument, the workhorse pattern for updating fields in place:

buoyancy(T, S; g = 9.81, α = 2e-4, β = 8e-4) = g * (α * T - β * S)   # one-line, with keyword arguments

function normalize!(field)                          # the ! signals: this overwrites `field`
    field .-= mean(field)                           # broadcasting again, in place
    field ./= std(field)
    return field
end

normalize!(copy(c))                                 # operate on a copy so `c` survives for later cells
buoyancy(4.0, 35.0)

# ## Anonymous functions
#
# A function does not need a name. The arrow form `(args) -> body` makes one on the spot — exactly how you
# hand an *initial condition* or a *forcing* to the model, as a function of space:

initial_temperature = (x, y, z) -> 4.0 + 0.01z      # warm at the surface, cooling with depth
initial_temperature(0, 0, -100)

# They are most common as an argument to another function. `map` applies one to every element and `filter`
# keeps the ones that pass a test — each takes a small anonymous function as its first argument:

map(z -> initial_temperature(0, 0, z), z_faces)     # the temperature profile on our stretched grid

filter(d -> d < -100, column)                       # keep only the interfaces below 100 m

# ## Multiple dispatch — why the stack composes
#
# A function is one *operation*; its *methods* are concrete implementations chosen by the types of the
# arguments. New types slot into existing functions, so independent packages cooperate without ever
# having been written for one another. A tiny ocean-flavored example:

abstract type Stratification end
struct Stable   <: Stratification end
struct Unstable <: Stratification end

mixing(::Stable)   = "weak diffusive mixing"
mixing(::Unstable) = "vigorous convective mixing"

mixing(Stable()), mixing(Unstable())

# Adding a *third* regime is a new type plus one method — the existing `mixing` calls, and any code built
# on top of them, are untouched:

struct Neutral <: Stratification end
mixing(::Neutral) = "mechanical mixing only"
mixing(Neutral())

# Dispatch chooses on *all* the arguments at once — something single-dispatch object systems cannot do.
# In Python a method belongs to one object, so `a.encounter(b)` dispatches on the type of `a` alone, and
# the second type becomes a manual `isinstance` ladder you have to maintain by hand:
#
# ```python
# class WarmWater:
#     def encounter(self, other):            # keyed to WarmWater (self), blind to `other`
#         if isinstance(other, SeaIce):
#             return "basal melt"
#         elif isinstance(other, ColdAir):
#             return "heat loss, then convection"
#         raise NotImplementedError          # and every new pair edits this one method
# ```
#
# Julia keys the behaviour to *both* types — one method per pair, no ladder. We can even group the types
# and their interactions into a *module*, the unit Julia uses to namespace a package; `export` makes the
# chosen names available once we bring the module into scope with `using`:

module AirSeaExchange
    export WarmWater, SeaIce, ColdAir, encounter

    struct WarmWater end
    struct SeaIce    end
    struct ColdAir   end

    encounter(::WarmWater, ::SeaIce)    = "basal melt"
    encounter(::ColdAir,   ::SeaIce)    = "ice growth"
    encounter(::ColdAir,   ::WarmWater) = "heat loss, then convection"
    encounter(::T, ::T) where {T}       = "no exchange — same medium on both sides"   # one method, all (X, X)
    encounter(a, b)                     = encounter(b, a)                              # commutative: each pair once
end

using .AirSeaExchange

# The `where {T}` method matches any medium paired with *itself*, and the last method makes `encounter`
# commutative — so the three named pairs cover both orderings, and like media need no special case:

encounter(WarmWater(), SeaIce()), encounter(SeaIce(), WarmWater()), encounter(ColdAir(), ColdAir())

# Extending the model is a new type and a single method, written from *outside* the module and touching
# nothing that already exists — a calving glacier that meets warm water, in either order:

struct Glacier end
AirSeaExchange.encounter(::Glacier, ::WarmWater) = "frontal melt and calving"

encounter(Glacier(), WarmWater()), encounter(WarmWater(), Glacier())

# This is exactly the mechanism behind the showcase: one `advection` operator serving every field type, one
# model assembled from independently-written ocean, ice, and atmosphere components — each free to add its
# own types and methods to functions it does not own.
#
# ## Composability: code that never heard of your types
#
# The real dividend of dispatch is that a function written for plain numbers keeps working on types it
# was never designed for. `Measurements.jl` supplies a number that carries an uncertainty, written `a ± b`;
# our own `buoyancy`, unchanged, propagates that uncertainty through every operation inside it:

using Measurements

buoyancy(4.0 ± 0.5, 35.0 ± 0.1)            # error bars in, error bars out — and we wrote no special code

# Neither we nor the authors of `Measurements` arranged this — it is multiple dispatch composing two
# independent pieces of code, the property the whole stack is built on. We will watch it survive a *whole
# field update*, too, once we have built one — just below.
#
# ## Putting it together: one diffusion step
#
# Here is a genuine piece of a model — periodic diffusion of the tracer `c` — written with nothing but the
# ideas above. A periodic Laplacian by array shifts, then an explicit Euler step, in place:

laplacian(c, Δx) = (circshift(c, -1) .- 2 .* c .+ circshift(c, 1)) ./ Δx^2

function diffuse!(c, κ, Δx, Δt, steps)
    for _ in 1:steps
        c .+= κ * Δt .* laplacian(c, Δx)            # one fused broadcast per step
    end
    return c
end

κ = 0.5
Δt = 0.2 * Δx^2 / κ                                 # a stable explicit time step
c_diffused = diffuse!(copy(c), κ, Δx, Δt, 500)
@sprintf("peak amplitude: %.3f → %.3f after 500 steps", maximum(c), maximum(c_diffused))

# Notice what we did *not* do: no type annotations, no manual loops over indices, no separate "fast
# version." This single function is already specialized by the compiler — and, as the last section
# shows, it already runs on a GPU.
#
# And the composability promise from above survives the field update: hand `diffuse!` a field of
# *uncertain* values and the very same function — never given a method for `Measurements` — carries an
# error bar at every grid point as it steps:

uncertain_field = c .± 0.05                 # each tracer value gains an uncertainty of ±0.05
diffuse!(uncertain_field, κ, Δx, Δt, 50)
uncertain_field[64]                         # the midpoint, with its propagated uncertainty
#
# ## Seeing it: a first figure
#
# `CairoMakie` draws figures from the same arrays. The high-wavenumber wave decays, as diffusion demands:

using CairoMakie

fig = Figure(size = (760, 380))
ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "tracer", title = "explicit diffusion after 500 steps")
lines!(ax, x, c;          linewidth = 3, label = "initial")
lines!(ax, x, c_diffused; linewidth = 3, label = "diffused")
axislegend(ax)
save("julia_essentials_diffusion.png", fig)
nothing #hide

# ![](julia_essentials_diffusion.png)
#
# ## Moving to the GPU: the array type does the work
#
# A GPU array is a different *type* with the *same* interface. Move the data with `CuArray` and every
# broadcast and `circshift` above dispatches to a GPU kernel — `diffuse!` runs unchanged. We guard on
# `CUDA.functional()` so the notebook still runs top-to-bottom on a laptop with no GPU:

using CUDA

if CUDA.functional()
    c_gpu = CuArray(c)                              # the only line that mentions the GPU
    diffuse!(c_gpu, κ, Δx, Δt, 500)                 # identical call, now on the device
    @info "GPU result agrees with CPU" agree = Array(c_gpu) ≈ c_diffused
else
    @info "No GPU here — `diffuse!` ran on the CPU; the same call runs on a device unchanged."
end

# ## Writing the loop yourself: one kernel, every device
#
# Broadcasting hides the loop. Sometimes you want to *write* it — a stencil, a bespoke update — without
# giving up that portability. [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl)
# lets you write the body once and run it on whatever backend the data lives on. A `@kernel` describes the
# work of *one* grid point; `@index(Global)` tells each launched instance which point it owns:

using KernelAbstractions

@kernel function _diffuse_step!(c, cⁿ, α, N)
    i = @index(Global)
    west = ifelse(i == 1, N, i - 1)                 # periodic wrap, branch-free
    east = ifelse(i == N, 1, i + 1)
    @inbounds c[i] = cⁿ[i] + α * (cⁿ[east] - 2cⁿ[i] + cⁿ[west])
end

# The leading underscore is the stack's convention: `_diffuse_step!` is the raw kernel, and a plain-named
# launcher wraps it. The launcher reads the backend off the data with `get_backend`, so the *same* function
# runs on a CPU `Array` or a GPU `CuArray` — the device is chosen by the type, never written by hand:

function diffuse_step!(c, α, steps)
    backend = get_backend(c)                        # CPU() for an Array, the GPU for a CuArray
    kernel! = _diffuse_step!(backend, 64)           # 64 = workgroup size
    cⁿ = similar(c)
    for _ in 1:steps
        cⁿ .= c                                     # snapshot, so the stencil reads old neighbours
        kernel!(c, cⁿ, α, length(c); ndrange = length(c))
    end
    KernelAbstractions.synchronize(backend)
    return c
end

α = κ * Δt / Δx^2
c_kernel = diffuse_step!(copy(c), α, 500)
@info "hand-written kernel matches the broadcast version" agree = c_kernel ≈ c_diffused

# One kernel, chosen hardware: this is the `@index`-based pattern the whole NumericalEarth stack uses to run
# every operator on CPUs and on GPUs from any vendor.
#
# ## How much does the GPU actually buy us?
#
# A climate model spends most of its time applying *stencils* — each cell reading its neighbours — to large
# fields: a memory-bandwidth problem, exactly where a GPU's order-of-magnitude advantage lives. We measure
# it on a 2-D Laplacian at model resolution; `BenchmarkTools` keeps the minimum over many runs, and `$`
# interpolates the data so we time the kernel, not the global lookup:

using BenchmarkTools

N = 1024
ϕ = rand(N, N)                              # a synthetic field, ~1 M cells
stencil(ϕ) = circshift(ϕ, (1, 0)) .+ circshift(ϕ, (-1, 0)) .+
             circshift(ϕ, (0, 1)) .+ circshift(ϕ, (0, -1)) .- 4 .* ϕ

cpu_time = @belapsed stencil($ϕ)

if CUDA.functional()
    ϕ_gpu = CuArray(ϕ)
    gpu_time = @belapsed CUDA.@sync stencil($ϕ_gpu)   # @sync waits for the asynchronous GPU launch
    @info @sprintf("%d×%d Laplacian — CPU %.2f ms, GPU %.3f ms, speedup %.0f×",
                   N, N, 1e3cpu_time, 1e3gpu_time, cpu_time / gpu_time)
else
    @info @sprintf("%d×%d Laplacian — CPU %.2f ms. No GPU here; on a device this stencil is typically tens of times faster.",
                   N, N, 1e3cpu_time)
end

# The same `stencil` function, the same call — only the array type changed. On GPU hardware that speedup
# is what makes eddy-resolving and large-eddy simulation affordable at all.
#
