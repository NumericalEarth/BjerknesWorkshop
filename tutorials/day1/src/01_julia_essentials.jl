# # Julia essentials: enough to drive an ocean model
#
# *Monday morning — right after the showcase.*
#
# You just watched three simulations that are, underneath, short Julia scripts. This notebook gives you
# exactly the Julia you need to read and write those scripts — no more. We are not here to make you a
# software engineer; we are here to make the configuration files of the rest of the week *readable*, and
# to let you change them with confidence. Everything runs on your laptop. The last section gives a first
# taste of the GPU, which the afternoon session develops in full.
#
# We lean on four ideas, in order: **arrays and broadcasting** (how fields are represented and updated),
# **functions** (how operations are named), **multiple dispatch** (why independent pieces compose), and a
# closing **finite-difference step** that ties them together and runs unchanged on a GPU.
#
# ## Packages and the REPL
#
# Julia code lives in *packages*, activated per-project with an environment. This notebook's environment
# is `tutorials/day1`; `using` brings names into scope. We keep the imports modest:

using Statistics          # mean, std — small conveniences from the standard library
using Printf              # @sprintf, for clean numeric output

# `Pkg` manages environments. You do not need it now — the environment is already instantiated — but the
# two commands you will reach for later are `Pkg.activate("tutorials/dayN")` and `Pkg.instantiate()`.
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
# Julia keys the behaviour to *both* types — one method per pair, no ladder, and new pairs are new
# methods that touch nothing existing:

struct WarmWater end
struct SeaIce    end
struct ColdAir   end

encounter(::WarmWater, ::SeaIce)    = "basal melt"
encounter(::ColdAir,   ::SeaIce)    = "ice growth"
encounter(::ColdAir,   ::WarmWater) = "heat loss, then convection"

encounter(WarmWater(), SeaIce()), encounter(ColdAir(), WarmWater())

# This is exactly the mechanism behind the showcase: one `advection` operator serving every field type,
# one model assembled from independently-written ocean, ice, and atmosphere components.
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
# ## A first taste of the GPU
#
# A GPU array is a different *type* with the *same* interface. Move the data with `CuArray`, and every
# broadcast and `circshift` above dispatches to a GPU kernel — `diffuse!` runs unchanged. We guard on
# `CUDA.functional()` so this notebook still runs top-to-bottom on a laptop with no GPU:

using CUDA

if CUDA.functional()
    c_gpu = CuArray(c)                              # the only line that mentions the GPU
    diffuse!(c_gpu, κ, Δx, Δt, 500)                 # identical call, now on the device
    @info "GPU result agrees with CPU" agree = Array(c_gpu) ≈ c_diffused
else
    @info "No GPU here — `diffuse!` ran on the CPU. The same call runs on a device unchanged."
end

# That is the entire premise of the afternoon: write portable array (and, soon, *kernel*) code once, and
# choose the hardware with a single line at the top.
#
# ## Benchmarking: how much does the GPU actually buy us?
#
# A climate model spends most of its time applying *stencils* — each cell reading its neighbours — to
# large fields. That is a memory-bandwidth problem, exactly where a GPU's order-of-magnitude advantage
# lives. Let us measure it on a two-dimensional Laplacian at model resolution. `BenchmarkTools` runs the
# expression many times and keeps the minimum; `$` interpolates the data so we time the kernel, not the
# global lookup:

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

# The same `stencil` function, the same call — only the array type changed. On the workshop's GPU nodes
# the speedup is what makes eddy-resolving and large-eddy simulation affordable at all.
#
