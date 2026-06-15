# # Thursday shared infrastructure
#
# *Boundary heterogeneity writes turbulence into the fluid.*
#
# This file collects the utilities reused by all three Thursday case studies:
#
# 1. **A crack in the ice** — atmospheric turbulence over a sea-ice lead (Breeze).
# 2. **Beneath the crack** — ocean turbulence, brine rejection, and waves (Oceananigans).
# 3. **Fjords as boundary conditions** — 100 m atmospheric flow over coastal Norway (Breeze, terrain-following).
#
# It is written as a Literate source: it renders to a script, a notebook, and a
# documentation page, but the canonical artifact is this `.jl` file. The other
# case studies `include` the generated *script*, so everything here must run
# top-to-bottom with no plotting side effects.
#
# Each case study is hardcoded to run a single, developed configuration sized to
# complete on one H100 — there are no run-class or environment-variable knobs. The
# `ThursdayLES` module below holds the smooth lead/land masks that encode boundary
# heterogeneity, architecture selection, and memory/output bookkeeping.

module ThursdayLES

using Printf
using CUDA
using Oceananigans

export write_once!,
       gpu_report,
       smooth_step, top_hat, lead_mask, edge_taper,
       memory_report, format_gib, bilinear

# Each case writes its output with a plain, descriptive filename (e.g.
# `"free_convection.jld2"`) into the current working directory — exactly as a
# standalone Oceananigans/Breeze script would. The deployment workflow runs each
# case from its own artifacts directory, so those bare filenames land there with
# no path bookkeeping; run a case by hand and the files appear next to you.

# A one-shot device report for the top of a run log. (The cases construct their
# architecture directly with `GPU()` — every Thursday run is a GPU run.)

function gpu_report()
    if CUDA.functional()
        dev = CUDA.device()
        @info "CUDA device" name = CUDA.name(dev) total_memory_GiB = CUDA.totalmem(dev) / 2^30
        CUDA.versioninfo()
    else
        @info "CUDA not functional — running on CPU."
    end
    return nothing
end

# Force a single write from an output writer (e.g. to snapshot static fields once
# before time stepping). `write_output!` lives in a submodule and is not exported.
write_once!(writer, model) = Oceananigans.OutputWriters.write_output!(writer, model)

# ## Boundary heterogeneity: smooth masks
#
# The central Thursday idea is that the *boundary* writes structure into the fluid.
# We encode boundaries — a sea-ice lead, a coastline — as smooth masks in `[0, 1]`
# rather than sharp step functions, so the surface forcing stays well-resolved on
# the LES grid and does not seed grid-scale noise at the edges.
#
# `smooth_step(r, δ)` is a `tanh` ramp of width `δ` centered at `r = 0`.

@inline smooth_step(r, δ) = (1 + tanh(r / δ)) / 2

# `top_hat` is the product of two opposing ramps: ≈ 1 inside a band of the given
# `width` centered at `center`, falling smoothly to 0 over a transition of width
# `edge` on each side.

@inline function top_hat(x; center = 0, width, edge)
    rising  = smooth_step(x - (center - width / 2), edge)
    falling = smooth_step((center + width / 2) - x, edge)
    return rising * falling
end

# `lead_mask` is the canonical "crack in the ice": a top-hat in the across-lead
# direction `x`, uniform along the lead in `y`. The same mask multiplies every
# surface flux, so a single function defines the geometry of the heterogeneity.

@inline lead_mask(x, y = 0; center = 0, width, edge) = top_hat(x; center, width, edge)

# `edge_taper` smoothly suppresses a field within `taper_width` of the domain edges
# in `x` and `y`. The Norway case uses it to turn the outer rim of a periodic domain
# into a numerical buffer.

@inline function edge_taper(x, y, Lx, Ly; taper_width)
    tx = smooth_step(x + Lx/2 - taper_width, taper_width/3) *
         smooth_step(Lx/2 - taper_width - x, taper_width/3)
    ty = smooth_step(y + Ly/2 - taper_width, taper_width/3) *
         smooth_step(Ly/2 - taper_width - y, taper_width/3)
    return tx * ty
end

# ## Gridded-data lookup
#
# `bilinear(arr, xs, ys)` returns a function `(x, y) -> value` that bilinearly
# interpolates the array `arr` defined on the (sorted, evenly spaced) coordinate
# vectors `xs`, `ys`, clamping to the domain edges. The Norway case uses it to
# evaluate cached topography / land-mask arrays at arbitrary grid points — both
# when carving the terrain (simulation) and when reconstructing the static fields
# for the figures (visualization).

function bilinear(arr, xs, ys)
    x0, x1 = first(xs), last(xs); y0, y1 = first(ys), last(ys)
    nx, ny = length(xs), length(ys)
    dx = (x1 - x0) / (nx - 1); dy = (y1 - y0) / (ny - 1)
    return function (x, y)
        fx = clamp((x - x0) / dx, 0, nx - 1 - 1e-6)
        fy = clamp((y - y0) / dy, 0, ny - 1 - 1e-6)
        i = floor(Int, fx) + 1; j = floor(Int, fy) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        @inbounds (arr[i, j]   * (1 - tx) * (1 - ty) + arr[i+1, j]   * tx * (1 - ty) +
                   arr[i, j+1] * (1 - tx) * ty       + arr[i+1, j+1] * tx * ty)
    end
end

# ## Memory accounting
#
# Before committing to a grid we estimate the device memory a run will use.
# `memory_report` reports per-field and total field memory for an `Nx×Ny×Nz` grid;
# the *working* memory of a real run is several times this.

function memory_report(Nx, Ny, Nz; FT = Float32, nfields = 8, working_multiplier = 4)
    cells = Nx * Ny * Nz
    gib_per_field = cells * sizeof(FT) / 2^30
    field_gib = nfields * gib_per_field
    working_gib = working_multiplier * field_gib
    @info "Memory estimate" Nx Ny Nz cells million_cells = cells / 1e6 FT nfields gib_per_field=format_gib(gib_per_field) prognostic_field_GiB=format_gib(field_gib) estimated_working_GiB=format_gib(working_gib)
    return (; cells, gib_per_field, field_gib, working_gib)
end

format_gib(x) = @sprintf("%.2f GiB", x)

end # module ThursdayLES

# When a case study `include`s the generated `00_common.jl` script, it gets the
# `ThursdayLES` module in scope. The cases then do `using .ThursdayLES`.
nothing #hide
