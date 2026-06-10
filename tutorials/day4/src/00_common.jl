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
using Dates
using CUDA
using Oceananigans

export RunConfig, output_name, slice_name, movie_name, figure_name, write_once!,
       choose_architecture, gpu_report,
       smooth_step, top_hat, lead_mask, edge_taper,
       memory_report, format_gib, run_stamp

# ## Output, movie, and figure paths
#
# `RunConfig` is a tiny bookkeeping struct: a case name plus the on-disk layout.
# Output filenames embed the case name so the cases never clobber each other.

# When the deployment workflow drives a case it sets `CASE_OUTPUT_DIR` to a
# per-run artifacts directory; every output, movie, and figure then lands there.
# Run standalone (no env var) and the defaults reproduce the original
# `thursday/{output,movies,figures}` layout. `run_class` is accepted and ignored
# so the gallery source (which passes `run_class = ...`) constructs cleanly.

_case_output_root() = get(ENV, "CASE_OUTPUT_DIR", "")

_default_output_dir() = (r = _case_output_root(); isempty(r) ? joinpath("thursday", "output") : r)
_default_movie_dir()  = (r = _case_output_root(); isempty(r) ? joinpath("thursday", "movies")  : r)
_default_figure_dir() = (r = _case_output_root(); isempty(r) ? joinpath("thursday", "figures") : r)

Base.@kwdef struct RunConfig
    case_name  :: String
    output_dir :: String = _default_output_dir()
    movie_dir  :: String = _default_movie_dir()
    figure_dir :: String = _default_figure_dir()
    run_class  :: Symbol = Symbol(get(ENV, "RUN_CLASS", "production"))
end

RunConfig(case_name::String; kw...) = RunConfig(; case_name, kw...)

function output_name(config::RunConfig, label = nothing; ext = "jld2")
    mkpath(config.output_dir)
    stem = isnothing(label) ? config.case_name : string(config.case_name, "_", label)
    return joinpath(config.output_dir, string(stem, ".", ext))
end

slice_name(config::RunConfig; ext = "jld2") = output_name(config, "slices"; ext)

function movie_name(config::RunConfig, label; ext = "mp4")
    mkpath(config.movie_dir)
    return joinpath(config.movie_dir, string(label, ".", ext))
end

function figure_name(config::RunConfig, label; ext = "png")
    mkpath(config.figure_dir)
    return joinpath(config.figure_dir, string(label, ".", ext))
end

# ## Architecture selection
#
# Production runs are GPU runs. We probe CUDA at runtime and fall back to CPU so
# the scripts still construct on a head node with no device. Importing CUDA at
# module scope (not inside the function) avoids a world-age error when querying it.

function choose_architecture()
    if CUDA.functional()
        CUDA.allowscalar(false)
        return GPU()
    else
        @warn "CUDA not functional; running on CPU."
        return CPU()
    end
end

# A one-shot device report for the top of a run log. No-op (with a note) on CPU.

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

# A tiny reproducibility stamp for the end of a run log.

function run_stamp(config::RunConfig)
    return (; case = config.case_name, finished = string(now()), host = gethostname())
end

end # module ThursdayLES

# When a case study `include`s the generated `00_common.jl` script, it gets the
# `ThursdayLES` module in scope. The cases then do `using .ThursdayLES`.
nothing #hide
