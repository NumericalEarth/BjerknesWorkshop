#!/usr/bin/env julia
# # generate_literate_outputs.jl
#
# Turn the canonical Literate `.jl` sources under `tutorials/dayN/src/` into the
# plain executable scripts under `tutorials/dayN/scripts/` that the deployment
# runner actually launches (`Literate.script`). Optionally also emit Jupyter
# notebooks (`Literate.notebook`) when `GENERATE_NOTEBOOKS` is truthy.
#
# Which days/cases we generate for is decided by the same selection logic the
# runner uses (`selected_cases` honoring `RUN_CASES` / `RUN_DAYS`), so the two
# scripts always agree on scope. In addition to the selected *cases*, we always
# regenerate the shared helper sources for the touched days — `00_common.jl`
# (every case `include`s the generated common *script*) and, for day 4, the
# `03a_prepare_norway_topography.jl` helper that case `03` depends on. Those
# helpers are not registry cases, so they would otherwise never be emitted.
#
# Usage:
#   julia --project=. scripts/generate_literate_outputs.jl
#
# Environment:
#   RUN_CASES / RUN_DAYS    select which cases to generate scripts for
#   GENERATE_NOTEBOOKS=1    also emit .ipynb notebooks beside the scripts
#   FORCE_REGENERATE=1      regenerate even if the script is newer than its source

using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "TutorialWorkflow.jl"))
using .TutorialWorkflow

using Literate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_truthy(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")
_truthy(envvar::AbstractString, default::Bool) =
    haskey(ENV, envvar) ? _truthy(ENV[envvar]) : default

"Does `dst` need (re)generating from `src`? Yes if missing, stale, or forced."
function _needs_regen(src::AbstractString, dst::AbstractString)
    _truthy("FORCE_REGENERATE", false) && return true
    isfile(dst) || return true
    isfile(src) || return false             # nothing to do; let caller warn
    return mtime(src) > mtime(dst)
end

"""
    generate_script(src, outdir; notebook)

Run `Literate.script` (and optionally `Literate.notebook`) for a single source
file into `outdir`, returning the path to the generated script. Skips work when
the script is already up to date. `execute = false` for notebooks: we never run
the heavy science at generation time.
"""
function generate_script(src::AbstractString, outdir::AbstractString; notebook::Bool = false)
    mkpath(outdir)
    dst = joinpath(outdir, basename(src))
    if !isfile(src)
        @warn "Source missing; cannot generate" src
        return dst
    end
    if _needs_regen(src, dst)
        Literate.script(src, outdir; documenter = false, credit = false)
        @info "Generated script" src dst
    else
        @info "Script up to date; skipping" dst
    end
    if notebook
        try
            Literate.notebook(src, outdir; execute = false, documenter = false, credit = false)
            @info "Generated notebook" src nb = joinpath(outdir, replace(basename(src), r"\.jl$" => ".ipynb"))
        catch err
            @warn "Notebook generation failed (continuing)" src exception = (err, catch_backtrace())
        end
    end
    return dst
end

# Shared helper sources that are NOT registry cases but must be generated into
# the scripts dir alongside the cases of a given day, because the case scripts
# `include(joinpath(@__DIR__, "00_common.jl"))` the *generated* helper. Keyed by
# day. Paths are repo-relative.
function _helper_sources(day::Int)
    if day == 3
        return [joinpath("tutorials", "day3", "src", "00_common.jl")]
    elseif day == 4
        return [
            joinpath("tutorials", "day4", "src", "00_common.jl"),
            joinpath("tutorials", "day4", "src", "03a_prepare_norway_topography.jl"),
        ]
    else
        return String[]
    end
end

_scripts_dir(day::Int) = joinpath(ROOT, "tutorials", "day$day", "scripts")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    notebooks = _truthy("GENERATE_NOTEBOOKS", false)

    all_cases = case_registry(ROOT)
    selected = selected_cases(all_cases)

    if isempty(selected)
        @info "No cases selected for generation (RUN_CASES/RUN_DAYS); nothing to do."
        return 0
    end

    days = sort!(unique(c.day for c in selected))
    @info "Generating Literate outputs" days notebooks n_cases = length(selected)

    generated = String[]

    # 1. Per-day shared helpers (00_common, day4 03a) for every touched day.
    for day in days
        outdir = _scripts_dir(day)
        for helper in _helper_sources(day)
            src = joinpath(ROOT, helper)
            push!(generated, generate_script(src, outdir; notebook = notebooks))
        end
    end

    # 2. Selected cases.
    for case in selected
        outdir = _scripts_dir(case.day)
        src = joinpath(ROOT, case.source)
        push!(generated, generate_script(src, outdir; notebook = notebooks))
    end

    @info "Literate generation complete" n_files = length(generated)
    println()
    println("Generated scripts:")
    for g in generated
        rel = relpath(g, ROOT)
        mark = isfile(g) ? "  ok " : "MISS "
        println("  [$mark] $rel")
    end
    println()

    return 0
end

exit(main())
