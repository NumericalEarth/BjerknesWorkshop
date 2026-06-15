# # render_all_viz.jl — execute the visualization tutorials (workshop env)
#
# **Phase A of the two-phase docs build.** Run this with the *workshop* project
# (`--project=.`) — it has Oceananigans / CairoMakie / Breeze / NumericalEarth, which
# the docs env does not — *before* `docs/make.jl`. The two run as separate processes
# on purpose: the docs-env build and this heavy workshop process must not hold memory
# at the same time (the head node has little RAM).
#
# For every `tutorials/dayN/src/<name>_viz.jl` whose parent case has a cached
# successful production run, this:
#   1. stages `00_common.jl` into the output dir (Literate `cd`s there to execute, so
#      `include(joinpath(@__DIR__, "00_common.jl"))` must resolve there),
#   2. points `CASE_OUTPUT_DIR` at the case's cached production artifacts so the viz
#      loads the real production output,
#   3. renders with `Literate.markdown(...; execute = true)` — the code runs here
#      (CPU, seconds): the figure embeds inline as a base64 PNG, diagnostics print
#      inline, and no `@example` survives (Documenter never re-executes it),
#   4. base64-embeds the recorded `<video src="*.mp4">` so the page is self-contained,
#   5. writes the result to `docs/src/dayN/<name>_viz.md`, which `make.jl` appends to
#      the parent simulation page.
#
# The expensive `run!` already happened on the GPU; nothing here re-runs the LES.

import Literate
using Base64

const REPO_ROOT = abspath(get(ENV, "REPO_ROOT", pwd()))
const SRC_DIR   = joinpath(REPO_ROOT, "docs", "src")

include(joinpath(REPO_ROOT, "src", "TutorialWorkflow.jl"))
using .TutorialWorkflow

# Cached-output artifacts dir for the case whose simulation source is `simbase`
# (e.g. "05_intro_atmosphere_convection.jl"), or `nothing`.
function artifacts_dir_for(simbase)
    reg = TutorialWorkflow.case_registry(REPO_ROOT)
    idx = findfirst(c -> basename(c.source) == simbase, reg)
    idx === nothing && return nothing
    info = try TutorialWorkflow.safe_latest_success(reg[idx]; root = REPO_ROOT) catch; nothing end
    info === nothing && return nothing
    adir = isempty(info.artifacts_dir) ? joinpath(info.run_dir, "artifacts") : info.artifacts_dir
    return isdir(adir) ? adir : nothing
end

function render_one(vizfile, outdir, artifacts_dir)
    mkpath(outdir)
    # Stage the day's shared helper next to the viz only if it has one (day-4 cases
    # `include("00_common.jl")`; self-contained viz like the day-1 Breeze tutorial do not).
    common_src = joinpath(dirname(vizfile), "00_common.jl")
    staged_common = joinpath(outdir, "00_common.jl")
    isfile(common_src) && cp(common_src, staged_common; force = true)

    # The viz reads its simulation output by plain filename (e.g.
    # `FieldTimeSeries("free_convection.jld2", …)`). Literate executes inside
    # `cd(outdir)`, so stage the case's cached JLD2 there before rendering; the
    # cleanup below removes them again so they are never committed.
    for f in readdir(artifacts_dir)
        endswith(f, ".jld2") && cp(joinpath(artifacts_dir, f), joinpath(outdir, f); force = true)
    end

    withenv("CASE_OUTPUT_DIR" => artifacts_dir, "THURSDAY_REPO_ROOT" => REPO_ROOT) do
        Literate.markdown(vizfile, outdir; execute = true, credit = false,
                          flavor = Literate.DocumenterFlavor())
    end

    mdpath = joinpath(outdir, replace(basename(vizfile), r"\.jl$" => ".md"))
    md = read(mdpath, String)
    for m in collect(eachmatch(r"src=\"([^\"]+\.mp4)\"", md))
        rel = m.captures[1]
        mp4 = nothing
        # Prefer the movie the viz just recorded in `outdir`; only fall back to the
        # artifacts dir. A stale movie left in the artifacts dir must never win.
        for c in (joinpath(outdir, rel), joinpath(artifacts_dir, rel), rel)
            isfile(c) && (mp4 = c; break)
        end
        mp4 === nothing && (@warn "viz mp4 not found for embedding" rel; continue)
        md = replace(md, "src=\"$rel\"" => "src=\"data:video/mp4;base64,$(base64encode(read(mp4)))\"")
    end
    write(mdpath, md)

    rm(staged_common; force = true)
    for f in readdir(outdir; join = true)
        (endswith(f, ".jld2") || endswith(f, ".mp4")) && rm(f; force = true)
    end
    println("rendered viz → ", mdpath)
    return nothing
end

n = 0
for day in 1:9
    srcdir = joinpath(REPO_ROOT, "tutorials", "day$day", "src")
    isdir(srcdir) || continue
    outdir = joinpath(SRC_DIR, "day$day")
    for f in sort(readdir(srcdir))
        endswith(f, "_viz.jl") || continue
        only = get(ENV, "RENDER_VIZ", "")          # e.g. RENDER_VIZ=03_norway to re-render one case
        isempty(only) || occursin(only, f) || continue
        simbase = replace(f, "_viz.jl" => ".jl")
        adir = artifacts_dir_for(simbase)
        if adir === nothing
            @warn "no cached run for $(simbase); skipping its visualization"
            continue
        end
        try
            render_one(joinpath(srcdir, f), outdir, adir); global n += 1
        catch err
            @error "viz render failed" vizfile = f exception = (err, catch_backtrace())
        end
        GC.gc(true)   # free the case's FieldTimeSeries/figures before the next (low-RAM nodes)
    end
end
println("render_all_viz: rendered $n visualization page(s)")
