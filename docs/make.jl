#!/usr/bin/env julia
#
# docs/make.jl — build the Bjerknes Workshop tutorial documentation.
#
# Pipeline (CPU only, no GPU, NEVER runs a simulation):
#
#   1. include the core workflow module (TutorialWorkflow).
#   2. for each selected DOC_DAY, render the Literate sources under
#      tutorials/dayN/src/*.jl into docs/src/dayN/*.md with
#      `Literate.markdown(...; execute = false)`.
#
#      CRITICAL: `execute = false` only stops *Literate* from running the code.
#      Documenter still EXECUTES every ```@example``` block at `makedocs` time.
#      The day-4 science sources contain `run!(simulation)`, `JLD2Writer`, and a
#      `record(...)` movie call — if those landed in @example blocks Documenter
#      would launch the H100 LES during the doc build. So the Literate
#      `postprocess` here DEMOTES every generated ```@example``` fence to a plain,
#      non-executed ```julia``` fence. The narrative and code are shown verbatim
#      but nothing in the tutorial body ever runs at build time.
#
#   3. append, to each rendered day-4 page, a "Results" section built from a
#      small hidden ```@setup``` block plus ```@example``` blocks that ONLY call
#      safe loaders (`safe_latest_success`, `safe_artifact`). Those helpers embed
#      figures/movies from the latest *successful* cached run, copying the file
#      into the page build directory and returning a Markdown image/video link;
#      when an artifact is missing they render a colored admonition card
#      (success / warning "previous success" / danger "no output" /
#      "movie unavailable") instead of throwing. No `@assert isfile`, ever.
#
#   4. write the status summary pages (write_summary_pages!).
#   5. makedocs(...) with remotes = nothing, warnonly = true,
#      size_threshold = nothing (figure pages exceed the default cap).
#
# Environment knobs:
#   DOC_DAYS  — which days to document (all | 3 | 4 | 3,4 | none). Default all.

using Documenter
using Literate

const DOCS_DIR = @__DIR__
const REPO_ROOT = abspath(joinpath(DOCS_DIR, ".."))
const SRC_DIR = joinpath(DOCS_DIR, "src")

# Mark that we are in the doc-render phase, never the run phase. Any tutorial
# code that (despite the demotion below) checks this can guard itself.
ENV["DOCS_PHASE"] = "render"

include(joinpath(REPO_ROOT, "src", "TutorialWorkflow.jl"))
using .TutorialWorkflow
using Base64

# ============================================================================
# Literate postprocess: neutralize execution.
# ============================================================================

# Turn every `@example <name>` / `@repl` / `@eval` fence emitted by Literate's
# DocumenterFlavor into a plain `julia` fence so Documenter renders but does NOT
# execute it. This is what stops the LES from launching at build time.
#
# IMPORTANT: Literate (and CommonMark) emit a code fence with *at least* three
# backticks, and use MORE than three whenever the fenced content itself contains
# a run of backticks (e.g. nested ```@raw``` / docstrings). A day-4 science page
# routinely comes out with four-backtick ````@example```` fences. Matching only
# the literal "```@example" prefix therefore MISSES those fences, leaving
# `run!(simulation)` / `JLD2Writer` / `record(...)` in live @example blocks that
# Documenter would execute at build time. So we match a fence of N≥3 backticks
# followed by the Documenter block tag and rewrite ONLY the tag to `julia`,
# preserving the backtick run length so the matching closing fence still pairs.
const _EXEC_FENCE = r"^(`{3,})@(example|repl|eval)\b.*$"

function demote_example_blocks(content::AbstractString)
    out = IOBuffer()
    for line in eachline(IOBuffer(content); keep = true)
        stripped = rstrip(line)
        m = match(_EXEC_FENCE, stripped)
        if m !== nothing
            # opening fence of an executed block -> same-width plain julia fence
            println(out, m.captures[1] * "julia")
        else
            print(out, line)
        end
    end
    return String(take!(out))
end

# Because nothing executes at build time, the inline `![](figure.png)` references in
# the Literate sources (which point at files the *script* produces in its working
# directory) would render as broken images. Drop those lines; the cached figures and
# movies are embedded by the appended Results section instead.
const _LOCAL_IMAGE_LINE = r"^!\[[^\]]*\]\((?!https?://)[^)]+\)\s*$"

function strip_local_images(content::AbstractString)
    out = IOBuffer()
    for line in eachline(IOBuffer(content); keep = true)
        occursin(_LOCAL_IMAGE_LINE, rstrip(line)) || print(out, line)
    end
    return String(take!(out))
end

# For day-2 (and any INLINE_ASSET_DAYS) the Literate source writes its own figures/movies in the script's
# working directory and references them with `![](file)`. Rather than copy those files (whose relative
# links are fragile under prettyurls), replace each reference with a base64 data-URI <video>/<img> in a
# raw-HTML block — the same self-contained embedding the cached-artifact Results sections use. Returns a
# postprocess closure bound to the script's working directory.
function embed_local_assets(assetdir::AbstractString)
    return function (content::AbstractString)
        out = IOBuffer()
        for line in eachline(IOBuffer(content); keep = true)
            m = match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", rstrip(line))
            if m === nothing || startswith(m.captures[1], "http")
                print(out, line)
                continue
            end
            ref = m.captures[1]
            asset = joinpath(assetdir, ref)
            if !isfile(asset)
                @warn "inline asset not found; leaving reference as-is" ref asset
                print(out, line)
                continue
            end
            ext = lowercase(splitext(ref)[2])
            data = base64encode(read(asset))
            if ext in (".mp4", ".webm", ".ogg")
                mime = ext == ".mp4" ? "video/mp4" : ext == ".webm" ? "video/webm" : "video/ogg"
                print(out, "```@raw html\n<video controls width=\"100%\" src=\"data:", mime,
                      ";base64,", data, "\"></video>\n```\n")
            else
                mime = ext in (".jpg", ".jpeg") ? "image/jpeg" : ext == ".gif" ? "image/gif" : "image/png"
                print(out, "```@raw html\n<img style=\"max-width:100%;height:auto\" src=\"data:", mime,
                      ";base64,", data, "\">\n```\n")
            end
        end
        return String(take!(out))
    end
end

# ============================================================================
# Results section appended to each day-N page (safe, cached-only embedding).
# ============================================================================

# The setup block is hidden (```@setup```), runs at build time on CPU, and only
# does file IO: it reads the JSON status pointers and copies cached artifacts
# into the page build directory. It defines `results_md(slug)` which the visible
# ```@example``` block renders via Markdown.parse. Nothing here runs a sim.
#
# `slug => (figures, movies, title)` describing which optional artifacts to try
# to embed for each case. Figures embed as images; movies as an HTML5 <video>
# (with a download link fallback). Missing files become admonition cards.
const _RESULTS_SPEC = Dict(
    "gpu_computing" => (
        title = "GPU computing and a 2D turbulence solver",
        figures = String[],
        movies = ["two_dimensional_turbulence.mp4"],
    ),
    "distributed_convection" => (
        title = "Distributed nonhydrostatic LES",
        figures = String[],
        movies = String[],
    ),
    "internal_tide" => (
        title = "Internal tide over a sill",
        figures = ["internal_tide_domain.png"],
        movies = ["internal_tide.mp4"],
    ),
    "baroclinic_instability" => (
        title = "Baroclinic instability in a channel",
        figures = ["baroclinic_instability_energy.png"],
        movies = ["baroclinic_instability.mp4"],
    ),
    "capsizing_iceberg" => (
        title = "Capsizing iceberg",
        figures = ["iceberg_tilt.png"],
        movies = ["capsizing_iceberg.mp4"],
    ),
    "lead_atmosphere" => (
        title = "Atmospheric turbulence over a sea-ice lead",
        figures = ["atmosphere_lead_final_slice.png"],
        movies = ["lead_atmosphere_plume.mp4"],
    ),
    "lead_ocean_waves" => (
        title = "Ocean turbulence below a lead with surface waves",
        figures = ["ocean_lead_nowaves_final_slice.png", "ocean_lead_waves_final_slice.png"],
        movies = ["ocean_lead_nowaves.mp4", "ocean_lead_waves.mp4"],
    ),
    "norway_100m" => (
        title = "Norway 100 m prescribed fluxes",
        figures = ["norway_final_w_slice.png"],
        movies = ["norway_100m_prescribed_fluxes.mp4"],
    ),
    "intro_atmosphere" => (
        title = "Intro: 2D atmospheric free convection",
        figures = ["intro_atmosphere_convection_final.png"],
        movies = ["intro_atmosphere_convection.mp4"],
    ),
    "intro_ocean" => (
        title = "Intro: 2D ocean free convection",
        figures = ["intro_ocean_convection_final.png"],
        movies = ["intro_ocean_convection.mp4"],
    ),
    "intro_coupled" => (
        title = "Intro: 2D coupled air–sea convection",
        figures = ["intro_coupled_convection_final.png"],
        movies = ["intro_coupled_convection.mp4"],
    ),
    "warm_filament" => (
        title = "A warm filament writes a cloud street",
        figures = ["coupled_warm_filament_final.png"],
        movies = ["coupled_warm_filament.mp4"],
    ),
    "smoke_case" => (
        title = "Smoke case",
        figures = ["summary.png"],
        movies = String[],
    ),
    "hybrid_physics_ml" => (
        title = "Hybrid physics + ML",
        figures = ["hybrid_physics_ml.png"],
        movies = String[],
    ),
    "differentiable_esms" => (
        title = "Differentiable ESMs",
        figures = ["differentiable_esms.png"],
        movies = String[],
    ),
)

# Append a Results section to a rendered page for the given slug. The page lives
# at docs/src/dayN/<file>.md; figures get copied alongside it at build time.
function append_results_section!(page_md::AbstractString, slug::AbstractString)
    spec = get(_RESULTS_SPEC, slug, nothing)
    spec === nothing && return nothing

    reg = TutorialWorkflow.case_registry(REPO_ROOT)
    idx = findfirst(c -> c.slug == slug, reg)
    case = idx === nothing ? nothing : reg[idx]

    _admon(kind, title, body) =
        string("!!! ", kind, " \"", title, "\"\n\n    ", body, "\n\n")
    _safe(f) = try f() catch; nothing end

    io = IOBuffer()
    print(io, "\n## Results\n\n")
    print(io, "The figures and movies below come from the most recent **successful** run of ",
              "this case recorded by the deployment workflow. Nothing here launches a ",
              "simulation; the docs build only loads cached artifacts and embeds them inline.\n\n")

    # Status banner (static, evaluated now at make.jl time).
    ok = case === nothing ? nothing : _safe(() -> TutorialWorkflow.safe_latest_success(case; root = REPO_ROOT))
    if ok !== nothing && ok.status == "success"
        cur = something(_safe(() -> TutorialWorkflow.outputs_are_current(case; root = REPO_ROOT)), false)
        when = isempty(ok.finished) ? ok.started : ok.finished
        note = cur ? "Outputs are current." :
            "Outputs are from a previous configuration (parameters/source/manifest changed)."
        print(io, _admon("info", "Last successful run: " * ok.run_id, "Finished " * when * ". " * note))
    else
        attempt = case === nothing ? nothing : _safe(() -> TutorialWorkflow.latest_attempt(case; root = REPO_ROOT))
        if attempt === nothing
            print(io, _admon("danger", "No run yet",
                "This case has not been run by the deployment workflow."))
        else
            print(io, _admon("warning", "No successful run (last status: " * attempt.status * ")",
                "Showing last-known-good artifacts below if any exist."))
        end
    end

    _artifact(name) = case === nothing ? nothing :
        _safe(() -> TutorialWorkflow.safe_artifact(case, name; root = REPO_ROOT))

    # Figures: inline as base64 data-URI images so they always render in the
    # built site regardless of Documenter's prettyurls asset handling.
    for name in spec.figures
        src = _artifact(name)
        if src === nothing || !isfile(src)
            print(io, _admon("warning", "Figure unavailable",
                "`" * name * "` was not found in the latest successful run."))
        else
            # Use raw HTML (not markdown ![](...)) so Documenter does not try to
            # resolve the data URI as a local file path.
            print(io, "```@raw html\n<img alt=\"", name,
                  "\" style=\"max-width:100%;height:auto\" src=\"data:image/png;base64,",
                  base64encode(read(src)), "\">\n```\n\n")
        end
    end

    # Movies: inline as a base64 data-URI <video> in a raw-HTML block.
    for name in spec.movies
        src = _artifact(name)
        if src === nothing || !isfile(src)
            print(io, _admon("warning", "Movie unavailable",
                "`" * name * "` was not found in the latest successful run."))
        else
            print(io, "```@raw html\n<video controls width=\"100%\" src=\"data:video/mp4;base64,",
                  base64encode(read(src)), "\"></video>\n```\n\n")
        end
    end

    open(page_md, "a") do f
        write(f, String(take!(io)))
    end
    return nothing
end

# ============================================================================
# Render the Literate sources for the selected days.
# ============================================================================

# slug lookup by generated/source file, so we know which Results spec to append.
function _slug_for_source(source_file::AbstractString)
    reg = TutorialWorkflow.case_registry(REPO_ROOT)
    base = basename(source_file)
    for c in reg
        basename(c.source) == base && return c.slug
    end
    return nothing
end

# Days whose tutorial pages embed their own figures/movies inline: the Literate source writes them in its
# working directory (tutorials/dayN/) and references them with `![](file)`. These bypass the
# cached-artifact Results section; the references are base64-embedded at render time. A source is skipped
# until every asset it references exists, so a page only publishes once its movie has been produced.
const INLINE_ASSET_DAYS = (1, 2)

function _inline_assets_ready(source::AbstractString, assetdir::AbstractString)
    for m in eachmatch(r"!\[[^\]]*\]\(([^)]+)\)", read(source, String))
        ref = m.captures[1]
        startswith(ref, "http") && continue
        isfile(joinpath(assetdir, ref)) || return (false, ref)
    end
    return (true, "")
end

# Day-N page titles (the Literate `# # Title` first line becomes the page H1).
function render_day(day::Int)
    srcdir = joinpath(REPO_ROOT, "tutorials", "day$day", "src")
    outdir = joinpath(SRC_DIR, "day$day")
    isdir(srcdir) || return String[]
    rm(outdir; force = true, recursive = true)   # drop stale pages from earlier structures
    mkpath(outdir)

    inline = day in INLINE_ASSET_DAYS
    assetdir = joinpath(REPO_ROOT, "tutorials", "day$day")
    postprocess = inline ? embed_local_assets(assetdir) ∘ demote_example_blocks :
                           strip_local_images ∘ demote_example_blocks

    pages = String[]
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)))
    for f in files
        # Skip the shared infrastructure include (00_common) and the topo prep
        # helper (03a) — they are includes / data prep, not standalone pages.
        startswith(f, "00_") && continue
        startswith(f, "03a_") && continue

        source = joinpath(srcdir, f)

        if inline
            ready, missing_ref = _inline_assets_ready(source, assetdir)
            if !ready
                @info "Skipping $f until its inline asset exists" missing_ref
                continue
            end
        end

        Literate.markdown(source, outdir;
            execute = false,
            credit = false,
            flavor = Literate.DocumenterFlavor(),
            postprocess = postprocess)

        mdname = replace(f, r"\.jl$" => ".md")
        mdpath = joinpath(outdir, mdname)

        if !inline
            slug = _slug_for_source(f)
            slug === nothing || append_results_section!(mdpath, slug)
        end

        push!(pages, joinpath("day$day", mdname))
    end
    return pages
end

doc_days = selected_doc_days()
@info "Documenting days" doc_days

day_pages = Dict{Int,Vector{String}}()
for d in doc_days
    day_pages[d] = render_day(d)
end

# ============================================================================
# Status summary pages (docs/src/status/{index,day3,day4}.md).
# ============================================================================

write_summary_pages!(case_registry(REPO_ROOT); root = REPO_ROOT)

# ============================================================================
# Assemble the navigation tree.
# ============================================================================

function _nav_for_day(day::Int)
    pages = get(day_pages, day, String[])
    isempty(pages) && return nothing
    label = day == 1 ? "Day 1 — GPU computing" :
            day == 2 ? "Day 2 — One day in the high-latitude ocean" :
            day == 3 ? "Day 3 — Hybrid physics & differentiable ESMs" :
            day == 4 ? "Day 4 — Boundary heterogeneity & turbulence" :
            "Day $day"
    return label => pages
end

pages = Any["Home" => "index.md"]
for d in sort(collect(keys(day_pages)))
    nav = _nav_for_day(d)
    nav === nothing || push!(pages, nav)
end

status_pages = ["Overview" => "status/index.md"]
for d in sort(collect(keys(day_pages)))
    sp = joinpath("status", "day$d.md")
    isfile(joinpath(SRC_DIR, sp)) && push!(status_pages, "Day $d" => sp)
end
push!(pages, "Run status" => status_pages)

# ============================================================================
# Build.
# ============================================================================

makedocs(
    sitename = "Bjerknes Workshop Tutorials",
    authors = "Bjerknes Workshop",
    root = DOCS_DIR,
    source = "src",
    build = "build",
    clean = true,
    modules = Module[],            # not documenting a package's docstrings
    doctest = false,
    checkdocs = :none,
    warnonly = true,               # missing refs/cache warn, don't abort
    remotes = nothing,             # non-package repo: avoid remote-link errors
    format = Documenter.HTML(
        prettyurls = true,
        size_threshold = nothing,  # figure pages exceed the default cap
        edit_link = nothing,
        repolink = nothing,
        assets = String[],
    ),
    pages = pages,
)

@info "Documentation built" build_dir = joinpath(DOCS_DIR, "build")
