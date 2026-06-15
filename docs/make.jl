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
        title = "Intro: a first taste of the atmosphere",
        figures = ["thermal_bubble.png", "free_convection.png", "lee_waves.png", "mountain_clouds.png"],
        movies = ["thermal_bubble.mp4", "free_convection.mp4", "lee_waves.mp4", "mountain_clouds.mp4"],
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

# Pedagogical page order per day (by filename stem). The filenames were numbered
# in authoring order, not teaching order — the intro convection trio (05–07) was
# added after the advanced cases (01–03), so a plain filename sort buries "A first
# taste of convection" mid-list. List the intended order here; files not listed
# sort after, alphabetically. Other days fall back to filename order.
const _PAGE_ORDER = Dict(
    4 => ["07_intro_coupled_convection",             # 2D coupled
          "01_atmospheric_turbulence_over_a_sea_ice_lead",
          "02_ocean_turbulence_below_a_lead_with_surface_waves",
          "08_coupled_warm_filament",                # flagship coupled
          "03_norway_100m_prescribed_fluxes",        # flagship terrain
          "04_gallery_and_discussion",
          "99_smoke_case"],
)

function _page_rank(day::Int, f::AbstractString)
    order = get(_PAGE_ORDER, day, String[])
    stem = replace(f, r"\.jl$" => "")
    i = findfirst(==(stem), order)
    return i === nothing ? (length(order) + 1, stem) : (i, "")
end

# Days whose tutorial pages embed their own figures/movies inline: the Literate source writes them in its
# working directory (tutorials/dayN/) and references them with `![](file)`. These bypass the
# cached-artifact Results section; the references are base64-embedded at render time. A source is skipped
# until every asset it references exists, so a page only publishes once its movie has been produced.
const INLINE_ASSET_DAYS = (1, 2)

# Individual sources that inline-embed their own assets even outside INLINE_ASSET_DAYS — the global-ocean
# sim moved to day 4 but, like the day-1/2 examples, embeds the movie it writes in its working directory:
const INLINE_ASSET_SOURCES = ("09_global_ocean.jl",)

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
    # Drop stale pages from earlier structures, but PRESERVE the `*_viz.md` pages that
    # Phase A (scripts/render_all_viz.jl) just rendered — render_day appends them below.
    # (A blanket `rm(outdir)` here wipes those before they can be attached, dropping every
    # tutorial's movies.)
    if isdir(outdir)
        for x in readdir(outdir; join = true)
            endswith(x, "_viz.md") || rm(x; force = true, recursive = true)
        end
    end
    mkpath(outdir)

    assetdir = joinpath(REPO_ROOT, "tutorials", "day$day")

    pages = String[]
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)); by = f -> _page_rank(day, f))
    for f in files
        # Skip the shared infrastructure include (00_common), the topo prep helper
        # (03a), and the `_viz.jl` visualization halves (rendered into their parent
        # simulation page, not as standalone pages).
        startswith(f, "00_") && continue
        startswith(f, "03a_") && continue
        endswith(f, "_viz.jl") && continue
        # 04_gpu_computing is retained as a script but dropped from the day-1 lineup.
        f == "04_gpu_computing.jl" && continue

        source = joinpath(srcdir, f)

        # Inline-embed a source's own figures/movies for whole INLINE_ASSET_DAYS, or for individual
        # INLINE_ASSET_SOURCES (the global-ocean sim moved to day 4 but still embeds its movie).
        # EXCEPTION: a source with a `<stem>_viz.jl` sibling uses the render_all_viz viz-page
        # mechanism (e.g. the day-1 Breeze tutorial), so it is NOT inline even within an
        # INLINE_ASSET_DAY — otherwise its rendered movies would never be appended.
        has_viz = isfile(joinpath(srcdir, replace(f, r"\.jl$" => "_viz.jl")))
        inline = !has_viz && ((day in INLINE_ASSET_DAYS) || (f in INLINE_ASSET_SOURCES))
        postprocess = inline ? embed_local_assets(assetdir) ∘ demote_example_blocks :
                               strip_local_images ∘ demote_example_blocks

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

            # If `scripts/render_all_viz.jl` (Phase A, workshop env) pre-rendered an
            # executed visualization page for this case, append it (it carries the inline
            # figure, diagnostics, and base64 animation). Otherwise fall back to the static
            # cached-artifact Results section.
            vizmd = joinpath(outdir, replace(f, r"\.jl$" => "_viz.md"))
            if isfile(vizmd)
                open(mdpath, "a") do io
                    println(io)
                    write(io, read(vizmd, String))
                end
                rm(vizmd; force = true)
            elseif slug !== nothing
                append_results_section!(mdpath, slug)
            end
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
    label = day == 1 ? "Day 1 — Julia and interactive Earth system modeling" :
            day == 2 ? "Day 2 — Realistic simulations using Julia" :
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

# Strip base64 data-URI payloads from the search index. We embed figures/movies as
# base64 in `@raw html` blocks (so Documenter doesn't mistake data URIs for file
# paths), but Documenter then indexes those megabyte-scale strings into
# `search_index.js` — pushing it past GitHub's 100 MB single-file limit on gh-pages.
# Nobody searches base64, so dropping the payloads (keeping the surrounding text)
# leaves search fully functional and shrinks the index by ~1000×.
let search_index = joinpath(DOCS_DIR, "build", "search_index.js")
    if isfile(search_index)
        before = filesize(search_index)
        text = read(search_index, String)
        text = replace(text, r"data:(?:image/[a-z]+|video/mp4);base64,[A-Za-z0-9+/=]+" => "")
        write(search_index, text)
        @info "Stripped base64 from search index" MB_before = round(before/1e6; digits=1) MB_after = round(filesize(search_index)/1e6; digits=2)
    end
end
