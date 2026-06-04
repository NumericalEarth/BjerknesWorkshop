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

    figs = repr(spec.figures)
    movs = repr(spec.movies)

    # NOTE: the @setup block is evaluated by Documenter in the page build dir.
    # It uses only TutorialWorkflow + Base file IO. REPO_ROOT is interpolated as
    # a string literal so the helpers can locate the status pointers.
    setup = """

    ## Results

    The figures and movies below come from the most recent **successful** run of
    this case recorded by the deployment workflow. If a run has not completed (or
    its artifacts are missing) a status card is shown instead — the documentation
    build never launches a simulation.

    ```@setup $(slug)_results
    import Main.TutorialWorkflow as TW

    const _REPO_ROOT = raw"$(REPO_ROOT)"
    const _SLUG = "$(slug)"
    const _FIGURES = $(figs)
    const _MOVIES = $(movs)

    _case() = begin
        reg = TW.case_registry(_REPO_ROOT)
        idx = findfirst(c -> c.slug == _SLUG, reg)
        idx === nothing ? nothing : reg[idx]
    end

    # Copy a cached artifact into the page build dir; return its basename (a
    # path relative to the rendered page) or nothing if unavailable.
    function _localize(name)
        c = _case()
        c === nothing && return nothing
        src = TW.safe_artifact(c, name; root = _REPO_ROOT)
        src === nothing && return nothing
        dst = joinpath(@__DIR__, name)
        try
            cp(src, dst; force = true)
        catch
            return nothing
        end
        return name
    end

    _admonition(kind, title, body) =
        string("!!! ", kind, " \\"", title, "\\"\\n\\n    ", body, "\\n")

    # Status banner from the latest success / latest attempt pointers.
    function _banner()
        c = _case()
        c === nothing && return _admonition("danger", "Unknown case",
            "No registry entry for `" * _SLUG * "`.")
        ok = TW.safe_latest_success(c; root = _REPO_ROOT)
        if ok !== nothing && ok.status == "success"
            cur = TW.outputs_are_current(c; root = _REPO_ROOT)
            when = isempty(ok.finished) ? ok.started : ok.finished
            note = cur ? "Outputs are current." :
                "Outputs are from a previous configuration (parameters/source/manifest changed)."
            return _admonition("info", "Last successful run: " * ok.run_id,
                "Finished " * when * ". " * note)
        end
        attempt = try
            TW.latest_attempt(c; root = _REPO_ROOT)
        catch
            nothing
        end
        if attempt === nothing
            return _admonition("danger", "No run yet",
                "This case has not been run by the deployment workflow.")
        else
            return _admonition("warning",
                "No successful run (last status: " * attempt.status * ")",
                "Showing the last known good artifacts if any exist below.")
        end
    end

    function _figure_md(name)
        local rel = _localize(name)
        rel === nothing && return _admonition("warning", "Figure unavailable",
            "`" * name * "` was not found in the latest successful run.")
        return string("![", name, "](", rel, ")\\n")
    end

    function _movie_md(name)
        local rel = _localize(name)
        rel === nothing && return _admonition("warning", "Movie unavailable",
            "`" * name * "` was not found in the latest successful run.")
        return string(
            "```@raw html\\n",
            "<video controls width=\\"100%\\" src=\\"", rel, "\\"></video>\\n",
            "```\\n",
            "[Download ", name, "](", rel, ")\\n")
    end

    function results_md()
        io = IOBuffer()
        print(io, _banner(), "\\n")
        for f in _FIGURES
            print(io, _figure_md(f), "\\n")
        end
        for m in _MOVIES
            print(io, _movie_md(m), "\\n")
        end
        return String(take!(io))
    end
    nothing
    ```

    ```@example $(slug)_results
    using Markdown
    Markdown.parse(results_md())
    ```
    """

    open(page_md, "a") do io
        # de-indent the heredoc (it is indented by 4 for readability above)
        for line in eachline(IOBuffer(setup); keep = true)
            print(io, replace(line, r"^    " => ""))
        end
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

# Day-N page titles (the Literate `# # Title` first line becomes the page H1).
function render_day(day::Int)
    srcdir = joinpath(REPO_ROOT, "tutorials", "day$day", "src")
    outdir = joinpath(SRC_DIR, "day$day")
    isdir(srcdir) || return String[]
    mkpath(outdir)

    pages = String[]
    files = sort(filter(f -> endswith(f, ".jl"), readdir(srcdir)))
    for f in files
        # Skip the shared infrastructure include (00_common) and the topo prep
        # helper (03a) — they are includes / data prep, not standalone pages.
        startswith(f, "00_") && continue
        startswith(f, "03a_") && continue

        source = joinpath(srcdir, f)
        Literate.markdown(source, outdir;
            execute = false,
            credit = false,
            flavor = Literate.DocumenterFlavor(),
            postprocess = demote_example_blocks)

        mdname = replace(f, r"\.jl$" => ".md")
        mdpath = joinpath(outdir, mdname)

        slug = _slug_for_source(f)
        slug === nothing || append_results_section!(mdpath, slug)

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
    label = day == 3 ? "Day 3 — Hybrid physics & differentiable ESMs" :
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
