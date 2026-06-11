#!/usr/bin/env julia
#
# Standalone single-page Literate → Documenter build, for publishing one tutorial (with its movie) as a
# self-contained GitHub-Pages page. Nothing executes at build time — the code is shown verbatim and the
# cached figures/movie are embedded from the assets directory.
#
# Usage:
#   julia --project=docs docs/standalone.jl <source.jl> <page_dir> <assets_dir> "<site title>"
#
# e.g.  julia --project=docs docs/standalone.jl \
#           tutorials/day2/src/02_barents_sea_regional.jl barents_page tutorials/day2 "The Barents Sea"

using Literate, Documenter

source, page_dir, assets_dir, sitename = ARGS[1], ARGS[2], ARGS[3], ARGS[4]

src_out = joinpath(page_dir, "src")
rm(page_dir; force = true, recursive = true)
mkpath(src_out)

# Demote every executable Documenter fence (```@example / @repl / @eval, of any backtick width) to a plain
# ```julia fence, so Documenter renders the code but never runs the (GPU) simulation at build time:
const EXEC_FENCE = r"^(`{3,})@(example|repl|eval)\b.*$"

function demote_examples(content::AbstractString)
    out = IOBuffer()
    for line in eachline(IOBuffer(content); keep = true)
        m = match(EXEC_FENCE, rstrip(line))
        m === nothing ? print(out, line) : println(out, m.captures[1] * "julia")
    end
    return String(take!(out))
end

Literate.markdown(source, src_out;
                  name = "index", execute = false, credit = false,
                  flavor = Literate.DocumenterFlavor(), postprocess = demote_examples)

# Copy every local asset the page references (`![](file)`) from assets_dir into the page source tree, so
# Documenter finds the figures and movie and renders the .mp4 as an HTML5 <video>:
markdown = read(joinpath(src_out, "index.md"), String)
for m in eachmatch(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    ref = m.captures[1]
    startswith(ref, "http") && continue
    asset = joinpath(assets_dir, ref)
    isfile(asset) ? cp(asset, joinpath(src_out, ref); force = true) : @warn "asset not found" ref asset
end

makedocs(; sitename,
         root = abspath(page_dir), source = "src", build = "build", clean = true,
         remotes = nothing, warnonly = true, doctest = false, checkdocs = :none,
         format = Documenter.HTML(prettyurls = false, size_threshold = nothing,
                                  edit_link = nothing, repolink = nothing, assets = String[]),
         pages = ["Home" => "index.md"])

@info "Standalone page built" page = abspath(joinpath(page_dir, "build", "index.html"))
