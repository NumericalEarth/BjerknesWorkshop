# # Smoke case: a tiny end-to-end exercise of the deployment pipeline
#
# *Boundary heterogeneity writes turbulence into the fluid — but cheaply.*
#
# This is not a science case. It is a deliberately trivial Literate source that
# the deployment workflow can run in seconds, on CPU, with no GPU and no heavy
# dependencies, to prove that the whole machine turns: a case is selected, its
# script is generated, launched in a subprocess with `CASE_OUTPUT_DIR` set, its
# artifacts are written, the run is recorded, and the docs pick it up.
#
# It writes exactly two artifacts into `CASE_OUTPUT_DIR` (falling back to
# `thursday/output` when run standalone):
#
# * `fields.jld2` — a couple of tiny arrays (the *required* output).
# * `summary.png` — a one-panel figure (an *optional* output).

using JLD2
using Printf
using Dates

# Resolve the output directory the same way the real cases do: honor
# `CASE_OUTPUT_DIR` when the workflow sets it, else write beside the Thursday
# outputs so the script is runnable on its own.

output_dir = get(ENV, "CASE_OUTPUT_DIR", joinpath("thursday", "output"))
mkpath(output_dir)

# ## A trivial "simulation"
#
# A decaying sinusoid sampled on a small grid — enough to have something to save
# and plot, fast enough to finish instantly anywhere.

x = range(0, 2π, length = 64)
t = range(0, 1, length = 16)
field = [exp(-tj) * sin(xi) for xi in x, tj in t]

fields_path = joinpath(output_dir, "fields.jld2")
jldsave(fields_path; x = collect(x), t = collect(t), field)
@info "Smoke case wrote required artifact" fields_path size = size(field)

# ## Optional summary figure
#
# Guarded so the required artifact is written even if plotting is unavailable.

try
    using CairoMakie
    fig = Figure(size = (640, 360))
    ax = Axis(fig[1, 1], xlabel = "x", ylabel = "amplitude", title = "Smoke case: decaying sinusoid")
    lines!(ax, collect(x), field[:, 1], label = "t = 0")
    lines!(ax, collect(x), field[:, end], label = @sprintf("t = %.2f", last(t)))
    axislegend(ax)
    summary_path = joinpath(output_dir, "summary.png")
    save(summary_path, fig)
    @info "Smoke case wrote optional artifact" summary_path
catch err
    @warn "Smoke case skipped optional figure (CairoMakie unavailable?)" exception = err
end

@info "Smoke case complete." host = gethostname() finished = string(now())
nothing #hide
