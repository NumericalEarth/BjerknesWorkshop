# # Hybrid physics + machine learning (placeholder)
#
# *Day 3, case 1 of 2.*
#
# The real tutorial will build a hybrid model: a physical core with a small
# learned closure correcting its tendencies. For now this is a runnable
# placeholder that produces one figure so the deployment pipeline has something
# to render. It writes `hybrid_physics_ml.png` into `CASE_OUTPUT_DIR`.

using CairoMakie

include(joinpath(@__DIR__, "00_common.jl"))
using .Day3Common

# ## A toy "learned correction"
#
# A physical prediction (a damped oscillation) plus a synthetic ML correction that
# nudges it toward a "truth" signal. Nothing is actually trained — this only
# illustrates the shape of the eventual story.

t = range(0, 6π, length = 400)
truth   = @. exp(-0.1t) * sin(t) + 0.15 * sin(3t)
physics = @. exp(-0.1t) * sin(t)
hybrid  = @. physics + 0.15 * sin(3t)   # the "learned" correction

fig = Figure(size = (760, 420))
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "state",
          title = "Hybrid physics + ML (placeholder)")
lines!(ax, collect(t), truth,   label = "truth",        linewidth = 2)
lines!(ax, collect(t), physics, label = "physics only", linestyle = :dash)
lines!(ax, collect(t), hybrid,  label = "hybrid",        linewidth = 2)
axislegend(ax, position = :rt)

path = joinpath(case_output_dir(), "hybrid_physics_ml.png")
save(path, fig)
@info "Day 3 case 1 (hybrid physics + ML) complete" path

nothing #hide
