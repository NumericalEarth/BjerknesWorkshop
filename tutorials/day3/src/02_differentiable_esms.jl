# # Differentiable Earth-system models (placeholder)
#
# *Day 3, case 2 of 2.*
#
# The real tutorial will differentiate through a small Earth-system component to
# calibrate a parameter by gradient descent. For now this is a runnable
# placeholder that fakes a loss-vs-iteration curve so the pipeline has a figure to
# render. It writes `differentiable_esms.png` into `CASE_OUTPUT_DIR`.

using CairoMakie

include(joinpath(@__DIR__, "00_common.jl"))
using .Day3Common

# ## A toy calibration trace
#
# A synthetic optimization: the loss decays as the (pretend) gradient steps drive
# a parameter toward its true value. No autodiff is invoked here yet.

iters = 0:40
loss  = @. 1.0 * exp(-0.18 * iters) + 0.02
param = @. 3.0 - 2.0 * exp(-0.18 * iters)   # converging toward 1.0

fig = Figure(size = (820, 380))
ax1 = Axis(fig[1, 1], xlabel = "iteration", ylabel = "loss",
           title = "Calibration loss", yscale = log10)
lines!(ax1, collect(iters), loss, linewidth = 2)
ax2 = Axis(fig[1, 2], xlabel = "iteration", ylabel = "parameter",
           title = "Parameter convergence")
lines!(ax2, collect(iters), param, linewidth = 2)
hlines!(ax2, [1.0], linestyle = :dash, color = :gray, label = "true value")
axislegend(ax2, position = :rt)

path = joinpath(case_output_dir(), "differentiable_esms.png")
save(path, fig)
@info "Day 3 case 2 (differentiable ESMs) complete" path

nothing #hide
