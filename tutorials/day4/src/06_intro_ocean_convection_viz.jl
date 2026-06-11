# # Visualizing: 2D ocean free convection
#
# *This is the **visualization** half of the case. The simulation ran on a GPU
# before this page was built and cached its output; everything here executes live
# during the docs build, reading that cached output to draw the figures and record
# the animation — so these are the genuine production-resolution results.*

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics
include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES


# ## Visualization
#
# A vertical velocity transect `w(x, z, t)` shows the cold dense plumes plunging
# from the surface; the temperature transect `T(x, z, t)` shows the cold anomalies
# they carry down and the mixed layer deepening over time. We build a movie and a
# final-frame figure.

w_ts = FieldTimeSeries("ocean_convection.jld2", "w")
T_ts = FieldTimeSeries("ocean_convection.jld2", "T")
times = w_ts.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

xw, _, zw = nodes(w_ts)

n = Observable(Nt)
wn = @lift interior(w_ts[$n], :, 1, :)
Tn = @lift interior(T_ts[$n], :, 1, :)
title = @lift "Ocean free convection — t = " * prettytime(times[$n])

fig = Figure(size = (1000, 700))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
axw = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "w (m s⁻¹)")
axT = Axis(fig[2, 1], xlabel = "x (m)", ylabel = "z (m)", title = "T (°C)")

wlim = max(1e-5, maximum(abs, interior(w_ts[Nt])))
Tn_last = interior(T_ts[Nt], :, 1, :)
Tlims = (minimum(Tn_last), maximum(Tn_last))

hmw = heatmap!(axw, xw, zw, wn, colormap = :balance, colorrange = (-wlim, wlim))
hmT = heatmap!(axT, xw, zw, Tn, colormap = :thermal, colorrange = Tlims)
Colorbar(fig[1, 2], hmw)
Colorbar(fig[2, 2], hmT)

save("ocean_convection.png", fig)

# ## Final state
#
fig

# ## Animation
#

record(fig, "ocean_convection.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
@info "Wrote movie" "ocean_convection.mp4"


# ```@raw html
# <video autoplay loop muted playsinline controls src="ocean_convection.mp4" style="max-width:100%"></video>
# ```

