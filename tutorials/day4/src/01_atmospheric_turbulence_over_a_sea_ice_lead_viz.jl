# # Visualizing: atmospheric turbulence over a sea-ice lead
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
# A vertical transect of vertical velocity `w(x, z, t)` shows the plume rising
# over the lead until the inversion caps it and it spreads downwind; the
# potential-temperature transect shows the warm anomaly leaning and advecting over
# the downwind ice. We build a movie from the high-cadence slices and a
# final-frame figure.

using CairoMakie

w_xz = FieldTimeSeries("lead_atmosphere_slices.jld2", "w_xz")
θ_xz = FieldTimeSeries("lead_atmosphere_slices.jld2", "θ_xz")
qˡ_xz = FieldTimeSeries("lead_atmosphere_slices.jld2", "qˡ_xz")
times = w_xz.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

xw, _, zw = nodes(w_xz)
xkm = xw ./ 1e3
zkm = zw ./ 1e3

n = Observable(Nt)
wn = @lift interior(w_xz[$n], :, 1, :)
θn = @lift interior(θ_xz[$n], :, 1, :)
qln = @lift interior(qˡ_xz[$n], :, 1, :) .* 1e3   # g/kg
title = @lift "Sea-ice lead plume — t = " * prettytime(times[$n])

fig = Figure(size = (1100, 950))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
axw = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = "w (m s⁻¹)")
axθ = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "θ (K)")
axq = Axis(fig[3, 1], xlabel = "x (km)", ylabel = "z (km)", title = "cloud liquid qˡ (g kg⁻¹) — the lead fog")

wlim = max(1e-3, maximum(abs, interior(w_xz[Nt])))
qlmax = max(1e-4, maximum(interior(qˡ_xz[Nt])) * 1e3)
hmw = heatmap!(axw, xkm, zkm, wn, colormap = :balance, colorrange = (-wlim, wlim))
hmθ = heatmap!(axθ, xkm, zkm, θn, colormap = :thermal)
hmq = heatmap!(axq, xkm, zkm, qln, colormap = :dense, colorrange = (0, qlmax))
Colorbar(fig[1, 2], hmw)
Colorbar(fig[2, 2], hmθ)
Colorbar(fig[3, 2], hmq)

save("lead_atmosphere.png", fig)

# ## Final state
#
fig

# ## Animation
#

record(fig, "lead_atmosphere.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
@info "Wrote movie" "lead_atmosphere.mp4"


# ```@raw html
# <video autoplay loop muted playsinline controls src="lead_atmosphere.mp4" style="max-width:100%"></video>
# ```

