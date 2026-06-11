# # Visualizing the 2D free-convection run
#
# *This is the **visualization** half of the intro atmosphere case. The
# [simulation](05_intro_atmosphere_convection.md) ran on a GPU before this page was
# built and cached its output; everything below executes live during the docs build —
# it only reads that cached output, makes the figures, and records the animation. So
# the plots and the movie you see are the genuine **production-resolution** results,
# rendered by code that actually ran when this page was generated.*
#
# We load the saved slices as `FieldTimeSeries`, draw the final state, and animate the
# whole run.

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

# Load the cached vertical-velocity and potential-temperature slices.

w_t = FieldTimeSeries("free_convection.jld2", "w")
θ_t = FieldTimeSeries("free_convection.jld2", "θ")
times = w_t.times
Nt = length(times)

println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

# A glance at the developed boundary layer: peak vertical velocity and the
# potential-temperature range at the final time.

w_final = interior(w_t[Nt], :, 1, :)
θ_final = interior(θ_t[Nt], :, 1, :)
@printf("Final frame: max|w| = %.2f m/s, θ ∈ [%.1f, %.1f] K\n",
        maximum(abs, w_final), minimum(θ_final), maximum(θ_final))

# ## Final state
#
# Two stacked panels: vertical velocity `w(x, z)` (a symmetric blue–red colormap, so
# updrafts are red and downdrafts blue) shows the thermals punching upward with
# compensating subsidence between them; potential temperature `θ(x, z)` shows the warm
# thermals and the deepening, well-mixed boundary layer beneath the stable cap.

xw, _, zw = nodes(w_t)
xkm = xw ./ 1e3
zkm = zw ./ 1e3

n = Observable(Nt)
wn = @lift interior(w_t[$n], :, 1, :)
θn = @lift interior(θ_t[$n], :, 1, :)
title = @lift "2D free convection — t = " * prettytime(times[$n])

fig = Figure(size = (1100, 750))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
axw = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = "vertical velocity w (m s⁻¹)")
axθ = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "potential temperature θ (K)")

wlim = max(1e-3, maximum(abs, interior(w_t[Nt])))
hmw = heatmap!(axw, xkm, zkm, wn, colormap = :balance, colorrange = (-wlim, wlim))
hmθ = heatmap!(axθ, xkm, zkm, θn, colormap = :thermal)
Colorbar(fig[1, 2], hmw)
Colorbar(fig[2, 2], hmθ)

save("free_convection.png", fig)
fig

# ## Animation
#
# Stepping the observable through every frame animates the spin-up of the convective
# boundary layer.

CairoMakie.record(fig, "free_convection.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="free_convection.mp4" style="max-width:100%"></video>
# ```
