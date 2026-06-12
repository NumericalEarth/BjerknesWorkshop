# # Visualizing: convection over a flat surface vs over a mountain
#
# *This is the **visualization** half of the intro atmosphere case. The two simulations
# (flat control and Witch of Agnesi hill) ran on a GPU before this page was built and
# cached their output; everything below executes live during the docs build, reading
# that cached output — so the figures and movie are the genuine production results.*

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

flat_w = FieldTimeSeries("flat_convection.jld2", "w")
flat_θ = FieldTimeSeries("flat_convection.jld2", "θ")
agn_w  = FieldTimeSeries("agnesi_convection.jld2", "w")
agn_θ  = FieldTimeSeries("agnesi_convection.jld2", "θ")

times = flat_w.times
Nt = min(length(times), length(agn_w.times))
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[Nt]))

xw, _, zw = nodes(flat_w)
xkm = xw ./ 1e3
zkm = zw ./ 1e3

# The hill, for outlining the terrain on the Agnesi panels.

h₀, a = 600, 1e3
agnesi(x) = h₀ / (1 + x^2 / a^2)
hill_km = [agnesi(x) / 1e3 for x in xw]

# ## The comparison: same forcing, different boundary
#
# Four panels, one observable time. Left column: the flat control — free convection,
# thermals rising from the uniformly heated surface. Right column: the identical
# forcing with the Agnesi hill — the boundary layer still convects, but the crest
# accelerates the flow, mountain waves radiate into the stable air aloft, and the lee
# is a turbulent wake. Color limits are fixed across the whole run and **shared between
# the two columns**, so the comparison is honest.

n = Observable(Nt)
wf = @lift interior(flat_w[$n], :, 1, :)
θf = @lift interior(flat_θ[$n], :, 1, :)
wa = @lift interior(agn_w[$n], :, 1, :)
θa = @lift interior(agn_θ[$n], :, 1, :)
title = @lift "Convection: flat vs mountain — t = " * prettytime(times[$n])

wlim = max(maximum(abs, flat_w), maximum(abs, agn_w))
θmin = min(minimum(flat_θ), minimum(agn_θ))
θmax = θmin + 12   # focus the θ colormap on the boundary layer, not the sponge

fig = Figure(size = (1300, 760))
Label(fig[0, 1:3], title, fontsize = 18, tellwidth = false)

ax_wf = Axis(fig[1, 1], ylabel = "z (km)", title = "flat: vertical velocity w (m s⁻¹)")
ax_wa = Axis(fig[1, 2], title = "mountain: vertical velocity w (m s⁻¹)")
ax_θf = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "flat: potential temperature θ (K)")
ax_θa = Axis(fig[2, 2], xlabel = "x (km)", title = "mountain: potential temperature θ (K)")

hmw = heatmap!(ax_wf, xkm, zkm, wf, colormap = :balance, colorrange = (-wlim, wlim))
heatmap!(ax_wa, xkm, zkm, wa, colormap = :balance, colorrange = (-wlim, wlim))
hmθ = heatmap!(ax_θf, xkm, zkm, θf, colormap = :thermal, colorrange = (θmin, θmax))
heatmap!(ax_θa, xkm, zkm, θa, colormap = :thermal, colorrange = (θmin, θmax))

for ax in (ax_wa, ax_θa)
    band!(ax, xkm, zero(hill_km), hill_km, color = :grey25)
end
for ax in (ax_wf, ax_wa, ax_θf, ax_θa)
    ylims!(ax, 0, 6)   # hide the sponge; the physics lives below 6 km
end

Colorbar(fig[1, 3], hmw)
Colorbar(fig[2, 3], hmθ)

save("flat_vs_agnesi_convection.png", fig)
fig

# ## Animation
#
# Every saved frame at full playback rate. Watch the two columns diverge from the same
# start: the flat run grows a classic convective boundary layer; the mountain run adds
# a standing wave over the crest and a churning lee-side wake.

record(fig, "flat_vs_agnesi_convection.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="flat_vs_agnesi_convection.mp4" style="max-width:100%"></video>
# ```

# ## What the mountain adds
#
# A quick quantitative read: the peak vertical velocity in each run over time. The flat
# run's `max|w|` is set by the convective velocity scale `w★`; the mountain run rides
# above it once the wave and wake spin up.

wmax_flat = [maximum(abs, flat_w[i]) for i in 1:Nt]
wmax_agn  = [maximum(abs, agn_w[i])  for i in 1:Nt]

fig2 = Figure(size = (800, 420))
ax2 = Axis(fig2[1, 1], xlabel = "time (hours)", ylabel = "max |w| (m s⁻¹)",
           title = "Peak vertical velocity: the mountain's contribution")
lines!(ax2, times[1:Nt] ./ 3600, wmax_flat, linewidth = 2, label = "flat (free convection)")
lines!(ax2, times[1:Nt] ./ 3600, wmax_agn,  linewidth = 2, label = "mountain (convection + waves)")
axislegend(ax2, position = :rb)

save("flat_vs_agnesi_wmax.png", fig2)
fig2
