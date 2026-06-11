# # Visualizing: 2D coupled air–sea convection
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
# A two-panel figure stacks the two fluids across the interface at `z = 0`:
# atmospheric vertical velocity `w(x, z, t)` on top (thermals rising over the warm
# sea) and ocean vertical velocity `w(x, z, t)` on the bottom (cooled water sinking
# from the surface). The shared blue–red colormap makes the *symmetry* of
# convection above and below the interface visible. We build a movie and a
# final-frame figure.

using CairoMakie

atmos_file = "coupled_convection_atmosphere.jld2"
ocean_file = "coupled_convection_ocean.jld2"

wa_t = FieldTimeSeries(atmos_file, "w_a")
wo_t = FieldTimeSeries(ocean_file, "w_o")
times = wa_t.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

xa, _, za = nodes(wa_t)
xo, _, zo = nodes(wo_t)
xa_km = xa ./ 1e3
xo_km = xo ./ 1e3

n = Observable(Nt)
wan = @lift interior(wa_t[$n], :, 1, :)
won = @lift interior(wo_t[$n], :, 1, :)
title = @lift "Coupled air–sea convection — t = " * prettytime(times[$n])

## Symmetric color limits per fluid, taken from the final frame.
wa_lim = max(1e-3, maximum(abs, interior(wa_t[Nt])))
wo_lim = max(1e-5, maximum(abs, interior(wo_t[Nt])))

fig = Figure(size = (1100, 800))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
axa = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (m)",
           title = "atmosphere: vertical velocity w (m s⁻¹)")
axo = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (m)",
           title = "ocean: vertical velocity w (m s⁻¹)")

hma = heatmap!(axa, xa_km, za, wan, colormap = :balance, colorrange = (-wa_lim, wa_lim))
hmo = heatmap!(axo, xo_km, zo, won, colormap = :balance, colorrange = (-wo_lim, wo_lim))
Colorbar(fig[1, 2], hma)
Colorbar(fig[2, 2], hmo)

save("coupled_convection.png", fig)

# ## Final state
#
fig

# ## Animation
#

record(fig, "coupled_convection.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
@info "Wrote movie" "coupled_convection.mp4"


# ```@raw html
# <video autoplay loop muted playsinline controls src="coupled_convection.mp4" style="max-width:100%"></video>
# ```

