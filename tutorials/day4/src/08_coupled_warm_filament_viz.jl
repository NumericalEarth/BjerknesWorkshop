# # Visualizing: the warm-filament cloud street
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

config = RunConfig("08_coupled_warm_filament")

# ## Visualization
#
# A multi-panel figure tells the coupled story in one frame:
#
# - top-left: the **cloud street** — near-surface cloud liquid `qˡ(x, y)` (the band of
#   cloud over the warm filament, leaning downwind);
# - top-right: near-surface vertical velocity `w(x, y)` (the convective updrafts);
# - middle-left: **SST** `T(x, y)` (the warm filament, being eroded);
# - middle-right: the **air–sea sensible heat flux** `Q(x, y)` (localized over the
#   filament);
# - bottom-left: across-filament atmospheric transect `w(y, z)` (the rising plume);
# - bottom-right: across-filament ocean transect `w(y, z)` (the sinking response).
#
# We build a movie from the time series and save the final frame.

using CairoMakie
_safelim(x, fallback) = (m = maximum(abs, x); isfinite(m) && m > 0 ? m : fallback)

atmos_file = output_name(config, "atmosphere")
ocean_file = output_name(config, "ocean")
flux_file  = output_name(config, "fluxes")

qˡxy = FieldTimeSeries(atmos_file, "qˡ_xy")
wxy  = FieldTimeSeries(atmos_file, "w_xy")
wyz  = FieldTimeSeries(atmos_file, "w_yz")
Txy  = FieldTimeSeries(ocean_file, "T_xy")
woyz = FieldTimeSeries(ocean_file, "w_yz")

times = qˡxy.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

xa, ya, _  = nodes(qˡxy)
_,  yaz, za = nodes(wyz)
xo, yo, _  = nodes(Txy)
_,  yoz, zo = nodes(woyz)

xkm  = xa  ./ 1e3
ykm  = ya  ./ 1e3
yazkm = yaz ./ 1e3
xokm = xo  ./ 1e3
yokm = yo  ./ 1e3
yozkm = yoz ./ 1e3

Qfields = isfile(flux_file)
if Qfields
    Qts = FieldTimeSeries(flux_file, "Q_sensible")
    xq, yq, _ = nodes(Qts)
    xqkm = xq ./ 1e3
    yqkm = yq ./ 1e3
end

n = Observable(Nt)
qln = @lift interior(qˡxy[$n], :, :, 1) .* 1e3   # g/kg
wn  = @lift interior(wxy[$n], :, :, 1)
Tn  = @lift interior(Txy[$n], :, :, 1)
wyzn = @lift interior(wyz[$n], 1, :, :)
woyzn = @lift interior(woyz[$n], 1, :, :)
title = @lift "A warm filament writes a cloud street — t = " * prettytime(times[$n])

## Color limits from the final frame.
qlmax = _safelim(interior(qˡxy[Nt]) .* 1e3, 1.0)
wlim  = _safelim(interior(wxy[Nt]), 1e-3)
wyzlim = _safelim(interior(wyz[Nt]), 1e-3)
woyzlim = _safelim(interior(woyz[Nt]), 1e-5)
Tmin = minimum(interior(Txy[Nt]))
Tmax = (m=maximum(interior(Txy[Nt])); isfinite(m) ? m : 1.0)

fig = Figure(size = (1300, 1100))
Label(fig[0, 1:4], title, fontsize = 18, tellwidth = false)

axq = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)",
           title = "cloud liquid qˡ at z ≈ 800 m (g kg⁻¹) — the cloud street")
axw = Axis(fig[1, 3], xlabel = "x (km)", ylabel = "y (km)",
           title = "near-surface w (m s⁻¹)")
axT = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "y (km)",
           title = "SST T (°C) — the warm filament")
axF = Axis(fig[2, 3], xlabel = "x (km)", ylabel = "y (km)",
           title = "sensible heat flux Q (W m⁻²)")
axwa = Axis(fig[3, 1], xlabel = "y (km)", ylabel = "z (m)",
            title = "atmosphere transect w (m s⁻¹)")
axwo = Axis(fig[3, 3], xlabel = "y (km)", ylabel = "z (m)",
            title = "ocean transect w (m s⁻¹)")

hmq = heatmap!(axq, xkm, ykm, qln, colormap = :dense, colorrange = (0, qlmax))
hmw = heatmap!(axw, xkm, ykm, wn, colormap = :balance, colorrange = (-wlim, wlim))
hmT = heatmap!(axT, xokm, yokm, Tn, colormap = :thermal, colorrange = (Tmin, Tmax))
hmwa = heatmap!(axwa, yazkm, za, wyzn, colormap = :balance, colorrange = (-wyzlim, wyzlim))
hmwo = heatmap!(axwo, yozkm, zo, woyzn, colormap = :balance, colorrange = (-woyzlim, woyzlim))

Colorbar(fig[1, 2], hmq)
Colorbar(fig[1, 4], hmw)
Colorbar(fig[2, 2], hmT)
Colorbar(fig[3, 2], hmwa)
Colorbar(fig[3, 4], hmwo)

if Qfields
    Qn = @lift interior(Qts[$n], :, :, 1)
    Qmax = _safelim(interior(Qts[Nt]), 1.0)
    hmF = heatmap!(axF, xqkm, yqkm, Qn, colormap = :balance, colorrange = (-Qmax, Qmax))
    Colorbar(fig[2, 4], hmF)
end

save(figure_name(config, "coupled_warm_filament_final"), fig)

# ## Final state
#
fig

# ## Animation
#

record(fig, movie_name(config, "coupled_warm_filament"), 1:Nt; framerate = 12) do i
    n[] = i
end
@info "Wrote movie" movie_name(config, "coupled_warm_filament")


# ```@raw html
# <video autoplay loop muted playsinline controls src="coupled_warm_filament.mp4" style="max-width:100%"></video>
# ```

