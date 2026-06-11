# # Visualizing: coupled flow over Lofoten
#
# *This is the **visualization** half of the Norway case. The simulation ran on a GPU
# before this page was built and cached its output; everything here executes live
# during the docs build, reading that cached output (and the topography artifact) to
# draw the figures and record the animation — the genuine production-resolution result.*

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics
using JLD2

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES


# Load the cached near-surface wind slices, the `w` transect, and the land surface
# saturation, plus the static terrain/land-mask (reconstructed on the slice grid from
# the cached topography artifact — the same bilinear interpolation the simulation used).

u_xy = FieldTimeSeries("norway_slices.jld2", "u_xy")
v_xy = FieldTimeSeries("norway_slices.jld2", "v_xy")
w_xz = FieldTimeSeries("norway_slices.jld2", "w_xz")
𝒮_ts = FieldTimeSeries("norway_land.jld2", "𝒮")
times = w_xz.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

repo = get(ENV, "THURSDAY_REPO_ROOT", pwd())
topo = load(joinpath(repo, "thursday", "data", "norway_lofoten_100m_topography.jld2"))
xt, yt, h_data, land_data = topo["x"], topo["y"], topo["h"], topo["land_mask"]

function bilinear(arr, xs, ys)
    x0, x1 = first(xs), last(xs); y0, y1 = first(ys), last(ys)
    nx, ny = length(xs), length(ys)
    dx = (x1 - x0) / (nx - 1); dy = (y1 - y0) / (ny - 1)
    return function (x, y)
        fx = clamp((x - x0) / dx, 0, nx - 1 - 1e-6)
        fy = clamp((y - y0) / dy, 0, ny - 1 - 1e-6)
        i = floor(Int, fx) + 1; j = floor(Int, fy) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        @inbounds (arr[i, j]   * (1 - tx) * (1 - ty) + arr[i+1, j]   * tx * (1 - ty) +
                   arr[i, j+1] * (1 - tx) * ty       + arr[i+1, j+1] * tx * ty)
    end
end
h_fun, land_fun = bilinear(h_data, xt, yt), bilinear(land_data, xt, yt)

xs, ys, _ = nodes(u_xy)
xz_x, _, xz_z = nodes(w_xz)
xkm, ykm = xs ./ 1e3, ys ./ 1e3
hh    = [h_fun(x, y)       for x in xs, y in ys]
water = [1 - land_fun(x, y) for x in xs, y in ys]

# ## The terrain and the four-panel coupled flow
#
# (top-left) terrain with the **land/water mask** — the boundary the flow reads;
# (top-right) near-surface wind speed — gap jets threading the fjords, wakes behind the
# islands; (bottom-left) a vertical `w` transect — the mountain-wave train under the
# sponge; (bottom-right) surface saturation `𝒮` — wet fjords vs dry land, the moisture
# heterogeneity driving the differential surface flux.

n = Observable(Nt)
speed = @lift sqrt.(interior(u_xy[$n], :, :, 1).^2 .+ interior(v_xy[$n], :, :, 1).^2)
wn    = @lift interior(w_xz[$n], :, 1, :)
𝒮n    = @lift interior(𝒮_ts[$n], :, :, 1)
title = @lift "Coupled flow over Lofoten — t = " * prettytime(times[$n])

fig = Figure(size = (1150, 1000))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)

ax_terr = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "terrain (m) + water (cyan)", aspect = 1)
hmt = heatmap!(ax_terr, xkm, ykm, hh, colormap = :terrain, colorrange = (0, maximum(hh)))
contourf!(ax_terr, xkm, ykm, water; levels = [0.5, 1.0], colormap = [(:cyan, 0.0), (:cyan, 0.55)])
Colorbar(fig[1, 0], hmt, label = "elevation (m)")

ax_spd = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "y (km)", title = "near-surface wind speed (m s⁻¹)", aspect = 1)
hms = heatmap!(ax_spd, xkm, ykm, speed, colormap = :speed)
Colorbar(fig[1, 3], hms)

ax_w = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "w transect (m s⁻¹)")
wlim = max(1e-3, maximum(abs, interior(w_xz[Nt])))
hmw = heatmap!(ax_w, xz_x ./ 1e3, xz_z ./ 1e3, wn, colormap = :balance, colorrange = (-wlim, wlim))
Colorbar(fig[2, 0], hmw)

ax_𝒮 = Axis(fig[2, 2], xlabel = "x (km)", ylabel = "y (km)", title = "surface saturation 𝒮 (wet fjords / dry land)", aspect = 1)
hm𝒮 = heatmap!(ax_𝒮, xkm, ykm, 𝒮n, colormap = :dense, colorrange = (0, 1))
Colorbar(fig[2, 3], hm𝒮)

save("norway.png", fig)
fig

# ## Animation

record(fig, "norway.mp4", 1:Nt; framerate = 12) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="norway.mp4" style="max-width:100%"></video>
# ```
