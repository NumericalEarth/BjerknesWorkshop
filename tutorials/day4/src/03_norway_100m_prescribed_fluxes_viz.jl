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
using NCDatasets

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES


# Load the cached near-surface wind slices, the `w` transect, and the land surface
# saturation, plus the static terrain/land-mask (reconstructed on the slice grid from
# the cached topography artifact — the same bilinear interpolation the simulation used).

u_xy = FieldTimeSeries("norway_slices.jld2", "u_xy")
v_xy = FieldTimeSeries("norway_slices.jld2", "v_xy")
w_xy = FieldTimeSeries("norway_slices.jld2", "w_xy")   # near-surface vertical velocity
w_xz = FieldTimeSeries("norway_slices.jld2", "w_xz")
times = w_xz.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

repo = get(ENV, "THURSDAY_REPO_ROOT", pwd())
topo = load(joinpath(repo, "thursday", "data", "norway_lofoten_100m_topography.jld2"))
xt, yt, h_data, land_data = topo["x"], topo["y"], topo["h"], topo["land_mask"]

## `bilinear` lives in ThursdayLES — the same interpolator the simulation used to
## carve the terrain, reused here to reconstruct the static terrain/water fields.
h_fun, land_fun = bilinear(h_data, xt, yt), bilinear(land_data, xt, yt)

xs, ys, _ = nodes(u_xy)
xz_x, _, xz_z = nodes(w_xz)
xkm, ykm = xs ./ 1e3, ys ./ 1e3
hh    = [h_fun(x, y)       for x in xs, y in ys]
water = [1 - land_fun(x, y) for x in xs, y in ys]

# ## Where in the world: the Lofoten archipelago
#
# Lofoten is a ~160 km chain of granite peaks rising straight from the sea off the coast
# of **northern Norway, near 68° N** — well inside the Arctic Circle, fronting the
# Norwegian Sea. Steep mountains (many 600–1300 m) stand directly over deep, narrow
# fjords, with essentially no coastal plain. In winter, cold continental or Arctic air
# spilling over this wall of rock and water makes it a natural laboratory for
# orographic flow, gap/fjord jets, and surface-flux heterogeneity. The locator below
# (global ETOPO 2022 relief) marks the 100 km × 100 km LES domain in that setting.

lon0, lat0 = 13.7, 68.05        # domain center (≈ UTM33N fetch center)
dlon, dlat = 1.25, 0.46         # ≈ ±50 km box at this latitude
box_lon, box_lat = (lon0 - dlon, lon0 + dlon), (lat0 - dlat, lat0 + dlat)

## Find the locally cached global ETOPO relief (downloaded by ClimaOcean/NumericalEarth).
function find_etopo()
    for depot in (joinpath(homedir(), ".julia"), "/shared/julia_depot", get(ENV, "JULIA_DEPOT_PATH", ""))
        isempty(depot) && continue
        sp = joinpath(depot, "scratchspaces")
        isdir(sp) || continue
        for sub in readdir(sp; join = true)
            ed = joinpath(sub, "ETOPO")
            isdir(ed) || continue
            for f in readdir(ed; join = true)
                endswith(f, ".nc") && return f
            end
        end
    end
    return nothing
end

## Build the locator map; if the relief file is unavailable, skip it gracefully so the
## rest of the page still renders.
figL = try
    etopo = find_etopo()
    etopo === nothing && error("ETOPO relief not found in any depot scratchspace")
    ds = NCDataset(etopo)
    ## NCDatasets returns CF arrays as Union{Missing,Float32}; Makie can't use a
    ## Missing-union as an axis dimension, so coalesce to plain Float64.
    elon = Float64.(coalesce.(ds["lon"][:], NaN))
    elat = Float64.(coalesce.(ds["lat"][:], NaN))
    iL = findall(l -> 2 ≤ l ≤ 26, elon)      # Scandinavia / Norwegian Sea window
    jL = findall(l -> 57 ≤ l ≤ 72, elat)
    Z = Float64.(coalesce.(Array(ds["z"][iL, jL]), NaN))   # (lon, lat), metres
    close(ds)
    elons, elats = elon[iL], elat[jL]

    zmax = maximum(abs, filter(isfinite, Z))
    fig_loc = Figure(size = (760, 740))
    axL = Axis(fig_loc[1, 1], xlabel = "longitude (°E)", ylabel = "latitude (°N)", aspect = DataAspect(),
               title = "Lofoten, northern Norway — the 100 km LES domain (red)")
    hmL = heatmap!(axL, elons, elats, Z; colormap = :oleron, colorrange = (-zmax, zmax))
    contour!(axL, elons, elats, Z; levels = [0.0], color = :black, linewidth = 0.7)  # coastline
    bx = [box_lon[1], box_lon[2], box_lon[2], box_lon[1], box_lon[1]]
    by = [box_lat[1], box_lat[1], box_lat[2], box_lat[2], box_lat[1]]
    lines!(axL, bx, by; color = :red, linewidth = 3)
    text!(axL, lon0, box_lat[2] + 0.25; text = "LES domain", color = :red,
          align = (:center, :bottom), fontsize = 14)
    Colorbar(fig_loc[1, 2], hmL, label = "elevation / depth (m)")
    save("norway_locator.png", fig_loc)
    fig_loc
catch err
    @warn "Locator map skipped: $err"
    nothing
end
figL

# ## The terrain and the coupled flow — four views
#
# (top-left) terrain with the **land/water mask** — the boundary the flow reads. This
# wet/dry surface forcing is what drives the differential heat/moisture flux; the
# surface saturation itself barely changes over 90 min (the soil hydrology evolves on
# an hours timescale), so we show it here as a *static* boundary rather than animate it.
# (top-right) near-surface wind speed — gap jets threading between the islands, wakes
# behind the peaks; (bottom-left) a vertical `w` transect through **the lowest 5 km**,
# where the boundary-layer convection and the base of the mountain-wave train live;
# (bottom-right) near-surface **vertical velocity** `w` in plan view — convective
# plumes over the warm water and terrain-forced ascent/descent over the peaks.
#
# All colour limits are computed **once over the whole run** and held fixed, so the
# colour scale does not flicker frame-to-frame.

k5 = findlast(z -> z ≤ 5000, xz_z)            # crop the transect to the lowest 5 km
speed_max = maximum(maximum(sqrt.(interior(u_xy[i], :, :, 1).^2 .+ interior(v_xy[i], :, :, 1).^2)) for i in 1:Nt)
w_lim     = maximum(maximum(abs, interior(w_xz[i], :, 1, 1:k5)) for i in 1:Nt)
wxy_lim   = maximum(maximum(abs, interior(w_xy[i], :, :, 1)) for i in 1:Nt)

n = Observable(Nt)
speed = @lift sqrt.(interior(u_xy[$n], :, :, 1).^2 .+ interior(v_xy[$n], :, :, 1).^2)
wn    = @lift interior(w_xz[$n], :, 1, 1:k5)
wxy   = @lift interior(w_xy[$n], :, :, 1)
title = @lift "Coupled flow over Lofoten — t = " * prettytime(times[$n])

fig = Figure(size = (1150, 1000))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)

ax_terr = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "terrain (m) + water (cyan)", aspect = 1)
hmt = heatmap!(ax_terr, xkm, ykm, hh, colormap = :terrain, colorrange = (0, maximum(hh)))
contourf!(ax_terr, xkm, ykm, water; levels = [0.5, 1.0], colormap = [(:cyan, 0.0), (:cyan, 0.55)])
Colorbar(fig[1, 0], hmt, label = "elevation (m)")

ax_spd = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "y (km)", title = "near-surface wind speed (m s⁻¹)", aspect = 1)
hms = heatmap!(ax_spd, xkm, ykm, speed, colormap = :speed, colorrange = (0, speed_max))
Colorbar(fig[1, 3], hms)

ax_w = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "w transect, lowest 5 km (m s⁻¹)")
hmw = heatmap!(ax_w, xz_x ./ 1e3, xz_z[1:k5] ./ 1e3, wn, colormap = :balance, colorrange = (-w_lim, w_lim))
Colorbar(fig[2, 0], hmw)

ax_wxy = Axis(fig[2, 2], xlabel = "x (km)", ylabel = "y (km)", title = "near-surface vertical velocity w (m s⁻¹)", aspect = 1)
hmwxy = heatmap!(ax_wxy, xkm, ykm, wxy, colormap = :balance, colorrange = (-wxy_lim, wxy_lim))
Colorbar(fig[2, 3], hmwxy)

save("norway.png", fig)
fig

# ## Animation
#
# Slowed to 6 frames per second so the gap jets, lee eddies, and the mountain-wave
# train have time to read on screen.

record(fig, "norway.mp4", 1:Nt; framerate = 6) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="norway.mp4" style="max-width:100%"></video>
# ```

# ## Turbulence and terrain: how the mountains stir the flow
#
# Does the terrain actually *enhance* the turbulence? We quantify it with the
# near-surface **turbulent kinetic energy** built from the temporal fluctuations of the
# horizontal wind, `e = ½⟨u′² + v′²⟩`, where `u′ = u − ū(x,y)` is the departure from the
# time-mean wind at each point (a per-column Reynolds decomposition over the run). High
# `e` marks where the flow is most variable — shear layers off the peaks, separating
# gap jets, and lee eddies.

ū = sum(interior(u_xy[i], :, :, 1) for i in 1:Nt) ./ Nt
v̄ = sum(interior(v_xy[i], :, :, 1) for i in 1:Nt) ./ Nt
tke = sum(0.5 .* ((interior(u_xy[i], :, :, 1) .- ū).^2 .+ (interior(v_xy[i], :, :, 1) .- v̄).^2)
          for i in 1:Nt) ./ Nt

## Bin the TKE by terrain elevation to show the enhancement quantitatively.
hmax  = maximum(hh)
edges = range(0, hmax; length = 9)
binmid = 0.5 .* (edges[1:end-1] .+ edges[2:end])
tke_binned = [ (m = (hh .>= edges[k]) .& (hh .< edges[k+1]); any(m) ? mean(tke[m]) : NaN)
               for k in 1:length(edges)-1 ]

tke_sea  = mean(tke[water .> 0.5])
tke_land = mean(tke[water .<= 0.5])
@printf("Near-surface TKE: domain mean %.2f, peak %.1f m² s⁻²; land %.2f vs water %.2f m² s⁻²\n",
        mean(tke), maximum(tke), tke_land, tke_sea)

figT = Figure(size = (1250, 520))
axTm = Axis(figT[1, 1], xlabel = "x (km)", ylabel = "y (km)", aspect = 1,
            title = "near-surface TKE  ½⟨u′²+v′²⟩  (m² s⁻²)")
hmT = heatmap!(axTm, xkm, ykm, tke, colormap = :inferno)
contour!(axTm, xkm, ykm, hh; levels = range(200, hmax; length = 5), color = (:white, 0.45), linewidth = 0.6)
Colorbar(figT[1, 2], hmT)

axTb = Axis(figT[1, 3], xlabel = "terrain elevation (m)", ylabel = "mean TKE (m² s⁻²)",
            title = "TKE grows with terrain height")
scatterlines!(axTb, binmid, tke_binned, color = :firebrick, markersize = 10)

save("norway_tke.png", figT)
figT
