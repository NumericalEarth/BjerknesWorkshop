# # Visualize the Nærøyfjord ocean response (stagnant → down-fjord mixing)
#
# Reads the ocean run output (`10_naeroyfjord_ocean.jl`) and renders the fresh surface
# lens being eroded as the wind rotates onto the fjord axis: surface salinity at three
# times (initial / cross-fjord / down-fjord) and an along-fjord salinity transect, plus a
# movie of the surface salinity.

using Oceananigans
using JLD2
using Printf
using Statistics
using CairoMakie

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
const datadir = joinpath(repo_root, "thursday", "data")
const figdir  = joinpath(repo_root, "thursday", "figures")
mkpath(figdir)

run_tag = get(ENV, "RUN_TAG", "ocean4h")
fname = joinpath(datadir, "naeroyfjord_ocean_$(run_tag).jld2")
isfile(fname) || error("Missing $fname — run 10_naeroyfjord_ocean.jl (RUN_TAG=$run_tag) first.")

S_xy = FieldTimeSeries(fname, "S_xy")
S_yz = FieldTimeSeries(fname, "S_yz")
w_xy = FieldTimeSeries(fname, "w_xy")
times = S_xy.times
Nt = length(times)
@info "Loaded ocean series" fname Nt t_end_hours = round(times[end] / 3600, digits = 2)

xc = xnodes(S_xy) ./ 1e3
yc = ynodes(S_xy) ./ 1e3
zc = znodes(S_yz)
yt = ynodes(S_yz) ./ 1e3

## Salinity range for a shared colormap (drop land NaN/zeros).
finite_S(f) = filter(s -> isfinite(s) && s > 1, vec(interior(f)))
allS = vcat((finite_S(S_xy[i]) for i in 1:Nt)...)
Smin, Smax = quantile(allS, 0.01), quantile(allS, 0.99)

# ## Three-panel snapshot: initial, mid (cross-fjord), late (down-fjord)
i1, i2, i3 = 1, max(1, Nt ÷ 3), Nt
fig = Figure(size = (1200, 900))
for (col, it) in enumerate((i1, i2, i3))
    th = round(times[it] / 3600, digits = 2)
    ax = Axis(fig[1, col]; title = "Surface S — t = $(th) h", xlabel = "cross-fjord x (km)",
              ylabel = col == 1 ? "along-fjord y (km)" : "", aspect = DataAspect())
    heatmap!(ax, xc, yc, interior(S_xy[it], :, :, 1); colormap = :haline, colorrange = (Smin, Smax))
end
axt = Axis(fig[2, 1:3]; title = "Along-fjord salinity transect (mid-channel) — final",
           xlabel = "along-fjord y (km)", ylabel = "z (m)")
hm = heatmap!(axt, yt, zc, interior(S_yz[Nt], 1, :, :); colormap = :haline, colorrange = (Smin, Smax))
Colorbar(fig[2, 4], hm, label = "S (g/kg)")
Label(fig[0, :], "Nærøyfjord ocean: fresh lens eroded as wind rotates down-fjord", fontsize = 18)

snap = joinpath(figdir, "naeroyfjord_ocean_$(run_tag).png")
save(snap, fig)
@info "Saved snapshot" snap

# ## Movie of surface salinity
n = Observable(1)
Sn = @lift interior(S_xy[$n], :, :, 1)
titlestr = @lift @sprintf("Nærøyfjord surface salinity — t = %.2f h", times[$n] / 3600)
figm = Figure(size = (700, 800))
axm = Axis(figm[1, 1]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hmm = heatmap!(axm, xc, yc, Sn; colormap = :haline, colorrange = (Smin, Smax))
Colorbar(figm[1, 2], hmm, label = "S (g/kg)")
Label(figm[0, 1], titlestr, fontsize = 16)
movie = joinpath(figdir, "naeroyfjord_ocean_$(run_tag).mp4")
record(figm, movie, 1:Nt; framerate = 8) do i; n[] = i; end
@info "Saved movie" movie
nothing #hide
