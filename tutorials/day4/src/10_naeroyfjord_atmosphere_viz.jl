# # Visualize the Nærøyfjord atmosphere (cross-fjord blocking → down-fjord gap jet)
#
# Reads the rotating-wind atmosphere run slices and shows the near-surface wind speed
# when the geostrophic wind is cross-fjord (blocked) vs down-fjord (a gap jet accelerates
# along the axis), plus a vertical w transect.

using Oceananigans
using JLD2
using Printf
using Statistics
using CairoMakie

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
const datadir = joinpath(repo_root, "thursday", "data")
const figdir  = joinpath(repo_root, "thursday", "figures")
mkpath(figdir)

run_tag = get(ENV, "RUN_TAG", "rotate600")
fname = joinpath(datadir, "naeroyfjord_atmos_slices_$(run_tag).jld2")
isfile(fname) || error("Missing $fname")

u = FieldTimeSeries(fname, "u_xy")
v = FieldTimeSeries(fname, "v_xy")
w = FieldTimeSeries(fname, "w_xy")
times = u.times
Nt = length(times)
xc = xnodes(u) ./ 1e3
yc = ynodes(u) ./ 1e3
@info "Loaded atmos slices" Nt t_end_min = round(times[end] / 60, digits = 1)

speed(i) = sqrt.(interior(u[i], :, :, 1).^2 .+ interior(v[i], :, :, 1).^2)
smax = maximum(speed(Nt))

## Early (cross-fjord) vs late (down-fjord) near-surface wind speed.
fig = Figure(size = (1100, 750))
for (col, it, lab) in ((1, max(1, Nt ÷ 6), "cross-fjord"), (2, Nt, "down-fjord"))
    th = round(times[it] / 60, digits = 0)
    ax = Axis(fig[1, col]; title = "|u| near surface — $(lab) (t=$(th) min)",
              xlabel = "cross-fjord x (km)", ylabel = col == 1 ? "along-fjord y (km)" : "",
              aspect = DataAspect())
    hm = heatmap!(ax, xc, yc, speed(it); colormap = :speed, colorrange = (0, smax))
    col == 2 && Colorbar(fig[1, 3], hm, label = "m/s")
end
Label(fig[0, :], "Nærøyfjord atmosphere: cross-fjord wind blocked → down-fjord gap jet", fontsize = 17)
snap = joinpath(figdir, "naeroyfjord_atmosphere_$(run_tag).png")
save(snap, fig)
@info "Saved snapshot" snap

## Movie of near-surface wind speed through the rotation.
n = Observable(1)
Sp = @lift speed($n)
ttl = @lift @sprintf("Nærøyfjord near-surface wind speed — t = %.0f min", times[$n] / 60)
figm = Figure(size = (650, 800))
axm = Axis(figm[1, 1]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hmm = heatmap!(axm, xc, yc, Sp; colormap = :speed, colorrange = (0, smax))
Colorbar(figm[1, 2], hmm, label = "m/s")
Label(figm[0, 1], ttl, fontsize = 15)
movie = joinpath(figdir, "naeroyfjord_atmosphere_$(run_tag).mp4")
record(figm, movie, 1:Nt; framerate = 10) do i; n[] = i; end
@info "Saved movie" movie
nothing #hide
