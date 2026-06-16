# # Visualize the coupled Nærøyfjord run
#
# The coupled story in one figure: the near-surface **atmospheric wind** (blocked
# cross-fjord, a gap jet down-fjord), the **air–sea heat flux** it drives at the surface,
# and the **ocean surface salinity** responding (the fresh lens mixing as the wind aligns).
# Reads the three output files written by `10_naeroyfjord_coupled_les.jl`.

using Oceananigans
using JLD2
using Printf
using Statistics
using CairoMakie

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
const datadir = joinpath(repo_root, "thursday", "data")
const figdir  = joinpath(repo_root, "thursday", "figures")
mkpath(figdir)

run_tag = get(ENV, "RUN_TAG", "coupled90")
af = joinpath(datadir, "naeroyfjord_coupled_atmos_$(run_tag).jld2")
ff = joinpath(datadir, "naeroyfjord_coupled_flux_$(run_tag).jld2")
of = joinpath(datadir, "naeroyfjord_coupled_ocean_$(run_tag).jld2")
for f in (af, ff, of); isfile(f) || error("Missing $f"); end

ua = FieldTimeSeries(af, "u_xy"); va = FieldTimeSeries(af, "v_xy")
Q  = FieldTimeSeries(ff, "Qsens")
So = FieldTimeSeries(of, "S_xy"); Soyz = FieldTimeSeries(of, "S_yz")
times = ua.times; Nt = length(times)
xa = xnodes(ua) ./ 1e3; ya = ynodes(ua) ./ 1e3
xo = xnodes(So) ./ 1e3; yo = ynodes(So) ./ 1e3
@info "Loaded coupled series" Nt t_end_min = round(times[end] / 60, digits = 1)

spd(i) = sqrt.(interior(ua[i], :, :, 1).^2 .+ interior(va[i], :, :, 1).^2)
smax = maximum(spd(Nt))
finiteS = filter(s -> isfinite(s) && s > 1, vcat((vec(interior(So[i])) for i in 1:Nt)...))
Smin, Smax = quantile(finiteS, 0.02), quantile(finiteS, 0.98)
Qabs = maximum(abs, filter(isfinite, vec(interior(Q[Nt]))))

# Final-time triptych: wind | air–sea heat flux | ocean surface salinity.
fig = Figure(size = (1350, 720))
th = round(times[Nt] / 60, digits = 0)
ax1 = Axis(fig[1, 1]; title = "Atmosphere |u| (m/s)", xlabel = "x (km)", ylabel = "along-fjord y (km)", aspect = DataAspect())
hm1 = heatmap!(ax1, xa, ya, spd(Nt); colormap = :speed, colorrange = (0, smax)); Colorbar(fig[1, 2], hm1)
ax2 = Axis(fig[1, 3]; title = "Air–sea sensible heat (W/m²)", xlabel = "x (km)", aspect = DataAspect())
hm2 = heatmap!(ax2, xa, ya, interior(Q[Nt], :, :, 1); colormap = :balance, colorrange = (-Qabs, Qabs)); Colorbar(fig[1, 4], hm2)
ax3 = Axis(fig[1, 5]; title = "Ocean surface S (g/kg)", xlabel = "x (km)", aspect = DataAspect())
hm3 = heatmap!(ax3, xo, yo, interior(So[Nt], :, :, 1); colormap = :haline, colorrange = (Smin, Smax)); Colorbar(fig[1, 6], hm3)
Label(fig[0, :], "Coupled Nærøyfjord at t = $(th) min (down-fjord): wind → flux → ocean mixing", fontsize = 18)
snap = joinpath(figdir, "naeroyfjord_coupled_$(run_tag).png")
save(snap, fig); @info "Saved snapshot" snap

# Movie: atmosphere wind speed (left) and ocean surface salinity (right) evolving together.
n = Observable(1)
WS = @lift spd($n); SS = @lift interior(So[$n], :, :, 1)
ttl = @lift @sprintf("Coupled Nærøyfjord — t = %.0f min", times[$n] / 60)
figm = Figure(size = (1100, 720))
axw = Axis(figm[1, 1]; title = "Atmosphere |u| (m/s)", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hw = heatmap!(axw, xa, ya, WS; colormap = :speed, colorrange = (0, smax)); Colorbar(figm[1, 2], hw)
axs = Axis(figm[1, 3]; title = "Ocean surface S (g/kg)", xlabel = "x (km)", aspect = DataAspect())
hs = heatmap!(axs, xo, yo, SS; colormap = :haline, colorrange = (Smin, Smax)); Colorbar(figm[1, 4], hs)
Label(figm[0, :], ttl, fontsize = 16)
movie = joinpath(figdir, "naeroyfjord_coupled_$(run_tag).mp4")
record(figm, movie, 1:Nt; framerate = 8) do i; n[] = i; end
@info "Saved movie" movie
nothing #hide
