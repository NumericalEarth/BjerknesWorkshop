# Robust last-frame figure of the coupled fjord smoke: scalar fields only (no Face/Center
# velocity issues). SST, vertical T transect + MLD, TKE transect, surface sensible heat flux.
using Breeze, Oceananigans, Oceananigans.Units, CairoMakie, JLD2, Statistics, Printf

cd("/shared/home/greg/Projects/BjerknesWorkshop/output/day4/coupled_fjord_smoke")
st = load("coupled_fjord_statics.jld2")
lon = st["lon"]; lat = st["lat"]; water = st["water"]; hter = st["h"]

T_xy = FieldTimeSeries("coupled_fjord_ocean.jld2", "T_xy")
T_xz = FieldTimeSeries("coupled_fjord_ocean.jld2", "T_xz")
e_xz = FieldTimeSeries("coupled_fjord_ocean.jld2", "e_xz")
Qs   = FieldTimeSeries("coupled_fjord_fluxes.jld2", "Q_sensible")
times = T_xy.times; n = length(times)
xz_lon, _, xz_z = nodes(T_xz)

# SST, masked to water
sst = interior(T_xy[n], :, :, 1) |> Array
sst[water .< 0.5] .= NaN
sstvals = filter(!isnan, sst)

Tt = interior(T_xz[n], :, 1, :) |> Array          # (lon, z)
ee = interior(e_xz[n], :, 1, :) |> Array
qq = interior(Qs[n], :, :, 1) |> Array
qq[water .< 0.5] .= NaN

@printf("SST(water): %.2f–%.2f °C  | TKE max %.2e | Qsens(water) %.0f..%.0f W/m²\n",
        minimum(sstvals), maximum(sstvals), maximum(ee), minimum(filter(!isnan,qq)), maximum(filter(!isnan,qq)))

fig = Figure(size = (1250, 950))
Label(fig[0, 1:2], @sprintf("Coupled Sunnmøre fjord — t = %s (smoke, 160², wind rotating cross→along)", prettytime(times[n])), fontsize = 18, tellwidth = false)

ax1 = Axis(fig[1,1], title = "sea-surface temperature (°C)", xlabel="lon (°E)", ylabel="lat (°N)", aspect=DataAspect())
h1 = heatmap!(ax1, lon, lat, sst; colormap=:thermal, colorrange=(floor(minimum(sstvals)), 10))
contour!(ax1, lon, lat, water; levels=[0.5], color=:black, linewidth=0.8)
Colorbar(fig[1,0], h1)

ax2 = Axis(fig[1,2], title="surface sensible heat flux (W m⁻²)", xlabel="lon (°E)", ylabel="lat (°N)", aspect=DataAspect())
qm = maximum(abs, filter(!isnan, qq))
h2 = heatmap!(ax2, lon, lat, qq; colormap=:balance, colorrange=(-qm, qm))
contour!(ax2, lon, lat, water; levels=[0.5], color=:black, linewidth=0.8)
Colorbar(fig[1,3], h2)

ax3 = Axis(fig[2,1], title="ocean temperature transect (°C)", xlabel="lon (°E)", ylabel="z (m)")
h3 = heatmap!(ax3, xz_lon, xz_z, Tt; colormap=:thermal)
Colorbar(fig[2,0], h3)

ax4 = Axis(fig[2,2], title="ocean TKE transect — mixing (m² s⁻²)", xlabel="lon (°E)", ylabel="z (m)")
h4 = heatmap!(ax4, xz_lon, xz_z, ee; colormap=:inferno)
Colorbar(fig[2,3], h4)

save("coupled_fjord_quicklook.png", fig)
println("saved coupled_fjord_quicklook.png")
