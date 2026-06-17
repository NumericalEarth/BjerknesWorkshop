# # Visualizing: ocean mixing response to a rotating wind in a coupled fjord
#
# *This is the **visualization** half of the coupled fjord case. The simulation
# ran on a GPU before this page was built and cached its output; everything here
# executes live during the docs build, reading that cached output to draw the
# figures and record the animation — so these are the genuine production-resolution
# results.*
#
# **Scientific question:** As the wind rotates from **cross-valley** to **along-valley**,
# how does ocean mixing respond? Cross-valley winds drive Ekman upwelling/downwelling
# against the fjord sidewalls; along-valley winds drive down-fjord surface currents and
# a deeper mixed layer through direct shear production. We track the transition through
# mixed-layer depth, TKE from the CATKE closure, and the evolving temperature structure.

using Breeze   # registers the terrain-following grid types so atmosphere slices deserialize
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics
using JLD2

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

# ## Load the cached output
#
# Four JLD2 files carry everything we need. All dynamic fields are `FieldTimeSeries`;
# the statics file holds single snapshots (terrain height, bathymetry, water mask).

atmos_file  = "coupled_fjord_atmos.jld2"
ocean_file  = "coupled_fjord_ocean.jld2"
flux_file   = "coupled_fjord_fluxes.jld2"
static_file = "coupled_fjord_statics.jld2"

## --- atmosphere ---
u_atm_xy = FieldTimeSeries(atmos_file, "u_xy")
v_atm_xy = FieldTimeSeries(atmos_file, "v_xy")
w_atm_xy = FieldTimeSeries(atmos_file, "w_xy")
w_atm_xz = FieldTimeSeries(atmos_file, "w_xz")

## --- ocean ---
T_oce_xy  = FieldTimeSeries(ocean_file, "T_xy")
S_oce_xy  = FieldTimeSeries(ocean_file, "S_xy")
u_oce_xy  = FieldTimeSeries(ocean_file, "u_xy")
v_oce_xy  = FieldTimeSeries(ocean_file, "v_xy")
T_oce_xz  = FieldTimeSeries(ocean_file, "T_xz")
w_oce_xz  = FieldTimeSeries(ocean_file, "w_xz")
e_oce_xz  = FieldTimeSeries(ocean_file, "e_xz")
h_mld_ts  = FieldTimeSeries(ocean_file, "h_mld")

## --- surface fluxes ---
tau_x_ts   = FieldTimeSeries(flux_file, "tau_x")
tau_y_ts   = FieldTimeSeries(flux_file, "tau_y")
Q_sens_ts  = FieldTimeSeries(flux_file, "Q_sensible")
Q_lat_ts   = FieldTimeSeries(flux_file, "Q_latent")

times = T_oce_xy.times
Nt    = length(times)
println("Loaded ", Nt, " frames spanning ",
        prettytime(times[1]), " – ", prettytime(times[end]))

# ## Load static fields (terrain, bathymetry, water mask)

statics = jldopen(static_file, "r") do f
    h     = f["h"]      # terrain height (m), on atmosphere xy grid
    depth = f["depth"]  # ocean bathymetry (m, positive down)
    water = f["water"]  # water fraction ∈ [0, 1], on ocean xy grid
    (; h, depth, water)
end
h_terrain = statics.h        # (Nxa, Nya) array
depth_bat = statics.depth    # (Nxo, Nyo)
water_mask = statics.water   # (Nxo, Nyo)
nothing #hide

# ## Grid coordinates

xa, ya, _    = nodes(u_atm_xy)
xz_xa, _, xz_za = nodes(w_atm_xz)

xo, yo, _    = nodes(T_oce_xy)
xz_xo, _, xz_zo = nodes(T_oce_xz)

xa_km  = xa  ./ 1e3
ya_km  = ya  ./ 1e3
xo_km  = xo  ./ 1e3
yo_km  = yo  ./ 1e3
xz_xa_km = xz_xa ./ 1e3
xz_xo_km = xz_xo ./ 1e3
nothing #hide

# ## Colour limits — computed once over the whole run, held fixed
#
# We sweep every saved frame *before* building the figure so no panel changes its
# colour scale between frames.

println("Computing global colour limits …")

speed_atm_max = maximum(
    maximum(sqrt.(interior(u_atm_xy[i], :, :, 1).^2 .+
                  interior(v_atm_xy[i], :, :, 1).^2))
    for i in 1:Nt)

T_oce_lims = let
    tmin = minimum(minimum(interior(T_oce_xy[i])) for i in 1:Nt)
    tmax = maximum(maximum(interior(T_oce_xy[i])) for i in 1:Nt)
    (tmin, tmax)
end

e_lim = maximum(maximum(interior(e_oce_xz[i])) for i in 1:Nt)
e_lim = max(e_lim, 1e-6)   # guard against all-zero first frame

T_xz_lims = let
    tmin = minimum(minimum(interior(T_oce_xz[i])) for i in 1:Nt)
    tmax = maximum(maximum(interior(T_oce_xz[i])) for i in 1:Nt)
    (tmin, tmax)
end

tau_lim = maximum(
    maximum(sqrt.(interior(tau_x_ts[i], :, :, 1).^2 .+
                  interior(tau_y_ts[i], :, :, 1).^2))
    for i in 1:Nt)
tau_lim = max(tau_lim, 1e-4)

println("  wind speed max    = ", @sprintf("%.2f m s⁻¹",  speed_atm_max))
println("  SST range         = ", @sprintf("%.2f – %.2f °C", T_oce_lims...))
println("  TKE max           = ", @sprintf("%.2e m² s⁻²",  e_lim))
println("  wind-stress max   = ", @sprintf("%.4f N m⁻²",  tau_lim))

# ## Instantaneous wind-direction helper
#
# Domain-mean atmospheric near-surface wind direction, in degrees from North (0° = N,
# 90° = E, 180° = S, 270° = W). We add the cross→along label: in a fjord orientated
# roughly along the x-axis, cross-valley is ~N/S (wind_dir ≈ 0° or 180°) and
# along-valley is ~E/W (wind_dir ≈ 90° or 270°).

function mean_wind_dir(i)
    ū = mean(interior(u_atm_xy[i], :, :, 1))
    v̄ = mean(interior(v_atm_xy[i], :, :, 1))
    # meteorological convention: direction wind is coming FROM
    deg = mod(atand(ū, v̄) + 180, 360)   # atan(u,v) gives "going-to" angle East-of-N
    return deg
end

wind_dirs = [mean_wind_dir(i) for i in 1:Nt]   # pre-compute for the summary figure
nothing #hide

# ## Mixed-layer depth from the `h_mld` output field
#
# The simulation writes `h_mld` directly from CATKE. We also compute a temperature-
# threshold fallback: depth where the horizontally-averaged ocean temperature profile
# first drops by `ΔT = 0.2 °C` below the surface value. Both diagnostics are
# domain-averaged over water cells only (using the `water_mask`).

ΔT_threshold = 0.2   # °C

function mld_from_temperature(T_xz_frame)
    # T_xz_frame: interior array (Nx, 1, Nz) or (Nx, Nz) — we work with (Nz,)
    # profile = horizontal mean along the fjord transect
    Tdat = interior(T_xz_frame, :, 1, :)   # (Nx, Nz)
    prof = vec(mean(Tdat; dims = 1))        # (Nz,) from surface to bottom
    Nz_loc = length(prof)
    T_surf = prof[Nz_loc]                   # xz_zo is ascending (neg), last = shallowest
    mld = abs(xz_zo[1])                     # default: full depth
    for k in Nz_loc:-1:1
        if T_surf - prof[k] > ΔT_threshold
            mld = abs(xz_zo[k])
            break
        end
    end
    return mld
end

## Pre-compute both MLD diagnostics for the summary figure.
mld_direct  = [mean(interior(h_mld_ts[i], :, :, 1)[water_mask .> 0.5]) for i in 1:Nt]
mld_fallback = [mld_from_temperature(T_oce_xz[i]) for i in 1:Nt]

## Use the direct output; fall back to the temperature-based estimate wherever it
## looks unphysical (e.g. all-zero before the mixed layer develops).
mld_use = [mld_direct[i] > 1.0 ? mld_direct[i] : mld_fallback[i] for i in 1:Nt]

println("MLD range: ", @sprintf("%.1f – %.1f m", minimum(mld_use), maximum(mld_use)))

# ## Pre-compute scalar time series for the summary figure

## Domain-mean near-surface ocean TKE along the transect
mean_tke_ts = [mean(interior(e_oce_xz[i])) for i in 1:Nt]

## Wind-stress magnitude (domain-mean)
tau_mag_ts = [mean(sqrt.(interior(tau_x_ts[i], :, :, 1).^2 .+
                          interior(tau_y_ts[i], :, :, 1).^2)) for i in 1:Nt]
nothing #hide

# ## Arrow subsampling helper
#
# We overlay a regular sparse grid of wind/current arrows on the heatmaps.
# `subsample_arrows` picks every `stride`-th point in both dimensions.

function subsample_arrows(x, y, U, V; stride = 12)
    Nx, Ny = length(x), length(y)
    ix = 1:stride:Nx
    iy = 1:stride:Ny
    xs_arr = vec([x[i] for i in ix, j in iy])
    ys_arr = vec([y[j] for i in ix, j in iy])
    us_arr = vec([U[i, j] for i in ix, j in iy])
    vs_arr = vec([V[i, j] for i in ix, j in iy])
    return xs_arr, ys_arr, us_arr, vs_arr
end

# ## Animation: `coupled_fjord.mp4`
#
# Four panels per frame:
#
# **(a)** Near-surface atmospheric wind speed with sparse wind-vector arrows overlaid
# on the terrain/water background — this panel makes the rotating wind direction
# immediately visible.
#
# **(b)** Sea-surface temperature (°C, TEOS-10) with surface-current arrows from the
# ocean model.
#
# **(c)** Vertical ocean temperature transect along the main fjord axis, with the
# mixed-layer depth drawn as a white line.
#
# **(d)** CATKE turbulent kinetic energy transect — the panel that directly shows
# *where* mixing is occurring in the water column.
#
# The frame title carries the simulation time and the instantaneous mean wind direction.

## Arrow sub-sampling strides (tune if the grids are very fine/coarse)
atm_stride = max(1, div(length(xa), 20))
oce_stride = max(1, div(length(xo), 20))

## Build crop index for the transect: deepest resolved ocean level
kbot_oz = findfirst(z -> z > -2000, xz_zo)   # top 2 km of ocean (xz_zo < 0)
kbot_oz = isnothing(kbot_oz) ? length(xz_zo) : kbot_oz
krange_oz = kbot_oz:length(xz_zo)            # indices from deepest to shallowest

n = Observable(1)

## --- panel (a): atmosphere wind speed + arrows ---
speed_n = @lift sqrt.(interior(u_atm_xy[$n], :, :, 1).^2 .+
                      interior(v_atm_xy[$n], :, :, 1).^2)
arrow_atm_n = @lift begin
    U = interior(u_atm_xy[$n], :, :, 1)
    V = interior(v_atm_xy[$n], :, :, 1)
    subsample_arrows(xa_km, ya_km, U, V; stride = atm_stride)
end

## --- panel (b): SST + surface-current arrows ---
T_sst_n = @lift interior(T_oce_xy[$n], :, :, 1)
arrow_oce_n = @lift begin
    U = interior(u_oce_xy[$n], :, :, 1)
    V = interior(v_oce_xy[$n], :, :, 1)
    subsample_arrows(xo_km, yo_km, U, V; stride = oce_stride)
end

## --- panel (c): ocean T transect + MLD overlay ---
T_xz_n = @lift interior(T_oce_xz[$n], :, 1, krange_oz)
mld_line_n = @lift begin
    mld_val = mld_direct[$n] > 1.0 ? mld_direct[$n] : mld_fallback[$n]
    fill(-mld_val, length(xz_xo_km))
end

## --- panel (d): CATKE TKE transect ---
e_xz_n = @lift interior(e_oce_xz[$n], :, 1, krange_oz)

## --- frame title ---
title_n = @lift let
    t   = times[$n]
    dir = wind_dirs[$n]
    label = if 45 < dir ≤ 135 || 225 < dir ≤ 315
        "along-valley"
    else
        "cross-valley"
    end
    @sprintf("Coupled fjord  —  t = %s  |  wind: %.0f° (%s)",
             prettytime(t), dir, label)
end

## Build figure
fig = Figure(size = (1280, 1080))
Label(fig[0, 1:4], title_n, fontsize = 16, tellwidth = false)

## (a) atmosphere wind speed
ax_a = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)",
            title = "near-surface wind speed (m s⁻¹)", aspect = DataAspect())
hm_a = heatmap!(ax_a, xa_km, ya_km, speed_n;
                colormap = :speed, colorrange = (0, speed_atm_max))
## terrain contours for geographic context
contour!(ax_a, xa_km, ya_km, h_terrain;
         levels = [200.0, 600.0, 1000.0], color = (:white, 0.5), linewidth = 0.7)
## wind arrows
arrows!(ax_a,
        @lift($arrow_atm_n[1]), @lift($arrow_atm_n[2]),
        @lift($arrow_atm_n[3]), @lift($arrow_atm_n[4]);
        color = :black, lengthscale = 0.8, arrowsize = 8)
Colorbar(fig[1, 2], hm_a, label = "speed (m s⁻¹)")

## (b) SST + current arrows
ax_b = Axis(fig[1, 3], xlabel = "x (km)", ylabel = "y (km)",
            title = "sea-surface temperature (°C)", aspect = DataAspect())
hm_b = heatmap!(ax_b, xo_km, yo_km, T_sst_n;
                colormap = :thermal, colorrange = T_oce_lims)
arrows!(ax_b,
        @lift($arrow_oce_n[1]), @lift($arrow_oce_n[2]),
        @lift($arrow_oce_n[3]), @lift($arrow_oce_n[4]);
        color = (:white, 0.7), lengthscale = 0.8, arrowsize = 8)
Colorbar(fig[1, 4], hm_b, label = "T (°C)")

## (c) ocean temperature transect + MLD
ax_c = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "depth (m)",
            title = "fjord temperature transect (°C)")
hm_c = heatmap!(ax_c, xz_xo_km, xz_zo[krange_oz], T_xz_n;
                colormap = :thermal, colorrange = T_xz_lims)
lines!(ax_c, xz_xo_km, mld_line_n; color = :white, linewidth = 2,
       label = "MLD")
axislegend(ax_c; position = :lt, labelsize = 11)
Colorbar(fig[2, 2], hm_c, label = "T (°C)")

## (d) CATKE TKE transect
ax_d = Axis(fig[2, 3], xlabel = "x (km)", ylabel = "depth (m)",
            title = "CATKE TKE transect (m² s⁻²)")
hm_d = heatmap!(ax_d, xz_xo_km, xz_zo[krange_oz], e_xz_n;
                colormap = :inferno, colorrange = (0, e_lim))
Colorbar(fig[2, 4], hm_d, label = "e (m² s⁻²)")

save("coupled_fjord.png", fig)
fig

# ## Animation

record(fig, "coupled_fjord.mp4", 1:Nt; framerate = 24, compression = 30) do i
    n[] = i
end
@info "Wrote animation" "coupled_fjord.mp4"

# ```@raw html
# <video autoplay loop muted playsinline controls src="coupled_fjord.mp4" style="max-width:100%"></video>
# ```

# ## Summary figure: ocean mixing response to the rotating wind
#
# This figure is the scientific payoff. The top three panels share the same time axis
# (labelled as both elapsed time and mean wind direction) and show:
#
# **(i)** domain-mean mixed-layer depth over water cells — deepens as the wind swings
# along-valley and inertial shear accumulates;
#
# **(ii)** mean near-surface ocean TKE — a proxy for instantaneous mixing intensity;
#
# **(iii)** wind-stress magnitude and direction — the forcing that drives (i) and (ii).
#
# The bottom panel shows horizontally-averaged ocean temperature profiles at a few times
# spanning the wind rotation, revealing the mixed-layer deepening directly.

## Select a handful of times that span the cross→along rotation for the profiles.
## We divide the run into 5 equally spaced snapshots.
profile_indices = unique(round.(Int, range(1, Nt; length = 5)))

## Build the x-axis for the time-series: elapsed time in hours.
t_hours = times ./ 3600

## Horizontally averaged temperature profiles (mean over the transect x-axis)
T_profiles = [vec(mean(interior(T_oce_xz[i], :, 1, :); dims = 1))
              for i in profile_indices]

fig_mix = Figure(size = (1100, 900))

## Shared x-axis (time in hours)
ax_mld = Axis(fig_mix[1, 1], ylabel = "MLD (m)",
              title = "Domain-mean mixed-layer depth over water",
              xticklabelsvisible = false, yreversed = true)
ax_tke = Axis(fig_mix[2, 1], ylabel = "mean TKE (m² s⁻²)",
              title = "Near-surface ocean TKE",
              xticklabelsvisible = false, yscale = log10)
ax_tau = Axis(fig_mix[3, 1], xlabel = "time (hours)",
              title = "Wind-stress magnitude (N m⁻²)")
ax_dir = Axis(fig_mix[3, 1]; yaxisposition = :right,
              ylabel = "wind direction (°)", ylabelcolor = :firebrick)

## --- (i) MLD ---
lines!(ax_mld, t_hours, mld_use; color = :steelblue, linewidth = 2)
scatter!(ax_mld, t_hours[profile_indices], mld_use[profile_indices];
         color = :orange, markersize = 10, label = "profile snapshots")
axislegend(ax_mld; position = :lb, labelsize = 11)

## --- (ii) mean TKE ---
lines!(ax_tke, t_hours, mean_tke_ts; color = :darkorange, linewidth = 2)

## --- (iii) wind-stress magnitude + direction ---
lines!(ax_tau, t_hours, tau_mag_ts; color = :black, linewidth = 2,
       label = "|τ| (N m⁻²)")
lines!(ax_dir, t_hours, wind_dirs; color = :firebrick, linewidth = 1.5,
       linestyle = :dash, label = "wind dir (°)")
axislegend(ax_tau; position = :lt, labelsize = 11)
axislegend(ax_dir; position = :rt, labelsize = 11)

## --- (iv) temperature profiles ---
ax_prof = Axis(fig_mix[1:3, 3], xlabel = "T (°C)", ylabel = "depth (m)",
               title = "Horizontally-averaged T profiles\n(ocean, °C)",
               yreversed = true)
cmap_prof = cgrad(:plasma, length(profile_indices); categorical = true)
for (k, idx) in enumerate(profile_indices)
    t_label = @sprintf("t = %.1f h, dir = %.0f°", times[idx] / 3600, wind_dirs[idx])
    lines!(ax_prof, T_profiles[k], abs.(xz_zo);
           color = cmap_prof[k], linewidth = 2, label = t_label)
    ## mark MLD on each profile
    mld_k = mld_use[idx]
    scatter!(ax_prof, [T_profiles[k][argmin(abs.(abs.(xz_zo) .- mld_k))]],
             [mld_k]; color = cmap_prof[k], markersize = 8, marker = :diamond)
end
axislegend(ax_prof; position = :lb, labelsize = 10, nbanks = 1)

## Column spacer so the time-series axes don't squeeze the profile panel
colgap!(fig_mix.layout, 1, 30)

save("coupled_fjord_mixing.png", fig_mix)
fig_mix

# ```@raw html
# <img src="coupled_fjord_mixing.png" style="max-width:100%"/>
# ```
