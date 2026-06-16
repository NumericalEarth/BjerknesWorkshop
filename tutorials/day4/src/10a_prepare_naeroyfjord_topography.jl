# # Preparing the Nærøyfjord topography & bathymetry
#
# *Coupled air–sea LES of a fjord — case-10 setup.*
#
# The Nærøyfjord case (`10_...`) runs a **coupled atmosphere–ocean** simulation over the
# narrow, steep-walled fjord that ends at **Gudvangen** (≈ 60.876° N, 6.845° E). Like the
# Lofoten case (`03a`), the simulation tutorial must **not** download or reproject data
# live; *this* preprocessing source produces a cached artifact
#
# ```text
# thursday/data/naeroyfjord_topography.jld2
# ```
#
# holding `(; x, y, h, h_terrain, depth, land_mask, ocean_mask, taper_mask,
# source_metadata)` on a metric grid whose `y`-axis is **aligned with the fjord axis**.
#
# ## What is different from the Lofoten case
#
#  1. **It needs the seafloor, not just the walls.** A fjord case needs *bathymetry*
#     (the water depth) as well as the terrain. Kartverket serves the seafloor only as
#     WMS imagery, so we get the fjord depth from **EMODnet Bathymetry** (a combined
#     land+sea DTM; we keep its negative/seafloor part) and the steep walls from the
#     far finer **Kartverket DTM**, fusing them into one signed elevation `h` (land > 0,
#     water < 0). See `src/FjordBathymetry.jl` and `src/KartverketDEM.jl`.
#  2. **It is UTM zone 32, not 33.** Gudvangen (6.8° E) is in UTM zone 32N — the
#     generalized `KartverketDEM` handles the zone (endpoint, EPSG, projection).
#  3. **The box is rotated to the fjord axis.** The fjord runs ≈ NNE from Gudvangen, so
#     we rotate the metric box by `fjord_azimuth` and put the **head (Gudvangen) at the
#     south (−y) end**. Then "down-fjord" is a clean `−y` direction for the rotating-wind
#     forcing in `10_...`.
#
# ## Modes
#
#  - **Synthetic** (`TOPO_SOURCE=synthetic`, the default): a parametric Nærøyfjord
#    (`FjordBathymetry.synthetic_naeroyfjord`) — runs with zero network access.
#  - **Real** (`TOPO_SOURCE=real`): Kartverket walls + EMODnet seafloor, fused.
#
# Both write the same artifact schema, so `10_...` does not care which was used.

using Oceananigans
using Oceananigans.Units
using JLD2
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "..", "..", "..", "src", "FjordBathymetry.jl"))
using .FjordBathymetry

const repo_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TOPO_SOURCE = Symbol(get(ENV, "TOPO_SOURCE", "synthetic"))

## The real path needs KartverketDEM (which pulls NumericalEarth). Include it at TOP
## LEVEL (not inside the function) so its methods are visible in the run-time world —
## an in-function `include` defines them in a newer world age and they aren't callable.
if TOPO_SOURCE === :real
    include(joinpath(repo_root, "src", "KartverketDEM.jl"))
    using .KartverketDEM
end

Random.seed!(6845)   # center_lon mnemonic

# ## Site and box geometry
#
# The metric box: `x` is **cross-fjord**, `y` is **along-fjord** (pointing NNE, toward
# the fjord mouth; the Gudvangen head is at −y). `fjord_azimuth` is the compass bearing
# (deg, clockwise from north) of the `+y` axis; the box is anchored at `(center_lat,
# center_lon)` and rotated into UTM zone 32 metres.

const utm_zone     = 32
const center_lat   = 60.905     # ~mid-fjord, so the head and a reach of channel both fit
const center_lon   = 6.880
const fjord_azimuth = 34.0       # deg, +y axis bearing (Gudvangen→mouth ≈ NNE)

const Lx = get(ENV, "FJORD_LX", "") == "" ?  8kilometers : parse(Float64, ENV["FJORD_LX"])
const Ly = get(ENV, "FJORD_LY", "") == "" ? 16kilometers : parse(Float64, ENV["FJORD_LY"])
const Δ  = get(ENV, "FJORD_DX", "") == "" ? 50meters     : parse(Float64, ENV["FJORD_DX"])
const taper_width = 1.5kilometers   # outer numerical buffer (flattened rim)

# Smoothing: the **smallest** wall smoothing that keeps the fjord open while bounding the
# atmosphere's resolved terrain slope. Nærøyfjord walls are near-vertical; over-smoothing
# fills the ~250–500 m channel, so we keep this tight (a few grid cells) and *report* the
# resulting channel width and max slope below. (Overridable for the smoothing study.)
const smoothing_length = get(ENV, "FJORD_SMOOTH", "") == "" ? 120meters : parse(Float64, ENV["FJORD_SMOOTH"])
const max_slope_target = 1.5

Nx = Int(round(Lx / Δ))
Ny = Int(round(Ly / Δ))
x = range(-Lx/2, Lx/2, length = Nx)
y = range(-Ly/2, Ly/2, length = Ny)

const datadir = joinpath(repo_root, "thursday", "data")
const figdir  = joinpath(repo_root, "thursday", "figures")
mkpath(datadir); mkpath(figdir)
## Optional filename tag (e.g. FJORD_TAG=real) so synthetic/real artifacts can coexist.
const TAG = get(ENV, "FJORD_TAG", "")
const _sfx = isempty(TAG) ? "" : "_" * TAG
const artifact_path = joinpath(datadir, "naeroyfjord_topography$(_sfx).jld2")

# ## Box → UTM rotation
#
# `+y` points at bearing `α`; `+x` is 90° clockwise from `+y`. A box point `(xi, yj)`
# (metres, recentred) maps to UTM zone-32 easting/northing about the box-centre anchor.

const α = deg2rad(fjord_azimuth)
@inline function box_to_utm(xi, yj, E₀, N₀)
    E = E₀ + xi * cos(α) + yj * sin(α)
    N = N₀ - xi * sin(α) + yj * cos(α)
    return E, N
end

# ## Real source: Kartverket walls (UTM) + EMODnet seafloor (lat–lon), fused
#
# Deferred `include` of `KartverketDEM` (which pulls NumericalEarth/Oceananigans) so the
# synthetic path stays light. The Kartverket window must cover the *rotated* box, so its
# half-width is the box half-diagonal plus a margin. EMODnet is requested over the
# lat–lon bounding box of the rotated corners.

function prepare_from_real(x, y)
    KV = KartverketDEM   # included at top level (see above)

    E₀, N₀ = KV.latlon_to_utm(center_lat, center_lon; zone = utm_zone)

    ## Kartverket window: cover the rotated box (half-diagonal) + margin.
    half_diag = 0.5 * hypot(last(x) - first(x), last(y) - first(y))
    hw = ceil(half_diag + 1000)
    @info "Fetching Kartverket DTM (zone $utm_zone) window" center_lat center_lon halfwidth=hw
    md = KV.kartverket_metadatum(; center_lat, center_lon, halfwidth = Float64(hw),
                                 resolution = Float64(Δ), zone = utm_zone)
    kart = KV.kartverket_height_function(md; recenter = false)   # interp(E, N), land ≥ 0, ocean 0

    ## EMODnet lat–lon bbox of the rotated box corners (+ margin).
    corners = [(first(x), first(y)), (last(x), first(y)), (first(x), last(y)), (last(x), last(y))]
    latlons = [KV.utm_to_latlon(box_to_utm(cx, cy, E₀, N₀)...; zone = utm_zone) for (cx, cy) in corners]
    lats = first.(latlons); lons = last.(latlons)
    m = 0.01
    @info "Fetching EMODnet seafloor window" lat = (minimum(lats), maximum(lats)) lon = (minimum(lons), maximum(lons))
    epath = FjordBathymetry.emodnet_download(; lat_min = minimum(lats) - m, lat_max = maximum(lats) + m,
                                             lon_min = minimum(lons) - m, lon_max = maximum(lons) + m,
                                             dir = datadir, filename = "emodnet_naeroyfjord.txt")
    emod = FjordBathymetry.emodnet_height_function(epath)        # interp(lon, lat), signed

    ## Fuse: water (EMODnet < 0) keeps the seafloor depth; land takes the fine Kartverket height.
    h = Array{Float64}(undef, length(x), length(y))
    for j in eachindex(y), i in eachindex(x)
        E, N = box_to_utm(x[i], y[j], E₀, N₀)
        lat, lon = KV.utm_to_latlon(E, N; zone = utm_zone)
        he = emod(lon, lat)
        h[i, j] = he < 0 ? he : max(kart(E, N), 0.0)
    end
    return h
end

# ## Shared post-processing: separable Gaussian smoother (no external deps)

function gaussian_smooth(h, x, y; smoothing_length)
    Nx, Ny = size(h)
    dx = step(x); dy = step(y)
    rx = max(1, round(Int, smoothing_length / dx)); ry = max(1, round(Int, smoothing_length / dy))
    σx = smoothing_length / dx / 2; σy = smoothing_length / dy / 2
    wx = [exp(-(k^2) / (2σx^2)) for k in -rx:rx]; wx ./= sum(wx)
    wy = [exp(-(k^2) / (2σy^2)) for k in -ry:ry]; wy ./= sum(wy)
    tmp = similar(h); out = similar(h)
    @inbounds for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for (m, kk) in enumerate(-rx:rx); acc += wx[m] * h[clamp(i + kk, 1, Nx), j]; end
        tmp[i, j] = acc
    end
    @inbounds for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for (m, kk) in enumerate(-ry:ry); acc += wy[m] * tmp[i, clamp(j + kk, 1, Ny)]; end
        out[i, j] = acc
    end
    return out
end

@inline _ramp(r, δ) = (1 + tanh(r / δ)) / 2
@inline function _taper(xi, yj, Lx, Ly, tw)
    sx = _ramp(xi + Lx/2 - tw, tw/3) * _ramp(Lx/2 - tw - xi, tw/3)
    sy = _ramp(yj + Ly/2 - tw, tw/3) * _ramp(Ly/2 - tw - yj, tw/3)
    return sx * sy
end
edge_taper_mask(x, y, Lx, Ly; taper_width) =
    [_taper(x[i], y[j], Lx, Ly, taper_width) for i in 1:length(x), j in 1:length(y)]

# ## Build the raw signed elevation field

h_signed = if TOPO_SOURCE === :real
    try
        prepare_from_real(x, y)
    catch err
        @warn "Real (Kartverket+EMODnet) path failed; falling back to synthetic fjord." exception = (err, catch_backtrace())
        synthetic_naeroyfjord(x, y; Lx = Float64(Lx), Ly = Float64(Ly))
    end
else
    synthetic_naeroyfjord(x, y; Lx = Float64(Lx), Ly = Float64(Ly))
end

# Masks and water depth (positive down).
land_mask  = Float64.(h_signed .> 0)
ocean_mask = Float64.(h_signed .< 0)
depth_raw  = max.(-h_signed, 0.0)        # ≥ 0 water depth; 0 over land

# ## Atmosphere terrain: smooth the walls minimally, then taper the rim to flat
#
# The atmosphere LES carves `h_terrain ≥ 0` (land; the water surface is 0). We mask ocean
# to 0 *before* smoothing (so the Gaussian rounds the coastal cliffs rather than bleeding
# inland height across a sharp post-smoothing step), then taper the outer rim flat.

h_land   = max.(h_signed, 0.0)
h_smooth = gaussian_smooth(h_land, x, y; smoothing_length)
taper_mask = edge_taper_mask(x, y, Lx, Ly; taper_width)
h_terrain = h_smooth .* taper_mask

# Ocean depth: a light smooth keeps the bottom well-conditioned without filling the channel.
depth = gaussian_smooth(depth_raw, x, y; smoothing_length = max(Δ, smoothing_length / 2)) .* (depth_raw .> 0)

# ## Diagnostics — fjord preservation vs slope (the "don't over-smooth" check)
#
# Report the resolved terrain slope (must stay tractable for the terrain-following
# atmosphere) and the open-water channel width at the head and mid-fjord (must survive
# the smoothing). These are the numbers that decide the smoothing length.

dhdx = maximum(abs, diff(h_terrain, dims = 1)) / step(x)
dhdy = maximum(abs, diff(h_terrain, dims = 2)) / step(y)
function channel_width_at(jj)
    col = ocean_mask[:, jj]
    n = count(>(0), col)
    return n * step(x)
end
j_head = max(1, round(Int, Ny * 0.15)); j_mid = Ny ÷ 2
@info "Topography prepared" TOPO_SOURCE Nx Ny smoothing_length max_resolved_slope = max(dhdx, dhdy) target = max_slope_target
@info "Fjord geometry" max_wall_m = round(maximum(h_terrain)) max_depth_m = round(maximum(depth)) water_fraction = round(sum(ocean_mask) / length(ocean_mask), digits = 3) channel_width_head_m = round(channel_width_at(j_head)) channel_width_mid_m = round(channel_width_at(j_mid))

source_metadata = (; source = TOPO_SOURCE, center_lat, center_lon, utm_zone, fjord_azimuth,
                     Lx = Float64(Lx), Ly = Float64(Ly), Δ = Float64(Δ),
                     smoothing_length = Float64(smoothing_length),
                     taper_width = Float64(taper_width), created = string(now()))

# ## Save the artifact

jldsave(artifact_path;
        x = collect(x), y = collect(y),
        h = h_signed, h_terrain, depth,
        land_mask, ocean_mask, taper_mask, source_metadata)
@info "Saved topography artifact" artifact_path

# ## Validation figure
#
# Signed bathymetry+terrain, the atmosphere terrain, the water depth, and the land/ocean
# mask — the figure validates the fjord geometry before any GPU run.

using CairoMakie

xkm = collect(x) ./ 1e3; ykm = collect(y) ./ 1e3
fig = Figure(size = (1300, 1000))
ax1 = Axis(fig[1, 1]; title = "Signed elevation h (m): land>0, sea<0", xlabel = "cross-fjord x (km)", ylabel = "along-fjord y (km)", aspect = DataAspect())
ax2 = Axis(fig[1, 2]; title = "Atmosphere terrain h_terrain (m)", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
ax3 = Axis(fig[2, 1]; title = "Water depth (m)", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
ax4 = Axis(fig[2, 2]; title = "Land / ocean mask", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())

cr = maximum(abs, h_signed)
hm1 = heatmap!(ax1, xkm, ykm, h_signed; colormap = :topo, colorrange = (-cr, cr))
hm2 = heatmap!(ax2, xkm, ykm, h_terrain; colormap = :terrain)
hm3 = heatmap!(ax3, xkm, ykm, depth; colormap = :deep)
heatmap!(ax4, xkm, ykm, land_mask; colormap = :grays)
Colorbar(fig[1, 0], hm1); Colorbar(fig[2, 0], hm3, label = "m")
Label(fig[0, :], "Nærøyfjord ($(TOPO_SOURCE)) — Gudvangen head at −y; smoothing $(Int(smoothing_length)) m", fontsize = 18)

figpath = joinpath(figdir, "naeroyfjord_topography$(_sfx).png")
save(figpath, fig)
@info "Saved validation figure" figpath
fig
