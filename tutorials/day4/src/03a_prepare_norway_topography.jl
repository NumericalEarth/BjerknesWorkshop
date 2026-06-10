# # Preparing the Norway topography
#
# *Boundary heterogeneity writes turbulence into the fluid — case 3 setup.*
#
# The Norway case study (`03_...`) runs a real-terrain atmospheric LES over a
# 100 km × 100 km patch of coastal northern Norway (Lofoten) at 100 m resolution.
# The simulation tutorial must **not** download or reproject DEM data live — that
# pulls in a heavy GDAL/PROJ stack that is fragile on the GPU production machine.
# Instead, *this* preprocessing source produces a cached artifact:
#
# ```text
# thursday/data/norway_lofoten_100m_topography.jld2
# ```
#
# holding `(; x, y, h_raw, h, land_mask, ocean_mask, taper_mask, source_metadata)`.
#
# ## Three modes
#
# 1. **Kartverket** (`TOPO_SOURCE=kartverket`): real Lofoten terrain from the
#    Norwegian Mapping Authority national DTM, fetched by Web Coverage Service
#    through the workshop's custom NumericalEarth dataset (`src/KartverketDEM.jl`).
#    The DTM is served in UTM 33N (metres) and returned as NetCDF (read by
#    `NCDatasets`) — **no GDAL/PROJ needed**, just network access. This is the
#    recommended real-terrain path; see `KartverketDEM` for how it extends
#    NumericalEarth's metadata system with a projected, windowed dataset.
# 2. **Generic DEM** (`TOPO_SOURCE=dem`): the documented GLO-30/ASTER pipeline
#    below — download tiles, reproject to local UTM, crop, resample. This requires
#    `ArchGDAL`/`Rasters` and is left as a stub for a dev machine.
# 3. **Synthetic** (`TOPO_SOURCE=synthetic`, the default): a Lofoten-flavored
#    idealized terrain — steep coastal massifs, fjord incisions, islands, and a
#    land/sea split — produced with no external dependencies. This makes the whole
#    Thursday workflow runnable with zero data access.
#
# All modes write the *same* artifact schema, so `03_...` does not care which was
# used.

using Oceananigans.Units
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

## The workshop's custom Kartverket DTM dataset (extends NumericalEarth metadata).
include(joinpath(@__DIR__, "..", "..", "..", "src", "KartverketDEM.jl"))

Random.seed!(68_13)   # center_lat, center_lon mnemonic

const TOPO_SOURCE = Symbol(get(ENV, "TOPO_SOURCE", "synthetic"))

# ## Domain definition
#
# Lofoten / coastal northern Norway: steep mountains, fjords, islands, and sharp
# land/sea contrasts inside a 100 km square.

const center_lat = 68.15
const center_lon = 13.70
const Lx = 100kilometers
const Ly = 100kilometers
const Δ  = 100meters          # target horizontal resolution
const taper_width = 12kilometers   # outer numerical buffer

Nx = Int(Lx ÷ Δ)
Ny = Int(Ly ÷ Δ)

x = range(-Lx/2, Lx/2, length = Nx)
y = range(-Ly/2, Ly/2, length = Ny)

const datadir = joinpath("thursday", "data")
mkpath(datadir)
const artifact_path = joinpath(datadir, "norway_lofoten_100m_topography.jld2")

# ## Terrain smoothing target
#
# The LES cannot resolve arbitrarily steep slopes; we smooth raw topography to a
# maximum resolved slope of ≈0.3–0.5 over a smoothing length of 300–500 m, and
# taper the outer 12 km to flat so the periodic rim is a clean numerical buffer.

const max_slope = 0.4
const smoothing_length = 400meters

# ## Real-DEM pipeline (documented; runs on a dev machine with GDAL)
#
# This function sketches the production preprocessing. It is intentionally guarded
# behind `TOPO_SOURCE=dem` and a `try` so the synthetic path stays dependency-free.

function prepare_from_dem(x, y)
    @info "Loading DEM and reprojecting to local UTM (requires ArchGDAL/Rasters)…"
    error("""
          Real-DEM preprocessing is not wired up in this environment.
          On a dev machine with network + GDAL:
            1. Download a DEM covering ($(center_lat)°N, $(center_lon)°E) ± 60 km
               (e.g. Copernicus GLO-30, ASTER GDEM, or Kartverket DTM).
            2. Reproject to the local UTM zone (33N for Lofoten).
            3. Crop a 100 km × 100 km square centered on the domain.
            4. Resample to $(Int(Δ)) m (e.g. bilinear).
            5. Return the raw height array h_raw[i, j] on the (x, y) grid.
          Then the shared smoothing / mask / taper steps below produce the artifact.
          """)
end

# ## Real Kartverket DTM (via the custom NumericalEarth dataset)
#
# Build a `Metadatum` for a metric window centered on the domain (`halfwidth = Lx/2`)
# at the case resolution, fetch the DTM via WCS, and sample its height function onto
# the case `(x, y)` grid (box-centred metres). Kartverket returns ocean as 0, so the
# shared `h_raw .> 0` land mask below works directly. No reprojection — the DTM is
# already in the UTM 33N metric frame the LES grid uses.

function prepare_from_kartverket(x, y)
    @info "Fetching Kartverket DTM via WCS and sampling onto the case grid…" center_lat center_lon
    metadatum = KartverketDEM.kartverket_metadatum(; center_lat, center_lon,
                                                   halfwidth = Float64(Lx / 2),
                                                   resolution = Float64(Δ))
    h_fun = KartverketDEM.kartverket_height_function(metadatum)   # box-centred metric coords
    Nx, Ny = length(x), length(y)
    return Float64[h_fun(x[i], y[j]) for i in 1:Nx, j in 1:Ny]
end

# ## Synthetic Lofoten-flavored terrain (dependency-free fallback)
#
# A deterministic pseudo-terrain: a few steep coastal massifs, a couple of fjord
# incisions cutting inland from a roughly NW–SE coastline, and offshore islands.
# Heights are in meters; ocean is negative (will be clamped to zero land height
# with an ocean mask).

function synthetic_lofoten(x, y)
    Nx, Ny = length(x), length(y)
    h = zeros(Float64, Nx, Ny)

    ## Coastline: land to the SE of a tilted line, ocean to the NW.
    coast(xi, yj) = (xi + yj) / √2   # signed distance along NW–SE normal (m)

    ## A handful of massifs (center_x, center_y, height, radius) in meters.
    massifs = [(-10e3,  8e3, 1100.0, 6e3),
               ( 12e3, -6e3, 1400.0, 7e3),
               ( 28e3,  20e3, 900.0, 5e3),
               (-25e3, -18e3, 1000.0, 6e3),
               (  2e3,  30e3, 750.0, 4e3)]

    ## Fjords: narrow low corridors cutting inland (center line + width).
    fjords = [(-5e3, 0.5, 1.5e3, 22e3),    # (x-intercept, slope, half-width, reach)
              (18e3, -0.8, 1.2e3, 26e3)]

    for i in 1:Nx, j in 1:Ny
        xi, yj = x[i], y[j]
        land = smooth_step(coast(xi, yj) + 6e3, 4e3)   # 1 over land, 0 over ocean

        elev = 0.0
        for (cx, cy, hmax, r) in massifs
            elev += hmax * exp(-((xi - cx)^2 + (yj - cy)^2) / (2r^2))
        end
        ## ridged detail
        elev *= 1 + 0.25 * sin(xi / 1.8e3) * cos(yj / 2.1e3)

        ## carve fjords
        for (x0, s, hw, reach) in fjords
            dline = abs(xi - (x0 + s * yj))
            inland = smooth_step(reach - sqrt(xi^2 + yj^2), 6e3)
            elev *= 1 - 0.85 * inland * exp(-(dline^2) / (2hw^2))
        end

        ## land/ocean split: ocean gets a shallow negative shelf
        h[i, j] = land * max(elev, 0.0) + (1 - land) * (-50.0)
    end
    return h
end

# ## Shared post-processing: smooth, mask, zero ocean, taper
#
# A simple separable Gaussian smoother (no external deps) limits resolved slope.

function gaussian_smooth(h, x, y; smoothing_length)
    Nx, Ny = size(h)
    dx = step(x); dy = step(y)
    rx = max(1, round(Int, smoothing_length / dx))
    ry = max(1, round(Int, smoothing_length / dy))
    σx = smoothing_length / dx / 2
    σy = smoothing_length / dy / 2
    wx = [exp(-(k^2) / (2σx^2)) for k in -rx:rx]; wx ./= sum(wx)
    wy = [exp(-(k^2) / (2σy^2)) for k in -ry:ry]; wy ./= sum(wy)

    tmp = similar(h)
    @inbounds for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for (m, kk) in enumerate(-rx:rx)
            ii = clamp(i + kk, 1, Nx)
            acc += wx[m] * h[ii, j]
        end
        tmp[i, j] = acc
    end
    out = similar(h)
    @inbounds for j in 1:Ny, i in 1:Nx
        acc = 0.0
        for (m, kk) in enumerate(-ry:ry)
            jj = clamp(j + kk, 1, Ny)
            acc += wy[m] * tmp[i, jj]
        end
        out[i, j] = acc
    end
    return out
end

# Build the raw height field.
h_raw = if TOPO_SOURCE === :kartverket
    try
        prepare_from_kartverket(x, y)
    catch err
        @warn "Kartverket path failed; falling back to synthetic terrain." exception = err
        synthetic_lofoten(x, y)
    end
elseif TOPO_SOURCE === :dem
    try
        prepare_from_dem(x, y)
    catch err
        @warn "DEM path failed; falling back to synthetic terrain." exception = err
        synthetic_lofoten(x, y)
    end
else
    synthetic_lofoten(x, y)
end

# Masks: land where raw elevation > 0.
land_mask  = Float64.(h_raw .> 0)
ocean_mask = 1 .- land_mask

# Smooth, clamp ocean to zero height, then taper the rim to flat.
h_smooth = gaussian_smooth(max.(h_raw, 0.0), x, y; smoothing_length)
h_smooth .*= land_mask   # ocean topography is exactly zero

taper_mask = [edge_taper(x[i], y[j], Lx, Ly; taper_width) for i in 1:Nx, j in 1:Ny]
h = h_smooth .* taper_mask

# Report resolved slope so we know the smoothing target was met.
dhdx = maximum(abs, diff(h, dims = 1)) / step(x)
dhdy = maximum(abs, diff(h, dims = 2)) / step(y)
@info "Topography prepared" TOPO_SOURCE Nx Ny max_height = maximum(h) max_resolved_slope = max(dhdx, dhdy) target = max_slope

source_metadata = (; source = TOPO_SOURCE,
                     center_lat, center_lon,
                     Lx = Float64(Lx), Ly = Float64(Ly), Δ = Float64(Δ),
                     smoothing_length = Float64(smoothing_length),
                     taper_width = Float64(taper_width),
                     created = string(ThursdayLES.now()))

# ## Save the artifact

jldsave(artifact_path;
        x = collect(x), y = collect(y),
        h_raw, h, land_mask, ocean_mask, taper_mask, source_metadata)
@info "Saved topography artifact" artifact_path

# ## Validation figure
#
# Raw vs. smoothed topography, the land/ocean mask, and the taper — the figure
# alone validates much of the setup before any expensive run.

using CairoMakie

xkm = collect(x) ./ 1e3
ykm = collect(y) ./ 1e3

fig = Figure(size = (1200, 1000))
ax1 = Axis(fig[1, 1], title = "Raw topography (m)", xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
ax2 = Axis(fig[1, 2], title = "Smoothed + tapered h (m)", xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
ax3 = Axis(fig[2, 1], title = "Land / ocean mask", xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
ax4 = Axis(fig[2, 2], title = "Taper mask", xlabel = "x (km)", ylabel = "y (km)", aspect = 1)

hm1 = heatmap!(ax1, xkm, ykm, h_raw, colormap = :terrain)
hm2 = heatmap!(ax2, xkm, ykm, h, colormap = :terrain)
heatmap!(ax3, xkm, ykm, land_mask, colormap = :grays)
heatmap!(ax4, xkm, ykm, taper_mask, colormap = :grays)
Colorbar(fig[1, 0], hm1, label = "m")

figpath = joinpath("thursday", "figures", "norway_raw_vs_smoothed_topography.png")
mkpath(dirname(figpath))
save(figpath, fig)
@info "Saved validation figure" figpath
fig
