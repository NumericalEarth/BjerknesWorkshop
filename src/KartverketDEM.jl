"""
    KartverketDEM

A **custom NumericalEarth dataset** for the Norwegian Mapping Authority
(*Kartverket*) national digital terrain model, exposed through the workshop so the
Norway 100 m LES (`tutorials/day4/03_norway...`) can run on *real* Lofoten
topography instead of synthetic terrain.

This module is a self-contained worked example of **extending NumericalEarth's
metadata system with your own dataset**: `KartverketDTM` subtypes
`NumericalEarth.DataWrangling.AbstractStaticBathymetry`, so it slots into the same
`Metadatum` machinery as the built-in `ETOPO2022`, `GEBCO2024`, … — you just
implement a handful of trait methods plus a `download`.

Two things make Kartverket different from the built-in (lat–lon, single-file)
bathymetry datasets, and both are instructive:

 1. **It is a *projected* dataset.** The DTM is served in UTM zone 33N
    (EPSG:25833), i.e. metres, not degrees. That is a *feature* here: the Norway
    LES runs on a metric `RectilinearGrid` with terrain-following coordinates, and
    the terrain is set by sampling a height function `h(x, y)` at the grid's metric
    nodes (`Breeze.TerrainFollowingDiscretization.materialize_terrain!`). A
    projected DEM maps onto that box directly — no reprojection needed — whereas
    NumericalEarth's lat–lon `regrid_bathymetry` is aimed at geographic grids.

 2. **It is fetched by Web Coverage Service (WCS), windowed.** The full 1 m DTM is
    ~10¹² points; we request only the box we need, at the resolution we need, via a
    WCS 1.0.0 `GetCoverage` returning NetCDF (read by `NCDatasets` — no GDAL).

Data source: Kartverket Høydedata, coverage `nhm_dtm_topo_25833`
(<https://hoydedata.no>, <https://wcs.geonorge.no>). Licensed CC-BY 4.0.
"""
module KartverketDEM

using NCDatasets
using Downloads
using Printf
using Statistics

using Oceananigans
using NumericalEarth
using NumericalEarth.DataWrangling: AbstractStaticBathymetry, Metadata

import NumericalEarth.DataWrangling:
    default_download_directory,
    metadata_filename,
    dataset_variable_name,
    metadata_url,
    construct_native_grid,
    validate_dataset_coverage

import NumericalEarth: regrid_topography, regrid_bathymetry

export KartverketDTM, KartverketWindow, kartverket_metadatum,
       latlon_to_utm33n, kartverket_topography, kartverket_height_function

# ============================================================================
# WGS84 (lat, lon) → UTM zone 33N (EPSG:25833) forward projection
# ============================================================================
#
# Standard transverse-Mercator series (Snyder, USGS PP-1395), accurate to a few
# mm over the Lofoten extent — enough to anchor the metric box on the DTM. We roll
# it by hand so the workshop env needs no Proj/GDAL dependency.

const _UTM33_λ₀  = deg2rad(15.0)   # central meridian of UTM zone 33
const _UTM33_k₀  = 0.9996          # scale factor
const _UTM33_FE  = 500_000.0       # false easting
const _WGS84_a   = 6_378_137.0     # semi-major axis (m)
const _WGS84_f   = 1 / 298.257223563
const _WGS84_e²  = 2 * _WGS84_f - _WGS84_f^2   # first eccentricity squared

"""
    latlon_to_utm33n(lat, lon) -> (easting, northing)

Project geographic `(lat, lon)` in degrees (WGS84) to UTM zone 33N eastings and
northings in metres (EPSG:25833, northern hemisphere).
"""
function latlon_to_utm33n(lat, lon)
    a  = _WGS84_a
    e² = _WGS84_e²
    φ  = deg2rad(lat)
    λ  = deg2rad(lon)

    e′² = e² / (1 - e²)
    N   = a / sqrt(1 - e² * sin(φ)^2)
    T   = tan(φ)^2
    C   = e′² * cos(φ)^2
    A   = (λ - _UTM33_λ₀) * cos(φ)

    M = a * ((1 - e²/4 - 3e²^2/64 - 5e²^3/256) * φ
             - (3e²/8 + 3e²^2/32 + 45e²^3/1024) * sin(2φ)
             + (15e²^2/256 + 45e²^3/1024) * sin(4φ)
             - (35e²^3/3072) * sin(6φ))

    easting = _UTM33_FE + _UTM33_k₀ * N * (A
              + (1 - T + C) * A^3 / 6
              + (5 - 18T + T^2 + 72C - 58e′²) * A^5 / 120)

    northing = _UTM33_k₀ * (M + N * tan(φ) * (A^2 / 2
               + (5 - T + 9C + 4C^2) * A^4 / 24
               + (61 - 58T + T^2 + 600C - 330e′²) * A^6 / 720))

    return easting, northing
end

# ============================================================================
# The dataset and its metric window
# ============================================================================

"""
    KartverketDTM(; coverage = "nhm_dtm_topo_25833")

The Kartverket national DTM as a NumericalEarth static-bathymetry dataset. `coverage`
is the WCS coverage id (the default is the 1 m topographic DTM in EPSG:25833).
"""
struct KartverketDTM <: AbstractStaticBathymetry
    coverage :: String
end

KartverketDTM(; coverage = "nhm_dtm_topo_25833") = KartverketDTM(coverage)

Base.summary(d::KartverketDTM) = "KartverketDTM(\"$(d.coverage)\")"
Base.show(io::IO, d::KartverketDTM) = print(io, summary(d))

"""
    KartverketWindow(; center_lat, center_lon, halfwidth, resolution)

A square metric window on the DTM, used as a `Metadatum`'s `region`. `center_lat`,
`center_lon` (degrees) are projected to UTM 33N to anchor the box; `halfwidth` and
`resolution` are in metres. The fetched grid is `(2·halfwidth / resolution)²` cells.
"""
struct KartverketWindow{T}
    center_easting  :: T
    center_northing :: T
    halfwidth       :: T
    resolution      :: T
end

function KartverketWindow(; center_lat, center_lon, halfwidth, resolution)
    E₀, N₀ = latlon_to_utm33n(center_lat, center_lon)
    return KartverketWindow(promote(E₀, N₀, float(halfwidth), float(resolution))...)
end

## UTM bounding box (xmin, ymin, xmax, ymax) and pixel counts of a window.
utm_bbox(w::KartverketWindow) = (w.center_easting - w.halfwidth, w.center_northing - w.halfwidth,
                                 w.center_easting + w.halfwidth, w.center_northing + w.halfwidth)
npixels(w::KartverketWindow) = round(Int, 2w.halfwidth / w.resolution)

# ============================================================================
# NumericalEarth metadata traits (mirror the ETOPO/IBCAO pattern)
# ============================================================================

const _kartverket_cache = joinpath(@__DIR__, "..", "thursday", "data")

default_download_directory(::KartverketDTM) = _kartverket_cache
Base.size(::KartverketDTM, name = :bottom_height) = (0, 0, 1)  # window-dependent; see Downloads.download
dataset_variable_name(::KartverketDTM) = "Band1"               # WCS NetCDF elevation band

## A region-stamped filename so different windows cache separately.
function metadata_filename(::KartverketDTM, name, date, region::KartverketWindow)
    E = round(Int, region.center_easting); N = round(Int, region.center_northing)
    L = round(Int, region.halfwidth); Δ = round(Int, region.resolution)
    return @sprintf("kartverket_dtm_E%d_N%d_L%d_d%d.nc", E, N, L, Δ)
end
metadata_filename(::KartverketDTM, name, date, ::Nothing) = "kartverket_dtm.nc"

const _WCS_BASE = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-25833"

"""
    wcs_url(dataset, window) -> String

Build the WCS 1.0.0 `GetCoverage` URL that returns the windowed DTM as NetCDF in
EPSG:25833. ArcGIS Server (which backs this service) is reliable with WCS 1.0.0's
explicit `bbox`/`width`/`height`, where it rejects 2.0.1 `subset` syntax.
"""
function wcs_url(dataset::KartverketDTM, w::KartverketWindow)
    xmin, ymin, xmax, ymax = utm_bbox(w)
    n = npixels(w)
    return string(_WCS_BASE,
        "?service=WCS&version=1.0.0&request=GetCoverage",
        "&coverage=", dataset.coverage,
        "&crs=EPSG:25833",
        @sprintf("&bbox=%.3f,%.3f,%.3f,%.3f", xmin, ymin, xmax, ymax),
        "&width=", n, "&height=", n,
        "&format=NetCDF")
end

# ============================================================================
# Metadatum constructor + download
# ============================================================================

"""
    kartverket_metadatum(; center_lat, center_lon, halfwidth, resolution,
                         coverage = "nhm_dtm_topo_25833", dir = <thursday/data>)

Build a `NumericalEarth.DataWrangling.Metadatum` for a metric window of the
Kartverket DTM. `center_lat/center_lon` (deg) anchor the box; `halfwidth` and
`resolution` are metres.
"""
function kartverket_metadatum(; center_lat, center_lon, halfwidth, resolution,
                              coverage = "nhm_dtm_topo_25833",
                              dir = _kartverket_cache)
    # Build the Metadatum manually so we control the (custom, projected) region type.
    dataset = KartverketDTM(; coverage)
    region  = KartverketWindow(; center_lat, center_lon, halfwidth, resolution)
    name    = :bottom_height
    filename = metadata_filename(dataset, name, nothing, region)
    return NumericalEarth.DataWrangling.Metadata(name, dataset, nothing, region, dir, filename)
end

import NumericalEarth.DataWrangling: metadata_path

"""
    Downloads.download(metadatum) -> filepath

Fetch the windowed Kartverket DTM via WCS into the metadatum's cache path (skipped
if already present). Returns the local NetCDF path.
"""
function Downloads.download(metadatum::NumericalEarth.DataWrangling.Metadata{<:KartverketDTM})
    filepath = metadata_path(metadatum)
    mkpath(dirname(filepath))
    if !isfile(filepath)
        url = wcs_url(metadatum.dataset, metadatum.region)
        @info "Downloading Kartverket DTM window via WCS" coverage=metadatum.dataset.coverage filepath
        Downloads.download(url, filepath)
        _check_netcdf(filepath)
    end
    return filepath
end

## WCS errors come back as HTML with a 200-or-400; make sure we actually got NetCDF.
function _check_netcdf(filepath)
    magic = open(filepath) do io; read(io, min(4, filesize(filepath))); end
    # NetCDF classic ("CDF\x01/\x02") or HDF5 ("\x89HDF"). Anything else is an error page.
    ok = length(magic) ≥ 3 && (magic[1:3] == b"CDF" || (length(magic) == 4 && magic == UInt8[0x89,0x48,0x44,0x46]))
    if !ok
        body = String(read(filepath))
        rm(filepath; force = true)
        error("Kartverket WCS did not return NetCDF (got $(length(body)) bytes). First 200 chars:\n" *
              first(replace(body, r"<[^>]*>" => " "), 200))
    end
    return nothing
end

# ============================================================================
# Reading the DTM and building a height function for the metric box
# ============================================================================

"""
    kartverket_topography(metadatum) -> (x, y, h)

Download (if needed) and read the windowed DTM. Returns UTM-33N easting `x`,
northing `y` (metres, monotonically increasing), and elevation `h[i, j]` in metres
(ocean is 0). `h` is oriented so `h[i, j]` is at `(x[i], y[j])`.
"""
function kartverket_topography(metadatum::NumericalEarth.DataWrangling.Metadata{<:KartverketDTM})
    filepath = Downloads.download(metadatum)
    x, y, h = NCDataset(filepath) do ds
        x = Array{Float64}(ds["x"][:])
        y = Array{Float64}(ds["y"][:])
        H = coalesce.(Array(ds[dataset_variable_name(metadatum.dataset)]), 0.0f0)
        # WCS NetCDF is (x, y); elevation already (Nx, Ny). Ensure y ascending.
        if length(y) > 1 && y[2] < y[1]
            y = reverse(y); H = H[:, end:-1:1]
        end
        x, y, Array{Float64}(H)
    end
    return x, y, h
end

"""
    kartverket_height_function(metadatum; recenter = true) -> h(x, y)

Return a bilinear interpolant of the windowed DTM. With `recenter = true` (the
default) the returned function takes **box-centred metric coordinates** `(x, y)`
where `(0, 0)` is the window centre — exactly what the Norway LES grid uses
(`x ∈ [-L/2, L/2]`). With `recenter = false` it takes absolute UTM-33N
eastings/northings. Out-of-range queries clamp to the edge.
"""
function kartverket_height_function(metadatum::NumericalEarth.DataWrangling.Metadata{<:KartverketDTM};
                                    recenter = true)
    xs, ys, h = kartverket_topography(metadatum)
    E₀ = metadatum.region.center_easting
    N₀ = metadatum.region.center_northing
    interp = _bilinear(h, xs, ys)
    return recenter ? (x, y) -> interp(x + E₀, y + N₀) : interp
end

# ============================================================================
# `regrid_topography` compatibility (NumericalEarth.Bathymetry)
# ============================================================================
#
# NumericalEarth's regridding pipeline (`regrid_bathymetry` → `regrid_topography`)
# needs three things from a dataset: a *native grid*, the NetCDF *variable name*,
# and a *coverage check*. The DTM window and the LES grid are both metric and
# rectilinear (UTM 33N metres, recentred on the window), so Oceananigans'
# `interpolate!` regrids between them directly — no lat–lon detour.

dataset_variable_name(m::Metadata{<:KartverketDTM}) = dataset_variable_name(m.dataset)

## The native grid of the windowed DTM: one cell per DTM pixel, centres at the
## pixel coordinates, recentred so (0, 0) is the window centre — the same frame
## the LES grid uses. Requires the file (downloads on demand; idempotent).
function construct_native_grid(metadatum::Metadata{<:KartverketDTM}, region::KartverketWindow,
                               arch; halo = (10, 10, 1))
    filepath = Downloads.download(metadatum)
    x, y = NCDataset(filepath) do ds
        Array{Float64}(ds["x"][:]), Array{Float64}(ds["y"][:])
    end
    (length(y) > 1 && y[2] < y[1]) &&
        error("Kartverket WCS NetCDF has descending northings; the regrid pipeline assumes " *
              "ascending coordinates. Re-fetch the window (recent WCS responses are ascending).")
    Δx = (x[end] - x[1]) / (length(x) - 1)
    Δy = (y[end] - y[1]) / (length(y) - 1)
    E₀, N₀ = region.center_easting, region.center_northing
    return RectilinearGrid(arch; size = (length(x), length(y)), halo = (halo[1], halo[2]),
                           x = (x[1] - Δx/2 - E₀, x[end] + Δx/2 - E₀),
                           y = (y[1] - Δy/2 - N₀, y[end] + Δy/2 - N₀),
                           topology = (Bounded, Bounded, Flat))
end

## The window must cover the target grid (both in recentred metric coordinates).
function validate_dataset_coverage(grid, metadatum::Metadata{<:KartverketDTM})
    w = metadatum.region
    x₋, x₊ = Oceananigans.Grids.x_domain(grid)
    y₋, y₊ = Oceananigans.Grids.y_domain(grid)
    if x₋ < -w.halfwidth || x₊ > w.halfwidth || y₋ < -w.halfwidth || y₊ > w.halfwidth
        error("Target grid ($(x₋)..$(x₊), $(y₋)..$(y₊)) m exceeds the Kartverket window " *
              "(±$(w.halfwidth) m). Fetch a wider window with kartverket_metadatum.")
    end
    return nothing
end

"""
    regrid_topography(target_grid, metadatum::Metadata{<:KartverketDTM}; kw...)

Regrid the windowed Kartverket DTM onto `target_grid` (a metric `RectilinearGrid`
recentred on the window) with NumericalEarth's bathymetry-regridding pipeline, then
clamp to land elevation (ocean → 0). `major_basins` is disabled — this is terrain,
not bathymetry. Returns a `Field{Center, Center, Nothing}` on `target_grid`.
"""
function regrid_topography(target_grid, metadatum::Metadata{<:KartverketDTM};
                           interpolation_passes = 1, cache = true)
    elevation = regrid_bathymetry(target_grid, metadatum;
                                  interpolation_passes, major_basins = Inf, cache)
    parent(elevation) .= max.(parent(elevation), 0)   # land elevation; ocean → 0
    return elevation
end

## Bilinear interpolation over a regular (xs, ys) grid; clamps to the domain edge.
function _bilinear(arr, xs, ys)
    x0, x1 = first(xs), last(xs); y0, y1 = first(ys), last(ys)
    nx, ny = length(xs), length(ys)
    dx = (x1 - x0) / (nx - 1); dy = (y1 - y0) / (ny - 1)
    return function (x, y)
        fx = clamp((x - x0) / dx, 0, nx - 1 - 1e-9)
        fy = clamp((y - y0) / dy, 0, ny - 1 - 1e-9)
        i = floor(Int, fx) + 1; j = floor(Int, fy) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        @inbounds (arr[i, j]   * (1 - tx) * (1 - ty) + arr[i+1, j]   * tx * (1 - ty) +
                   arr[i, j+1] * (1 - tx) * ty       + arr[i+1, j+1] * tx * ty)
    end
end

end # module
