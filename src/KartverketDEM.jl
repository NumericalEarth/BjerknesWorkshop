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
       latlon_to_utm, latlon_to_utm33n, utm_central_meridian,
       kartverket_topography, kartverket_height_function

# ============================================================================
# WGS84 (lat, lon) → UTM zone N (EPSG:258zz) forward projection
# ============================================================================
#
# Standard transverse-Mercator series (Snyder, USGS PP-1395), accurate to a few
# mm over a workshop-sized extent — enough to anchor the metric box on the DTM. We
# roll it by hand so the workshop env needs no Proj/GDAL dependency.
#
# The projection is parametrized by the UTM zone so the same machinery serves both
# **zone 33N** (Lofoten, ~13.7° E — the original Norway case) and **zone 32N**
# (western Norway / Sognefjord–Nærøyfjord, ~6.8° E — the fjord case). Each Norwegian
# UTM zone has its own Geonorge national-DTM coverage and WCS endpoint (`…-258zz`),
# so anchoring in the *correct* zone keeps scale distortion small (a few cm/m), where
# forcing everything through zone 33 would distort a 6.8° E box by ~8° of longitude.

const _UTM_k₀  = 0.9996          # scale factor
const _UTM_FE  = 500_000.0       # false easting
const _WGS84_a   = 6_378_137.0     # semi-major axis (m)
const _WGS84_f   = 1 / 298.257223563
const _WGS84_e²  = 2 * _WGS84_f - _WGS84_f^2   # first eccentricity squared

## Central meridian of a UTM zone (degrees): zone 32 → 9° E, zone 33 → 15° E.
utm_central_meridian(zone::Integer) = 6 * zone - 183

"""
    latlon_to_utm(lat, lon; zone = 33) -> (easting, northing)

Project geographic `(lat, lon)` in degrees (WGS84) to UTM zone `zone` (northern
hemisphere) eastings and northings in metres (EPSG:258`zone`). Defaults to zone 33
(Lofoten); pass `zone = 32` for western Norway (Sognefjord / Nærøyfjord).
"""
function latlon_to_utm(lat, lon; zone::Integer = 33)
    a  = _WGS84_a
    e² = _WGS84_e²
    φ  = deg2rad(lat)
    λ  = deg2rad(lon)
    λ₀ = deg2rad(utm_central_meridian(zone))

    e′² = e² / (1 - e²)
    N   = a / sqrt(1 - e² * sin(φ)^2)
    T   = tan(φ)^2
    C   = e′² * cos(φ)^2
    A   = (λ - λ₀) * cos(φ)

    M = a * ((1 - e²/4 - 3e²^2/64 - 5e²^3/256) * φ
             - (3e²/8 + 3e²^2/32 + 45e²^3/1024) * sin(2φ)
             + (15e²^2/256 + 45e²^3/1024) * sin(4φ)
             - (35e²^3/3072) * sin(6φ))

    easting = _UTM_FE + _UTM_k₀ * N * (A
              + (1 - T + C) * A^3 / 6
              + (5 - 18T + T^2 + 72C - 58e′²) * A^5 / 120)

    northing = _UTM_k₀ * (M + N * tan(φ) * (A^2 / 2
               + (5 - T + 9C + 4C^2) * A^4 / 24
               + (61 - 58T + T^2 + 600C - 330e′²) * A^6 / 720))

    return easting, northing
end

## Backward-compatible alias: the original zone-33-only entry point.
latlon_to_utm33n(lat, lon) = latlon_to_utm(lat, lon; zone = 33)

"""
    utm_to_latlon(easting, northing; zone = 33) -> (lat, lon)

Inverse of [`latlon_to_utm`](@ref): map UTM zone `zone` (northern hemisphere)
eastings/northings in metres back to geographic `(lat, lon)` in degrees (WGS84). Uses
the standard inverse transverse-Mercator series (Snyder, USGS PP-1395). Needed to
sample a geographic (lat–lon) dataset — e.g. EMODnet bathymetry — onto the metric UTM
grid the fjord LES runs on.
"""
function utm_to_latlon(easting, northing; zone::Integer = 33)
    a  = _WGS84_a
    e² = _WGS84_e²
    k₀ = _UTM_k₀
    λ₀ = deg2rad(utm_central_meridian(zone))

    x = easting - _UTM_FE
    y = northing                      # northern hemisphere: no false northing

    # NB: write every coefficient × e₁ with an explicit `*` — `3e1` etc. would lex as a
    # floating-point literal (3×10¹), not 3·e₁. (Hence the subscript name `e₁`, too.)
    e₁  = (1 - sqrt(1 - e²)) / (1 + sqrt(1 - e²))
    M   = y / k₀
    μ   = M / (a * (1 - e²/4 - 3e²^2/64 - 5e²^3/256))
    φ1  = (μ + (3*e₁/2 - 27*e₁^3/32) * sin(2μ)
             + (21*e₁^2/16 - 55*e₁^4/32) * sin(4μ)
             + (151*e₁^3/96) * sin(6μ)
             + (1097*e₁^4/512) * sin(8μ))

    e′² = e² / (1 - e²)
    C1  = e′² * cos(φ1)^2
    T1  = tan(φ1)^2
    N1  = a / sqrt(1 - e² * sin(φ1)^2)
    R1  = a * (1 - e²) / (1 - e² * sin(φ1)^2)^1.5
    D   = x / (N1 * k₀)

    φ = φ1 - (N1 * tan(φ1) / R1) * (D^2/2
            - (5 + 3T1 + 10C1 - 4C1^2 - 9e′²) * D^4/24
            + (61 + 90T1 + 298C1 + 45T1^2 - 252e′² - 3C1^2) * D^6/720)

    λ = λ₀ + (D - (1 + 2T1 + C1) * D^3/6
            + (5 - 2C1 + 28T1 - 3C1^2 + 8e′² + 24T1^2) * D^5/120) / cos(φ1)

    return rad2deg(φ), rad2deg(λ)
end

# ============================================================================
# The dataset and its metric window
# ============================================================================

"""
    KartverketDTM(; zone = 33, coverage = "nhm_dtm_topo_258`zone`")

The Kartverket national DTM as a NumericalEarth static-bathymetry dataset. `zone` is
the UTM zone (33 for Lofoten, 32 for western Norway / Nærøyfjord) — it selects both the
projection central meridian and the Geonorge WCS endpoint/coverage. `coverage` is the
WCS coverage id; by default the 1 m topographic DTM in EPSG:258`zone`.
"""
struct KartverketDTM <: AbstractStaticBathymetry
    zone     :: Int
    coverage :: String
end

KartverketDTM(; zone::Integer = 33, coverage = "nhm_dtm_topo_258$(zone)") =
    KartverketDTM(Int(zone), coverage)

Base.summary(d::KartverketDTM) = "KartverketDTM(zone=$(d.zone), \"$(d.coverage)\")"
Base.show(io::IO, d::KartverketDTM) = print(io, summary(d))

"""
    KartverketWindow(; center_lat, center_lon, halfwidth, resolution, zone = 33)

A square metric window on the DTM, used as a `Metadatum`'s `region`. `center_lat`,
`center_lon` (degrees) are projected to UTM zone `zone` to anchor the box; `halfwidth`
and `resolution` are in metres. The fetched grid is `(2·halfwidth / resolution)²` cells.
The `zone` is stored so the windowed coordinates can be interpreted in the right CRS.
"""
struct KartverketWindow{T}
    center_easting  :: T
    center_northing :: T
    halfwidth       :: T
    resolution      :: T
    zone            :: Int
end

function KartverketWindow(; center_lat, center_lon, halfwidth, resolution, zone::Integer = 33)
    E₀, N₀ = latlon_to_utm(center_lat, center_lon; zone)
    return KartverketWindow(promote(E₀, N₀, float(halfwidth), float(resolution))..., Int(zone))
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

## A region-stamped filename so different windows (and zones) cache separately.
function metadata_filename(::KartverketDTM, name, date, region::KartverketWindow)
    E = round(Int, region.center_easting); N = round(Int, region.center_northing)
    L = round(Int, region.halfwidth); Δ = round(Int, region.resolution)
    return @sprintf("kartverket_dtm_z%d_E%d_N%d_L%d_d%d.nc", region.zone, E, N, L, Δ)
end
metadata_filename(::KartverketDTM, name, date, ::Nothing) = "kartverket_dtm.nc"

## Each Norwegian UTM zone has its own national-DTM WCS endpoint and EPSG code.
wcs_base(zone::Integer) = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-258$(zone)"
epsg_code(zone::Integer) = 25800 + zone

"""
    wcs_url(dataset, window) -> String

Build the WCS 1.0.0 `GetCoverage` URL that returns the windowed DTM as NetCDF in
EPSG:258`zone` (the dataset's UTM zone). ArcGIS Server (which backs this service) is
reliable with WCS 1.0.0's explicit `bbox`/`width`/`height`, where it rejects 2.0.1
`subset` syntax.
"""
function wcs_url(dataset::KartverketDTM, w::KartverketWindow)
    xmin, ymin, xmax, ymax = utm_bbox(w)
    n = npixels(w)
    return string(wcs_base(dataset.zone),
        "?service=WCS&version=1.0.0&request=GetCoverage",
        "&coverage=", dataset.coverage,
        "&crs=EPSG:", epsg_code(dataset.zone),
        @sprintf("&bbox=%.3f,%.3f,%.3f,%.3f", xmin, ymin, xmax, ymax),
        "&width=", n, "&height=", n,
        "&format=NetCDF")
end

# ============================================================================
# Metadatum constructor + download
# ============================================================================

"""
    kartverket_metadatum(; center_lat, center_lon, halfwidth, resolution,
                         zone = 33, coverage = "nhm_dtm_topo_258`zone`",
                         dir = <thursday/data>)

Build a `NumericalEarth.DataWrangling.Metadatum` for a metric window of the
Kartverket DTM. `center_lat/center_lon` (deg) anchor the box; `halfwidth` and
`resolution` are metres. `zone` is the UTM zone (33 for Lofoten, 32 for western
Norway / Nærøyfjord) — it selects the projection and the Geonorge endpoint/coverage.
"""
function kartverket_metadatum(; center_lat, center_lon, halfwidth, resolution,
                              zone::Integer = 33,
                              coverage = "nhm_dtm_topo_258$(zone)",
                              dir = _kartverket_cache)
    # Build the Metadatum manually so we control the (custom, projected) region type.
    dataset = KartverketDTM(; zone, coverage)
    region  = KartverketWindow(; center_lat, center_lon, halfwidth, resolution, zone)
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
