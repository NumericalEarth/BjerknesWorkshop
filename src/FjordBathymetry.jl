"""
    FjordBathymetry

Seafloor (and synthetic-fjord) bathymetry for the Nærøyfjord coupled LES
(`tutorials/day4/10_naeroyfjord...`).

The Norway terrain case (`03_norway...`) gets its **above-water** terrain from the
Kartverket 1 m DTM (`KartverketDEM`). A *fjord* case also needs the **seafloor** — the
fjord depth — which Kartverket serves only as WMS imagery / depth contours, not as an
open coverage raster. This module fills that gap with two paths, mirroring the
`kartverket / synthetic` split in `03a`:

 1. **EMODnet Bathymetry** (`:emodnet`). The pan-European Digital Terrain Model
    (`ows.emodnet-bathymetry.eu`, coverage `emodnet__mean`) is a *combined land + sea*
    elevation grid at 1/16′ (~115 m N–S, ~55 m E–W at 61° N): **positive over land,
    negative below the sea surface**. We fetch a windowed subset by WCS 2.0.1
    `GetCoverage` in the service's `text/plain` ASCII-grid format — which carries the
    affine georeferencing in its header — so we parse it with **pure Julia, no
    GDAL/PROJ**. We keep only the *negative* (seafloor) part; the steep walls come from
    the far finer Kartverket DTM. The coverage is in geographic (lat–lon) coordinates,
    so callers sample it via `KartverketDEM.utm_to_latlon` to land on the metric UTM
    grid.

 2. **Synthetic Nærøyfjord** (`:synthetic`, the dependency-free default). A parametric
    fjord: a gently meandering channel that runs north from the Gudvangen head,
    narrow and shallow at the head and widening/deepening seaward, between steep walls
    that rise ~1200 m. It writes the *same* signed-elevation convention (land > 0,
    water < 0) so the whole pipeline runs with zero network access.

Data source: EMODnet Bathymetry Consortium (2022/2024), <https://emodnet.ec.europa.eu/>,
CC-BY. Kartverket DTM via [`KartverketDEM`](@ref).
"""
module FjordBathymetry

using Downloads
using Printf
using Statistics

export emodnet_download, read_emodnet, emodnet_height_function,
       synthetic_naeroyfjord

# ============================================================================
# EMODnet Bathymetry — windowed WCS fetch (text/plain ASCII grid, no GDAL)
# ============================================================================

const _EMODNET_WCS = "https://ows.emodnet-bathymetry.eu/wcs"

"""
    emodnet_download(; lat_min, lat_max, lon_min, lon_max,
                     coverage = "emodnet__mean", dir, filename) -> path

Fetch the EMODnet DTM over a geographic box via WCS 2.0.1 `GetCoverage` in
`text/plain` (an ASCII grid with the affine transform in its header). Cached: skipped
if `filename` already exists in `dir`. Returns the local path.
"""
function emodnet_download(; lat_min, lat_max, lon_min, lon_max,
                          coverage = "emodnet__mean", dir, filename)
    mkpath(dir)
    path = joinpath(dir, filename)
    if !isfile(path)
        url = string(_EMODNET_WCS,
            "?service=WCS&version=2.0.1&request=GetCoverage",
            "&coverageId=", coverage,
            @sprintf("&subset=Lat(%.6f,%.6f)", lat_min, lat_max),
            @sprintf("&subset=Long(%.6f,%.6f)", lon_min, lon_max),
            "&format=text/plain")
        @info "Downloading EMODnet bathymetry window via WCS" coverage path
        Downloads.download(url, path)
        _check_ascii_grid(path)
    end
    return path
end

## WCS errors come back as XML/HTML; make sure we actually got the ASCII grid.
function _check_ascii_grid(path)
    head = open(path) do io; String(read(io, min(200, filesize(path)))); end
    if !occursin("Grid bounds", head) && !occursin("Band 0", head)
        body = String(read(path)); rm(path; force = true)
        error("EMODnet WCS did not return an ASCII grid. First 200 chars:\n" *
              first(replace(body, r"<[^>]*>" => " "), 200))
    end
    return nothing
end

"""
    read_emodnet(path) -> (lons, lats, H)

Parse the EMODnet `text/plain` coverage at `path`. Returns longitude `lons` and
latitude `lats` (degrees, **ascending**) and the elevation matrix `H[i, j]` at
`(lons[i], lats[j])` in metres (positive land, negative seafloor). The affine
georeferencing (pixel size + upper-left origin, GDAL convention) is read from the
header's `elt_*` parameters.
"""
function read_emodnet(path)
    lines = readlines(path)

    ## Affine transform from the WKT-ish header parameters.
    grab(tag) = begin
        idx = findfirst(l -> occursin("elt_$(tag)", l), lines)
        idx === nothing && error("EMODnet grid: missing affine parameter elt_$(tag) in $path")
        m = match(r"elt_[0-9]_[0-9]\"\s*,\s*([-+0-9.eE]+)", lines[idx])
        m === nothing && error("EMODnet grid: could not parse elt_$(tag) from: $(lines[idx])")
        parse(Float64, m.captures[1])
    end
    e00 = grab("0_0")   # Δlon per pixel (east step)
    e02 = grab("0_2")   # upper-left lon (corner)
    e11 = grab("1_1")   # Δlat per pixel (north step, negative → rows go south)
    e12 = grab("1_2")   # upper-left lat (corner)

    ## Data rows follow the "Band 0:" marker; collect whitespace-separated floats.
    b0 = findfirst(l -> occursin("Band 0", l), lines)
    b0 === nothing && error("EMODnet grid: no 'Band 0:' marker in $path")
    rows = Vector{Vector{Float64}}()
    for k in (b0 + 1):length(lines)
        toks = split(strip(lines[k]))
        isempty(toks) && continue
        # Stop at any trailing non-numeric section.
        all(t -> tryparse(Float64, t) !== nothing, toks) || break
        push!(rows, parse.(Float64, toks))
    end
    isempty(rows) && error("EMODnet grid: no numeric data after 'Band 0:' in $path")

    nrows = length(rows)
    ncols = length(rows[1])
    raw = Array{Float64}(undef, nrows, ncols)   # raw[row(top→down), col(left→right)]
    for j in 1:nrows
        length(rows[j]) == ncols ||
            error("EMODnet grid: ragged row $j ($(length(rows[j])) ≠ $ncols)")
        raw[j, :] .= rows[j]
    end

    ## Pixel-center coordinates (GDAL affine, +0.5 to centre); reorder to ascending lat.
    lons = [e02 + (i - 0.5) * e00 for i in 1:ncols]
    lats_desc = [e12 + (j - 0.5) * e11 for j in 1:nrows]     # row 0 is northernmost
    lats = reverse(lats_desc)
    H = Array{Float64}(undef, ncols, nrows)                  # H[i, j] at (lons[i], lats[j])
    for j in 1:nrows, i in 1:ncols
        H[i, nrows - j + 1] = raw[j, i]
    end
    return lons, lats, H
end

"""
    emodnet_height_function(path) -> h(lon, lat)

Bilinear interpolant of the EMODnet window read from `path`, as a function of geographic
`(lon, lat)` in degrees. Returns metres (positive land, negative seafloor); queries
outside the window clamp to the edge.
"""
function emodnet_height_function(path)
    lons, lats, H = read_emodnet(path)
    interp = _bilinear(H, lons, lats)
    return (lon, lat) -> interp(lon, lat)
end

# ============================================================================
# Synthetic Nærøyfjord — a parametric fjord channel (dependency-free)
# ============================================================================

@inline _smoothstep(r, δ) = (1 + tanh(r / δ)) / 2

"""
    synthetic_naeroyfjord(x, y; Lx, Ly, kw...) -> h[i, j]

A deterministic, parametric Nærøyfjord-flavoured signed-elevation field on the metric
grid `(x, y)` (recentred UTM metres; `x` cross-fjord, `y` along-fjord pointing north).
Land is positive, water negative. The fjord is a gently meandering channel that runs
north, **shallow and narrow at the Gudvangen head** (small `y`) and widening/deepening
seaward (large `y`), flanked by steep ~1200 m walls. Keyword arguments tune the
geometry; the defaults give a Nærøyfjord-scale fjord on a ~10–20 km box.
"""
function synthetic_naeroyfjord(x, y; Lx, Ly,
                               wall_height   = 1200.0,   # m, ridge-top elevation
                               wall_slope    = 1300.0,   # m, e-folding of the wall rise
                               half_width_head = 130.0,  # m, channel half-width at the head
                               half_width_sea  = 380.0,  # m, channel half-width seaward
                               depth_head      = 12.0,   # m, water depth at the head
                               depth_sea       = 320.0,  # m, water depth seaward
                               meander_amp     = 350.0,  # m, centerline meander amplitude
                               meander_wave    = 9.0e3,  # m, meander wavelength
                               head_taper      = 2.0e3)  # m, along-fjord head closure scale
    Nx, Ny = length(x), length(y)
    h = zeros(Float64, Nx, Ny)

    y0 = first(y)                       # south end ≈ the Gudvangen head
    Ly_span = last(y) - first(y)
    ## Seaward fraction s ∈ [0,1]: 0 at the head (south), 1 at the seaward (north) end.
    sfrac(yj) = clamp((yj - y0) / Ly_span, 0, 1)

    @inbounds for j in 1:Ny, i in 1:Nx
        xi, yj = x[i], y[j]
        s = sfrac(yj)

        ## Meandering centerline and along-fjord-varying channel geometry.
        xc = meander_amp * sin(2π * yj / meander_wave)
        d  = xi - xc                                   # cross-channel distance from axis
        w  = half_width_head + (half_width_sea - half_width_head) * s
        D  = depth_head + (depth_sea - depth_head) * s  # max depth seaward

        ## Head closure: the channel only exists north of the head taper.
        open = _smoothstep(yj - (y0 + head_taper), head_taper / 2)

        ## Channel (U-shaped) vs walls. `inside` ≈ 1 within the channel, 0 outside.
        inside = _smoothstep(w - abs(d), max(20.0, 0.15w)) * open
        depth  = D * (1 - (clamp(d / w, -1, 1))^2)       # U-shaped cross-section

        ## Wall elevation: rises steeply away from the channel edge, with mild ridges.
        dwall  = max(abs(d) - w, 0.0)
        ridge  = 1 + 0.18 * sin(xi / 1.7e3) * cos(yj / 2.3e3)
        eland  = wall_height * (1 - exp(-dwall / wall_slope)) * ridge

        ## Blend: water inside the (open) channel, land on the walls / closed head.
        h[i, j] = inside * (-depth) + (1 - inside) * max(eland, 0.0)
    end
    return h
end

# ============================================================================
# Bilinear interpolation over a regular (xs, ys) grid; clamps to the domain edge.
# ============================================================================

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

end # module FjordBathymetry
