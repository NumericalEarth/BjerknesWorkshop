# # Preparing the Sunnmøre idealized bathymetry
#
# *Boundary heterogeneity writes turbulence into the fluid — coupled-ocean case setup.*
#
# The coupled Sunnmøre case puts a prognostic ocean in the fjord and sea cells beneath
# the atmospheric LES.  The ocean needs a **bathymetry** (water depth) field on the
# *same* 50 km × 50 km, 100 m horizontal grid as the topography artifact produced by
# `03a_prepare_norway_topography.jl`.
#
# ## Design choice: idealized depth, real coastline planform
#
# We do **not** use a real bathymetric dataset (GEBCO, EMODnet, etc.) here.  Fetching
# and reprojecting ocean-floor data introduces the same fragile GDAL/PROJ dependency
# chain that motivated the synthetic fallback in `03a`.  More importantly, the coupled
# LES does not need the exact depth profile — it needs:
#
#  1. **The right coastline mask**: which cells are ocean vs. land.  We read this
#     directly from the topography artifact (the `ocean_mask` and `land_mask` arrays),
#     so the atmosphere and ocean share an identical land/sea boundary.
#  2. **A smooth, well-conditioned depth field**: no abrupt steps that stress the
#     ocean immersed-boundary solver.  A saturating ramp from the shoreline outward
#     achieves this without real data.
#
# The output artifact is:
#
# ```text
# thursday/data/sunnmore_50km_bathymetry.jld2
# ```
#
# holding `(; x, y, depth, ocean_mask, land_mask, source_metadata)`.

using Oceananigans.Units    # meters, kilometers — lightweight, no GPU needed
using JLD2
using Printf
using Dates

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

# ## Domain definition
#
# These must match the topography artifact exactly.  If they differ, the coordinate
# sanity-check below will error loudly before we write anything.

const center_lat  = 62.35
const center_lon  = 6.15
const Lx          = 50kilometers
const Ly          = 50kilometers
const Δ           = 100meters
const taper_width = 6kilometers

# ## Depth model parameters
#
# The idealized depth follows a saturating ramp.  We Gaussian-smooth the binary
# ocean mask (1 = ocean, 0 = land) with scale `coast_smooth` to produce a
# dimensionless "coastal distance proxy" φ ∈ [0, 1] — φ ≈ 0 at the shoreline
# and φ → 1 far offshore.  We then map through:
#
# ```
# depth(i,j) = max_depth * (1 - exp(-φ(i,j) / d0)),    d0 = shelf_width / coast_smooth
# ```
#
# This saturating, strictly monotone function gives:
#  - **0 at the shoreline** (φ ≈ 0, land cells forced to 0 via ocean_mask)
#  - **→ max_depth far offshore** (φ → 1, depth → max_depth * (1 - exp(-1/d0)))
#  - **smooth everywhere** — no sharp steps, well-conditioned for the ocean IB solver
#
# The e-folding scale `d0 = shelf_width / coast_smooth` controls how quickly depth
# ramps up from the shore: with `shelf_width = 1500 m` and `coast_smooth = 4000 m`,
# `d0 ≈ 0.375`, so the depth reaches ~63% of `max_depth` within roughly one
# `coast_smooth` radius of the shoreline.

const max_depth   = 300meters     # asymptotic depth far from shore (m)
const shelf_width = 1500meters    # e-folding distance of the depth ramp (m)

# Smoothing scale for the "coastal distance" proxy.  Larger = wider ramp from shore.
# 4 km spreads the nearshore ramp over ≈ 4 grid cells, keeping it well-resolved.

const coast_smooth = 4kilometers

# ## Paths

const repo_root    = get(ENV, "THURSDAY_REPO_ROOT", pwd())
const topo_path    = joinpath(repo_root, "thursday", "data", "sunnmore_50km_100m_topography.jld2")
const datadir      = joinpath(repo_root, "thursday", "data")
const figdir       = joinpath(repo_root, "thursday", "figures")
const artifact_path = joinpath(datadir, "sunnmore_50km_bathymetry.jld2")

# ## Gaussian smoother (separable, no external deps)
#
# Identical in structure to the one in `03a`; copied here so this file is
# self-contained at runtime (it does not `include` the topo preprocessing script).

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

# ## Depth construction from masks
#
# Takes the ocean and taper masks (and grid vectors) and returns the depth field.
# Separated into a function so we can call it on both the real artifact masks and
# the synthetic stand-in mask during logic validation.

function build_depth(ocean_mask, taper_mask, x, y)
    ## 1. Gaussian-smooth the ocean mask → "coastal distance proxy" φ ∈ [0, 1].
    ##    Land cells smear into water: φ ≈ 0 at the shoreline, φ → 1 far offshore.
    ##    The Gaussian scale `coast_smooth` sets the physical width of the ramp.
    φ = gaussian_smooth(Float64.(ocean_mask), x, y; smoothing_length = coast_smooth)
    φ = clamp.(φ, 0.0, 1.0)

    ## 2. Map through a saturating function to get depth (m).
    ##    We need depth = 0 at φ = 0 and depth → max_depth at φ = 1.
    ##    The e-folding scale in φ-space is d0 = shelf_width / coast_smooth:
    ##    a shelf_width of 1500 m over a 4 km Gaussian means we reach 63% of
    ##    max_depth by the time φ = shelf_width / coast_smooth ≈ 0.375.
    d0 = shelf_width / coast_smooth   # dimensionless e-folding scale in φ
    depth_raw = max_depth .* (1 .- exp.(-φ ./ d0))

    ## 3. Zero depth on land cells.
    depth_masked = depth_raw .* ocean_mask

    ## 4. Apply taper mask so the domain rim matches the atmosphere buffer.
    depth = depth_masked .* taper_mask

    return depth
end

# ## Source of masks
#
# Runtime: load the real topography artifact.  Authoring / offline validation: synthesize
# a coastline-ramp stand-in so the logic can be tested without the artifact.

const BATHY_SOURCE = Symbol(get(ENV, "BATHY_SOURCE", "topography_artifact"))

Nx = Int(Lx ÷ Δ)
Ny = Int(Ly ÷ Δ)
x  = range(-Lx/2, Lx/2, length = Nx)
y  = range(-Ly/2, Ly/2, length = Ny)

# ## Synthetic stand-in masks (for offline validation; matches `03a`'s synthetic fjord)
#
# A tilted land/sea split with an island cluster — enough geometry to exercise the
# depth ramp logic without any JLD2 dependency.

function synthetic_masks(x, y)
    Nx, Ny = length(x), length(y)
    land_mask  = zeros(Float64, Nx, Ny)
    ocean_mask = zeros(Float64, Nx, Ny)

    ## Coastline: land to the SE of a tilted line, ocean to the NW.
    coast(xi, yj) = (xi + yj) / √2   # signed distance along NW–SE normal

    ## A small island cluster offshore (always ocean on coastline side).
    island(xi, yj) = exp(-((xi + 15e3)^2 + (yj - 8e3)^2) / (2 * (4e3)^2))

    for i in 1:Nx, j in 1:Ny
        xi, yj = x[i], y[j]
        land_val = smooth_step(coast(xi, yj) + 6e3, 4e3)
        ## Island counts as land if island function > 0.5.
        land_val = max(land_val, smooth_step(island(xi, yj) - 0.5, 0.1))
        land_mask[i, j]  = clamp(land_val, 0.0, 1.0)
        ocean_mask[i, j] = 1.0 - land_mask[i, j]
    end
    ## Binarize: depth model only needs 0/1 masks.
    land_mask  = Float64.(land_mask  .> 0.5)
    ocean_mask = Float64.(ocean_mask .> 0.5)
    return land_mask, ocean_mask
end

if BATHY_SOURCE === :synthetic
    @info "Using synthetic stand-in masks (offline validation mode)."
    land_mask, ocean_mask = synthetic_masks(x, y)
    taper_mask = [edge_taper(x[i], y[j], Lx, Ly; taper_width) for i in 1:Nx, j in 1:Ny]
else
    ## ── Real artifact path ────────────────────────────────────────────────────
    isfile(topo_path) || error("""
        Missing topography artifact:
          $topo_path

        Run `03a_prepare_norway_topography.jl` first (or set BATHY_SOURCE=synthetic
        to run in offline validation mode with a synthetic stand-in mask).
        """)

    @info "Loading topography artifact for coastline masks…" topo_path
    topo = load(topo_path)

    ## Coordinate sanity check: artifact grid must match our domain constants.
    xt = topo["x"]; yt = topo["y"]
    Nxt, Nyt = length(xt), length(yt)
    Nxt == Nx && Nyt == Ny || error(
        "Grid size mismatch: artifact is $(Nxt)×$(Nyt), expected $(Nx)×$(Ny). " *
        "Check domain constants (Lx, Ly, Δ) against the topography artifact."
    )
    isapprox(xt[1], first(x); rtol = 1e-6) && isapprox(xt[end], last(x); rtol = 1e-6) ||
        error("x-coordinate mismatch between topography artifact and bathymetry domain.")
    isapprox(yt[1], first(y); rtol = 1e-6) && isapprox(yt[end], last(y); rtol = 1e-6) ||
        error("y-coordinate mismatch between topography artifact and bathymetry domain.")

    land_mask  = Float64.(topo["land_mask"])
    ocean_mask = Float64.(topo["ocean_mask"])
    taper_mask = Float64.(topo["taper_mask"])

    @info "Loaded masks from topography artifact" source = topo["source_metadata"].source land_fraction = sum(land_mask) / (Nx * Ny)
end

# ## Build the depth field

depth = build_depth(ocean_mask, taper_mask, x, y)

wet_cells  = sum(ocean_mask)
wet_frac   = wet_cells / (Nx * Ny)
@info "Bathymetry prepared" BATHY_SOURCE Nx Ny max_depth_computed = maximum(depth) wet_fraction = wet_frac shelf_width max_depth

# ## Save the artifact

mkpath(datadir)

source_metadata = (; source = :idealized,
                     max_depth   = Float64(max_depth),
                     shelf_width = Float64(shelf_width),
                     Lx          = Float64(Lx),
                     Ly          = Float64(Ly),
                     Δ           = Float64(Δ),
                     created     = string(now()))

jldsave(artifact_path;
        x = collect(x), y = collect(y),
        depth, ocean_mask, land_mask, source_metadata)
@info "Saved bathymetry artifact" artifact_path

# ## Validation figure
#
# Depth heatmap, ocean mask, and a mid-domain cross-section — mirrors the layout
# of the topography validation figure in `03a`.

using CairoMakie

xkm = collect(x) ./ 1e3
ykm = collect(y) ./ 1e3

fig = Figure(size = (1400, 1000))

ax1 = Axis(fig[1, 1], title = "Idealized bathymetry — depth (m)",
           xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
ax2 = Axis(fig[1, 2], title = "Ocean mask (1 = ocean, 0 = land)",
           xlabel = "x (km)", ylabel = "y (km)", aspect = 1)
ax3 = Axis(fig[2, 1:2], title = "Cross-section at y = 0 km",
           xlabel = "x (km)", ylabel = "depth (m)")

hm1 = heatmap!(ax1, xkm, ykm, depth, colormap = :deep)
hm2 = heatmap!(ax2, xkm, ykm, ocean_mask, colormap = :grays)
Colorbar(fig[1, 0], hm1, label = "m", flipaxis = false)

## Cross-section at the grid row nearest y = 0.
j_mid = argmin(abs.(collect(y)))
lines!(ax3, xkm, depth[:, j_mid], color = :steelblue, linewidth = 2, label = "depth")
land_cross = land_mask[:, j_mid]
depth_max = max(maximum(depth), 1.0)   # guard against all-land cross-section
band!(ax3, xkm, fill(0.0, Nx), land_cross .* depth_max .* 0.05,
     color = (:saddlebrown, 0.4), label = "land")
axislegend(ax3, position = :rt)
ylims!(ax3, -5, depth_max * 1.1)
ax3.yreversed = true

figpath = joinpath(figdir, "sunnmore_bathymetry.png")
mkpath(figdir)
save(figpath, fig)
@info "Saved validation figure" figpath
fig
