# # Shared setup for the two-phase coupled fjord experiment over Sunnmøre
#
# *A controlled comparison: identical atmosphere, terrain, and rotating geostrophic
# wind — only the SURFACE treatment of the water differs between the two phases.*
#
# This file is the SHARED foundation that both phase scripts `include`. It builds, on a
# **latitude–longitude grid** with real **ETOPO2022** relief over the Sunnmøre coast of
# western Norway:
#
#   * the domain/grid parameters and the signed relief field (terrain + bathymetry),
#   * the terrain-following compressible **atmosphere** with OPEN lateral boundaries and
#     a rotating geostrophic wind (cross-valley → along-valley),
#   * the immersed **ocean grid** and the 2D **land grid** both phases reference,
#   * the `rotate_geostrophic_wind!` callback.
#
# It is an `include` (NOT a module): everything below is an ordinary top-level binding
# the phase scripts can use directly. It runs standalone — constructing the atmosphere
# without error — so you can sanity-check the shared setup on its own.
#
# The two phases that build on it:
#
#   * **Phase 1** (`10a_atmosphere_land.jl`): atmosphere + a "land-as-ocean" `SlabLand`
#     (a wet, warm slab that mimics the sea surface), coupled with `AtmosphereLandModel`.
#   * **Phase 2** (`10b_atmosphere_ocean.jl`): atmosphere + a prognostic, CLOSED ocean,
#     coupled with `AtmosphereOceanModel` (no land model).
#
# Neither phase uses the `land_fraction` weighted-flux coupler — each phase has a single
# surface-type interface, so the blended coupling is not needed.

using Breeze
using NumericalEarth
using NumericalEarth: regrid_bathymetry, ETOPO2022
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: λnode, φnode, λnodes, φnodes
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, NormalFlowBoundaryCondition,
                                       FluxBoundaryCondition, PerturbationAdvection
using CUDA
using JLD2
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

_t0 = Ref(time_ns())
checkpoint(msg) = (@info @sprintf("⏱ %-28s %8.1f s", msg, 1e-9 * (time_ns() - _t0[])); flush(stderr))

Random.seed!(62)

arch = GPU()
gpu_report()
## Coupled runs: the EarthSystemModel clock is Float64, so all component grids match.
Oceananigans.defaults.FloatType = Float64
FT = Float64
nothing #hide

# ## Domain & grid (latitude–longitude)
#
# An ~90 km box over the Sunnmøre coast: south edge ≈ 62.0° N, east edge ≈ 7.3° E, so it
# opens onto the open Norwegian Sea in the NW (more ocean fetch) while the SE reaches the
# *heads* of several fjords — Hjørundfjorden and the Storfjorden→Sunnylvsfjorden/
# Geirangerfjorden system. `RUN_CLASS=production` selects the fine grid.

const PROD = get(ENV, "RUN_CLASS", "smoke") == "production"

center_lat = 62.40
center_lon = 6.43
## Half-spans in degrees for an ~90 km square (latitude ~111 km/°, longitude shrinks by
## cos) ⇒ south ≈ 62.0° N, east ≈ 7.3° E, NW corner ≈ (62.8° N, 5.56° E) in open water.
dlat = 45kilometers / 111320
dlon = 45kilometers / (111320 * cosd(center_lat))

Nλ = Nφ = PROD ? 384 : 160
Lz_a = 12kilometers   # atmosphere depth
Lz_o = 700meters      # ocean depth (deeper bathymetry is truncated to a flat floor)
Nz_o = PROD ? 40 : 20

longitude = (center_lon - dlon, center_lon + dlon)
latitude  = (center_lat - dlat, center_lat + dlat)

# ## Real relief from ETOPO2022 (terrain + bathymetry in one field)
#
# `regrid_bathymetry` maps the signed ETOPO2022 relief (land > 0, ocean < 0) straight
# onto our lat–lon grid — the canonical NumericalEarth pattern. We taper the relief to
# flat (sea level) within `taper_width` of the edges so the domain rim is a clean
# buffer, then split it into the ocean's immersed bottom and the atmosphere's terrain.

taper_width = 6kilometers

ocean_underlying = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ, Nz_o), halo = (7, 7, 7),
                                         longitude, latitude, z = (-Lz_o, 0),
                                         topology = (Bounded, Bounded, Bounded))

checkpoint("start")
relief = regrid_bathymetry(ocean_underlying; dataset = ETOPO2022(),
                           major_basins = Inf, height_above_water = nothing)
checkpoint("ETOPO relief regridded")

## Taper the signed relief to 0 (sea level) at the domain edges, using the metric
## offset of each lat–lon node from the box center. `edge_taper` (ThursdayLES) returns
## 1 in the interior and 0 at the rim.
λc = Array(λnodes(ocean_underlying, Center()))
φc = Array(φnodes(ocean_underlying, Center()))
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * cosd(center_lat)
Lx = 2 * dlon * m_per_deg_lon
Ly = 2 * dlat * m_per_deg_lat
taper = [edge_taper((λc[i] - center_lon) * m_per_deg_lon,
                    (φc[j] - center_lat) * m_per_deg_lat, Lx, Ly; taper_width)
         for i in 1:Nλ, j in 1:Nφ]

relief_cpu = Array(interior(relief, :, :, 1)) .* taper      # signed, tapered
land_elev  = max.(relief_cpu, 0.0)                          # terrain height (≥0)
bottom_cpu = min.(relief_cpu, 0.0)                          # sea floor z (≤0)
land_frac_cpu = FT.(relief_cpu .> 0)                        # 1 land, 0 ocean

@info "Relief" max_terrain = maximum(land_elev) max_depth = -minimum(bottom_cpu) water_fraction = sum(land_frac_cpu .== 0) / length(land_frac_cpu)

# ## Ocean grid: immersed bottom from the bathymetry

bottom_field = Field{Center, Center, Nothing}(ocean_underlying)
set!(bottom_field, bottom_cpu)
ocean_grid = ImmersedBoundaryGrid(ocean_underlying, GridFittedBottom(bottom_field))
checkpoint("ocean immersed grid")

# ## Atmosphere grid: terrain-following, real terrain carved in
#
# Fine ~120 m cells through the boundary layer, coarsening aloft; terrain-following
# surfaces relax to flat with height (`TwoLevelDecay`) under a 4 km sponge.

z_faces = PiecewiseStretchedDiscretization(z = [0, 3000, 6000, Int(Lz_a)], Δz = [120, 120, 400, 800])
Nz_a = length(z_faces) - 1
z_coord = TerrainFollowingVerticalDiscretization(z_faces;
              formulation = TwoLevelDecay(large_scale_height = Lz_a/2, small_scale_height = Lz_a/8))

atmos_grid = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ, Nz_a), halo = (5, 5, 5),
                                   longitude, latitude, z = z_coord,
                                   topology = (Bounded, Bounded, Bounded))

## Terrain height as a function of (λ, φ): bilinear interpolation of the tapered land
## elevation over the lat–lon nodes (materialize_terrain! evaluates it on the CPU).
terrain_fun = bilinear(land_elev, λc, φc)
materialize_terrain!(atmos_grid, (λ, φ) -> terrain_fun(λ, φ))
checkpoint("terrain materialized")

land_grid = LatitudeLongitudeGrid(arch; size = (Nλ, Nφ), halo = (atmos_grid.Hx, atmos_grid.Hy),
                                  longitude, latitude, topology = (Bounded, Bounded, Flat))

# ## Atmosphere component (compressible, terrain-following, coupling-ready)

θ₀ = 280; p₀ = 1e5; N²_a = 1.0e-4; g = 9.81
potential_temperature_profile(z) = θ₀ * exp(N²_a * z / g)

time_discretization = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5,
                          open_boundary_relaxation = 0.5,
                          sponge = UpperSponge(damping_rate = 0.01, depth = 4kilometers))
dynamics = CompressibleDynamics(time_discretization;
                                slope_stencil = SlopeInsideInterpolation(),
                                surface_pressure = p₀,
                                reference_potential_temperature = potential_temperature_profile)

# ### Rotating large-scale wind (geostrophic; sweeps cross → along valley)
#
# `geostrophic_forcings(uᵍ, vᵍ)` drives the flow toward a geostrophic wind under
# Coriolis. We rotate it from cross-valley to along-valley over `t_rotate` by updating
# the stored geostrophic velocity each step (callback below).

## Inertial period at this latitude (2π/f ≈ 13.5 h at 62.35° N) sets the natural
## timescale: we rotate the wind cross→along over ¼ inertial period and run for ½.
f_coriolis = 2 * 7.2921e-5 * sind(center_lat)
inertial_period = 2π / f_coriolis

U_mag    = 12.0
α_valley = 0.0                  # fjord-axis angle (rad from east); 0 ⇒ along-valley = eastward
t_rotate = PROD ? inertial_period / 4 : 3hours
wind_params = (; U_mag, α_valley, t_rotate)
wind_angle(t, p) = (p.α_valley + π/2) + clamp(t / p.t_rotate, 0, 1) * (p.α_valley - (p.α_valley + π/2))
target_u(t, p) = p.U_mag * cos(wind_angle(t, p))
target_v(t, p) = p.U_mag * sin(wind_angle(t, p))

uᵍ0(z) = target_u(0.0, wind_params)
vᵍ0(z) = target_v(0.0, wind_params)
geostrophic = geostrophic_forcings(uᵍ0, vᵍ0)

# ### Open lateral boundaries (let the wind pass through instead of piling up)
#
# The geostrophic forcing is still the primary interior wind driver; the lateral
# boundaries are opened so flow can EXIT rather than reflect off closed walls. We set
# **Open** (`NormalFlowBoundaryCondition`) conditions on the boundary-normal momentum
# components — `ρu` at west/east, `ρv` at south/north — whose value is the prescribed
# large-scale wind. The prognostic momentum is DENSITY-WEIGHTED (ρu, ρv), so the BC value
# is `ρ_ref · u` (mirroring Breeze's `test/open_boundary_momentum.jl`, `ρu_value = ρ_b·U_bg`).
# `ρ_ref` is a representative low-level reference air density. Because the wind ROTATES,
# the value is time-dependent: we use a `ContinuousBoundaryFunction` (a `(coord, z, t, p)`
# function, GPU-safe) that reads the clock time `t` and reuses `target_u`/`target_v` so the
# wall value tracks the same cross→along rotation as the interior forcing each halo fill.
# These Open BCs are what arm `open_boundary_relaxation` (it is a no-op otherwise).
ρ_ref = 1.2  # representative low-level reference air density [kg/m³]
obc_params = (; wind_params..., ρ_ref)

## west/east are x-normal ⇒ the boundary function takes (φ, z, t, p); south/north are
## y-normal ⇒ (λ, z, t, p). The wall value is the density-weighted prescribed wind.
ρu_obc_value(φ, z, t, p) = p.ρ_ref * target_u(t, p)
ρv_obc_value(λ, z, t, p) = p.ρ_ref * target_v(t, p)

## `atmosphere_simulation` pre-wires the BOTTOM momentum BCs to 2D coupling-flux fields
## (`FluxBoundaryCondition(ρτˣ/ρτʸ)`) and merges user BCs with a PLAIN `merge`, so a user
## `ρu`/`ρv` entry REPLACES that field's whole BC set. The coupler later reads
## `momentum.ρu.boundary_conditions.bottom.condition` back to fill the surface stress. So we
## create the coupling flux fields ourselves and include them as the `bottom` BC alongside
## the lateral open BCs — preserving the air–sea/air–land momentum coupling. This trick is
## needed in BOTH phases so the lateral open BCs do not clobber the bottom coupling flux.
ρτˣ = Field{Center, Center, Nothing}(atmos_grid)
ρτʸ = Field{Center, Center, Nothing}(atmos_grid)
ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρτˣ),
                                 west = NormalFlowBoundaryCondition(ρu_obc_value; parameters = obc_params),
                                 east = NormalFlowBoundaryCondition(ρu_obc_value; parameters = obc_params))
ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρτʸ),
                                 south = NormalFlowBoundaryCondition(ρv_obc_value; parameters = obc_params),
                                 north = NormalFlowBoundaryCondition(ρv_obc_value; parameters = obc_params))

atmosphere = atmosphere_simulation(atmos_grid; dynamics,
                                   momentum_advection = WENO(order = 9),
                                   scalar_advection = WENO(order = 5),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(latitude = center_lat),
                                   forcing = geostrophic,
                                   boundary_conditions = (; ρu = ρu_bcs, ρv = ρv_bcs))
checkpoint("atmosphere built")

δθ = 0.3; zδ = 500; qᵗ₀ = 2e-3
ϵ() = rand() - 0.5
θᵢ(λ, φ, z) = potential_temperature_profile(z) + δθ * ϵ() * (z < zδ)
set!(atmosphere.model, ρ = atmosphere.model.dynamics.terrain_reference_density,
     θ = θᵢ, u = target_u(0.0, wind_params), v = target_v(0.0, wind_params), w = 0, qᵗ = qᵗ₀,
     enforce_mass_conservation = false)
Oceananigans.TimeSteppers.update_state!(atmosphere.model)
checkpoint("atmosphere initialized")

# ## Rotate the geostrophic wind cross → along valley (shared by both phases)
#
# The u-forcing stores vᵍ and the v-forcing stores uᵍ (geostrophic balance), so we swap
# them. Both phase scripts add this as a callback.
function rotate_geostrophic_wind!(sim)
    t = sim.model.clock.time
    a = sim.model.atmosphere.model
    set!(a.forcing.ρu.forcing.geostrophic_velocity, target_v(t, wind_params))
    set!(a.forcing.ρv.forcing.geostrophic_velocity, target_u(t, wind_params))
    return nothing
end

# ## Shared sea-surface temperature
#
# Both phases use the same ~10 °C water surface so the comparison is controlled: in
# Phase 2 it is the ocean's initial mixed-layer temperature (in °C), and in Phase 1 it
# is the SlabLand deep reservoir / skin temperature (in Kelvin, 283.15 K = 10 °C).
T_sea_celsius = 10.0
T_sea_kelvin  = 273.15 + T_sea_celsius

checkpoint("shared setup complete")
nothing #hide
