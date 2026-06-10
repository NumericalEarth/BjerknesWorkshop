# # Sea ice in the Arctic: a pan-Arctic simulation over a slab ocean
#
# *The realistic Arctic: a seasonal cycle of sea ice driven by reanalysis.*
#
# Sea ice is where the polar ocean meets the atmosphere, and modeling it well is a question of getting
# three things to talk to each other: the ice itself (which grows, melts, drifts, and fractures), the
# ocean beneath (which supplies or removes heat at the base), and the atmosphere above (which cools the
# surface and pushes the pack around). Here we build a **pan-Arctic** sea-ice simulation in which the
# ocean is a single 50 m *slab* mixed layer — its temperature evolves with the surface heat flux, while its
# salinity stays fixed and it carries no currents — so the prognostic action is in the ice and the SST it
# floats on. The atmosphere is the JRA55-do
# reanalysis. This is the cheapest realistic configuration that still produces a genuine Arctic seasonal
# cycle: starting from the late-winter maximum, the marginal ice zone retreating through spring and
# summer, with leads opening under the wind.
#
# !!! warning "Hardware and data requirements"
#     This is a GPU tutorial. The first run downloads EN4 hydrography for the ocean state, ECCO for the initial
#     ice, and the JRA55-do reanalysis for the atmosphere (a few GB, cached for every later run). On the workshop
#     cluster the cache is pre-staged. On a laptop you can read along, or shrink the grid and run a short
#     CPU segment with some patience.
#
# !!! note "What the slab ocean does"
#     The ocean is a single 50 m mixed layer whose temperature evolves with the net surface heat flux. It
#     supplies the heat that melts the ice from below, sets the freezing point at the ice base through its
#     (fixed) salinity, and presents a quiescent surface that drags on the moving pack. Swapping the slab
#     for a prognostic `ocean_simulation` turns this into a fully coupled regional ocean–sea ice run — the
#     components are the same, only the ocean's complexity changes.

using NumericalEarth
using NumericalEarth.Oceans: SlabOcean
using NumericalEarth.EarthSystemModels: ocean_surface_salinity, ocean_surface_velocities
using Oceananigans
using Oceananigans.Units
using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using ClimaSeaIce.SeaIceDynamics: SeaIceMomentumEquation, SemiImplicitStress, SplitExplicitSolver
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology
using Dates, Printf, CUDA
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

# `regrid_bathymetry` crops the source bathymetry to the grid's geographic extent through
# `x_domain`/`y_domain`, which Oceananigans defines only for lat–lon grids. A rotated grid stores the
# *true geographic* node coordinates, so its bounding box is just their extrema (temporary local patch):
import Oceananigans.Grids: x_domain, y_domain
x_domain(grid::Oceananigans.Grids.OrthogonalSphericalShellGrid) = extrema(parent(grid.λᶠᶠᵃ))
y_domain(grid::Oceananigans.Grids.OrthogonalSphericalShellGrid) = extrema(parent(grid.φᶠᶠᵃ))

arch = GPU()   # CPU() also works, at reduced resolution

# ## A grid for the Arctic cap
#
# A latitude–longitude grid degenerates at the geographic poles — the zonal cell width vanishes, and with
# it the affordable time step — which is exactly the wrong behaviour for an Arctic-centered domain. We
# sidestep the singularity with a `RotatedLatitudeLongitudeGrid`: an ordinary lat–lon patch whose pole has
# been rotated onto the geographic *equator* (`north_pole = (0, 0)`). Positioned over rotated longitude
# 180°, the patch lands as a cap **centered on the real North Pole**, which becomes an ordinary,
# well-resolved ocean point rather than a coordinate singularity — the singular points are exiled to the
# equator. A half-width of `δ = 35°` brings the cap edges down to ~55°N, enough for the Arctic Ocean and
# its marginal seas. Crucially the grid carries the *true geographic* longitude and latitude of every node,
# so the plotting later needs no regridding. Since the slab ocean has no vertical structure we use a single
# vertical level, enough for the bathymetry to mark land from ocean.

Nx, Ny = 180, 180
δ = 35   # half-width of the cap, in rotated degrees
z = (-10meters, 0)

underlying_grid = RotatedLatitudeLongitudeGrid(arch;
                                               size = (Nx, Ny, 1),
                                               longitude = (180 - δ, 180 + δ),
                                               latitude = (-δ, δ),
                                               north_pole = (0, 0),
                                               halo = (5, 5, 1),
                                               z)

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 15, major_basins = Inf)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# ## The slab ocean
#
# A `SlabOcean` represents the upper ocean as a single well-mixed layer of fixed depth whose temperature
# evolves in response to the net surface heat flux, ``\partial_t T = -J^T / H``. We give it a 50 m mixed
# layer and initialize its temperature from the EN4 hydrographic objective analysis. Salinity is held
# fixed at 35 and there are no currents, so the ocean still sets the freezing point and a quiescent drag
# beneath the ice — but now its surface temperature warms and cools with the seasons.
#
# One wrinkle to reconcile: `SlabOcean` reports its temperature in *Kelvin* (it was built for
# atmosphere-only coupling), but ClimaSeaIce's ice–ocean exchange compares the ocean temperature against a
# freezing point computed in *Celsius*. We make this slab report Celsius instead — the atmosphere flux still
# gets Kelvin, because it converts through this same `temperature_units` hook:

using NumericalEarth.EarthSystemModels: DegreesCelsius
NumericalEarth.EarthSystemModels.temperature_units(::SlabOcean) = DegreesCelsius()

ocean = SlabOcean(grid; depth = 50)

date = DateTime(1993, 2, 1)   # late-winter Arctic, near the seasonal ice maximum

set!(ocean.temperature, Metadatum(:temperature; dataset = EN4Monthly(), date))

# ## The sea-ice model
#
# The ice carries slab thermodynamics — conductive growth and melt between the ocean base and the
# atmosphere-set surface temperature — and EVP dynamics, the elastic–viscous–plastic rheology that lets the
# pack resist convergence, fail in shear, and open leads. Two couplings to the ocean are explicit here,
# because the convenience constructor only wires them automatically for a *prognostic* ocean:
#
# 1. the **freezing boundary condition** at the base, `IceWaterThermalEquilibrium`, which reads the ocean
#    surface salinity to set the local freezing temperature, and
# 2. the **ice–ocean stress**, `SemiImplicitStress`, built on the (zero) ocean surface velocity — a
#    quiescent ocean that still exerts drag on moving ice.

sea_surface_salinity = ocean_surface_salinity(ocean)
sea_surface_u, sea_surface_v = ocean_surface_velocities(ocean)

bottom_heat_boundary_condition = IceWaterThermalEquilibrium(sea_surface_salinity)

ocean_ice_stress = SemiImplicitStress(uₑ = sea_surface_u, vₑ = sea_surface_v)

atmosphere_ice_stress = (u = Field{Face, Center, Nothing}(grid),
                         v = Field{Center, Face, Nothing}(grid))

dynamics = SeaIceMomentumEquation(grid;
                                  coriolis = HydrostaticSphericalCoriolis(),
                                  top_momentum_stress = atmosphere_ice_stress,
                                  bottom_momentum_stress = ocean_ice_stress,
                                  rheology = ElastoViscoPlasticRheology(),
                                  solver = SplitExplicitSolver(grid; substeps = 120))

sea_ice = sea_ice_simulation(grid; Δt = 15minutes,
                             advection = WENO(order = 7),
                             dynamics,
                             timestepper = :ForwardEuler,
                             bottom_heat_boundary_condition)

# We start the ice from the ECCO state estimate for the same date — a realistic January–February pack —
# rather than from open water, so the simulation begins near the seasonal maximum and we watch it melt back
# through the spring and summer:

ecco_ice = MetadataSet(:sea_ice_thickness, :sea_ice_concentration; dataset = ECCO4Monthly(), date)
set!(sea_ice.model, ecco_ice)

# ## The prescribed atmosphere
#
# JRA55-do supplies the atmospheric state — winds, air temperature, humidity, precipitation — and the
# downwelling radiation. The turbulent fluxes that actually cool the ice are computed interactively from
# similarity theory, using the evolving ice-surface temperature:

atmosphere = JRA55PrescribedAtmosphere(arch)
radiation  = JRA55PrescribedRadiation(arch)

# ## The coupled model
#
# `OceanSeaIceModel` owns the components and the interfaces between them: the atmosphere–ice turbulent and
# radiative fluxes, and the ice–ocean heat and salt exchange. The slab ocean evolves under those fluxes,
# so the coupling is two-way, and every exchanged flux is a `Field` you can output:

arctic = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(arctic; Δt = 15minutes, stop_time = 180days)

wall_time = Ref(time_ns())

function progress(sim)
    ℐ = sim.model.sea_ice.model
    h = ℐ.ice_thickness
    ℵ = ℐ.ice_concentration
    u, v = ℐ.velocities
    T = sim.model.ocean.temperature

    msg = @sprintf("time: %s, iter: %d, max(h): %.2f m, max(ℵ): %.2f, max|u|: %.2e m s⁻¹, extrema(T): (%.1f, %.1f) °C, wall: %s",
                   prettytime(sim), iteration(sim), maximum(h), maximum(ℵ),
                   maximum(abs, u), minimum(T), maximum(T),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    @info msg
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(5days))

# ## Output and run
#
# Ice thickness, concentration, and drift, daily:

h = sea_ice.model.ice_thickness
ℵ = sea_ice.model.ice_concentration
u, v = sea_ice.model.velocities

simulation.output_writers[:ice] = JLD2Writer(sea_ice.model, (; h, ℵ, u, v);
                                             filename = "arctic_sea_ice.jld2",
                                             schedule = TimeInterval(1days),
                                             overwrite_existing = true)

run!(simulation)

# ## An Arctic movie
#
# Because the rotated grid carries the true longitude and latitude of every node, each snapshot is drawn
# directly — no interpolation onto an intermediate grid. We plot on a GeoMakie `GeoAxis` in a
# polar-stereographic projection (`+proj=stere +lat_0=90`), with three panels following the pack: thickness,
# concentration, and ice speed with the drift drawn as arrows. We show *speed* rather than the eastward and
# northward velocity because a single vector component is ill-defined at the pole — it folds to a spurious
# sign change there — whereas speed is rotation-invariant and the arrows carry the direction. Land is
# painted over the fields in gray, with coastlines for orientation:

using CairoMakie
using GeoMakie

hi = FieldTimeSeries("arctic_sea_ice.jld2", "h")
ℵi = FieldTimeSeries("arctic_sea_ice.jld2", "ℵ")
ui = FieldTimeSeries("arctic_sea_ice.jld2", "u")
vi = FieldTimeSeries("arctic_sea_ice.jld2", "v")

λ = Array(λnodes(grid, Center(), Center(), Center()))   # 2D geographic longitude of every node
φ = Array(φnodes(grid, Center(), Center(), Center()))   # 2D geographic latitude
Nx, Ny = size(λ)
landmask = ifelse.(Array(interior(grid.immersed_boundary.bottom_height, :, :, 1)) .< 0, NaN, 1.0)

# The ice velocity is stored in the grid's local frame (along the rotated lon/lat axes). To draw the drift
# as physically-oriented arrows on the geographic map we rotate it into true east/north using the angle α
# between the grid's x-axis and geographic east, computed once from the 3D node positions:

X = @. cosd(φ) * cosd(λ); Y = @. cosd(φ) * sind(λ); Z = @. sind(φ)
α = zeros(Nx, Ny)
for j in 1:Ny, i in 2:Nx-1
    τ = (X[i+1, j] - X[i-1, j], Y[i+1, j] - Y[i-1, j], Z[i+1, j] - Z[i-1, j])   # grid-east tangent
    ê = (-sind(λ[i, j]), cosd(λ[i, j]), 0)
    n̂ = (-sind(φ[i, j]) * cosd(λ[i, j]), -sind(φ[i, j]) * sind(λ[i, j]), cosd(φ[i, j]))
    α[i, j] = atan(τ[1]*n̂[1] + τ[2]*n̂[2] + τ[3]*n̂[3], τ[1]*ê[1] + τ[2]*ê[2] + τ[3]*ê[3])
end
α[1, :] .= α[2, :]; α[Nx, :] .= α[Nx-1, :]

# Drift arrows are anchored at a fixed subset of wet cells (every 9th), so the animation updates them in place:
anchors = [(i, j) for i in 5:9:Nx, j in 5:9:Ny if isnan(landmask[i, j])]
arrow_longitude = [λ[i, j] for (i, j) in anchors]
arrow_latitude  = [φ[i, j] for (i, j) in anchors]

times = hi.times
n = Observable(length(times))
title = @lift "Arctic sea ice — day " * string(round(Int, times[$n] / days))

snapshot(fts) = @lift Array(interior(fts[$n], :, :, 1))
hₙ = snapshot(hi)
ℵₙ = snapshot(ℵi)

# Co-locate the staggered velocity to cell centers, rotate to geographic east/north, and keep the drift
# only where there is real ice (concentration ≥ 0.15); elsewhere the model still carries a noisy velocity:
drift = @lift begin
    u = Array(interior(ui[$n], :, :, 1))
    v = Array(interior(vi[$n], :, :, 1))
    uc = 0.5 .* (u[1:Nx, :] .+ u[2:Nx+1, :])
    vc = 0.5 .* (v[:, 1:Ny] .+ v[:, 2:Ny+1])
    ice = @. ($ℵₙ ≥ 0.15) & isnan(landmask)
    east  = @. ifelse(ice, uc * cos(α) - vc * sin(α), 0.0)
    north = @. ifelse(ice, uc * sin(α) + vc * cos(α), 0.0)
    speed = @. ifelse(ice, hypot(uc, vc), NaN)
    (; east, north, speed)
end
speedₙ = @lift $drift.speed
arrow_east  = @lift [(s = $drift.speed[i, j]; isfinite(s) && s > 0.03 ? $drift.east[i, j]  / s : NaN) for (i, j) in anchors]
arrow_north = @lift [(s = $drift.speed[i, j]; isfinite(s) && s > 0.03 ? $drift.north[i, j] / s : NaN) for (i, j) in anchors]

projection = "+proj=stere +lat_0=90 +lat_ts=70"

fig = Figure(size = (1580, 600))
fig[0, :] = Label(fig, title, fontsize = 22, tellwidth = false)

# Each field is drawn per-cell with `surface!` — filled contours leave thin polygon artifacts on this
# sheared curvilinear mesh. The gray land layer sits on top to hide the per-cell coastline, and the explicit
# `limits` keep the global coastlines from blowing up the polar view:
function surface_panel!(column, field, name; colorrange, colormap)
    ax = GeoAxis(fig[1, column]; dest = projection, limits = ((-180, 180), (50, 90)), title = name)
    sf = surface!(ax, λ, φ, field; colormap, colorrange, nan_color = :transparent, shading = NoShading)
    surface!(ax, λ, φ, landmask; colormap = [:gray20, :gray20], nan_color = :transparent, shading = NoShading)
    lines!(ax, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
    Colorbar(fig[1, column + 1], sf)
    return ax
end

surface_panel!(1, hₙ, "ice thickness [m]"; colorrange = (0, 4), colormap = Reverse(:blues))
surface_panel!(3, ℵₙ, "ice concentration"; colorrange = (0, 1), colormap = :ice)

# Speed is masked to the ice (NaN over open water), with the rotated unit drift vectors overlaid as arrows:
ax_speed = surface_panel!(5, speedₙ, "ice speed [m s⁻¹]"; colorrange = (0, 0.4), colormap = :amp)
arrows2d!(ax_speed, arrow_longitude, arrow_latitude, arrow_east, arrow_north;
          lengthscale = 2.6, color = :black, tipwidth = 7, tiplength = 8, shaftwidth = 1.6)

CairoMakie.record(fig, "arctic_sea_ice.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](arctic_sea_ice.mp4)
#
# Through the run the pack thins from its late-winter maximum, the marginal ice zone retreats toward the
# central Arctic, and the wind-driven drift opens and closes leads along the way — driven by a 50 m slab
# ocean and a reanalysis sky, with the ice and the sea-surface temperature as the prognostic fields.
#
# ## Where next
#
# - **A prognostic ocean.** Replace `SlabOcean` with `ocean_simulation(grid)` and pass it to
#   `OceanSeaIceModel`: the ice now feels a full mixed layer with currents and lateral heat transport —
#   the configuration of a fully coupled regional ocean–sea ice run.
# - **The marginal ice zone.** Increase the resolution and the rheology starts to resolve the linear
#   kinematic features — leads and ridges — that set the Arctic's winter heat loss.
