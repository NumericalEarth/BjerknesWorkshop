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
# cycle: ice thickening through winter, the marginal ice zone retreating in summer, leads opening under
# the wind.
#
# !!! warning "Hardware and data requirements"
#     This is a GPU tutorial. The first run downloads the EN4 (or ECCO) hydrography for the ocean state and
#     the JRA55-do reanalysis for the atmosphere (a few GB, cached for every later run). On the workshop
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

arch = CPU()   # CPU() works at reduced resolution

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

sea_ice = sea_ice_simulation(grid; Δt = 5minutes,
                             advection = WENO(order = 7),
                             dynamics,
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

simulation = Simulation(arctic; Δt = 5minutes, stop_time = 270days)

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
# directly — no interpolation onto an intermediate grid. We read the node coordinates once and plot the
# curvilinear fields on a GeoMakie `GeoAxis` in a polar-stereographic projection (`+proj=stere +lat_0=90`),
# masking land to gray and overlaying coastlines for orientation:

using CairoMakie
using GeoMakie

hi = FieldTimeSeries("arctic_sea_ice.jld2", "h"; backend = OnDisk())
ℵi = FieldTimeSeries("arctic_sea_ice.jld2", "ℵ"; backend = OnDisk())

λ = Array(λnodes(grid, Center(), Center(), Center()))   # 2D geographic longitude of every node
φ = Array(φnodes(grid, Center(), Center(), Center()))   # 2D geographic latitude
wet = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1)) .< 0

times = hi.times
n = Observable(length(times))
title = @lift "Arctic sea ice — day " * string(round(Int, times[$n] / days))

# Mask land to NaN so it draws in `nan_color` rather than as zero-ice ocean:
masked(field) = (data = Array(interior(field, :, :, 1)); data[.!wet] .= NaN; data)
hₙ = @lift masked(hi[$n])
ℵₙ = @lift masked(ℵi[$n])

projection = "+proj=stere +lat_0=90 +lat_ts=70"

fig = Figure(size = (1000, 560))
fig[0, :] = Label(fig, title, fontsize = 22, tellwidth = false)

ax_h = GeoAxis(fig[1, 1]; dest = projection, title = "ice thickness [m]")
ylims!(ax_h, 50, 90)
sf_h = surface!(ax_h, λ, φ, hₙ; colormap = Reverse(:blues), colorrange = (0, 4),
                nan_color = :gray20, shading = NoShading)
lines!(ax_h, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
Colorbar(fig[1, 2], sf_h)

ax_ℵ = GeoAxis(fig[1, 3]; dest = projection, title = "ice concentration")
ylims!(ax_ℵ, 50, 90)
sf_ℵ = surface!(ax_ℵ, λ, φ, ℵₙ; colormap = :ice, colorrange = (0, 1),
                nan_color = :gray20, shading = NoShading)
lines!(ax_ℵ, GeoMakie.coastlines(); color = :black, linewidth = 0.5)
Colorbar(fig[1, 4], sf_ℵ)

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
