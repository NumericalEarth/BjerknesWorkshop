# # Sea ice in the Arctic: a pan-Arctic simulation over a prescribed ocean
#
# *The realistic Arctic: a seasonal cycle of sea ice driven by reanalysis.*
#
# Sea ice is where the polar ocean meets the atmosphere, and modeling it well is a question of getting
# three things to talk to each other: the ice itself (which grows, melts, drifts, and fractures), the
# ocean beneath (which supplies or removes heat at the base), and the atmosphere above (which cools the
# surface and pushes the pack around). Here we build a **pan-Arctic** sea-ice simulation in which the
# ocean is *prescribed* — its temperature and salinity are held fixed from a hydrographic climatology, and
# it carries no currents — so that all the prognostic action is in the ice. The atmosphere is the JRA55-do
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
# !!! note "A prescribed ocean is not a no-ocean"
#     The ocean does not evolve, but it is still *there*: its surface salinity sets the freezing point at
#     the base of the ice, and its (here zero) surface velocity enters the ice–ocean drag. Swapping the
#     prescribed ocean for a prognostic `ocean_simulation` turns this into the fully coupled regional run
#     of the Barents Sea tutorial — the components are the same, only the ocean's status changes.

using NumericalEarth
using NumericalEarth.Oceans: PrescribedOcean
using Oceananigans
using Oceananigans.Units
using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using ClimaSeaIce.SeaIceDynamics: SeaIceMomentumEquation, SemiImplicitStress, SplitExplicitSolver
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology
using Dates, Printf, CUDA

arch = GPU()   # CPU() works at reduced resolution

# ## A grid centered on the pole
#
# A latitude–longitude grid degenerates at the geographic poles — the zonal cell width vanishes, and with
# it the affordable time step — which is exactly the wrong behaviour for an Arctic-centered domain. The
# `RotatedLatitudeLongitudeGrid` rotates the coordinate system so that its *own* pole sits over the equator
# (`north_pole = (180, 0)`), leaving a smooth, singularity-free mesh over the real North Pole. The
# `(latitude, longitude)` extents are given in the rotated frame; ``\pm 45`` degrees about the rotated
# equator covers the Arctic Ocean and its marginal seas down to the Nordic Seas and the Bering Strait.

Nx, Ny, Nz = 180, 180, 30

depth = 2000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/3, mutable = true)

underlying_grid = RotatedLatitudeLongitudeGrid(arch;
                                               size = (Nx, Ny, Nz),
                                               latitude = (-45, 45),
                                               longitude = (-45, 45),
                                               z,
                                               north_pole = (180, 0),
                                               halo = (5, 5, 4),
                                               topology = (Bounded, Bounded, Bounded))

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 15, major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# ## The prescribed ocean
#
# `PrescribedOcean` holds a sea-surface temperature, a sea-surface salinity, and a surface velocity as
# *prescribed* fields — they are read, never integrated. We initialize temperature and salinity from the
# EN4 hydrographic objective analysis and leave the velocity at zero, so the ocean is a fixed reservoir of
# heat and salt with no currents. We regrid the EN4 surface onto our grid through a temporary field, then
# hand the result to the prescribed ocean:

ocean = PrescribedOcean(grid)

date = DateTime(1993, 2, 1)   # late-winter Arctic, near the seasonal ice maximum

surface = Field{Center, Center, Nothing}(grid)

set!(surface, Metadatum(:temperature; dataset = EN4Monthly(), date))
set!(ocean; T = interior(surface))

set!(surface, Metadatum(:salinity; dataset = EN4Monthly(), date))
set!(ocean; S = interior(surface))

# ## The sea-ice model
#
# The ice carries slab thermodynamics — conductive growth and melt between the ocean base and the
# atmosphere-set surface temperature — and EVP dynamics, the elastic–viscous–plastic rheology that lets the
# pack resist convergence, fail in shear, and open leads. Two couplings to the prescribed ocean are
# explicit here, because the convenience constructor only wires them automatically for a *prognostic*
# ocean:
#
# 1. the **freezing boundary condition** at the base, `IceWaterThermalEquilibrium`, which reads the
#    prescribed sea-surface salinity to set the local freezing temperature, and
# 2. the **ice–ocean stress**, `SemiImplicitStress`, built here on the (zero) prescribed surface velocity —
#    a quiescent ocean that still exerts drag on moving ice.

sea_surface_salinity = ocean.sea_surface_salinity[1]
sea_surface_u = ocean.velocities.u[1]
sea_surface_v = ocean.velocities.v[1]

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
# radiative fluxes, and the ice–ocean heat and salt exchange. With a prescribed ocean the ocean side of
# those interfaces is a one-way read, but every exchanged flux is still a `Field` you can output:

arctic = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(arctic; Δt = 5minutes, stop_time = 270days)

wall_time = Ref(time_ns())

function progress(sim)
    ℐ = sim.model.sea_ice.model
    h = ℐ.ice_thickness
    ℵ = ℐ.ice_concentration
    u, v = ℐ.velocities

    msg = @sprintf("time: %s, iter: %d, max(h): %.2f m, max(ℵ): %.2f, max|u|: %.2e m s⁻¹, wall: %s",
                   prettytime(sim), iteration(sim), maximum(h), maximum(ℵ),
                   maximum(abs, u), prettytime(1e-9 * (time_ns() - wall_time[])))
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
# Ice concentration and thickness over the melt season. The grid is rotated, so we plot in the model's
# own indices rather than geographic coordinates — the pole sits in the middle of the frame:

using CairoMakie

hi = FieldTimeSeries("arctic_sea_ice.jld2", "h"; backend = OnDisk())
ℵi = FieldTimeSeries("arctic_sea_ice.jld2", "ℵ"; backend = OnDisk())

times = hi.times
n = Observable(length(times))

title = @lift "Arctic sea ice — day " * string(round(Int, times[$n] / day))

hₙ = @lift interior(hi[$n], :, :, 1)
ℵₙ = @lift interior(ℵi[$n], :, :, 1)

fig = Figure(size = (1000, 520))
fig[0, :] = Label(fig, title, fontsize = 22, tellwidth = false)

ax_h = Axis(fig[1, 1], title = "thickness [m]", aspect = 1)
hm_h = heatmap!(ax_h, hₙ, colormap = Reverse(:blues), colorrange = (0, 4), nan_color = :gray20)
Colorbar(fig[1, 2], hm_h)

ax_ℵ = Axis(fig[1, 3], title = "concentration", aspect = 1)
hm_ℵ = heatmap!(ax_ℵ, ℵₙ, colormap = :ice, colorrange = (0, 1), nan_color = :gray20)
Colorbar(fig[1, 4], hm_ℵ)

CairoMakie.record(fig, "arctic_sea_ice.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](arctic_sea_ice.mp4)
#
# Through the run the pack thins from its late-winter maximum, the marginal ice zone retreats toward the
# central Arctic, and the wind-driven drift opens and closes leads along the way — all of it forced by a
# fixed ocean and a reanalysis sky, with the ice as the only prognostic component.
#
# ## Where next
#
# - **A prognostic ocean.** Replace `PrescribedOcean` with `ocean_simulation(grid)` and pass it to
#   `OceanSeaIceModel`: the ice now feels an evolving mixed layer, and the configuration becomes the
#   coupled regional run used for the Barents Sea.
# - **The marginal ice zone.** Increase the resolution and the rheology starts to resolve the linear
#   kinematic features — leads and ridges — that set the Arctic's winter heat loss.
