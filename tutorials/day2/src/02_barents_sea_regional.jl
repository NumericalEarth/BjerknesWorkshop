# # The Barents Sea: a regional coupled ocean–sea ice simulation
#
# *A realistic regional coupled ocean–sea ice simulation.*
#
# Several threads of ocean–sea ice modeling meet in one place, and it happens to be a sea worth watching: the
# **Barents Sea**, where the Atlantic water that baroclinic eddies carry north meets the ice governed by slab
# thermodynamics. It is the region where the Arctic's "Atlantification"
# is unfolding fastest — the ice edge retreating as the Atlantic inflow warms
# ([Årthun et al., 2012](https://doi.org/10.1175/JCLI-D-11-00466.1);
# [Smedsrud et al., 2013](https://doi.org/10.1002/rog.20017)) — and it makes an ideal regional target: small
# enough to simulate at eddy-permitting resolution on one GPU, rich enough to contain an ice edge, a polar
# front, a warm inflow, and a shelf.
#
# [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl) supplies the unglamorous 90% that
# separates an idealized experiment from a real regional simulation: bathymetry regridding,
# state-estimate initial conditions, reanalysis forcing, bulk fluxes, and the ocean–ice coupling. Contrarily
# to the traditional regional-modeling workflow — namelists, preprocessing executables, grid generators — the
# configuration here *is the program*: the same script style you would use for an idealized setup.
#
# !!! warning "Hardware and data requirements"
#     This is a GPU tutorial: ~3 km resolution over the Barents Sea is a few million grid cells. The first run
#     downloads GLORYS12 and JRA55 data (a few GB, cached for every later run); GLORYS lives at the Copernicus
#     Marine Service, so you need a (free) account and a one-time `CopernicusMarine.login()`. On the workshop
#     cluster the cache is pre-staged. On a laptop you can still read along, or halve every resolution number
#     and run on the CPU with some patience.
#
# !!! warning "Development stack"
#     The open boundary conditions used below live on the `ss/open-boundary-conditions` branch of
#     Oceananigans, and the day-2 environment `dev`s that branch together with compat-adjusted ClimaSeaIce and
#     NumericalEarth checkouts (see the `[sources]` section of the environment's `Project.toml`). Expect this
#     section of the stack to evolve faster than the rest of the tutorial.

using NumericalEarth, Oceananigans, Oceananigans.Units
using Oceananigans.BoundaryConditions: Radiation, FlatherBoundaryCondition, NormalFlowBoundaryCondition
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Dates, CUDA, Printf
using CopernicusMarine   # enables the GLORYS download extension

arch = GPU()

# ## A regional grid
#
# A latitude–longitude box from the Lofoten basin to Franz Josef Land: longitude 5°E–60°E, latitude 63°N–78°N,
# at 1/8° — about 3.5 km zonally and 14 km meridionally at these latitudes, eddy-permitting for the ~5 km
# Barents deformation radius. The vertical grid concentrates 40 levels toward the surface over 4000 m, enough
# to hold the Norwegian Sea basin in the southwest corner; the Barents shelf itself sits at 200–400 m:

const λ₁, λ₂ =  5, 60
const φ₁, φ₂ = 63, 78

Nx = 8 * (λ₂ - λ₁)
Ny = 8 * (φ₂ - φ₁)
Nz = 40

depth = 4000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂),
                                        z,
                                        halo = (7, 7, 7))

# Downloads land in `DATA_DIR` when that environment variable is set, else each product's default cache:
dir_kw = (;) # haskey(ENV, "DATA_DIR") ? (; dir = ENV["DATA_DIR"]) : (;)

bathymetry = Metadatum(:bottom_height; dataset = ETOPO2022(), dir_kw...)
bottom_height = regrid_bathymetry(underlying_grid, bathymetry;
                                  minimum_depth = 15,
                                  interpolation_passes = 25,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height); active_cells_map = true)

# The same `ImmersedBoundaryGrid` as any idealized setup — Novaya Zemlya and the Norwegian coast are just very
# large sills. Let's look at the stage:

using CairoMakie, SixelTerm

h_bottom = Array(interior(grid.immersed_boundary.bottom_height, :, :, 1))
h_bottom[h_bottom .≥ 0] .= NaN

fig = Figure(size = (900, 350))
ax = Axis(fig[1, 1], xlabel = "longitude [°E]", ylabel = "latitude [°N]",
          title = "Barents Sea bathymetry")
hm = heatmap!(ax, range(λ₁, λ₂, Nx), range(φ₁, φ₂, Ny), h_bottom,
              colormap = :deep, colorrange = (-depth, 0))
Colorbar(fig[1, 2], hm, label = "bottom height [m]")
save("barents_bathymetry.png", fig)
nothing #hide

# ![](barents_bathymetry.png)
#
# ## Open boundary conditions from GLORYS12
#
# A regional domain has open edges: Atlantic water must flow in through the western boundary and Arctic water
# through the northern one. We treat them with genuine **open boundary conditions**, fed by the GLORYS12
# reanalysis (1/12°, daily), in the combination that regional modeling converged on decades ago:
#
# - the **barotropic mode** gets a Flather (1976) characteristic condition: the incoming Riemann invariant is
#   prescribed, the outgoing one radiates freely — surface gravity waves leave the domain instead of sloshing
#   back from the boundary;
# - **baroclinic velocities and tracers** get an
#   [Orlanski (1976)](https://doi.org/10.1016/0021-9991(76)90023-1) radiation condition with adaptive nudging
#   ([Marchesiello et al., 2001](https://doi.org/10.1016/S1463-5003(00)00013-5)): the boundary value follows a
#   locally-diagnosed phase speed on outflow, and relaxes toward the external data on inflow — strongly when
#   the flow enters, weakly when it leaves.
#
# The external data: GLORYS12 daily fields, pre-interpolated onto the model grid as `FieldTimeSeries` with a
# lazy, GPU-aware backend that keeps fifty days in memory and interpolates linearly in time. (The simulation
# clock starts at the first date below, so model time and dataset time agree.)
#
# We also crop the GLORYS download to the model footprint (a 1° margin past the grid covers the halos and the
# boundary interpolation) instead of pulling the global 1/12° fields:

dates   = DateTime(1993, 1, 1) : Day(1) : DateTime(1993, 3, 1)
dataset = GLORYSDaily()
region  = BoundingBox(longitude=(0, 80), latitude=(55, 85))

Tᵉˣᵗ = FieldTimeSeries(Metadata(:temperature;  dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
Sᵉˣᵗ = FieldTimeSeries(Metadata(:salinity;     dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
uᵉˣᵗ = FieldTimeSeries(Metadata(:u_velocity;   dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
vᵉˣᵗ = FieldTimeSeries(Metadata(:v_velocity;   dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
ηᵉˣᵗ = FieldTimeSeries(Metadata(:free_surface; dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
nothing #hide

# Discrete boundary functions hand the external values to the boundary machinery: each evaluates its
# `FieldTimeSeries` at the boundary index and the current clock time — the same zero-overhead pattern as every
# forcing and flux function:

@inline  west_obc(j, k, grid, clock, fields, φ) = @inbounds φ[1,           j, k, Oceananigans.Units.Time(clock.time)]
@inline  east_obc(j, k, grid, clock, fields, φ) = @inbounds φ[grid.Nx,     j, k, Oceananigans.Units.Time(clock.time)]
@inline north_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny,     k, Oceananigans.Units.Time(clock.time)]

@inline  east_u_obc(j, k, grid, clock, fields, φ) = @inbounds φ[grid.Nx+1, j, k, Oceananigans.Units.Time(clock.time)]
@inline north_v_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny+1, k, Oceananigans.Units.Time(clock.time)]
nothing #hide

# Radiation timescales à la Marchesiello: 1days on inflow (the boundary follows GLORYS closely where water
# enters), infinite on outflow (the interior solution leaves undisturbed). The south boundary is the Norwegian
# coast — land — and keeps the default wall:

u_obcs = FieldBoundaryConditions(
    west = NormalFlowBoundaryCondition(west_obc,   discrete_form = true, parameters = uᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east = NormalFlowBoundaryCondition(east_u_obc, discrete_form = true, parameters = uᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

v_obcs = FieldBoundaryConditions(
    north = NormalFlowBoundaryCondition(north_v_obc, discrete_form = true, parameters = vᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

T_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

S_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

# The Flather condition acts on the barotropic transports `U` and `V` — inside every barotropic substep of the
# split-explicit solver. Its external state is the 2-tuple `(Uᵉˣᵗ, ηᵉˣᵗ)`: the external *barotropic transport*
# `∫uᵉˣᵗ dz` and the external free-surface elevation. Feeding GLORYS here — not zero — is what lets the
# Atlantic inflow's depth-mean transport actually cross the boundary; with `(0, 0)` the Flather would radiate
# the barotropic mode toward rest and damp the very inflow this domain exists to capture. `Uᵉˣᵗ` is the column
# integral of the GLORYS velocity already loaded in `uᵉˣᵗ`/`vᵉˣᵗ`, with `immersed_peripheral_node` skipping
# the solid cells so the sum is exactly the wet-column transport (rather than trusting the dataset to zero its
# land points), and `ηᵉˣᵗ` is the GLORYS `zos` read at the boundary:

@inline wetcell(i, j, k, grid, ℓx, ℓy, ℓz) =
    !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline function vertical_integral(i, j, grid, u, t, Δz, ℓx, ℓy, ℓz)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = wetcell(i, j, k, grid, ℓx, ℓy, ℓz)
        U += ifelse(wet, u[i, j, k, t] * Δz(i, j, k, grid), zero(U))
    end
    return U
end

@inline function west_U_obc(j, k, grid, clock, fields, p)
    t = Oceananigans.Units.Time(clock.time)
    U = vertical_integral(1, j, grid, p.u, t, Δzᶠᶜᶜ, Face(), Center(), Center())
    return (U, @inbounds p.η[1, j, 1, t])
end

@inline function east_U_obc(j, k, grid, clock, fields, p)
    i = grid.Nx+1
    t = Oceananigans.Units.Time(clock.time)
    U = vertical_integral(i, j, grid, p.u, t, Δzᶠᶜᶜ, Face(), Center(), Center())
    return (U, @inbounds p.η[grid.Nx, j, 1, t])
end

@inline function north_V_obc(i, k, grid, clock, fields, p)
    j = grid.Ny+1
    t = Oceananigans.Units.Time(clock.time)
    V = vertical_integral(i, j, grid, p.v, t, Δzᶜᶠᶜ, Center(), Face(), Center())
    return (V, @inbounds p.η[i, j, 1, t])
end

U_obcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
    west = FlatherBoundaryCondition(west_U_obc, discrete_form = true, parameters = (u = uᵉˣᵗ, η = ηᵉˣᵗ)),
    east = FlatherBoundaryCondition(east_U_obc, discrete_form = true, parameters = (u = uᵉˣᵗ, η = ηᵉˣᵗ)))

V_obcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
    north = FlatherBoundaryCondition(north_V_obc, discrete_form = true, parameters = (v = vᵉˣᵗ, η = ηᵉˣᵗ)))

# ## ... and a sponge behind them
#
# Open boundary conditions are good at radiating what arrives perpendicularly and following the prescribed
# inflow; they are imperfect for everything else (oblique waves, boundary-trapped instabilities, slow drift).
# The standard belt-and-braces complement is a thin **sponge layer** just inside the open edges, restoring
# toward the same GLORYS12 data with `DatasetRestoring` — a 1-day timescale at the edge, fading to nothing
# within a couple of degrees:

@inline rim(ξ, edge, width) = exp(-(ξ - edge)^2 / 2width^2)
@inline sponge_mask(λ, φ, z, t) = max(rim(λ, λ₁, 2), rim(λ, λ₂, 2), rim(φ, φ₂, 1))

# Tracers relax on the gentle 1-day timescale. The *velocities* need a much stronger edge nudge: the radiation pins the boundary-normal
# velocity to GLORYS while the interior spins up its own flow. Therefore, to avoid mismatches, a ~20-minute velocity sponge keeps the
# near-boundary interior matched to the prescribed boundary and holds max|w| at the GLORYS-consistent floor.

FT = DatasetRestoring(Metadata(:temperature; dates, dataset, region, dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
Fu = DatasetRestoring(Metadata(:u_velocity;  dates, dataset, region, dir_kw...), grid; rate = 1/20minutes, mask = sponge_mask, inpainting=100)
Fv = DatasetRestoring(Metadata(:v_velocity;  dates, dataset, region, dir_kw...), grid; rate = 1/20minutes, mask = sponge_mask, inpainting=100)
FS = DatasetRestoring(Metadata(:salinity;    dates, dataset, region, dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)

# ## The ocean component
#
# The familiar hydrostatic model, assembled by `ocean_simulation` with realistic defaults — TEOS-10 equation
# of state, CATKE vertical mixing, WENO advection, split-explicit free surface — plus our open boundaries and
# sponge forcings. The lateral boundary conditions merge side-by-side with the defaults, so the surface fluxes
# (which the coupler owns), the bottom drag, and the immersed drag stay wired. At eddy-permitting resolution
# we leave the Gent–McWilliams parameterization *out*: a resolved baroclinic front shows what the resolved
# eddies can do by themselves:

closure = (CATKEVerticalDiffusivity(minimum_tke=1e-7)) #, HorizontalScalarBiharmonicDiffusivity(ν = 5e8))
time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)

ocean = ocean_simulation(grid;
                         free_surface = SplitExplicitFreeSurface(grid; substeps=80),
                         momentum_advection = WENOVectorInvariant(; order=5, time_discretization),
                         tracer_advection = WENO(; order=7, time_discretization, minimum_buffer_upwind_order=1),
                         closure,
                         forcing = (T = FT, S = FS, u = Fu, v = Fv),
                         boundary_conditions = (u = u_obcs, v = v_obcs,
                                                T = T_obcs, S = S_obcs,
                                                U = U_obcs, V = V_obcs))

# ## The sea-ice component
#
# Sea ice here is thermodynamics only — conductive growth and melt between the ocean below and the atmosphere
# above — assembled by `sea_ice_simulation` and wired to the ocean: the ice–ocean heat flux uses the model's
# evolving sea-surface salinity for the freezing point. We leave the ice *dynamics* off (`dynamics = nothing`),
# since open boundary conditions for the sea-ice momentum are not ready yet:

sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)

# ## Initial conditions: the Barents Sea in winter
#
# Mid-winter 1993 from GLORYS12 — the same reanalysis that feeds the boundaries, so the initial state and the
# boundary data agree from the first time step. Temperature and salinity go to the ocean, thickness and
# concentration to the ice; one `MetadataSet` feeds both models, each picking up the variables it owns:

set!(ocean.model, T = Tᵉˣᵗ[1], S = Sᵉˣᵗ[1])
set!(sea_ice.model, h = Metadatum(:sea_ice_thickness;     date=dates[1], dataset=ECCO4Monthly(), dir_kw...),
                    ℵ = Metadatum(:sea_ice_concentration; date=dates[1], dataset=ECCO4Monthly(), dir_kw...))

# ## The atmosphere and the coupled model
#
# JRA55 reanalysis supplies winds, temperature, humidity, precipitation, radiation and runoff; the turbulent
# fluxes are computed interactively from the evolving SST and ice surface temperature, exactly as in a coupled
# climate model run under the OMIP protocol. `EarthSystemModel` owns the components and every interface
# between them — each exchanged flux is a `Field` you can inspect and output; there is no hidden coupler:

atmosphere    = JRA55PrescribedAtmosphere(arch; dir_kw...)
radiation     = JRA55PrescribedRadiation(arch; dir_kw...)
land          = JRA55PrescribedLand(arch; dir_kw...)
coupled_model = EarthSystemModel(; ocean, sea_ice, land, atmosphere, radiation)

# Sixty days, watching the pack thin and the ice edge retreat from its mid-winter extent:

simulation = Simulation(coupled_model; Δt = 10minutes, stop_time = 60days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    sea_ice = sim.model.sea_ice
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S
    u, v, w = ocean.model.velocities
    h = sea_ice.model.ice_thickness
    msg = @sprintf("time: %s, iter: %d, extrema(T, S): (%.1f, %.1f) °C (%.1f, %.1f) psu, extrema(u) (%.2e, %.2e, %.2e) max(h): %.2f m, wall: %s",
                   prettytime(sim), iteration(sim),
                   extrema(T)..., extrema(S)..., maximum(abs, u), maximum(abs, v), maximum(abs, w), maximum(h),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    @info msg
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# Daily surface fields from both components:

u, v, w = ocean.model.velocities
h = sea_ice.model.ice_thickness
ℵ = sea_ice.model.ice_concentration
𝒱 = @at((Center, Center, Center), sqrt(u^2 + v^2))
he = h * ℵ
ocean_outputs = merge(ocean.model.tracers, (; 𝒱))

sea_ice_outputs = (; he)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            filename = "barents_ocean_surface.jld2",
                                            indices = (:, :, grid.Nz-2),
                                            schedule = TimeInterval(1days),
                                            overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              filename = "barents_sea_ice_surface.jld2",
                                              schedule = TimeInterval(1days),
                                              overwrite_existing = true)

# The big red button:

run!(simulation)

# ## Sixty days over the Barents
#
# Four daily surface fields — sea-surface temperature and salinity, the surface current speed, and the ice
# volume — the protagonists of the Barents Sea climate story in one frame:

To = FieldTimeSeries("barents_ocean_surface.jld2",   "T")
So = FieldTimeSeries("barents_ocean_surface.jld2",   "S")
Uo = FieldTimeSeries("barents_ocean_surface.jld2",   "𝒱")
hi = FieldTimeSeries("barents_sea_ice_surface.jld2", "he")

times = To.times
n = Observable(length(times))

title = @lift "Barents Sea — day " * string(round(Int, times[$n] / days))

Tₙ = @lift(To[$n])
Sₙ = @lift(So[$n])
Uₙ = @lift(Uo[$n])
hₙ = @lift(hi[$n])

fig = Figure(size = (1200, 650))
fig[0, 1:4] = Label(fig, title, fontsize = 20, tellwidth = false)

ax = Axis(fig[1, 1], ylabel = "latitude [°N]")
hm_T = heatmap!(ax, Tₙ, colormap = :thermal, colorrange = (-2, 8), nan_color = :gray80)
ax = Axis(fig[1, 3])
hm_S = heatmap!(ax, Sₙ, colormap = :haline, colorrange = (32.5, 35.5))
ax = Axis(fig[2, 1], xlabel = "longitude [°E]", ylabel = "latitude [°N]")
hm_U = heatmap!(ax, Uₙ, colormap = Reverse(:solar), colorrange = (0, 0.5))
ax = Axis(fig[2, 3], xlabel = "longitude [°E]")
hm_h = heatmap!(ax, hₙ, colormap = Reverse(:blues), colorrange = (0.01, 1.5), lowclip = :gray80)
Colorbar(fig[1, 2], hm_T, label = "SST [°C]")
Colorbar(fig[1, 4], hm_S, label = "SSS [psu]")
Colorbar(fig[2, 2], hm_U, label = "Surface speed [ms⁻¹]")
Colorbar(fig[2, 4], hm_h, label = "ice volume per area [m]")

CairoMakie.record(fig, "barents_sea.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](barents_sea.mp4)
#
# Things to look for: the warm Atlantic tongue entering from the southwest between the Norwegian coast and Bear
# Island, holding the southwestern Barents ice-free (the reason Murmansk is a year-round port); its northern
# extension — the West Spitsbergen Current, the far tip of the Gulf Stream — running up the western flank of
# Svalbard; the inflow breaking into live, resolved eddies that stir the polar front where it meets Arctic
# water; and, over the sixty days, the ice edge thinning and retreating north and east as that ocean heat
# works on the pack from below.
#
