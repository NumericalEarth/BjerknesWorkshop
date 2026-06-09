# # The Barents Sea: a regional coupled ocean–sea ice simulation
#
# *Tuesday — one day in the high-latitude ocean, part 6: all of it together.*
#
# Everything we touched today meets in one place, and it happens to be the sea outside the window: the
# **Barents Sea**, where the Atlantic water that the eddies of part 2 carry north meets the ice whose
# thermodynamics and dynamics we built in parts 3 and 4. It is the region where the Arctic's "Atlantification"
# is unfolding fastest — the ice edge retreating as the Atlantic inflow warms
# ([Årthun et al., 2012](https://doi.org/10.1175/JCLI-D-11-00466.1);
# [Smedsrud et al., 2013](https://doi.org/10.1002/rog.20017)) — and it makes an ideal regional target: small
# enough to simulate at eddy-permitting resolution on one GPU, rich enough to contain an ice edge, a polar
# front, a warm inflow, and a shelf.
#
# [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl) supplies the unglamorous 90% that
# separates today's idealized experiments from a real regional simulation: bathymetry regridding,
# state-estimate initial conditions, reanalysis forcing, bulk fluxes, and the ocean–ice coupling. Contrarily
# to the traditional regional-modeling workflow — namelists, preprocessing executables, grid generators — the
# configuration here *is the program*: the same script style we have used since this morning's sill.
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
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
using Oceananigans.Units: Time
using Dates, CUDA, Printf
using CopernicusMarine   # enables the GLORYS download extension

arch = GPU()

# ## A regional grid
#
# A latitude–longitude box from the Lofoten basin to Franz Josef Land: longitude 5°E–60°E, latitude 67°N–80°N,
# at 1/8° — about 3.5 km zonally and 14 km meridionally at these latitudes, eddy-permitting for the ~5 km
# Barents deformation radius. The vertical grid concentrates 40 levels toward the surface over 4000 m, enough
# to hold the Norwegian Sea basin in the southwest corner; the Barents shelf itself sits at 200–400 m:

λ₁, λ₂ = 5, 60    # longitude extent [°E]
φ₁, φ₂ = 60, 80   # latitude extent [°N]

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

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 15,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map = true)

# The same `ImmersedBoundaryGrid` as this morning's sill — Novaya Zemlya and the Norwegian coast are just very
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
# reanalysis (1/12°, monthly), in the combination that regional modeling converged on decades ago:
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
# The external data: GLORYS12 monthly fields, pre-interpolated onto the model grid as `FieldTimeSeries` with a
# lazy, GPU-aware backend that keeps two months in memory and interpolates linearly in time. (The simulation
# clock starts at the first date below, so model time and dataset time agree.)
#
# We also crop the GLORYS download to the model footprint (a 1° margin past the grid covers the halos and the
# boundary interpolation) instead of pulling the global 1/12° fields:

dates   = DateTime(1993, 1, 1) : Month(1) : DateTime(1994, 1, 1)
dataset = GLORYSMonthly()
region  = BoundingBox(longitude = (λ₁ - 1, λ₂ + 1), latitude = (φ₁ - 1, φ₂ + 1))

Tᵉˣᵗ = FieldTimeSeries(Metadata(:temperature;  dates, dataset, region), grid, inpainting=100)
Sᵉˣᵗ = FieldTimeSeries(Metadata(:salinity;     dates, dataset, region), grid, inpainting=100)
uᵉˣᵗ = FieldTimeSeries(Metadata(:u_velocity;   dates, dataset, region), grid, inpainting=100)
vᵉˣᵗ = FieldTimeSeries(Metadata(:v_velocity;   dates, dataset, region), grid, inpainting=100)
ηᵉˣᵗ = FieldTimeSeries(Metadata(:free_surface; dates, dataset, region), grid, inpainting=100)
nothing #hide

# Discrete boundary functions hand the external values to the boundary machinery: each evaluates its
# `FieldTimeSeries` at the boundary index and the current clock time — the same zero-overhead pattern as every
# forcing and flux function this week:

@inline  west_obc(j, k, grid, clock, fields, φ) = @inbounds φ[1,           j, k, Time(clock.time)]
@inline  east_obc(j, k, grid, clock, fields, φ) = @inbounds φ[grid.Nx,     j, k, Time(clock.time)]
@inline north_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny,     k, Time(clock.time)]

@inline east_u_obc(j, k, grid, clock, fields, φ)  = @inbounds φ[grid.Nx+1, j, k, Time(clock.time)]
@inline north_v_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny+1, k, Time(clock.time)]
nothing #hide

# Radiation timescales à la Marchesiello: a day on inflow (the boundary follows GLORYS closely where water
# enters), a month on outflow (the interior solution leaves undisturbed). The south boundary is the Norwegian
# coast — land — and keeps the default wall:

radiation = Radiation(inflow_timescale = 1days, outflow_timescale = 30days)

u_obcs = FieldBoundaryConditions(
    west = NormalFlowBoundaryCondition(west_obc,   discrete_form = true, parameters = uᵉˣᵗ, scheme = radiation),
    east = NormalFlowBoundaryCondition(east_u_obc, discrete_form = true, parameters = uᵉˣᵗ, scheme = radiation))

v_obcs = FieldBoundaryConditions(
    north = NormalFlowBoundaryCondition(north_v_obc, discrete_form = true, parameters = vᵉˣᵗ, scheme = radiation))

T_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = radiation),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = radiation),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Tᵉˣᵗ, scheme = radiation))

S_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = radiation),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = radiation),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Sᵉˣᵗ, scheme = radiation))

# The Flather condition acts on the barotropic transports `U` and `V` — inside every barotropic substep of the
# split-explicit solver. Its external state is the 2-tuple `(Uᵉˣᵗ, ηᵉˣᵗ)`: the external *barotropic transport*
# `∫uᵉˣᵗ dz` and the external free-surface elevation. Feeding GLORYS here — not zero — is what lets the
# Atlantic inflow's depth-mean transport actually cross the boundary; with `(0, 0)` the Flather would radiate
# the barotropic mode toward rest and damp the very inflow this domain exists to capture. `Uᵉˣᵗ` is the column
# integral of the GLORYS velocity already loaded in `uᵉˣᵗ`/`vᵉˣᵗ`, with `immersed_peripheral_node` skipping
# the solid cells so the sum is exactly the wet-column transport (rather than trusting the dataset to zero its
# land points), and `ηᵉˣᵗ` is the GLORYS `zos` read at the boundary:

@inline function west_U_obc(i, j, grid, clock, fields, p)
    t = isnothing(clock) ? 0 : Time(clock.time)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = !immersed_peripheral_node(1, j, k, grid, Face(), Center(), Center())
        U += ifelse(wet, p.u[1, j, k, t] * Δzᶠᶜᶜ(1, j, k, grid), zero(U))
    end
    return (U, @inbounds p.η[1, j, 1, t])
end

@inline function east_U_obc(i, j, grid, clock, fields, p)
    t = isnothing(clock) ? 0 : Time(clock.time)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = !immersed_peripheral_node(grid.Nx + 1, j, k, grid, Face(), Center(), Center())
        U += ifelse(wet, p.u[grid.Nx + 1, j, k, t] * Δzᶠᶜᶜ(grid.Nx + 1, j, k, grid), zero(U))
    end
    return (U, @inbounds p.η[grid.Nx, j, 1, t])
end

@inline function north_V_obc(i, j, grid, clock, fields, p)
    t = isnothing(clock) ? 0 : Time(clock.time)
    V = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = !immersed_peripheral_node(i, grid.Ny + 1, k, grid, Center(), Face(), Center())
        V += ifelse(wet, p.v[i, grid.Ny + 1, k, t] * Δzᶜᶠᶜ(i, grid.Ny + 1, k, grid), zero(V))
    end
    return (V, @inbounds p.η[i, grid.Ny, 1, t])
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
# toward the same GLORYS12 data with `DatasetRestoring` — a 5-day timescale at the edge, fading to nothing
# within a couple of degrees:

# (The south rim is mostly Norwegian coast and keeps its wall, but its open southwest corner — Norwegian Sea —
# relies entirely on the sponge, so it stays in the mask.)

@inline rim(ξ, edge, width) = exp(-(ξ - edge)^2 / 2width^2)

@inline sponge_mask(λ, φ, z, t) = max(rim(λ, 5, 2), rim(λ, 60, 2), rim(φ, 67, 0.5), rim(φ, 80, 0.5))

FT = DatasetRestoring(Metadata(:temperature; dates, dataset, region), grid; rate = 1/5days, mask = sponge_mask, inpainting=100)
Fu = DatasetRestoring(Metadata(:u_velocity;  dates, dataset, region), grid; rate = 1/5days, mask = sponge_mask, inpainting=100)
Fv = DatasetRestoring(Metadata(:v_velocity;  dates, dataset, region), grid; rate = 1/5days, mask = sponge_mask, inpainting=100)
FS = DatasetRestoring(Metadata(:salinity;    dates, dataset, region), grid; rate = 1/5days, mask = sponge_mask, inpainting=100)

# ## The ocean component
#
# The familiar hydrostatic model, assembled by `ocean_simulation` with realistic defaults — TEOS-10 equation
# of state, CATKE vertical mixing, WENO advection, split-explicit free surface — plus our open boundaries and
# sponge forcings. The lateral boundary conditions merge side-by-side with the defaults, so the surface fluxes
# (which the coupler owns), the bottom drag, and the immersed drag stay wired. At eddy-permitting resolution
# we leave the Gent–McWilliams parameterization *out*: the front of part 2 taught us what the resolved eddies
# can do by themselves:

@inline _area_scaled_biharmonic_viscosity(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) =
    Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

function area_scaled_biharmonic_viscosity(FT=Oceananigans.defaults.FloatType; timescale=15days)
    return HorizontalScalarBiharmonicDiffusivity(FT;
        ν = _area_scaled_biharmonic_viscosity,
        discrete_form = true,
        parameters = timescale)
end

ocean = ocean_simulation(grid;
                         free_surface = SplitExplicitFreeSurface(grid; substeps=100),
                         momentum_advection = WENO(order=5, minimum_buffer_upwind_order=1),
                         tracer_advection = WENO(order=5, minimum_buffer_upwind_order=1),
                         closure = (NumericalEarth.Oceans.default_ocean_closure(), area_scaled_biharmonic_viscosity()),
                         forcing = (T = FT, S = FS, u = Fu, v = Fv),
                         boundary_conditions = (u = u_obcs, v = v_obcs,
                                                T = T_obcs, S = S_obcs,
                                                U = U_obcs, V = V_obcs))

# ## The sea-ice component
#
# The two halves of sea ice we met in parts 3 and 4 — slab thermodynamics and EVP dynamics — assembled by
# `sea_ice_simulation` and wired to the ocean below: the ice–ocean heat flux uses the model's evolving
# sea-surface salinity for the freezing point, and the ice feels the surface currents as a bottom stress:

sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)

# ## Initial conditions: the Barents Sea in winter
#
# Mid-winter 1993 from GLORYS12 — the same reanalysis that feeds the boundaries, so the initial state and the
# boundary data agree from the first time step. Temperature and salinity go to the ocean, thickness and
# concentration to the ice; one `MetadataSet` feeds both models, each picking up the variables it owns:

set!(ocean.model, T = Tᵉˣᵗ[1], S = Sᵉˣᵗ[1])
set!(sea_ice.model, h = Metadatum(:sea_ice_thickness,     date=dates[1], dataset=ECCO4Monthly()),
                    ℵ = Metadatum(:sea_ice_concentration, date=dates[1], dataset=ECCO4Monthly()))

# ## The atmosphere and the coupled model
#
# JRA55 reanalysis supplies winds, temperature, humidity, precipitation, radiation and runoff; the turbulent
# fluxes are computed interactively from the evolving SST and ice surface temperature, exactly as in a coupled
# climate model run under the OMIP protocol. `EarthSystemModel` owns the components and every interface
# between them — each exchanged flux is a `Field` you can inspect and output; there is no hidden coupler:

atmosphere    = JRA55PrescribedAtmosphere(arch)
radiation     = JRA55PrescribedRadiation(arch)
coupled_model = EarthSystemModel(; ocean, sea_ice, atmosphere, radiation)

# Two months, from mid-winter into the spring freeze-up maximum:

simulation = Simulation(coupled_model; Δt = 1minutes, stop_time = 60days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    sea_ice = sim.model.sea_ice
    T = ocean.model.tracers.T
    h = sea_ice.model.ice_thickness
    msg = @sprintf("time: %s, iter: %d, extrema(T): (%.1f, %.1f) °C, max(h): %.2f m, wall: %s",
                   prettytime(sim), iteration(sim),
                   minimum(T), maximum(T), maximum(h),
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
𝒱 = @at((Center, Center, Center), sqrt(u^2 + v^2))
ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities, (; 𝒱))

sea_ice_outputs = (h = sea_ice.model.ice_thickness,
                   ℵ = sea_ice.model.ice_concentration,
                   u = sea_ice.model.velocities.u,
                   v = sea_ice.model.velocities.v)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            filename = "barents_ocean_surface.jld2",
                                            indices = (:, :, grid.Nz),
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
# Sea-surface temperature with the ice cover drawn on top — the two protagonists of the Barents Sea climate
# story in one frame:

To = FieldTimeSeries("barents_ocean_surface.jld2",   "T")
Uo = FieldTimeSeries("barents_ocean_surface.jld2",   "𝒱")
hi = FieldTimeSeries("barents_sea_ice_surface.jld2", "h")
ℵi = FieldTimeSeries("barents_sea_ice_surface.jld2", "ℵ")

times = To.times

land_mask = interior(To.grid.immersed_boundary.bottom_height, :, :, 1) .≥ 0

n = Observable(length(times))

title = @lift "Barents Sea — day " * string(round(Int, times[$n] / days))

Tₙ = @lift begin
    T = interior(To, :, :, 1, $n)
    T[land_mask] .= NaN
    T
end

Uₙ = @lift begin
    U = interior(Uo, :, :, 1, $n)
    U[land_mask] .= NaN
    U
end


iceₙ = @lift begin
    hℵ = interior(hi, :, :, 1, $n) .* interior(ℵi, :, :, 1, $n)
    hℵ[land_mask] .= NaN
    hℵ[hℵ .< 0.05] .= NaN
    hℵ
end

λ = range(λ₁, λ₂, Nx)
φ = range(φ₁, φ₂, Ny)

fig = Figure(size = (1200, 450))
fig[0, 1:4] = Label(fig, title, fontsize = 20, tellwidth = false)

ax = Axis(fig[1, 1], xlabel = "longitude [°E]", ylabel = "latitude [°N]")
hm_T = heatmap!(ax, λ, φ, Tₙ, colormap = :thermal, colorrange = (-2, 8), nan_color = :gray80)
ax = Axis(fig[1, 3])
hm_h = heatmap!(ax, λ, φ, iceₙ, colormap = Reverse(:blues), colorrange = (0, 3))
ax = Axis(fig[2, 1])
hm_U = heatmap!(ax, λ, φ, Uₙ, colormap = Reverse(:solar), colorrange = (0, 0.5))
Colorbar(fig[1, 2], hm_T, label = "SST [°C]")
Colorbar(fig[1, 4], hm_h, label = "ice volume per area [m]")
Colorbar(fig[2, 2], hm_h, label = "Surface speed [ms⁻¹]")

CairoMakie.record(fig, "barents_sea.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](barents_sea.mp4)
#
# Things to look for, with the day's tutorials in mind: the warm Atlantic tongue entering between Bear Island
# and the Norwegian coast and holding the southwestern Barents ice-free (the reason Murmansk is a year-round
# port); the ice edge sitting along the polar front where that inflow meets Arctic water; eddies — live,
# resolved relatives of part 2 — stirring the front; leads and ridges from part 4's rheology opening and
# closing in the pack as storms pass through the JRA55 winds; and part 3's thermodynamics quietly thickening
# the ice in the cold northeastern corner.
#
# ## Things to try
#
# !!! tip "Watching the fluxes"
#     The coupler's flux fields are all addressable —
#     `coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat`, for instance. Adding the
#     ice–ocean and atmosphere–ocean heat fluxes to the output writers turns the simulation into a flux
#     laboratory: where does the ocean lose most heat, over the open Barents or through the leads?
#
# !!! tip "Atlantification, accelerated"
#     Initializing and forcing the boundaries from a different year — replace 1993 throughout with 2015, say
#     (GLORYS12 covers 1993–2021) — changes the Atlantic inflow temperature at the western boundary. How far
#     east does the ice edge sit after sixty days, compared with 1993?
#
# !!! tip "Anatomy of an open boundary"
#     Three ablations, one lesson each: walls + sponge only (drop the `boundary_conditions`) — watch boundary
#     reflections contaminate the interior; open boundaries without the sponge (drop the `forcing`) — watch
#     the slow drift the radiation conditions cannot hold back; and zeroing the Flather external state
#     (`FlatherBoundaryCondition((0, 0))` in place of the GLORYS transport and `zos`) — watch the Atlantic
#     inflow's barotropic transport get damped right at the boundary, the very error this case now avoids.
#
# !!! tip "The eddy dividend"
#     The 1/8° grid is eddy-*permitting*. At 1/16° (one number at the top) the Barents becomes properly
#     eddy-resolving — and a candidate for the distributed techniques of part 7. Does the polar front sharpen?
#     Does the ice edge get filamented?
#
# !!! tip "From the Barents to the globe"
#     The same components in a global configuration — `TripolarGrid` instead of the regional box, no sponge
#     needed — is the `one_degree_simulation` example in the NumericalEarth repository, and the gateway to
#     Thursday's coupled simulations with an interactive atmosphere. The script is shorter than this one.
