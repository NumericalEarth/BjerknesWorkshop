# # NumericalEarth: a realistic global ocean–sea ice simulation
#
# *A realistic global ocean–sea ice simulation.*
#
# The idealized examples use rectangular domains, synthetic initial conditions, forcing written as one-line
# Julia functions. [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl) closes the gap to
# the real Earth: it wraps Oceananigans and ClimaSeaIce — the very models behind the idealized examples — and
# adds the unglamorous 90% of realistic modeling: bathymetry regridding, state-estimate initial conditions,
# reanalysis surface forcing, bulk flux computations, and the coupling glue between components.
#
# Contrarily to the traditional GCM workflow — namelists, configuration files, a build system, and a queue of
# preprocessing executables — the configuration here *is the program*: the same hundred-line script style as
# the idealized examples, scaled up to a global, JRA55-forced, eddy-parameterized ocean–sea ice simulation. This
# is the configuration class that NumericalEarth runs for OMIP-style integrations, and what we set up below is,
# give or take resolution, a one-degree OMIP experiment.
#
# !!! warning "Hardware and data requirements"
#     This tutorial is meant for a GPU machine with internet access. The first run downloads the ECCO4 state
#     estimate, the JRA55-do reanalysis, and the ETOPO1 bathymetry (≈ 10 GB altogether, cached locally for
#     every later run). On the workshop cluster the data is pre-staged; on your laptop you can still read along
#     — and run a shortened CPU version by lowering the resolution at the top of the script.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using CUDA
using Printf
using Statistics

arch = GPU()   # CPU() works too — heroically — at reduced resolution

# ## The grid: a warped sphere
#
# One degree of horizontal resolution, 50 vertical levels concentrated near the surface by an
# `ExponentialDiscretization` (the top cell is a few meters thick, the abyssal ones a few hundred).
# Latitude–longitude grids degenerate at the poles — the zonal cell width vanishes and with it the affordable
# time step — so global ocean models warp the grid: the `TripolarGrid` places two artificial northern poles
# over land (Siberia and Canada), keeping every wet cell comfortably sized. The Arctic, dear to this audience,
# is a first-class citizen rather than a coordinate singularity:

Nx = 1440
Ny = 720
Nz = 40

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (7, 7, 7), z)

# (`mutable = true` makes the vertical coordinate a *z★* coordinate that breathes with the free surface —
# relevant for tides and shelf seas, free to keep on.)
#
# The real bathymetry comes from ETOPO1, regridded with a few smoothing passes; `major_basins = 2` keeps the
# two largest connected ocean basins and fills lakes and disconnected seas:

# Downloads land in `DATA_DIR` when that environment variable is set, else each product's default cache:
dir_kw = haskey(ENV, "DATA_DIR") ? (; dir = ENV["DATA_DIR"]) : (;)

bathymetry = Metadatum(:bottom_height; dataset = ETOPO2022(), dir_kw...)
bottom_height = regrid_bathymetry(underlying_grid, bathymetry;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map = true)

# The same `ImmersedBoundaryGrid` as the internal-tide sill — continents are just very large sills. With
# `active_cells_map = true` the GPU kernels iterate only over wet cells, which at one degree saves roughly a
# third of the work.
#
# ## Closures: the physics we cannot afford
#
# At one degree the mesoscale eddies of the baroclinic-instability tutorial are not resolved — so we put back,
# as a parameterization, exactly the eddy fluxes seen in the baroclinic-instability example: the
# Gent–McWilliams skew flux plus isopycnal (Redi) diffusion. For the surface boundary layer we use CATKE, a
# prognostic turbulent-kinetic-energy vertical mixing scheme:

vertical_mixing = CATKEVerticalDiffusivity(minimum_tke=1e-7)

# ## The ocean component
#
# `ocean_simulation` assembles a `HydrostaticFreeSurfaceModel` configured for realistic ocean modeling —
# temperature and salinity tracers, the TEOS-10 equation of state, split-explicit free surface, sensible
# defaults everywhere — and returns it wrapped in a `Simulation`. Every default can be overridden through
# keyword arguments, with the objects you already know:

free_surface       = SplitExplicitFreeSurface(grid; substeps = 70)
momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order = 7)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface, closure = vertical_mixing)

# ## The sea-ice component
#
# `sea_ice_simulation` builds the ClimaSeaIce model of the sea-ice examples — slab thermodynamics plus EVP
# dynamics — already wired to feel the ocean below:

sea_ice = sea_ice_simulation(grid, ocean; advection = tracer_advection)

# ## Initial conditions from a state estimate
#
# We initialize ocean and ice from the ECCO4 monthly climatology for January 1993. A `MetadataSet` describes
# *what* to fetch; `set!` regrids it onto our grid, downloading (and caching) the data on first use. The same
# line with `EN4Monthly()` or `GLORYSMonthly()` initializes from different products — try it when you wonder
# how much your spin-up remembers its parents:

date = DateTime(1993, 1, 1)
ecco_variables = (:temperature, :salinity, :sea_ice_thickness, :sea_ice_concentration)
ecco_set = MetadataSet(ecco_variables; dataset = ECCO4Monthly(), date, dir_kw...)

set!(ocean.model,   ecco_set)   # picks up temperature and salinity
set!(sea_ice.model, ecco_set)   # picks up thickness and concentration

# ## The atmosphere: prescribed reanalysis
#
# The JRA55-do reanalysis provides the atmospheric state (winds, temperature, humidity, precipitation), the
# downwelling radiation, and land runoff. *Prescribed* means the atmosphere does not feel the ocean — the OMIP
# protocol — but the turbulent fluxes are still computed interactively from similarity theory, using the
# evolving SST and ice surface temperature:

land       = JRA55PrescribedLand(arch; dir_kw...)
atmosphere = JRA55PrescribedAtmosphere(arch; dir_kw...)

ocean_surface = SurfaceRadiationProperties(albedo = LatitudeDependentAlbedo())
radiation     = JRA55PrescribedRadiation(arch; ocean_surface, dir_kw...)

# ## The coupled model
#
# `OceanSeaIceModel` owns the components and the *interfaces* between them: bulk turbulent fluxes at the
# atmosphere–ocean and atmosphere–ice interfaces, frazil and basal melt at the ice–ocean interface, solar
# penetration, the lot. Every exchanged flux is a `Field` you can inspect and output — there is no hidden
# coupler:

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)

simulation = Simulation(coupled_model; Δt = 30minutes, stop_time = 365days)

# A year at one degree takes a few hours on a single modern GPU. The progress callback keeps us honest while it
# runs:

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T

    msg = @sprintf("time: %s, iter: %d, max|u|: (%.1e, %.1e, %.1e) m s⁻¹, extrema(T): (%.1f, %.1f) °C, wall: %s",
                   prettytime(sim), iteration(sim),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   minimum(T), maximum(T),
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    @info msg
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(5days))

# ## Output
#
# Surface fields only, daily — the `indices` trick from the baroclinic channel, at planetary scale:

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)

sea_ice_outputs = (h = sea_ice.model.ice_thickness,
                   ℵ = sea_ice.model.ice_concentration)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            filename = "global_ocean_surface.jld2",
                                            indices = (:, :, grid.Nz),
                                            schedule = TimeInterval(1days),
                                            overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              filename = "global_sea_ice_surface.jld2",
                                              schedule = TimeInterval(1days),
                                              overwrite_existing = true)

# The big red button:

run!(simulation)

# ## A planetary movie
#
# Surface speed, surface temperature, and the sea-ice cover, with land masked to `NaN`:

using CairoMakie

uo = FieldTimeSeries("global_ocean_surface.jld2", "u"; backend = OnDisk())
vo = FieldTimeSeries("global_ocean_surface.jld2", "v"; backend = OnDisk())
To = FieldTimeSeries("global_ocean_surface.jld2", "T"; backend = OnDisk())
hi = FieldTimeSeries("global_sea_ice_surface.jld2", "h"; backend = OnDisk())
ℵi = FieldTimeSeries("global_sea_ice_surface.jld2", "ℵ"; backend = OnDisk())

times = To.times

land_mask = interior(To.grid.immersed_boundary.bottom_height, :, :, 1) .≥ 0

n = Observable(length(times))

title = @lift "global ocean and sea ice — day " * string(round(Int, times[$n] / day))

speedₙ = @lift begin
    s = @. sqrt(interior(uo[$n], :, :, 1)^2 + interior(vo[$n], :, :, 1)^2)
    s[land_mask] .= NaN
    s
end

Tₙ = @lift begin
    T = interior(To[$n], :, :, 1)
    T[land_mask] .= NaN
    T
end

iceₙ = @lift begin
    hℵ = interior(hi[$n], :, :, 1) .* interior(ℵi[$n], :, :, 1)
    hℵ[land_mask] .= NaN
    hℵ[hℵ .< 0.05] .= NaN  # show only where there is actual ice
    hℵ
end

fig = Figure(size = (1100, 1000))
fig[1, :] = Label(fig, title, fontsize = 22, tellwidth = false)

ax_s = Axis(fig[2, 1], title = "surface speed [m s⁻¹]")
hm_s = heatmap!(ax_s, speedₙ, colormap = :solar, colorrange = (0, 0.6), nan_color = :lightgray)
Colorbar(fig[2, 2], hm_s)

ax_T = Axis(fig[3, 1], title = "surface temperature [°C]")
hm_T = heatmap!(ax_T, Tₙ, colormap = :magma, colorrange = (-2, 30), nan_color = :lightgray)
Colorbar(fig[3, 2], hm_T)

ax_h = Axis(fig[4, 1], title = "sea ice volume per area [m]")
hm_h = heatmap!(ax_h, iceₙ, colormap = Reverse(:blues), colorrange = (0, 4), nan_color = :lightgray)
Colorbar(fig[4, 2], hm_h)

CairoMakie.record(fig, "global_ocean.mp4", 1:length(times), framerate = 12) do i
    n[] = i
end
nothing #hide

# ![](global_ocean.mp4)
#
# Things to look for, with Bjerknes eyes: the Gulf Stream and its (parameterized, therefore laminar-looking)
# North Atlantic Drift feeding the Norwegian Atlantic Current; the winter deepening of the marginal ice zone in
# the Barents Sea; the seasonal pulse of the Antarctic ice cover; and the equatorial current system responding
# to the JRA55 wind stress within the first weeks.
#
# ## Where to go from here
#
# - **Resolution.** The same script at `Nx, Ny = 1440, 600` is the quarter-degree, eddy-permitting
#   configuration — provided you also *remove* `eddy_closure` (the resolved eddies object to being
#   parameterized on top). That configuration is the subject of the distributed tutorial next door.
# - **Regional.** Cut a Nordic Seas domain with `LatitudeLongitudeGrid` (or a rotated
#   `RotatedLatitudeLongitudeGrid` centered on the pole) and the same components — see the Arctic experiment in
#   the NumericalEarth repository.
# - **Interactive atmosphere.** A prognostic atmosphere can replace `JRA55PrescribedAtmosphere`, and the same
#   `OceanSeaIceModel` becomes a small coupled ESM. It is, indeed, easier than you think.
#
