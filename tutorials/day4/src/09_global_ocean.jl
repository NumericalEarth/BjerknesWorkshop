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
using CairoMakie

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
Nz = 15

depth = 3000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (7, 7, 7), z)

# (`mutable = true` makes the vertical coordinate a *z★* coordinate that breathes with the free surface —
# relevant for tides and shelf seas, free to keep on.)
#
# It is worth looking at what "warped sphere" buys us, and a north-polar *stereographic* projection makes the
# point clear. On a plain latitude–longitude grid every meridian runs to the geographic North Pole: in the
# projection the cells become concentric rings that shrink onto a single point in the middle of the Arctic
# Ocean, where the zonal cell width — and the affordable time step — go to zero. The tripolar construction
# splits that singularity into two poles and places both over land (Siberia and Canada), so the meridians
# converge on the continents and every wet Arctic cell keeps a finite size:

using GeoMakie

projection = "+proj=stere +lat_0=90 +lat_ts=90"

fig = Figure(size = (1160, 620))

# A regular latitude–longitude grid: concentric rings collapsing onto one pole.

ax = GeoAxis(fig[1, 1]; dest = projection, limits = ((-180, 180), (15, 90)),
             title = "latitude–longitude grid: one pole, a singularity")
hidedecorations!(ax)
for φ₀ in 20:10:80
    lines!(ax, range(-180, 180, length = 361), fill(φ₀, 361); color = (:black, 0.65), linewidth = 0.7)
end
for λ₀ in -180:15:165
    lines!(ax, fill(λ₀, 181), range(0, 90, length = 181); color = (:black, 0.65), linewidth = 0.7)
end
lines!(ax, GeoMakie.coastlines(); color = (:dodgerblue, 0.9), linewidth = 1)

# The tripolar grid: the mesh steps around the Arctic Ocean and converges on two poles, both on land.

viz_grid = TripolarGrid(size = (90, 45, 1), halo = (5, 5, 5), z = (0, 1))
λ = Array(λnodes(viz_grid, Face(), Face(), Center()))
φ = Array(φnodes(viz_grid, Face(), Face(), Center()))
λ = vcat(λ, λ[1:1, :])   # close the periodic seam in longitude
φ = vcat(φ, φ[1:1, :])

ax = GeoAxis(fig[1, 2]; dest = projection, limits = ((-180, 180), (15, 90)),
             title = "tripolar grid: two poles, both over land")
hidedecorations!(ax)
for j in 1:2:size(λ, 2)
    lines!(ax, λ[:, j], φ[:, j]; color = (:black, 0.65), linewidth = 0.6)
end
for i in 1:3:size(λ, 1)
    lines!(ax, λ[i, :], φ[i, :]; color = (:black, 0.65), linewidth = 0.6)
end
lines!(ax, GeoMakie.coastlines(); color = (:dodgerblue, 0.9), linewidth = 1)
save("tripolar_grid.png", fig)
fig

# ![](tripolar_grid.png)
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
momentum_advection = WENOVectorInvariant(time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
tracer_advection   = WENO(order=7, time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5))

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

# ## Interface fluxes and Monin–Obukhov similarity theory
#
# `OceanSeaIceModel` computes the interface fluxes already at construction, so before time-stepping we can stop
# and look at what the coupler does. Every exchanged flux is a plain `Field` stored under
# `coupled_model.interfaces`; there is no hidden coupler state to dig into.
#
# The turbulent exchange of momentum, heat, and water vapor between the atmosphere and the surface below is
# written with bulk formulae,
#
# ```math
# \overline{\mathbf{u}' w'} = C_D \, U \, \Delta \mathbf{u}, \qquad
# \overline{w' \theta'} = C_\theta \, U \, \Delta \theta, \qquad
# \overline{w' q'} = C_q \, U \, \Delta q ,
# ```
#
# where ``\Delta`` denotes the air–surface difference and ``U`` a characteristic velocity scale. The transfer
# coefficients ``C_D``, ``C_\theta``, ``C_q`` are not constants — Monin–Obukhov similarity theory derives them
# from the structure of the surface layer. Introducing the characteristic scales
#
# ```math
# u_\star^2 \equiv |\overline{\mathbf{u}' w'}|, \qquad
# u_\star \theta_\star \equiv \overline{w' \theta'}, \qquad
# u_\star q_\star \equiv \overline{w' q'} ,
# ```
#
# similarity theory supposes that the near-surface shear depends only on height, ``\partial_z u = u_\star /
# (\kappa z)``, with ``\kappa`` the von Kármán constant. Integrating from a roughness length ``\ell_u`` up to
# the measurement height ``h``, and correcting for buoyancy fluxes through the stability function ``\psi_u``
# evaluated at ``\zeta = z / L_\star`` (where ``L_\star = -u_\star^2 / (\kappa b_\star)`` is the Monin–Obukhov
# length), the velocity difference becomes
#
# ```math
# \Delta u = \frac{u_\star}{\kappa}
#            \left[ \log \frac{h}{\ell_u} - \psi_u\!\left(\frac{h}{L_\star}\right) \right] .
# ```
#
# Over the ocean ``\ell_u`` itself depends on ``u_\star``, so this is a nonlinear equation for ``u_\star``,
# solved by fixed-point iteration. The friction velocity, the temperature scale, and the effective drag
# coefficient ``C_D = u_\star^2 / |\Delta u|^2`` are exactly the quantities that the interface stores.
#
# These quantities — the friction velocity, and the heat and momentum fluxes — are computed for the real
# global state already at construction, one value per surface cell. To look at them, we first move them off
# the tripolar grid.
#
# ## From the model grid to a map: conservative regridding
#
# These fluxes are computed on the tripolar grid, whose cells are curvilinear quadrilaterals of uneven
# size. To draw them on a map — or to compare with a product on a different mesh — we move them onto a regular
# longitude–latitude grid. Pointwise interpolation would not conserve the area integral of a flux, which is the
# property we usually care about (the total heat or freshwater exchanged). *Conservative* regridding instead
# computes, for each target cell, the area-weighted average of the source cells it overlaps; the global
# integral is preserved up to the mismatch at the domain edges.
#
# `ConservativeRegridding` builds a `Regridder` from the geometric overlap of two grids (`dst` first, `src`
# second). The weights depend only on the meshes, so a single regridder is reused for every field that lives on
# the same grid and location — and all our scalar fluxes are `(Center, Center)` fields on the same tripolar
# grid:

using ConservativeRegridding
using GeoMakie
using Oceananigans.Architectures: on_architecture

ao = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
io = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

wind_stress = Field(sqrt(ao.x_momentum^2 + ao.y_momentum^2))
compute!(wind_stress)

# The flux Fields live wherever the model runs; the regridding and plotting happen on the CPU:

sensible = on_architecture(CPU(), ao.sensible_heat)
latent   = on_architecture(CPU(), ao.latent_heat)
friction = on_architecture(CPU(), ao.friction_velocity)
stress   = on_architecture(CPU(), wind_stress)
ice_heat = on_architecture(CPU(), io.interface_heat)
ice_salt = on_architecture(CPU(), io.salt)

# A quarter-degree target, matching the simulation resolution:

latlon_grid = LatitudeLongitudeGrid(size = (1440, 720, 1),
                                    longitude = (-180, 180),
                                    latitude  = (-80, 90),
                                    z = (0, 1))

regridder = ConservativeRegridding.Regridder(Field{Center, Center, Nothing}(latlon_grid), sensible)

function regrid(field)
    out = Field{Center, Center, Nothing}(latlon_grid)
    ConservativeRegridding.regrid!(out, regridder, field)
    return out
end

# An ocean fraction, regridded the same way, masks the continents:

source_grid  = sensible.grid
bottom       = source_grid.immersed_boundary.bottom_height
ocean_source = Field{Center, Center, Nothing}(source_grid)
interior(ocean_source) .= ifelse.(interior(bottom) .< 0, 1, 0)
ocean_fraction = interior(regrid(ocean_source), :, :, 1)
mask(data) = ifelse.(ocean_fraction .> 0.5, data, NaN)

λ = λnodes(latlon_grid, Center(), Center(), Center())
φ = φnodes(latlon_grid, Center(), Center(), Center())

panels = [(sensible, "sensible heat (W m⁻²)",        :balance, (-200, 200)),
          (latent,   "latent heat (W m⁻²)",          :balance, (-300, 300)),
          (stress,   "wind stress (N m⁻²)",          :solar,   (0, 0.4)),
          (friction, "friction velocity u★ (m s⁻¹)", :solar,   (0, 0.5)),
          (ice_heat, "ice-ocean heat (W m⁻²)",       :balance, (-50, 50)),
          (ice_salt, "ice-ocean salt (psu m s⁻¹)",   :haline,  (0, 2e-6))]

fig = Figure(size = (1200, 1200))

for (k, (field, name, colormap, colorrange)) in enumerate(panels)
    i, j = fldmod1(k, 2)
    gl = GridLayout(fig[i, j])
    ax = GeoAxis(gl[1, 1]; dest = "+proj=robin", title = name)
    data = mask(interior(regrid(field), :, :, 1))
    sf = surface!(ax, λ, φ, data; colormap, colorrange, nan_color = :gray20, shading = NoShading)
    lines!(ax, GeoMakie.coastlines(); color = :black, linewidth = 0.4)
    Colorbar(gl[1, 2], sf)
end

save("interface_fluxes_map.png", fig)
fig

# ![](interface_fluxes_map.png)
#
# The maps read as the coupler's diagnosis of the January state estimate: strong latent and sensible heat loss
# over the warm western-boundary currents and along the sea-ice edge, the wind-stress imprint of the storm
# tracks and the trades, and a sharply localized ice–ocean heat flux confined to the marginal ice zones.
#
# ## Running forward

simulation = Simulation(coupled_model; Δt = 20minutes, stop_time = 365days)

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

𝒱 = @at((Center, Center, Center), sqrt(u^2 + v^2))
ocean_outputs = merge(ocean.model.tracers, (; 𝒱))

sea_ice_outputs = (h = sea_ice.model.ice_thickness,
                   ℵ = sea_ice.model.ice_concentration)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            filename = "global_ocean_surface.jld2",
                                            indices = (:, :, grid.Nz-1),
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
# Surface speed, surface temperature, and the sea-ice cover — rendered on the same quarter-degree
# longitude–latitude grid, regridding every frame conservatively from the tripolar output and drawing it on a
# GeoMakie map, with land masked:

using GeoMakie
using ConservativeRegridding
using Oceananigans.Architectures: on_architecture

Uo = FieldTimeSeries("global_ocean_surface.jld2", "𝒱")
To = FieldTimeSeries("global_ocean_surface.jld2", "T")
hi = FieldTimeSeries("global_sea_ice_surface.jld2", "h")
ℵi = FieldTimeSeries("global_sea_ice_surface.jld2", "ℵ")

times = To.times

movie_grid = LatitudeLongitudeGrid(size = (1440, 720, 1),
                                   longitude = (-180, 180),
                                   latitude  = (-80, 90),
                                   z = (0, 1))

# One regridder, shared by every (Center, Center) frame:

regridder = ConservativeRegridding.Regridder(Field{Center, Center, Nothing}(movie_grid),
                                             on_architecture(CPU(), Uo[1]))

function to_map(field)
    out = Field{Center, Center, Nothing}(movie_grid)
    ConservativeRegridding.regrid!(out, regridder, on_architecture(CPU(), field))
    return interior(out, :, :, 1)
end

# An ocean fraction on the target grid masks the continents:

ocean_source = Field{Center, Center, Nothing}(Uo.grid)
interior(ocean_source) .= ifelse.(interior(Uo.grid.immersed_boundary.bottom_height) .< 0, 1, 0)
land = .!(to_map(ocean_source) .> 0.5)

λ = λnodes(movie_grid, Center(), Center(), Center())
φ = φnodes(movie_grid, Center(), Center(), Center())

n = Observable(length(times))
title = @lift "global ocean and sea ice — day " * string(round(Int, times[$n] / days))

mask_land!(data) = (data[land] .= NaN; data)

speedₙ = @lift mask_land!(to_map(Uo[$n]))
Tₙ     = @lift mask_land!(to_map(To[$n]))
iceₙ   = @lift begin
    hℵ = to_map(hi[$n]) .* to_map(ℵi[$n])
    hℵ[land] .= NaN
    hℵ[hℵ .< 0.05] .= NaN   # show only where there is actual ice
    hℵ
end

projection = "+proj=robin"

fig = Figure(size = (1000, 1200))
fig[1, :] = Label(fig, title, fontsize = 22, tellwidth = false)

ax_s = GeoAxis(fig[2, 1]; dest = projection, title = "surface speed [m s⁻¹]")
hm_s = surface!(ax_s, λ, φ, speedₙ; colormap = :solar, colorrange = (0, 0.6), nan_color = :lightgray, shading = NoShading)
lines!(ax_s, GeoMakie.coastlines(); color = :black, linewidth = 0.4)
Colorbar(fig[2, 2], hm_s)

ax_T = GeoAxis(fig[3, 1]; dest = projection, title = "surface temperature [°C]")
hm_T = surface!(ax_T, λ, φ, Tₙ; colormap = :magma, colorrange = (-2, 30), nan_color = :lightgray, shading = NoShading)
lines!(ax_T, GeoMakie.coastlines(); color = :black, linewidth = 0.4)
Colorbar(fig[3, 2], hm_T)

ax_h = GeoAxis(fig[4, 1]; dest = projection, title = "sea ice volume per area [m]")
hm_h = surface!(ax_h, λ, φ, iceₙ; colormap = Reverse(:blues), colorrange = (0, 2), nan_color = :lightgray, shading = NoShading)
lines!(ax_h, GeoMakie.coastlines(); color = :black, linewidth = 0.4)
Colorbar(fig[4, 2], hm_h)

CairoMakie.record(fig, "global_ocean.mp4", 1:length(times), framerate = 12) do i
    @info "doing step $i"
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
# - **Resolution.** The same script at `Nx, Ny, Nz = 5760, 2880, 100` is the sixteenth-degree, eddy-resolving
#   configuration. That configuration is a good candidate to test distributed computing (`arch = Distributed(GPU)``)
# - **Prognostic atmosphere.** A prognostic atmosphere can replace `JRA55PrescribedAtmosphere`, and with 
#   `coupled_model = EarthSystemModel(; ocean, sea_ice, atmosphere, land, radiation)`, this simulation becomed
#    a small coupled ESM. (you can try using SpeedyWeather)
#
