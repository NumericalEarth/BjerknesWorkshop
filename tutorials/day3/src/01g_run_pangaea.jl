# # Generalisability: the learned roughness on Pangaea
#
# The strongest argument for learning the surface roughness as a *column-based*
# function of local surface variables — rather than hard-coding another boundary
# condition map — is that it generalises: the network doesn't know anything about
# Earth's geography, only about the relation between a column's surface state and
# its roughness. It will also work with significant land use change in climate simulations, 
# or for paleo simulations. Even paleo simulations with dinosaurs! So nothing stops us from 
# applying it to a planet Earth with Pangaea still existing!
#
# Here we do exactly that: we repeat the simulation of
# [`run_parameterization.jl`](run_parameterization.jl), but with **Pangaea**, using the
# boundary conditions provided by
# [PlanetPangaea.jl](https://github.com/SpeedyWeather/PlanetPangaea.jl) (credit to Greg
# Munday / U Oxford).

using SpeedyWeather, Lux, JLD2

include(joinpath(@__DIR__,"01e_parameterization.jl"))

# ## Get the Pangaea boundary conditions
#
# We download the six NetCDF files from the PlanetPangaea.jl repository (only once
# — files already on disk are kept). `Downloads` is a Julia standard library.

using Downloads

pangaea_dir = "pangaea"
pangaea_url = "https://raw.githubusercontent.com/SpeedyWeather/PlanetPangaea.jl/main/data/boundary_conditions"
pangaea_files = ("lsm.nc", "orography.nc", "vegetation.nc", "soil_moisture.nc", "sst.nc", "albedo.nc")

mkpath(pangaea_dir)
for file in pangaea_files
    local_path = joinpath(pangaea_dir, file)
    isfile(local_path) || Downloads.download(join((pangaea_url, file), "/"), local_path)
end
readdir(pangaea_dir)

# All fields come gap-free: the SST continues smoothly under land and the soil
# moisture under the ocean, so SpeedyWeather can interpolate them onto any model
# grid without `NaN` ever reaching a coastal cell with fractional land — the
# filled values only ever matter at coasts, since fluxes are weighted by the land
# fraction. (Earlier versions of the files masked these regions with `NaN`, which
# blew up the simulation within a time step and required filling them manually —
# fixed upstream.) The vegetation file also carries the leaf area index
# (`lai_hv`, `lai_lv`), which we don't use here.

# ## Import the trained model
#
# Exactly as in [`run_parameterization.jl`](run_parameterization.jl): `training.jl`
# saved the Lux network, its trained parameters, the test-mode states and the
# normalisation statistics of the dataloaders. Note that the network was trained
# purely on ERA5 — present-day Earth — and will now be asked about a Triassic
# supercontinent.

trained = JLD2.load("trained_model.jld2")
land_nn     = trained["model"]
land_params = trained["parameters"]
land_states = trained["states"]       ## test-mode states: dropout disabled
stats       = trained["stats"]

# Map the normalisation constants from the ERA5 short names to the descriptive
# names the scheme uses, as before:

era5_to_scheme = Dict(
    "bare_soil" => :bare_soil,
    "cvh"       => :vegetation_high,
    "cvl"       => :vegetation_low,
    "z"         => :geopotential,
    "sd"        => :snow_depth,
    "stl1"      => :soil_temperature,
    "swvl1"     => :soil_moisture,
)

scheme_names = Tuple(era5_to_scheme[name] for name in stats.input_names)
land_input_means = NamedTuple{scheme_names}(Tuple(stats.input_mean))
land_input_stds  = NamedTuple{scheme_names}(Tuple(stats.input_std))

land_output_mean = stats.target_mean[1]
land_output_std  = stats.target_std[1]

# ## Swap Earth for Pangaea
#
# Every boundary-condition component in SpeedyWeather follows the same pattern: it
# reads a NetCDF variable from `path`, by default from the SpeedyWeatherAssets
# repository (`from_assets = true`). Pointing `path` at a local file and setting
# `from_assets = false` replaces Earth with whatever planet the file describes.
#
# The PlanetPangaea files use the standard variable names (`lsm`, `orog`,
# `vegh`/`vegl`, `swl1`/`swl2`, `sst`, `alb`) on a regular 384×192 longitude–latitude
# grid.

arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc = 127, architecture = arch)

orography = EarthOrography(spectral_grid,
    path = joinpath(pangaea_dir, "orography.nc"), from_assets = false)

land_sea_mask = EarthLandSeaMask(spectral_grid,
    path = joinpath(pangaea_dir, "lsm.nc"), from_assets = false,
    FieldType = FullGaussianField)

ocean = SeasonalOceanClimatology(spectral_grid,
    path = joinpath(pangaea_dir, "sst.nc"), from_assets = false)

albedo = AlbedoClimatology(spectral_grid,
    path = joinpath(pangaea_dir, "albedo.nc"), from_assets = false)

vegetation = VegetationClimatology(spectral_grid,
    path = joinpath(pangaea_dir, "vegetation.nc"), from_assets = false)

soil_moisture = SeasonalSoilMoisture(spectral_grid,
    path = joinpath(pangaea_dir, "soil_moisture.nc"), from_assets = false)

land = LandModel(spectral_grid; vegetation, soil_moisture)

# ## Run Pangaea with the learned surface roughness
#
# The learned scheme is constructed exactly as on Earth — it is the same network,
# the same weights, the same normalisation. Only the continents underneath changes.

surface_roughness = LearnedSurfaceRoughness(
    spectral_grid, land_nn, land_params, land_states,
    land_input_means, land_input_stds;
    land_output_mean, land_output_std)

boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)

model = PrimitiveWetModel(spectral_grid;
    orography, land_sea_mask, ocean, albedo, land, boundary_layer)

simulation = initialize!(model)
run!(simulation, period = Day(20))

# ## Roughness of a supercontinent
#
# For the visual check we plot the learned surface roughness next to the high
# vegetation cover — the network's dominant predictor — interpolated to a regular
# grid. The roughness should trace the (fictional) vegetation belts and highlands
# of Pangaea, even though the network never saw a continent like this.

using CairoMakie

## helper: SpeedyWeather field -> (matrix, lon, lat) on its full regular grid
function to_lonlat(field)
    full = RingGrids.interpolate(RingGrids.full_grid_type(field.grid), field.grid.nlat_half, field)
    return Matrix(full), RingGrids.get_lond(full), RingGrids.get_latd(full)
end

z₀_sim, lond, latd = to_lonlat(simulation.variables.parameterizations.land.surface_roughness)
vegh_sim, _, _     = to_lonlat(model.land.vegetation.high_cover)

fig = Figure(size = (800, 650))
ax1 = Axis(fig[1, 1]; title = "Learned surface roughness on Pangaea (T$(spectral_grid.trunc))")
hm1 = heatmap!(ax1, lond, latd, log10.(max.(z₀_sim, 1f-6)); colorrange = (-4, 0.5))
Colorbar(fig[1, 2], hm1, label = "log₁₀ z₀ [m]")
ax2 = Axis(fig[2, 1]; title = "High vegetation cover (PlanetPangaea)", xlabel = "longitude")
hm2 = heatmap!(ax2, lond, latd, vegh_sim; colorrange = (0, 1), colormap = :Greens)
Colorbar(fig[2, 2], hm2, label = "cover fraction")
CairoMakie.save("surface_roughness_pangaea.png", fig)   ## qualified: JLD2 also exports `save`
fig

# Spatial generalisation in action: a parameterization learned from ERA5 maps,
# applied to a planet that hasn't existed for 250 million years — because the
# network learned the *function*, not the map. T-Rex would be proud! 
