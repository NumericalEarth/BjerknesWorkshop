# # Running the learned parameterization in SpeedyWeather.jl
#
# After training the surface roughness MLP offline and defining
# the `LearnedSurfaceRoughness` parameterization, we can
# now plug the trained network into an actual SpeedyWeather simulation.

using SpeedyWeather, Lux, JLD2
## using CUDA, cuDNN   # only needed for arch = SpeedyWeather.GPU()

include(joinpath(@__DIR__, "..", "src/01e_parameterization.jl"))

# ## Import the trained model
#
# `training.jl` saved everything we need with `save_model`: the Lux network, its
# trained parameters, the matching states in test mode (so dropout is switched
# off at inference), and the normalisation statistics of the dataloaders.

trained = JLD2.load("trained_model.jld2")
land_nn     = trained["model"]
land_params = trained["parameters"]
land_states = trained["states"]       ## test-mode states: dropout disabled
stats       = trained["stats"]

# ## Map the normalisation constants to the scheme's inputs
#
# The dataloader `stats` hold the input means and standard deviations as vectors
# ordered like `stats.input_names` (the ERA5 short names), while
# `LearnedSurfaceRoughness` expects NamedTuples keyed by the descriptive variable
# names used in `surface_roughness_land`. So we translate the names and rebuild
# the constants as NamedTuples:

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

# The network was trained to predict the *normalised log* surface roughness, so
# the scheme also needs the target mean/std to undo that transformation:

land_output_mean = stats.target_mean[1]
land_output_std  = stats.target_std[1]

# ## Run it in a CPU simulation
#
# Now we construct the scheme and hand it to the model via the `BoundaryLayer`.
# Everything else stays at its defaults — the default `LandModel` already carries
# the vegetation, snow, soil temperature and soil moisture variables our network
# uses as inputs.

arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(trunc = 32, architecture = arch)

surface_roughness = LearnedSurfaceRoughness(
    spectral_grid, land_nn, land_params, land_states,
    land_input_means, land_input_stds;
    land_output_mean, land_output_std)

boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)
model = PrimitiveWetModel(spectral_grid; boundary_layer)
simulation = initialize!(model)
run!(simulation, period=Day(10))

# As a quick sanity check, the range of the surface roughness our network
# predicted over land (in meters, after undoing the log and the normalisation;
# the zero minimum is simply the ocean points of the land field):

extrema(simulation.variables.parameterizations.land.surface_roughness)

# ## Comparing to ERA5
#
# For a simple visual comparison we plot the current surface roughness of the
# simulation next to the ERA5 surface roughness. Roughness
# spans several orders of magnitude, so we plot `log10` of both.

using NCDatasets, CairoMakie

era5_file = "/cluster/projects/nn9984k/speedy-data/era5-roughness-snapshot.nc"
#era5_file = "era5-roughness-snapshot.nc"
lon, lat, z₀_era5 = NCDataset(era5_file) do ds
    ds["longitude"][:], ds["latitude"][:], Float32.(ds["fsr"][:, :])
end

## shift ERA5 from -180–180°E to 0–360°E so both maps share the same layout
z₀_era5 = circshift(z₀_era5, (720, 0))
lon360  = mod.(circshift(lon, 720), 360)

## the simulation field as a (lon, lat) matrix on its full grid
z₀_field = simulation.variables.parameterizations.land.surface_roughness
z₀_full = RingGrids.interpolate(RingGrids.full_grid_type(z₀_field.grid), z₀_field.grid.nlat_half, z₀_field)
z₀_sim, lond, latd = Matrix(z₀_full), RingGrids.get_lond(z₀_full), RingGrids.get_latd(z₀_full)

clims = (-4, 0.5)   ## log₁₀ of roughness in meters
fig = Figure(size = (800, 650))
ax1 = Axis(fig[1, 1]; title = "Learned surface roughness (SpeedyWeather, T32)")
hm = heatmap!(ax1, lond, latd, log10.(max.(z₀_sim, 1f-6)); colorrange = clims)  ## ocean is 0 -> clipped
ax2 = Axis(fig[2, 1]; title = "ERA5 fsr (first snapshot)", xlabel = "longitude")
heatmap!(ax2, lon360, lat, log10.(z₀_era5); colorrange = clims)
Colorbar(fig[:, 2], hm, label = "log₁₀ surface roughness [m]")
CairoMakie.save("surface_roughness_comparison.png", fig)   ## qualified: JLD2 also exports `save`
fig

# This already looks good! You can also change the resolution of the simulation and retry it
# to see how it changes the roughness. To run it on GPU we need to slightly adjust the parameterization though, 
# we'll do that in the last (optional) notebook. Before that, we want to see how well it really generalizes what we learned! 