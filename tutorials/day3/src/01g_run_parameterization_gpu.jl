# # Running the learned parameterization on the GPU
#
# The column-wise `LearnedSurfaceRoughness` from `parameterization.jl` is not
# GPU-compatible: all columns share a single input buffer (a data race in a
# parallel kernel) and `Lux.apply` of a full `Chain` cannot be called from
# inside a GPU kernel. On the GPU, SpeedyWeather fuses all column
# parameterizations into one kernel over the grid points, so the network would
# have to run per-point inside that kernel.
#
# SpeedyWeather offers a second kind of parameterization for exactly this
# situation: a [global parameterization](https://speedyweather.github.io/SpeedyWeatherDocumentation/dev/parameterizations).
# It is called once per time step with the complete state — and is expected to
# organise its own computation (e.g. launch its own kernels). That is a perfect
# match for a neural network: we gather the inputs of *all* grid points into one
# `(7, npoints)` matrix, evaluate the network batched in a single
# `Lux.apply`, and write the result back with array broadcasts.

using CUDA, cuDNN              # load the GPU stack first
using SpeedyWeather, Lux, JLD2, Adapt

# ## The global scheme
#
# The struct mirrors the column-wise version, with two differences: the
# normalisation constants are stored as plain vectors on the device (ordered
# like the NN inputs, i.e. like the dataloader `stats` — no name mapping
# needed), and the input buffer is a `(7, npoints)` matrix for all grid points
# at once.

@kwdef struct LearnedSurfaceRoughnessGlobal{NF, V, M, LNN, LP, LS} <: SpeedyWeather.AbstractSurfaceRoughness
    "[OPTION] constant roughness length over ocean [m]"
    roughness_length_ocean::NF = 1.0e-4

    ## normalisation constants on the device, ordered like the NN input:
    ## [bare_soil, cvh, cvl, z, sd, stl1, swvl1]
    land_input_mean::V
    land_input_std::V
    land_output_mean::NF
    land_output_std::NF

    ## preallocated NN input matrix (7, npoints) on the device
    input_buffer::M

    ## NN structure, parameters and states
    land_nn::LNN
    land_params::LP
    land_states::LS
end

"""
    LearnedSurfaceRoughnessGlobal(SG, land_nn, land_params, land_states, stats; kwargs...)

Construct the global scheme directly from the dataloader `stats` saved by
`training.jl` — the stats vectors are already in the NN input order. `land_params`
and `land_states` are expected on the same device the simulation runs on.
"""
function LearnedSurfaceRoughnessGlobal(
        SG::SpectralGrid, land_nn, land_params, land_states, stats; kwargs...)

    expected = ["bare_soil", "cvh", "cvl", "z", "sd", "stl1", "swvl1"]
    stats.input_names == expected ||
        error("stats.input_names = $(stats.input_names), expected $(expected)")

    arch = SG.architecture
    input_buffer    = on_architecture(arch, zeros(Float32, 7, SG.npoints))
    land_input_mean = on_architecture(arch, Float32.(stats.input_mean))
    land_input_std  = on_architecture(arch, Float32.(stats.input_std))

    return LearnedSurfaceRoughnessGlobal{
        SG.NF, typeof(land_input_mean), typeof(input_buffer),
        typeof(land_nn), typeof(land_params), typeof(land_states)}(;
        land_input_mean, land_input_std,
        land_output_mean = stats.target_mean[1],
        land_output_std = stats.target_std[1],
        input_buffer, land_nn, land_params, land_states, kwargs...)
end

# One GPU subtlety: SpeedyWeather fuses all *column* parameterizations into a
# single GPU kernel and passes the (adapted) model components into it — and
# kernel arguments must be `isbits`. A Lux network is not (its Dropout states
# carry a mutable RNG). But the column kernel never evaluates our network — it
# only runs in the host-side global pass. So when the scheme is adapted for the
# kernel we simply strip the network, keeping the lightweight rest:

function Adapt.adapt_structure(to, scheme::LearnedSurfaceRoughnessGlobal)
    land_input_mean = Adapt.adapt(to, scheme.land_input_mean)
    land_input_std  = Adapt.adapt(to, scheme.land_input_std)
    input_buffer    = Adapt.adapt(to, scheme.input_buffer)
    return LearnedSurfaceRoughnessGlobal{
        typeof(scheme.land_output_mean), typeof(land_input_mean),
        typeof(input_buffer), Nothing, Nothing, Nothing}(
        scheme.roughness_length_ocean, land_input_mean, land_input_std,
        scheme.land_output_mean, scheme.land_output_std, input_buffer,
        nothing, nothing, nothing)
end

SpeedyWeather.initialize!(::LearnedSurfaceRoughnessGlobal, ::PrimitiveEquation) = nothing

# ## The global `parameterization!`
#
# A global parameterization implements `parameterization!(vars, scheme, model)`
# — without the grid index `ij` — and runs once per time step outside any
# kernel, so calling `Lux.apply` here is perfectly fine. Everything is
# formulated as array broadcasts, which work on CPU arrays and CUDA arrays
# alike.

function SpeedyWeather.parameterization!(vars::SpeedyWeather.Variables,
        scheme::LearnedSurfaceRoughnessGlobal, model::PrimitiveEquation)

    X = scheme.input_buffer
    land_vars = vars.parameterizations.land
    soil = vars.prognostic.land

    ## gather the inputs of all grid points, rows ordered like in training
    @views begin
        X[2, :] .= land_vars.vegetation_high
        X[3, :] .= land_vars.vegetation_low
        X[1, :] .= 1 .- X[2, :] .- X[3, :]              # bare-soil fraction
        X[4, :] .= vars.grid.geopotential[:, end]       # lowermost layer
        X[5, :] .= soil.snow_depth
        X[6, :] .= soil.soil_temperature[:, 1]          # top soil layer
        X[7, :] .= soil.soil_moisture[:, 1]
    end

    ## normalise all features at once
    X .= (X .- scheme.land_input_mean) ./ scheme.land_input_std

    ## one batched NN evaluation for the whole globe
    pred, _ = Lux.apply(scheme.land_nn, X, scheme.land_params, scheme.land_states)

    ## un-normalise, transform back from log space and write the output fields
    mask = model.land_sea_mask.mask
    z₀_land = vars.parameterizations.land.surface_roughness
    z₀_ocean = vars.parameterizations.ocean.surface_roughness
    z₀ = scheme.roughness_length_ocean

    ## mixing Fields and plain GPU arrays in one broadcast falls back to (slow,
    ## disallowed) scalar indexing — broadcast over the underlying arrays
    ## (`parent`) wherever the raw NN output `pred` is involved
    parent(z₀_land) .= ifelse.(parent(mask) .> 0,
        exp.(vec(pred) .* scheme.land_output_std .+ scheme.land_output_mean),
        zero(z₀))
    z₀_ocean .= ifelse.(mask .< 1, z₀, zero(z₀))
    vars.parameterizations.surface_roughness .= mask .* z₀_land .+ (1 .- mask) .* z₀_ocean
    return nothing
end

# Two dispatch details make this work inside SpeedyWeather. First, as a global
# parameterization the column-wise call must explicitly do nothing — otherwise
# the fused column kernel would try (and fail) to evaluate the scheme per grid
# point:

SpeedyWeather.parameterization!(ij, vars, scheme::LearnedSurfaceRoughnessGlobal, model) = nothing

# Second, SpeedyWeather only collects *top-level* model components for the
# global pass, and our scheme sits inside the `BoundaryLayer`. The default
# global call on a `BoundaryLayer` is a no-op, so we forward it to our scheme
# (the drag component has no global part). The global pass runs before the
# column pass, so the drag — a column parameterization that consumes the
# surface roughness — always sees the freshly computed fields:

SpeedyWeather.parameterization!(vars::SpeedyWeather.Variables,
    BL::BoundaryLayer{<:LearnedSurfaceRoughnessGlobal}, model::PrimitiveEquation) =
    SpeedyWeather.parameterization!(vars, BL.surface_roughness, model)

# ## Import the trained model
#
# Same as in the CPU script, except that the parameters and states are moved to
# the GPU. `stats` can be passed as-is — no name translation needed.

trained = JLD2.load("trained_model.jld2")
land_nn = trained["model"]
stats   = trained["stats"]

device = gpu_device()
land_params = device(trained["parameters"])
land_states = device(trained["states"])    ## test-mode states: dropout disabled

# ## Run it in a GPU simulation
#
# Now the same setup as on the CPU, just with the GPU architecture. (Setting
# `arch = SpeedyWeather.CPU()` and `device = cpu_device()` above runs the very
# same global scheme on the CPU — handy for testing.)

arch = SpeedyWeather.GPU()
spectral_grid = SpectralGrid(trunc = 65, architecture = arch)

surface_roughness = LearnedSurfaceRoughnessGlobal(
    spectral_grid, land_nn, land_params, land_states, stats)

boundary_layer = BoundaryLayer(spectral_grid; surface_roughness)
model = PrimitiveWetModel(spectral_grid; boundary_layer)
simulation = initialize!(model)
run!(simulation, steps = 2)

# And the same sanity check as before — the surface roughness the network
# predicted over land, in meters:

extrema(simulation.variables.parameterizations.land.surface_roughness)
