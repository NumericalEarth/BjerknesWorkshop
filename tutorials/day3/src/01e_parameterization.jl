# # Defining the learned parameterization 
#
# Here, we define the learned surface roughness parameteriztion for usage with SpeedyWeather.jl  
# In SpeedyWeather we have two different kinds of possible custom parameterizations, the typical  
# column-based parameterizations that only compute per column, and global parameterizations that 
# take the complete global field as an input. An example for a global parameterization in our model 
# is the solar zenith calculation. The surface roughness we will implement as a column-based 
# parameterization initially. The parameterization system is explained in detail in the [documentation](https://speedyweather.github.io/SpeedyWeatherDocumentation/dev/parameterizations#Define-your-own-parameterizations).
# 
# In case of defining a new surface roughness, we just need to replace the default one
# `ConstantSurfaceRoughness` that merely writes constant values into the allocated variables 
# * `vars.parameterizations.ocean.surface_roughness`,
# * `vars.parameterizations.land.surface_roughness`,
# * `vars.parameterizations.surface_roughness`.
# 
# Therefore, we don't need to worry about the tendency or flux computation with this parameterization.
# We just need implement 
# * A `struct` that holds the parameters (including the NN) 
# * A constructor for that `struct`
# * `SpeedyWeather.initialize!(::LearnedSurfaceRoughness, ::PrimitiveEquation)` that initializes the parameterization
# * Either a `SpeedyWeather.parameterization!(ij, vars, scheme::AbstractSurfaceRoughness, model)` or `SpeedyWeather.surface_roughness!(ij, vars, scheme::LearnedSurfaceRoughness, land_sea_mask)` that implements the actual computation
#
# Okay, let's go! First we define the `struct` and it's constructor:

using SpeedyWeather, Lux, Adapt, Random

@kwdef struct LearnedSurfaceRoughness{NF, V, M, LNN, LP, LS} <: SpeedyWeather.AbstractSurfaceRoughness
    "[OPTION] constant roughness length over ocean [m]"
    roughness_length_ocean::NF = 1.0e-4

    ## learned land roughness parameters
    ## Land normalisation parameters (NamedTuples keyed by the inputs used in
    ## surface_roughness_land) to normalize NN inputs
    land_input_means::M
    land_input_stds::M

    land_output_mean::NF = -5.031811f0
    land_output_std::NF = 2.4447718f0

    ## input buffer for NN input so that we don't need to allocate it every time
    land_input_buffer::V

    ## NN structure, parameters and states
    land_nn::LNN
    land_params::LP
    land_states::LS
end

function LearnedSurfaceRoughness(
        SG::SpectralGrid,
        land_nn, 
        land_params, 
        land_states, 
        land_input_means, 
        land_input_stds;
        kwargs...
    )

    ## we allocate the input buffer on the same device we are running the model on
    land_input_buffer = on_architecture(SG.architecture, zeros(Float32, 7))

    return LearnedSurfaceRoughness{
        SG.NF,
        typeof(land_input_buffer),
        typeof(land_input_means),
        typeof(land_nn),
        typeof(land_params),
        typeof(land_states),
    }(;
        land_input_means = land_input_means,
        land_input_stds = land_input_stds,
        land_input_buffer = land_input_buffer,
        land_nn = land_nn,
        land_params = land_params,
        land_states = land_states,
        kwargs...
    )
end

## this is some housekeeping we need for GPU compatability 
Adapt.@adapt_structure LearnedSurfaceRoughness

# Next, the `initialize!` in this case it's actually, completley trivial: we 
# can just do `nothing`. 

SpeedyWeather.initialize!(::LearnedSurfaceRoughness, ::PrimitiveEquation) =  nothing

# Now, the core computation that collects the input variables, normalizes them,
# applies the neural network and then writes the result into
# `vars.parameterizations.ocean.surface_roughness`,
# `vars.parameterizations.land.surface_roughness`, and `vars.parameterizations.surface_roughness`.

@inline function normalise(a, m, s)
    return (a - m) / s
end

Base.@propagate_inbounds function surface_roughness_land(ij, vars, scheme::LearnedSurfaceRoughness)
    
    ## we collect all inputs for the NN from our state variables
    ## this doesn't allocate it's just a shorthand 
    vₕ = vars.parameterizations.land.vegetation_high[ij]
    vₗ =  vars.parameterizations.land.vegetation_low[ij]
    vᵦ = 1 - vₕ - vₗ  # bare soil fraction
    g = vars.grid.geopotential[ij, end]
    sd = vars.prognostic.land.snow_depth[ij]
    soil_moisture = vars.prognostic.land.soil_moisture[ij, begin]  # currently top layer
    soil_temperature = vars.prognostic.land.soil_temperature[ij, begin]  # top layer (matches stl1)

    ## Normalise inputs
    vₕ = normalise(vₕ, scheme.land_input_means.vegetation_high, scheme.land_input_stds.vegetation_high)
    vₗ = normalise(vₗ, scheme.land_input_means.vegetation_low, scheme.land_input_stds.vegetation_low)
    vᵦ = normalise(vᵦ, scheme.land_input_means.bare_soil, scheme.land_input_stds.bare_soil)
    g = normalise(g, scheme.land_input_means.geopotential, scheme.land_input_stds.geopotential)
    sd = normalise(sd, scheme.land_input_means.snow_depth, scheme.land_input_stds.snow_depth)
    soil_moisture = normalise(soil_moisture, scheme.land_input_means.soil_moisture, scheme.land_input_stds.soil_moisture)
    soil_temperature = normalise(soil_temperature, scheme.land_input_means.soil_temperature, scheme.land_input_stds.soil_temperature)

    ## write into input buffer for NN 
    scheme.land_input_buffer[:] .= (vᵦ, vₕ, vₗ, g, sd, soil_temperature, soil_moisture)

    ## apply NN 
    prediction, _ = Lux.apply(scheme.land_nn, scheme.land_input_buffer, scheme.land_params, scheme.land_states)
    
    ## unnormalise and return 
    log_surface_roughness = (prediction[1] * scheme.land_output_std) + scheme.land_output_mean
    surface_roughness = exp(log_surface_roughness)
    return surface_roughness
end

Base.@propagate_inbounds function SpeedyWeather.surface_roughness!(ij, vars, scheme::LearnedSurfaceRoughness, land_sea_mask)
    land_fraction = land_sea_mask.mask[ij]

    ## Compute separate ocean and land surface roughness
    ## (ocean roughness where there is any ocean, land roughness where there is any land)
    vars.parameterizations.ocean.surface_roughness[ij] = ifelse(land_fraction < 1, scheme.roughness_length_ocean, zero(land_fraction))
    vars.parameterizations.land.surface_roughness[ij] = ifelse(land_fraction > 0, surface_roughness_land(ij, vars, scheme), zero(land_fraction))

    ## Blend the two via arithmetic average
    vars.parameterizations.surface_roughness[ij] = land_fraction * vars.parameterizations.land.surface_roughness[ij] + (1 - land_fraction) * vars.parameterizations.ocean.surface_roughness[ij]
    return nothing
end
