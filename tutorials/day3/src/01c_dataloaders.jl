# # Dataloaders
#
# First, we need to build dataloaders with  model-ready batches for training 
# our neural network parameterization. The land surface roughness parameterization
# runs column-wise or point-wise, so we take our seven input surface variables and
# output just a single scalar surface roughness value in that particular column. 
# Therefore the dataloaders should provide, for every land column ``i``, a pair
# ``\big(\mathbf{x}^{(i)},\, y^{(i)}\big)`` of the seven (standardised) input
# features and the single (log-transformed, standardised) target:
#
# ```math
# \mathbf{x}^{(i)} = \big(\mathbf{u}^{(i)} - \boldsymbol{\mu}\big) \oslash \boldsymbol{\sigma}
#   \;\in\; \mathbb{R}^{7},
# \qquad
# y^{(i)} = \frac{\log z_0^{(i)} - \mu_y}{\sigma_y} \;\in\; \mathbb{R},
# ```
#
# where ``\oslash`` denotes element-wise division, ``\boldsymbol{\mu}, \boldsymbol{\sigma}``
# (and ``\mu_y, \sigma_y``) are the per-feature mean and standard deviation over all
# samples, and the raw feature vector ``\mathbf{u}`` collects the seven surface
# variables, with the bare-soil fraction ``c_b`` derived rather than stored:
#
# ```math
# \mathbf{u} = \big(\, \underbrace{1 - c_h - c_l}_{c_b},\; c_h,\; c_l,\; \Phi,\; d,\; T,\; \theta \,\big)^\top .
# ```
#
# | symbol     | ERA5    | variable                     |
# |:----------:|:-------:|:-----------------------------|
# | ``c_b``    | derived | bare-soil fraction           |
# | ``c_h``    | `cvh`   | high-vegetation cover        |
# | ``c_l``    | `cvl`   | low-vegetation cover         |
# | ``\Phi``   | `z`     | surface geopotential         |
# | ``d``      | `sd`    | snow depth                   |
# | ``T``      | `stl1`  | top-layer soil temperature   |
# | ``\theta`` | `swvl1` | top-layer soil moisture      |
# | ``z_0``    | `fsr`   | surface roughness *(target)* |
#
# Stacking ``B`` columns gives each batch ``X \in \mathbb{R}^{7\times B}`` and
# ``Y \in \mathbb{R}^{1\times B}`` — the trailing dimension is the batch axis Lux expects.

# ## Pre-processing 
#
# We take in ERA5 data for all variables from 2022-2025 at its native resolution at 
# a roughly weekly temporal resolution (with an included drift to sample all hours 
# of the day). The pre-processing from the spatiatemporal fields was already done in 
# [`01b_preprocessing.jl`](01b_preprocessing.md) that you can find in the repository.
#
# For the preprocessing, we 
# * Merged the downloaded 3-hourly ERA5 data 
# * Subsampled it roughly weekly resolution 
# * Applied a land sea mask to only load the land data 
# * Filtered out any other missing or NaN data 
# * Saved the data directly in point-wise in a Zarr file 
#
# This combined and pre-processed data is available at
# `/cluster/projects/nn9984k/speedy-data/era5-roughness.zarr` on the cluster.

# ## Preparing the dataloaders 
#
# With this pre-processing already done, we are left with just some very basic 
# dataloaders for this. For the dataloaders we use `MLUtils.DataLoader` that 
# provide us with features e.g. for shuffling the data or also for distributed 
# computing. 

using Zarr, MLUtils, Random

# ### Standardisation helpers
#
# First, some utilities to standardize the data to zero mean / unit variance 
# as needed in NN input. We'll also save those means and variances later, to 
# use them to normalize and de-normalize the data on the fly when using it the
# parameterization in our model. 

"Per-feature (per-row) mean and standard deviation across all samples."
function feature_mean_std(X::AbstractMatrix)
    n = size(X, 2)
    μ = sum(X; dims = 2) ./ n
    σ = sqrt.(max.(sum(abs2, X .- μ; dims = 2) ./ max(n - 1, 1), 0))
    return vec(μ), vec(σ)
end

"Standardise rows of `X` with the given mean/std vectors (zero std -> left unscaled)."
function standardize(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector)
    σsafe = map(s -> iszero(s) ? one(s) : s, σ)
    return (X .- μ) ./ σsafe
end

# ### Loading the Zarr dataset
#
# `load_zarr_dataset` reads the whole Zarr store into memory as `Float32` arrays —
# the column-wise samples are small (a handful of features each), so even tens
# of millions of land points fit comfortably. 
# The feature order matches the 7 inputs the `LearnedSurfaceRoughness` network
# expects. The target can be transformed (e.g. `log`) before standardisation to
# regress in the space the parameterization predicts in.

"""
    load_zarr_dataset(zarr_path; normalize=true, target_transform=identity, add_bare_soil=true)

Read the column-wise dataset written by `preprocess_to_zarr`. Stacks the input
variables (in the stored `input_names` order) into `X :: (features, samples)` and
reads the target into `Y :: (targets, samples)`.

Keywords
  * `target_transform` – applied element-wise to the target before normalising,
    e.g. `log` to regress in log space (the parameterization predicts `log`
    surface roughness).
  * `normalize` – standardise inputs and target to zero mean / unit std.
  * `add_bare_soil` – prepend the bare-soil fraction `1 - cvh - cvl` as the first
    feature row. This is the 7th input the `LearnedSurfaceRoughness` NN expects (its
    input order is `[bare_soil, vegetation_high, vegetation_low, geopotential,
    snow_depth, soil_temperature, soil_moisture]`); it is derived here rather than
    stored, and "bare_soil" is prepended to the returned `input_names`.

Returns `(; X, Y, input_mean, input_std, target_mean, target_std, input_names, target_name)`.
The data is already clean (preprocessing dropped every NaN/missing sample).
"""
function load_zarr_dataset(zarr_path::AbstractString;
        normalize::Bool = true, target_transform = identity, add_bare_soil::Bool = true)

    g = zopen(zarr_path, "r")
    input_names = String.(g.attrs["input_names"])
    target_name = String(g.attrs["target_name"])

    ## one (features, samples) block per stored input variable
    feats = Matrix{Float32}[Float32.(g[name][:, :]) for name in input_names]

    ## bare-soil fraction isn't stored — derive it so X matches the NN's 7-input order
    if add_bare_soil
        ih = findfirst(==("cvh"), input_names)   # high vegetation cover
        il = findfirst(==("cvl"), input_names)   # low vegetation cover
        (isnothing(ih) || isnothing(il)) &&
            error("add_bare_soil needs 'cvh' and 'cvl' among inputs; got $(input_names)")
        pushfirst!(feats, 1f0 .- feats[ih] .- feats[il])
        input_names = ["bare_soil"; input_names]
    end

    X = reduce(vcat, feats)
    Y = target_transform.(Float32.(g[target_name][:, :]))

    input_mean, input_std   = feature_mean_std(X)
    target_mean, target_std = feature_mean_std(Y)

    if normalize
        X = standardize(X, input_mean, input_std)
        Y = standardize(Y, target_mean, target_std)
    end

    return (; X, Y, input_mean, input_std, target_mean, target_std, input_names, target_name)
end

# ## Train/validation loaders
#
# Then we code up the actual code to construct the data loaders.
# Given, the loaded data from the Zarr store by `load_zarr_dataset` we initialize
# the `MLUtils.DataLoader`s with a train and validation split. That we can iterate
# over during traning. 

"""
    pointwise_dataloaders(zarr_path; batchsize=1024, split=0.8, shuffle=true,
                          rng=Random.default_rng(), kwargs...)

Build train/validation `MLUtils.DataLoader`s from the preprocessed Zarr store and
return `(train_loader, val_loader, stats)`. Batches are `(x, y)` with
`x :: (features, batchsize)` and `y :: (targets, batchsize)`, ready for Lux.

`split` is the fraction of samples used for training (`1.0` -> no validation
split; the second loader then iterates an empty set). Remaining `kwargs` go to
`load_zarr_dataset` (`normalize`, `target_transform`).

`stats.input_mean / input_std` and `stats.target_mean / target_std` are the
normalisation constants the `LearnedSurfaceRoughness` scheme needs at inference.
"""
function pointwise_dataloaders(zarr_path::AbstractString;
        batchsize::Int = 1024, split::Real = 0.8, shuffle::Bool = true,
        rng::AbstractRNG = Random.default_rng(), kwargs...)

    data = load_zarr_dataset(zarr_path; kwargs...)
    X, Y = data.X, data.Y

    (Xtrain, Ytrain), (Xval, Yval) = splitobs((X, Y); at = split, shuffle = true)

    train_loader = DataLoader((Xtrain, Ytrain); batchsize, shuffle, partial = false, rng)
    val_loader   = DataLoader((Xval, Yval); batchsize, shuffle = false, partial = true)

    stats = (; data.input_mean, data.input_std, data.target_mean, data.target_std,
               data.input_names, data.target_name)
    return train_loader, val_loader, stats
end

# Next we actually run the training! 