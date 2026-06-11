# # Hybrid ML Example: Learning a surface roughness parameterization
#
# This repository is a small, end-to-end example of *hybrid* Earth-system modelling:
# we replace a hard-coded boundary condition in the atmospheric model
# [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl) with a small
# neural network that is **learned offline from data** and then run inside
# the model. 
#
# Original work on this was done by Greg Munday at U Oxford, this here is a slightly 
# simplified version of it. 

# ## Why does surface roughness matter?
#
# Surface fluxes of momentum, heat and moisture between the atmosphere and the
# surface are controlled by a *bulk drag coefficient*. For example, the surface
# sensible heat flux is modelled as
#
# ```math
# F = -\rho \, r \, U \, (T - T_\text{s}),
# ```
#
# where ``\rho`` is air density, ``U`` the near-surface wind speed, ``T - T_\text{s}``
# the air–surface temperature difference, and ``r`` the drag coefficient. That drag
# coefficient depends on how *rough* the surface is, through the surface roughness
# length ``z_0``:
#
# ```math
# r = F\!\left(\,\dots,\; r_\text{max} = \left(\frac{\kappa}{\log(z / z_0)}\right)^2 \right),
# ```
#
# with ``\kappa`` the von Kármán constant and ``z`` the height of the lowest model
# level. Physically, ``z_0`` is itself a function of the surface state —
# **vegetation, snow, sea ice, orography, and (over the ocean) the wind-driven waves**.
# This calculation runs in *every grid cell on every time step*. It's one of the many 
# column-based parameterizations in SpeedyWeather.jl, and most atmospheric GCMs. 

# ## What SpeedyWeather does per default
#
# By default SpeedyWeather uses a **constant** roughness length: roughly `0.5 m` over
# land and `0.1 mm` over the ocean. It is a single hard-coded number per surface type
# — it does not vary with vegetation, snow, or wind.

# ## Can we do better — learn ``z_0`` from data?
#
# Instead of hard-coding another boundary-condition field, we ask the network to
# learn the *function* that maps the local surface state to its roughness, using
# ERA5 reanalysis as the training target. With this we want a **spatial generalisation**: a
# learned function of surface variables can be applied wherever those variables are
# defined — under land-use change, in idealised scenarios, even on rearranged
# continents — rather than being tied to a fixed global map.

# ## A column-based parameterization
#
# Crucially, we learn a **column-based** model: the roughness in a grid cell is a
# function of that cell's own surface variables only, with no spatial context. The
# network sees one column at a time, so during training every grid point (at every
# time step) is an independent sample — turning one global snapshot into millions of
# training examples.
#
#
# ## The land predictors and target
#
# This example focuses on the **land** roughness (the ocean keeps a constant value).
# The network maps the following per-column inputs to the (log) surface roughness.
# The middle column is the corresponding ERA5 variable name used by the preprocessing.
#
# | NN input (feature)            | ERA5 variable        |
# |-------------------------------|----------------------|
# | high vegetation cover         | `cvh`                |
# | low vegetation cover          | `cvl`                |
# | bare-soil fraction `1−cvh−cvl`| derived from `cvh,cvl` |
# | surface geopotential          | `z`                  |
# | snow depth                    | `sd`                 |
# | top-layer soil temperature    | `stl1`               |
# | top-layer soil moisture       | `swvl1`              |
# | **target**: surface roughness | `fsr`                |
#
# These seven features feed a small multilayer perceptron whose scalar output is the
# normalised ``\log z_0``; at inference time it is un-normalised and exponentiated
# back to a roughness length in metres.

# ## Outline
#
# The remaining literate scripts walk through the full offline-learn / online-apply
# loop:
#
# 1. [`dataloaders.jl`](dataloaders.jl) — read that Zarr store and serve normalised
#    `(features, batch)` mini-batches ready for [Lux.jl](https://lux.csail.mit.edu/).
# 2. [`training.jl`](training.jl) — train the MLP offline (AdamW, early stopping on a
#    validation split) and save the trained weights with JLD2.
# 3. [`parameterization.jl`](parameterization.jl) (run via
#    [`run_parameterization.jl`](run_parameterization.jl)) — wrap the trained network
#    as a `LearnedSurfaceRoughness <: SpeedyWeather.AbstractSurfaceRoughness` and run
#    it **online**, replacing the constant boundary condition with a learned one.

