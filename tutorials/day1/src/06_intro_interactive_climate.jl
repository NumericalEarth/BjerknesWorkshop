# # Interactive Climate Modelling Basics with SpeedyWeather.jl (and Oceananigans.jl)
#
# After the Julia introduction, you are ready to directly dive into interactive climate 
# modelling with NumericalEarth, SpeedyWeather.jl and Oceananigans.jl. Here, we will 
# introduce some basic concepts that both models share and then later actually run some 
# simulations with SpeedyWeather. Let's go step by step through all ingredients that we need! 
#
# Note that for a lot of these examples, we have to specify `SpeedyWeather.` or `Oceananigans.` (e.g. `SpeedyWeather.run!` or `Oceananigans.run!`) to avoid ambiguity. This isn't necessary if you are just using one of the packages by itself

# ## Devices and architectures: Where do we run our model?
#
# First, we will have to set up the device we want to run our model on. 

using SpeedyWeather, Oceananigans

arch = SpeedyWeather.CPU()
## arch = SpeedyWeather.GPU()

# with this architecture, we can also transfer data between devices with this: 

A = rand(10)
A_cpu = SpeedyWeather.on_architecture(arch, A) 
##A_gpu = on_architecture(SpeedyWeather.GPU(), A)

# This works in the same manner for both SpeedyWeather.jl and Oceananigans.jl.
#
# ## Grids and discretization 
#
# Next, we will have to set up the grid we want to discretize our model on.
# * SpeedyWeather.jl is a spectral (or rather pseudo-spectral) model with variables discretized both in spectral space and in grid space. # * Oceananigans is based on a Finite Volume framework with equations discretized on a staggered C-grid (velocities are defined on edges and tracers in the center of the cell). 
# For SpeedyWeather we therefore have a `SpectralGrid` as an object that bundles spectral and grid discretization information, we initialize it with an argument for the spectral truncation `trunc` that sets the maximum degree and order of the spherical harmonics (and therefore the resolution), and the number of vertical layers `nlayers` that we discretize the model in.

spectral_grid = SpectralGrid(trunc=31, nlayers=8, architecture = arch)

# We also directly see these information summarized here in the return. 
# The `SpectralGrid` hold both the spectral discretization

spectral_grid.spectrum 

# and the grid discretization

spectral_grid.grid 

# This is analogous to defining e.g. a `LatitudeLongitudeGrid` in Oceananigans.jl: 

latitude_longitude_grid = LatitudeLongitudeGrid(size = (60, 30, 1),
                                                z = (-1, 0),
                                                topology = (Periodic, Bounded, Bounded))

# With Oceananigans you can also define a `RectilinearGrid` for box domains or use the `LatitudeLongitudeGrid` for regional problems by 
# initializing it with latitude and longitude bounds e.g. `longitude = (0, 60)` , `latitude = (-30, 30)`.

# ## Data on grids 
#
# Almost all data or arrays in SpeedyWeather.jl and Oceananigans.jl are associated with these discrete domains. 
# In SpeedyWeather we have a `LowerTriangularArray` that hold an array and a `Spectrum` and a `Field` that hold an array and a `Grid`. 
# In Oceananigans we just have a `Field`. Let's initialize one! 

A = rand(spectral_grid.grid)

# Well, that was easy! We can also use the `set!` function in both packages to set values: 

SpeedyWeather.set!(A, 1.0)

# In Oceananigans we have a `Field` well, but as Oceananigans makes use of staggered grids, 
# we also have to specificy where on the grid (e.g. `Center` or `Face`) the data is located. 
# We initialize an Oceananigans `Field` as follows:

Field{Center, Center, Center}(latitude_longitude_grid) 

# ## Models 
#
# Now that we have a grid, we can also construct a model! In SpeedyWeather.jl we have four models available: a `BarotropicModel`, a `ShallowWaterModel`, a `PrimitiveDryModel` and the full complexity `PrimitiveWetModel`. The model constructor takes in the discretization as an argument and then also allows for further configuration (more about that later): 

model = PrimitiveWetModel(spectral_grid)

# The model by itself doesn't hold all necessary pre-computed fields yet and also not all information required for an experiment. We get those by initializing the model which yields a `Simulation`:

simulation = SpeedyWeather.initialize!(model)

# We can now run the model!

SpeedyWeather.run!(simulation, period=Day(10), output=true)

# And just like that, we've run our first interactive atmospheric simulation! 

# ## And what about Oceananigans? 
# The analogous quick start to run a model is very similar. Again, as with SpeedyWeather we define the architecture, then grid with the resolution, then the model, then we initialize the `Simulation`, and finally we `run!` the model: 
## set the architecture
arch = Oceananigans.CPU()

## set the discretization
grid = RectilinearGrid(arch, size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))

## construct the model 
ocean_model = NonhydrostaticModel(grid; advection=WENO())

## set an initial condition 
ϵ(x, y) = 2rand() - 1
Oceananigans.set!(ocean_model, u=ϵ, v=ϵ)

## initialize the simulation and run it!
ocean_simulation = Oceananigans.Simulation(ocean_model; Δt=0.01, stop_time=4)
Oceananigans.run!(ocean_simulation)

# ## Visualizing the results
#
# We ran the SpeedyWeather experiment with `output=true`, so we actually generated by default a NetCDF file 
# that saves the results of the experiment in the `output` folder (Zarr and JLD2 output is also available).
# Let's have a look at the results: 

using NCDatasets, CairoMakie 

ds = NCDataset(joinpath(model.output.run_path, model.output.filename))

# In the return you also directly see all the variables, that we saved. Let's first just plot some snapshots of the vorticity, and then we also animate it!

i_layer = 8 # that's the closest layer to the surface
time_step = 30
heatmap(ds["vor"][:,:,i_layer,time_step])

# We can also animate the vorticity ourselves with CairoMakie: we wrap the time index in an
# `Observable`, build a heatmap from it, and then `record` updates it frame by frame.

function animate_field(data, filename; lon=axes(data, 1), lat=axes(data, 2),
                       label="", title="", time_steps=1:size(data, 3), framerate=10)
    ## this defines the iterator that will update the animation
    i_time = Observable(1)

    ## this is the array that is animated
    frame = @lift data[:, :, $i_time]

    data_max = maximum(abs, data) # symmetric colorrange around zero, fixed for all frames
    fig, ax, hm = heatmap(lon, lat, frame, colormap=:balance,
                          colorrange=(-data_max, data_max),
                          axis=(xlabel="Longitude [˚E]", ylabel="Latitude [˚N]"))
    Colorbar(fig[1, 2], hm; label)

    ## here, we do the actual animation:
    record(fig, filename, eachindex(time_steps); framerate) do t
        i_time[] = t
        ax.title = "$title, time step $(time_steps[t])"
    end
end

# Now we can animate the vorticity (rendering the animation might take a bit): 

n_steps = min(100, size(ds["vor"], 4))  # animate a slice of (up to) 100 time steps
vor = nomissing(ds["vor"][:, :, i_layer, 1:n_steps], NaN)

animate_field(vor, "vorticity.mp4"; lon=ds["lon"][:], lat=ds["lat"][:],
              label="Relative vorticity [1/s]", title="Relative vorticity")

# ## The variables
#
# Then, let's briefly look a bit under the hood of our model. To run the model, so far we intialize the `model` with 
# `initialize!(model)` this returns a `Simulation` which hold both the model itself, but also all the allocated variables
# at the current timestep in SpeedyWeather.jl (In Oceananigans the model holds the variables directly). 
# When we want to interact and modify the model, we have to work with those variables!
#
# Let's inspect them:

simulation.variables

# On the top of this list you see all our prognostic variables, then you have all our auxiliary and diagnostic variables such as `grid` versions of the prognostic variables, or intermediate values of the `dynamics` and `parameterizations` that are saved here to avoid that they are constantly re-allocated at every timestep. Any custom parameterization can also define new variables to be allocated under `parameterizations` in the `Variables`.

# ## Configuring a model 
#
# Of course, we don't always want to run the default configuration of a model. 
# A main advantage of our modelling approaches is that it is very easy to tailor a model to your needs, change parameters, but also whole model components, just a in a script. 
# Later, on Wednesday, we'll even see how we can define a new machine learned model component that we can then use in a model!
#
# The documentations of [SpeedyWeather](https://speedyweather.github.io/SpeedyWeatherDocumentation/dev/), [Oceananigans](https://clima.github.io/OceananigansDocumentation/dev/) and [Breeze](https://github.com/NumericalEarth/Breeze.jl) provide extensive resources on which components and processes are actually available.
# Here, we'll just configure a very basic model, which we will then run with SpeedyWeather.jl. 

# ### Change the time step
#
# The time step of SpeedyWeather is controlled through the time stepping method of the model. This model component
# needs to know the spatial resolution to pick a time step by default that is stable, but you can still control this.
# SpeedyWeather's time integration is based on the `Leapfrog` scheme, so you create such a component like this

spectral_grid = SpectralGrid(trunc = 32, architecture = SpeedyWeather.CPU())
time_stepping = Leapfrog(spectral_grid, Δt_at_T31=Minute(20))

# where the argument `Δt_at_T31` determines the timestep `Δt` (write `\Delta` then hit tab) relative to a truncation of
# 31 (called T31), the actual time step is then in `Δt_sec`, scaled linearly from T31 to whatever resolution you chose.
# You can provide any `Second`, `Minute`, `Hour` (but note that there is a stability limit above which your simulation quickly blows up).
# But do not forget to also pass this component to the model constructor, i.e.

model = PrimitiveWetModel(spectral_grid; time_stepping)

# where `; time_stepping` matches a keyword argument `time_stepping` with the variable of the same name. This is equivalent
# to `, time_stepping=time_stepping`.
# and then run a model with it 

simulation = SpeedyWeather.initialize!(model)
SpeedyWeather.run!(simulation, period=Day(10))

# ## Change the orography
#
# You can change the orography in several different ways, the most convenient is
# probably using the `set!` function with anonymous functions `(λ,φ) -> ...`
# of longitude ``\lambda`` and latitude ``\varphi`` as outlined [here](https://speedyweather.github.io/SpeedyWeather.jl/dev/orography/#Changing-orography-manually). In brief, _after_ model initialization
# (otherwise orography would also be initialized, overwriting your changes)

spectral_grid = SpectralGrid(trunc = 32, architecture = SpeedyWeather.CPU())
model = PrimitiveWetModel(spectral_grid)
simulation = SpeedyWeather.initialize!(model)


# set to a global constant
SpeedyWeather.set!(model, orography=0)

# add two 1500m ridges at +-30˚E from 60˚S to 60˚N
H, λ₀, φmax = 1500, 15, 60
#SpeedyWeather.set!(model, orography=(λ,φ) -> 2λ₀ < λ < 360-2λ₀ || abs(φ) > φmax ? 0 : H*sind(180*λ/2λ₀)^2)

# add a 1km zonal ridge between 60˚E and 300˚E
SpeedyWeather.set!(model, orography=(λ,φ) -> λ > 4λ₀ && λ < 360-4λ₀ && abs(φ) < 20 ? H*cosd(3φ)^2 : 0, add=true)

# add two Gaussian mountains
λ1, λ2  = (120, 240)    # longitude positions [˚E]
φ₀ = 45                 # latitude [˚N]
σ = 5                   # width [˚]

# first 6km mountain, radius=360/2π to have distance in ˚ again (not meters)
SpeedyWeather.set!(model, orography=(λ,φ) -> 4*H*exp(-spherical_distance((λ,φ), (λ1,φ₀), radius=360/2π)^2/2σ^2), add=true)

# and add second
SpeedyWeather.set!(model, orography=(λ,φ) -> 4*H*exp(-spherical_distance((λ,φ), (λ2,φ₀), radius=360/2π)^2/2σ^2), add=true)

# * If you are not familiar with Julia, the `?` command might be unknown to you. It's just a short hand for a one-line `if` clause. It's syntax is `value = condition ? true_value : false_value`.
# * The `->` assigns an anonymous function (like Python's `lambda` functions), with `(input_arg_1, input_args_2, ...) -> output` as its syntax.
# * `||` and `&&` are the usual short-circuit logical operators `OR` and `AND`.
#
# Whatever orography you construct, you can always check it with

heatmap(model.orography.orography, title="Orography [m]: Is it supposed to be a smiley?")

# You can try to play around with setting different orographies and then rerunning the model. What differences can you see?

## experiment a bit! how different does the integration look like with this orography?
## when you reinitialize the model, you have to change the orography after the `simulation = initialize!(model)`!
SpeedyWeather.run!(simulation, period=Day(20), output=true)

# Let's have a look if we see a trace of what we changed. The zonal wind ``u`` shows the
# imprint of the orography most clearly: the flow is blocked and deflected by the ridges
# and excites stationary waves downstream of the mountains. Compare the wind field with
# the orography we plotted above!

ds = NCDataset(joinpath(model.output.run_path, model.output.filename))

lon, lat = ds["lon"][:], ds["lat"][:]
i_layer = 8               # that's the closest layer to the surface
t_end = size(ds["u"], 4)  # last output time step, after the flow adjusted to the orography

u_end = nomissing(ds["u"][:, :, i_layer, t_end], NaN)
u_max = maximum(abs, u_end) # symmetric colorrange: easterlies blue, westerlies red

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], title="Surface zonal wind after 20 days",
          xlabel="Longitude [˚E]", ylabel="Latitude [˚N]")
hm = heatmap!(ax, lon, lat, u_end, colormap=:balance, colorrange=(-u_max, u_max))
Colorbar(fig[1, 2], hm, label="u [m/s]")
fig

# And we can reuse our `animate_field` function from above, this time animating the zonal
# wind over (up to) the last 100 output time steps:

n_steps = min(100, size(ds["u"], 4))
t_range = (size(ds["u"], 4) - n_steps + 1):size(ds["u"], 4)  # slice from the end of the run
u_anim = nomissing(ds["u"][:, :, i_layer, t_range], NaN)

animate_field(u_anim, "zonal_wind.mp4"; lon, lat, time_steps=t_range,
              label="Zonal wind [m/s]", title="Surface zonal wind")

# ## One Global GPU Example
#
# Finally, let's put everything together for one high-resolution global simulation on the GPU.
# This is a main selling point of interactive climate modelling with NumericalEarth:
# the model code stays exactly the same, we only exchange the `architecture` of the
# `SpectralGrid`. We guard the choice with `CUDA.functional()` so that this script also
# runs (much slower though!) on machines without an Nvidia GPU:

using CUDA
arch = CUDA.functional() ? SpeedyWeather.GPU() : SpeedyWeather.CPU()

## 1. define the resolution and grid, change what you like
spectral_grid = SpectralGrid(trunc=256, nlayers=16, architecture=arch)

## 2. create a model
model = PrimitiveWetModel(spectral_grid)

## 3. then, initialize the model
simulation = SpeedyWeather.initialize!(model)

## 4. and then run the model
n_days = 30
SpeedyWeather.run!(simulation, period=Day(n_days), output=true)

# How fast was that? The progress meter of the model's `feedback` component already
# tracks the wallclock time of the run (it is reinitialized at the start of the time
# stepping for benchmark accuracy), so we can read the elapsed time directly from it:

pm = model.feedback.progress_meter.core
elapsed = pm.tlast - pm.tinit             # wallclock time of the run in seconds
sypd = (n_days / 365.25) * 86400 / elapsed  # simulated years per wallclock day
println("$n_days simulated days took $(round(elapsed, digits=1)) s, that's $(round(sypd, digits=1)) simulated years per day.")

# Try to rerun this section with `arch = SpeedyWeather.CPU()` or with a different `trunc`
# and compare: how much faster is the GPU, and how does the gap change with resolution?

# Then let's visualize the results: at T256 (about 50 km grid spacing). Let's look at the last snapshot:

ds = NCDataset(joinpath(model.output.run_path, model.output.filename))

lon, lat = ds["lon"][:], ds["lat"][:]
i_layer = size(ds["vor"], 3)  # with nlayers=16 the lowest model layer is now 16, not 8!

n_steps = min(100, size(ds["vor"], 4))
t_range = (size(ds["vor"], 4) - n_steps + 1):size(ds["vor"], 4)  # slice from the end of the run
vor_anim = nomissing(ds["vor"][:, :, i_layer, t_range], NaN)
vor_max = maximum(abs, vor_anim)  # fixed, symmetric colorrange as before

fig = Figure(size=(900, 450))
ax = Axis(fig[1, 1], title="Surface relative vorticity after $n_days days",
          xlabel="Longitude [˚E]", ylabel="Latitude [˚N]")
hm = heatmap!(ax, lon, lat, vor_anim[:, :, end], colormap=:balance, colorrange=(-vor_max, vor_max))
Colorbar(fig[1, 2], hm, label="Vorticity [1/s]")
fig

# We reuse our `animate_field` function once more,
# again on the slice from the end of the run:

animate_field(vor_anim, "vorticity_gpu.mp4"; lon, lat, time_steps=t_range,
              label="Relative vorticity [1/s]", title="Surface relative vorticity")

# That's it! You have run atmospheric simulations from your laptop's CPU all the way to a
# high-resolution GPU run, configured grids, time stepping and orography along the way.
# Experiment a bit further and configure your own simulation!

