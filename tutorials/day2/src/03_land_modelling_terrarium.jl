
# # [An introduction to Terrarium](@id introduction)
#
# [Terrarium.jl](https://github.com/NumericalEarth/Terrarium.jl) is a new framework for
# physics- and data-driven **land modeling across scales** we started. It is built on the
# finite-volume numerics of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
# and developed alongside [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl)
# as the land component of a new, fully GPU- and autodiff-compatible Earth system model in
# Julia.
#
# We see Terrarium as a **toolkit** for assembling land models from modular
# process building blocks. Its goals are to be fast enough for global simulations on a
# laptop, scalable to HPC and GPUs, fully differentiable with Enzyme.jl, and friendly
# enough to actually be fun to use. 
#
# !!! warning "Under construction"
#     Terrarium is still in early development. And we are still working on some core physics implementations. 
#     If you would like to get involved, please don't hesitate to reach out!
#
# ## Why build on Oceananigans?
#
# Because Oceananigans' finite-volume operators are GPU-compatible and differentiable
# out of the box, Terrarium inherits both capabilities almost for free. The payoff is
# performance that scales from a single column to millions of grid cells:
#
# ![Simulated years per day (SYPD) versus number of grid cells, CPU vs GPU](assets/terrarium-perf.png)
#
#
# ## How Terrarium is organized
#
# ![The data layout of a Terrarium model](assets/terrarium-structure.png)
#
# Every Terrarium model lives on a **grid**. Horizontally, the domain is a collection of
# up to $N_h$ independent **columns** (single grid cells, or points on a global
# [`RingGrid`](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/ringgrids/)).
# Vertically, each column is discretized into $N_i$ **layers** spanning the subsurface.
# The physics is expressed as **kernel functions** that run
# over the grid through a **device interface**, so the exact same code runs on CPU or GPU.
#
# Three abstractions tie this together (see [Basic concepts](https://numericalearth.github.io/Terrarium.jl/dev/introduction/basic_concepts/)):
#
# - **Models** (`AbstractModel`, e.g. [`SoilModel`](@ref), [`LandModel`](@ref)) bundle a
#   `grid`, one or more processes, and an `initializer`.
# - **Processes** (`AbstractProcess`) are the building blocks — they declare state
#   variables and parameters and define the equations of motion via fused kernels.
# - **State variables** are realized as Oceananigans [`Field`](@ref)s on the grid and come
#   in three flavors: *prognostic* (advanced by the timestepper), *auxiliary* (derived each
#   step), and *input* (external forcing supplied by `InputSource`s).



# ## A first model: a single soil column
#
# The smallest useful model is a [`SoilModel`](@ref) on a [`ColumnGrid`](@ref): one (or
# more) laterally independent vertical columns. Here we use 10 exponentially spaced soil
# layers, in single precision, on the CPU.

using Terrarium

grid = ColumnGrid(CPU(), Float32, ExponentialSpacing(N = 10))
model = SoilModel(grid)

# We prescribe a constant 1 °C surface temperature, pick the [`ForwardEuler`](@ref)
# timestepper, and run for 10 days:

bcs = PrescribedSurfaceTemperature(:T_ub, 1.0)
integrator = initialize(model, ForwardEuler(eltype(grid)); boundary_conditions = bcs)
run!(integrator, period = Day(10))

# That is a complete (if minimal) Terrarium simulation. Swapping out the `grid` is all it
# takes to scale up — which is exactly what we do next.

# ## Going global
#
# To run over the whole planet we place the columns on a global
# [`FullGaussianGrid`](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/ringgrids/)
# from RingGrids.jl and wrap it in a [`ColumnRingGrid`](@ref). Switching `CPU()` to `GPU()`
# is the only change needed to run on the GPU. (The full example with a real land–sea mask
# and ERA5 forcing lives in [`examples/simulations/soil_heat_global.jl`](https://github.com/NumericalEarth/Terrarium.jl/blob/main/examples/simulations/soil_heat_global.jl).)

using CUDA
import RingGrids

NF = Float32
arch = CUDA.functional() ? GPU() : CPU()

rings = RingGrids.FullGaussianGrid(8)  # 16 latitude rings, 512 cells, ~9°
grid = ColumnRingGrid(arch, NF, ExponentialSpacing(N = 10), rings)
grid_lon, grid_lat = RingGrids.get_lonlats(grid.rings)  # radians
model = SoilModel(grid)

# We force the surface with a simple latitude-dependent climatology (warm equator, cold
# poles) plus a longitude-shifted daily cycle, mimicking the march of the sun around the
# globe. Boundary conditions can be plain functions of the cell coordinate `x` and time `t`.

mean_annual_temperature(lat) = 20 - abs(40 * sin(lat))  # °C, max at equator

function diurnal_bc(lon::AbstractVector, lat::AbstractVector; amplitude = 10)
    lon_d = on_architecture(arch, NF.(lon))      # move coordinates onto the device
    lat_d = on_architecture(arch, NF.(lat))
    ## inner function matching the (x, t) boundary-condition signature
    function bc_fn(x::NF, t::NF) where {NF}
        i = round(Int, x)                        # x indexes the ring-ordered column
        T₀ = mean_annual_temperature(lat_d[i])
        return T₀ + NF(amplitude) * sin(2π * t / NF(86400) - lon_d[i])
    end
    return bc_fn
end

bcs = PrescribedSurfaceTemperature(:T_ub, diurnal_bc(grid_lon, grid_lat))
integrator = initialize(model, ForwardEuler(NF); boundary_conditions = bcs)

# Run for 12 hours and look at the temperature of the uppermost soil layer. Terrarium
# `Field`s convert directly to RingGrids `Field`s, which plot nicely on a globe with
# `CairoMakie` + `GeoMakie`.

run!(integrator, period = Hour(12), Δt = 600.0)

using CairoMakie, GeoMakie
T_surface = RingGrids.Field(arch, interior(integrator.state.ground_temperature), grid)
heatmap(T_surface[:, 1, 1], title = "Uppermost soil layer temperature", colorrange = (-20, 20))

# ## A global, coupled experiment: Terrarium as SpeedyWeather's land
#
# Terrarium's `ColumnRingGrid` shares its horizontal grid with
# [SpeedyWeather.jl](https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/),
# so a Terrarium model can be dropped in as SpeedyWeather's **land component**: the
# atmosphere supplies surface forcing (radiation, near-surface temperature/humidity,
# precipitation) while Terrarium evolves the soil energy, water, and carbon state and
# returns surface fluxes — a genuinely two-way coupling. Here we run a higher-resolution
# experiment over a realistic Earth land–sea mask, following the
# [Terrarium coupling guide](https://speedyweather.github.io/SpeedyWeatherDocumentation/dev/land#Terrarium-coupling)
# in the SpeedyWeather docs. (Full example:
# [`examples/simulations/speedy_wet_land.jl`](https://github.com/NumericalEarth/Terrarium.jl/blob/main/examples/simulations/speedy_wet_land.jl).)

import SpeedyWeather

## Higher-resolution shared grid: a full Gaussian grid with 64 latitude rings
ring_grid = RingGrids.FullGaussianGrid(32)
spectral_grid = SpeedyWeather.SpectralGrid(ring_grid)

## Realistic Earth land–sea mask; Terrarium columns are placed on land points only
land_sea_mask = SpeedyWeather.EarthLandSeaMask(spectral_grid)
SpeedyWeather.load_mask!(land_sea_mask)
column_grid = ColumnRingGrid(CPU(), Float32, ExponentialSpacing(N = 15, Δz_min = 0.05), ring_grid, land_sea_mask.mask .> 0)

## Build the Terrarium land model (soil only here; vegetation = nothing) and wrap it
soil = SoilEnergyWaterCarbon(eltype(column_grid); hydrology = SoilHydrology(eltype(column_grid)))
terrarium_model = LandModel(column_grid; initializer = SoilInitializer(eltype(column_grid)), vegetation = nothing, soil)
land = SpeedyWeather.LandModel(spectral_grid, terrarium_model; timestepper = ForwardEuler(eltype(column_grid)), Δt = 300.0)

# The land component plugs into a full primitive-equation atmosphere from SpeedyWeather.jl,
# sharing the same land–sea mask:

model_coupled = SpeedyWeather.PrimitiveWetModel(
    spectral_grid;
    land,
    land_sea_mask,
    surface_heat_flux = SpeedyWeather.SurfaceHeatFlux(spectral_grid, land = SpeedyWeather.PrescribedLandHeatFlux()),
    surface_humidity_flux = SpeedyWeather.SurfaceHumidityFlux(spectral_grid, land = SpeedyWeather.PrescribedLandHumidityFlux()),
    time_stepping = SpeedyWeather.Leapfrog(spectral_grid, Δt_at_T31 = Minute(15)),
)
simulation = SpeedyWeather.initialize!(model_coupled)

# We advance the coupled model in short chunks, capturing a snapshot at each step of the
# **surface soil moisture** (top-layer saturation — a Terrarium land variable, living on the
# masked land grid) and the **surface relative vorticity** (lowest atmospheric layer — a
# SpeedyWeather diagnostic). The masked land field is mapped back onto the full ring grid
# with `RingGrids.Field`; ocean points become `NaN` and plot as blank.

using CairoMakie

nframes = 20
moisture_frames, vorticity_frames = Matrix{Float32}[], Matrix{Float32}[]
for _ in 1:nframes
    SpeedyWeather.run!(simulation, period = Hour(6))
    land_state = simulation.variables.prognostic.land.terrarium
    sat_top = interior(land_state.saturation_water_ice)[:, :, end:end]              # surface saturation
    ## map the masked land columns back onto the full ring grid; ocean points come back as NaN
    push!(moisture_frames, Float32.(Matrix(RingGrids.Field(CPU(), sat_top, column_grid)[:, 1, 1])))
    push!(vorticity_frames, Float32.(Matrix(simulation.variables.grid.vorticity[:, end])))  # lowest layer
end

## stack the per-step snapshots into (lon, lat, time) arrays for animation
moisture, vorticity = stack(moisture_frames), stack(vorticity_frames)

# Longitude/latitude axes shared by both panels, read from the full ring grid:
ζ_ref = simulation.variables.grid.vorticity[:, end]
lond, latd = RingGrids.get_lond(ζ_ref), RingGrids.get_latd(ζ_ref)

# We reuse the `animate_field` helper from the day-1 tutorial
# (`01_intro_interactive_climate.jl`): it wraps the time index in an `Observable`, builds a
# heatmap from it, and then `record`s the frames one by one. (Generalized here with optional
# `colormap`/`colorrange`; the defaults reproduce day 1's symmetric `:balance` styling.)
function animate_field(data, filename; lon = axes(data, 1), lat = axes(data, 2),
                       label = "", title = "", time_steps = 1:size(data, 3), framerate = 10,
                       colormap = :balance, colorrange = (-maximum(abs, data), maximum(abs, data)))
    ## this defines the iterator that will update the animation
    i_time = Observable(1)

    ## this is the array that is animated
    frame = @lift data[:, :, $i_time]

    fig, ax, hm = heatmap(lon, lat, frame; colormap, colorrange,
                          axis = (xlabel = "Longitude [˚E]", ylabel = "Latitude [˚N]"))
    Colorbar(fig[1, 2], hm; label)

    ## here, we do the actual animation:
    anim = CairoMakie.record(fig, filename, eachindex(time_steps); framerate) do t
        i_time[] = t
        ax.title = "$title, time step $(time_steps[t])"
    end

    return anim
end

# **Surface soil moisture** over land (ocean masked out). Saturation lies in [0, 1], so we pass
# a sequential colormap and range rather than the symmetric default:

animate_field(moisture, "soil_moisture.mp4"; lon = lond, lat = latd,
              label = "Surface soil moisture (saturation)", title = "Surface soil moisture",
              colormap = :viridis, colorrange = (0, 1))

# ![Surface soil moisture animation](soil_moisture.mp4)

# **Surface relative vorticity** of the coupled atmosphere — the synoptic weather systems
# evolving above the land. Here `animate_field`'s defaults (symmetric range, `:balance`) make
# the spatial structure clear regardless of units:

animate_field(vorticity, "vorticity.mp4"; lon = lond, lat = latd,
              label = "Surface relative vorticity", title = "Surface relative vorticity")

              
# ![Surface vorticity animation](vorticity.mp4)

# ## Where to go next: Terrarium's documentation
#
# - [Basic concepts](https://numericalearth.github.io/Terrarium.jl/dev/introduction/basic_concepts/) and the [Numerical core](https://numericalearth.github.io/Terrarium.jl/dev/introduction/numerical_core/) for the design of grids, fields, and kernels.
# - The [Soil](https://numericalearth.github.io/Terrarium.jl/dev/models/soil_model/) and [Land](https://numericalearth.github.io/Terrarium.jl/dev/models/land_model/) model pages for the full process catalog.
# - The [examples](https://github.com/NumericalEarth/Terrarium.jl/tree/main/examples) directory for runnable scripts, including differentiating a model end-to-end with Enzyme.jl.
