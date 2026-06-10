# # The differentiable ACC: adjoint sensitivities of the Southern Ocean
#
# *Wednesday — differentiable Earth-system models.*
#
# The Antarctic Circumpolar Current (ACC) carries roughly 150 Sv of water around
# Antarctica. Its strength is set by a
# delicate balance between wind forcing, buoyancy fluxes, and mesoscale eddy transport.
# A central question in climate science is *which aspects of the ocean's initial state
# and surface forcing most strongly influence the ACC transport.*
#
# This tutorial answers that question using automatic differentiation (AD) through a
# full 3-D ocean model. Starting from the idealized ACC configuration of
# [Abernathey et al. (2011)](https://doi.org/10.1175/JPO-D-11-0142.1), we
#
# 1. spin up a re-entrant channel to a statistically equilibrated state,
# 2. compute the zonal volume transport through the channel as a scalar cost function,
# 3. differentiate that transport **backwards through the model time-stepping** to obtain
#    the gradient with respect to every grid-cell value of temperature, salinity,
#    wind stress, and heat flux — the **adjoint** sensitivities.
#
# We use [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) for the ocean
# dynamics, [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) to JIT-compile the
# time-stepping via XLA (the same compiler used by JAX), and
# [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for reverse-mode AD.

# ## Packages
#
# When running as a notebook, activate the bundled environment first.
# When running as a plain script, pass `--project=tutorials/day3` to Julia instead.
#nb import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: HorizontalFormulation

using SeawaterPolynomials

using Reactant
using Oceananigans.Architectures: ReactantState
using Enzyme

using Printf
using Statistics
using FileIO, JLD2
using CairoMakie

# ## Configuration
#
# ### Reactant back-end
#
# By default we use an NVIDIA GPU via Reactant's XLA backend. To run on CPU instead
# (slower, but no GPU required), change the backend:
#
# ```julia
# # Reactant.set_default_backend("cpu")
# ```

using CUDA
Reactant.set_default_backend("gpu")

# Set the default floating-point precision for all Oceananigans fields:

Oceananigans.defaults.FloatType = Float64

# ### Loop lengths
#
# These constants set the number of time steps in each compiled kernel.
# They **must** be constants because Reactant traces loops at compile time
# (the XLA program is unrolled, not interpreted). Increase them for production runs.

const Ntimesteps = 5    # steps in the AD pass      (production: 25+)
const Nspinup    = 5    # spinup steps               (production: 100+)

# ### Grid size
#
# Small defaults for interactive use. For a credible ACC the values used in the paper
# are Nx = 80, Ny = 160, Nz = 32.

const Nx = 20
const Ny = 40
const Nz = 8

const x_midpoint = Nx ÷ 2 + 1   # zonal index at which we evaluate the transport

# ### Output directory

output_dir = get(ENV, "CASE_OUTPUT_DIR", "differentiable_channel_output")
isdir(output_dir) || mkdir(output_dir)
nothing #hide

# ## Physical parameters
#
# The ACC lives at roughly 60°S. We place it on a β-plane with `f₀ < 0` (Southern
# Hemisphere) and a mild meridional gradient of the Coriolis parameter. The buoyancy
# and wind forcing parameters are taken directly from Abernathey et al. (2011).

const f = -1e-4           # [s⁻¹]  Coriolis at the southern boundary
const β =  1e-11          # [m⁻¹ s⁻¹]

const α  = 2e-4           # [K⁻¹]  thermal expansion coefficient
const g  = 9.8061         # [m s⁻²]
const cᵖ = 3994.0         # [J K⁻¹]
const ρ  = 999.8          # [kg m⁻³]

const Lx = 1000kilometers
const Ly = 2000kilometers

# ### Vertical grid
#
# We use a surface-intensified stretched grid, with grid cells expanding
# geometrically toward the sea floor:

k_center  = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lz  = sum(Δz_center)

z_faces   = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz + 1] = 0.0

Δz_col    = reshape(z_faces[2:end] - z_faces[1:end-1], 1, :)

# ### Forcing and sponge layer parameters

parameters = (
    Ly        = Ly,
    Lz        = Lz,
    Qᵀ        = 10 / (ρ * cᵖ),                    # temperature flux magnitude [K m s⁻¹]
    Qᵇ        = 10 / (ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]
    y_shutoff = 5 / 6 * Ly,                        # northern limit of heat flux
    τ         = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ         = 1 / 30days,                        # bottom drag damping time-scale [s⁻¹]
    ΔB        = 8 * α * g,                         # surface buoyancy gradient [s⁻²]
    ΔT        = 8.0,                               # surface temperature difference [K]
    H         = Lz,
    h         = 1000.0,                            # stratification decay scale [m]
    y_sponge  = 19 / 20 * Ly,                      # southern edge of sponge layer
    λt        = 7.0days,                           # relaxation time scale [s]
)
nothing #hide

# ## Grid
#
# The channel has a partial meridional barrier ("Drake Passage gap") to produce an
# ACC-like jet and barotropic transport. The barrier is a ridge spanning most of the
# meridional extent of the domain, with a gap at mid-latitudes:

function wall_function(x, y)
    zonal = (x > 470kilometers) && (x < 530kilometers)
    gap   = (y < 400kilometers) || (y > 1000kilometers)
    return (Lz + 1) * zonal * gap - Lz
end

const halo_size = 4

function make_grid(architecture, Nx, Ny, Nz, z_faces)
    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size     = (Nx, Ny, Nz),
        halo     = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, wall_function)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
end
nothing #hide

# ## Model
#
# We use `HydrostaticFreeSurfaceModel` — appropriate for the mesoscale-resolving
# (or permitting) regime of this channel. The key choices are:
#
# * **Free surface**: split-explicit scheme with 10 barotropic substeps per baroclinic step.
# * **Advection**: third-order WENO, which provides implicit upwind diffusion exactly
#   where the flow develops grid-scale gradients.
# * **Closure**: a horizontal scalar diffusivity (representing subgrid mesoscale effects)
#   plus a vertical diffusivity with surface intensification, and a biharmonic horizontal
#   diffusivity to handle grid-scale noise.
# * **Tracers**: temperature `T`, salinity `S`, and TKE `e` for the CATKE vertical mixing.

function build_model(grid, Δt₀, parameters)

    # ---- Boundary conditions ----
    # Wind stress and heat flux are stored as 2-D fields so their values can be
    # set independently of the model and passed as initial conditions to the AD.
    temperature_flux_bc = FluxBoundaryCondition(Field{Center, Center, Nothing}(grid))
    u_stress_bc         = FluxBoundaryCondition(Field{Face,   Center, Nothing}(grid))
    v_stress_bc         = FluxBoundaryCondition(Field{Center, Face,   Nothing}(grid))

    @inline u_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * p.Lz * fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, fields, p) = @inbounds -p.μ * p.Lz * fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)
    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(top = v_stress_bc, bottom = v_drag_bc)

    # ---- Coriolis ----
    coriolis = BetaPlane(f₀ = f, β = β)

    # ---- Temperature relaxation forcing ----
    # A sponge layer at the southern boundary restores T to a prescribed profile,
    # preventing the channel from drifting away from the target stratification.
    @inline initial_temperature(z, p) =
        p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

    @inline function temperature_relaxation(i, j, k, grid, clock, fields, p)
        y        = ynode(j, grid, Center())
        z        = znode(k, grid, Center())
        target_T = initial_temperature(z, p)
        T        = @inbounds fields.T[i, j, k]
        return -1 / p.λt * mask(y, p) * (T - target_T)
    end

    FT = Forcing(temperature_relaxation, discrete_form = true, parameters = parameters)

    # ---- Diffusivities ----
    κh = 5e-5    # [m² s⁻¹] horizontal diffusivity
    νh = 500.0   # [m² s⁻¹] horizontal viscosity
    κz = 5e-5    # [m² s⁻¹] vertical diffusivity
    νz = 3e-3    # [m² s⁻¹] vertical viscosity

    # Surface-intensified vertical diffusivity:
    κz_field = Field{Center, Center, Center}(grid)
    κz_array = zeros(Nx, Ny, Nz)
    κz_add   = 5e-5
    for k in 1:Nz
        κz_array[:, :, k] .= κz + κz_add * exp(-(k - 1) / 5.0)
    end
    set!(κz_field, κz_array)

    horizontal_closure  = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    vertical_closure    = VerticalScalarDiffusivity(ν = νz, κ = κz_field)
    biharmonic_closure  = ScalarBiharmonicDiffusivity(HorizontalFormulation(),
                                                      Oceananigans.defaults.FloatType;
                                                      ν = 1e11)

    model = HydrostaticFreeSurfaceModel(
        grid;
        free_surface          = SplitExplicitFreeSurface(substeps = 10),
        momentum_advection    = WENO(order = 3),
        tracer_advection      = WENO(order = 3),
        buoyancy              = SeawaterBuoyancy(
                                    equation_of_state = LinearEquationOfState(
                                        Oceananigans.defaults.FloatType)),
        coriolis              = coriolis,
        closure               = (horizontal_closure, vertical_closure, biharmonic_closure),
        tracers               = (:T, :S, :e),
        boundary_conditions   = (T = T_bcs, u = u_bcs, v = v_bcs),
        forcing               = (T = FT,),
    )

    model.clock.last_Δt = Δt₀
    return model
end
nothing #hide

# ## Initial and boundary condition helpers
#
# Each of these returns an Oceananigans `Field`. Keeping them as `Field` objects (rather
# than plain arrays) is required for the Enzyme AD pass: Enzyme differentiates through
# the memory layout of `Field` directly.

function T_flux_init(grid, p)
    @inline temp_flux_fn(x, y) = ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    temp_flux = Field{Center, Center, Nothing}(grid)
    set!(temp_flux, temp_flux_fn)
    return temp_flux
end

function u_wind_stress_init(grid, p)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    set!(wind_stress, u_stress)
    return wind_stress
end

function v_wind_stress_init(grid, p)
    wind_stress = Field{Center, Face, Nothing}(grid)
    set!(wind_stress, 0)
    return wind_stress
end

function temperature_salinity_init(grid, parameters)
    ε(σ)           = σ * randn()
    Tᵢ_fn(x, y, z) = (parameters.ΔT
                       * (exp(z / parameters.h) - exp(-Lz / parameters.h))
                       / (1 - exp(-Lz / parameters.h))
                       + ε(1e-8))
    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)
    set!(Tᵢ, Tᵢ_fn)
    set!(Sᵢ, 35.0)
    return Tᵢ, Sᵢ
end
nothing #hide

# ## Reactant-compiled kernels
#
# Reactant's `@compile` macro traces through the Julia code and emits an XLA program.
# This program is then JIT-compiled by the XLA runtime (LLVM on CPU, ptxas on GPU) the
# first time it is called — a one-time cost amortised over many subsequent calls.
#
# We compile two kernels separately:
#
# 1. **`spinup_loop!`** — just the forward time-stepping, unrolled for `Nspinup` steps.
# 2. **`differentiate_tracer_error`** — the full forward + reverse (adjoint) pass,
#    unrolled for `Ntimesteps` steps.
#
# Reverse-mode AD unrolls the forward tape **and** the corresponding adjoint operations,
# so the compiled program is ``O(N_\text{timesteps})`` in length.

function spinup_loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true track_numbers = false for i = 1:Nspinup
        time_step!(model, Δt)
    end
    return nothing
end

function spinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)
    model.clock.iteration = 0
    model.clock.time      = 0
    spinup_loop!(model)
    return nothing
end

# ### The cost function: zonal volume transport
#
# The zonal transport through the mid-channel section (in Sverdrups) is our scalar
# objective ``J``. It weights the zonal velocity by the cell face area
# (``\Delta y \cdot \Delta z``):

function estimate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld)
    run_reentrant_channel_model!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    Nx, Ny, Nz = size(model.grid)
    zonal_transport = (model.velocities.u[x_midpoint, 1:Ny, 1:Nz]
                       .* model.grid.Δyᵃᶜᵃ) .* Δz
    return sum(zonal_transport) / 1e6  # Sverdrups
end

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true checkpointing = true track_numbers = false for i = 1:Ntimesteps
        time_step!(model, Δt)
    end
    return nothing
end

function run_reentrant_channel_model!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)
    model.clock.iteration = 0
    model.clock.time      = 0
    loop!(model)
    return nothing
end

# ### Enzyme wrapper
#
# `differentiate_tracer_error` wraps the cost function in an Enzyme
# `autodiff(..., ReverseWithPrimal, ...)` call. `Duplicated(x, dx)` pairs each primal
# input with its "shadow" accumulator: after the call, `dx` holds ``\partial J / \partial x``.

function differentiate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress,
                                    temp_flux, Δz, mld,
                                    dmodel, dTᵢ, dSᵢ, du_wind_stress, dv_wind_stress,
                                    dtemp_flux, dΔz, dmld)
    return autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model,         dmodel),
                    Duplicated(Tᵢ,           dTᵢ),
                    Duplicated(Sᵢ,           dSᵢ),
                    Duplicated(u_wind_stress, du_wind_stress),
                    Duplicated(v_wind_stress, dv_wind_stress),
                    Duplicated(temp_flux,     dtemp_flux),
                    Duplicated(Δz,           dΔz),
                    Duplicated(mld,          dmld))
end
nothing #hide

# ## Build the model and initial conditions

Δt₀ = 2.5minutes

architecture   = ReactantState()
grid           = make_grid(architecture, Nx, Ny, Nz, z_faces)
model          = build_model(grid, Δt₀, parameters)

T_flux        = T_flux_init(model.grid, parameters)
u_wind_stress = u_wind_stress_init(model.grid, parameters)
v_wind_stress = v_wind_stress_init(model.grid, parameters)
Tᵢ, Sᵢ       = temperature_salinity_init(model.grid, parameters)
mld           = Field{Center, Center, Nothing}(model.grid)
Δz            = Reactant.ConcreteRArray(reshape(Δz_col, :))

# Shadow (gradient accumulator) fields — initialized to zero:
dmodel         = Enzyme.make_zero(model)
dTᵢ            = Field{Center, Center, Center}(model.grid)
dSᵢ            = Field{Center, Center, Center}(model.grid)
du_wind_stress = Field{Face,   Center, Nothing}(model.grid)
dv_wind_stress = Field{Center, Face,   Nothing}(model.grid)
dT_flux        = Field{Center, Center, Nothing}(model.grid)
dmld           = Field{Center, Center, Nothing}(model.grid)
dΔz            = Enzyme.make_zero(Δz)

@info "Built $(summary(model))"
nothing #hide

# ### Channel geometry
#
# The partial meridional barrier mimicking Drake Passage is visible as a ridge spanning
# most of the domain, with a gap at mid-latitudes:

_xc, _yc, _ = nodes(grid, Center(), Center(), Center())
bh_arr = convert(Array, interior(model.grid.immersed_boundary.bottom_height))[:, :, 1]

fig_topo, ax_topo, hm_topo = heatmap(collect(_xc) .* 1e-3, collect(_yc) .* 1e-3, bh_arr;
    colormap = :deep,
    axis = (xlabel = "x (km)", ylabel = "y (km)", title = "Channel bottom depth"))
Colorbar(fig_topo[1, 2], hm_topo, label = "m")
save(joinpath(output_dir, "bottom_topography.png"), fig_topo)
nothing #hide

# ![](differentiable_channel_output/bottom_topography.png)

# ## Compilation
#
# Compiling the two kernels takes most of the wall time the first time the notebook is
# run. Subsequent calls are fast (only the XLA runtime overhead):

@info "Compiling spinup kernel…"
compile_tic = time()
rspinup! = @compile raise_first = true raise = true sync = true  spinup_reentrant_channel_model!(
    model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux)

@info "Compiling AD kernel…"
rdifferentiate! = @compile raise_first = true raise = true sync = true  differentiate_tracer_error(
    model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux, Δz, mld,
    dmodel, dTᵢ, dSᵢ, du_wind_stress, dv_wind_stress, dT_flux, dΔz, dmld)

@info @sprintf("Compilation done in %.1f s", time() - compile_tic)
nothing #hide

# ## Spinup
#
# Run the model forward for `Nspinup` time steps to move away from the cold
# (analytically stratified, zero-velocity) initial condition. We then copy the
# spun-up temperature and salinity back into `Tᵢ` and `Sᵢ` so the AD pass starts
# from a physical state.

@info "Running spinup…"
spinup_tic = time()
rspinup!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux)
set!(Tᵢ, model.tracers.T)
set!(Sᵢ, model.tracers.S)
@info @sprintf("Spinup done in %.1f s", time() - spinup_tic)
nothing #hide

# ### Spun-up state
#
# Surface temperature and sea-surface height after spinup — a sanity check before the
# more expensive AD pass:

T_surf = convert(Array, interior(model.tracers.T))[:, :, Nz]
η_surf = convert(Array, interior(model.free_surface.η))

fig_su = Figure(size = (1000, 450))
ax_su1 = Axis(fig_su[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "Surface T [°C]")
heatmap!(ax_su1, collect(_xc) .* 1e-3, collect(_yc) .* 1e-3, T_surf; colormap = :thermal)
ax_su2 = Axis(fig_su[1, 3], xlabel = "x (km)", ylabel = "y (km)", title = "SSH [m]")
heatmap!(ax_su2, collect(_xc) .* 1e-3, collect(_yc) .* 1e-3, η_surf; colormap = :balance)
save(joinpath(output_dir, "spinup_state.png"), fig_su)
nothing #hide

# ![](differentiable_channel_output/spinup_state.png)

# ## AD pass
#
# Run the AD pass. This simultaneously runs the forward model for `Ntimesteps` steps
# *and* the adjoint model that accumulates ``\partial J / \partial \theta`` for every
# input ``\theta``. The primal value (the transport itself) is returned alongside the
# gradients:

@info "Running AD pass…"
ad_tic = time()
result = rdifferentiate!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux, Δz, mld,
                         dmodel, dTᵢ, dSᵢ, du_wind_stress, dv_wind_stress, dT_flux, dΔz, dmld)
zonal_transport = convert(Float64, result[2])
@info @sprintf("AD pass done in %.1f s — J = %.4f Sv", time() - ad_tic, zonal_transport)
nothing #hide

# ## Save results
#
# We save the model state and all gradient fields to JLD2 so the visualization cells
# below can be rerun without re-running the simulation.

# Get grid node coordinates for plotting:
xc, yc, zc = nodes(grid, Center(), Center(), Center())
xu, yu, _   = nodes(grid, Face(),   Center(), Center())
xv, yv, _   = nodes(grid, Center(), Face(),   Center())
xζ, yζ, _   = nodes(grid, Face(),   Face(),   Center())
zw          = nodes(grid, Center(), Center(), Face())[3]

bottom_height = if isa(model.grid, ImmersedBoundaryGrid)
    model.grid.immersed_boundary.bottom_height
else
    bf = Field{Center, Center, Nothing}(model.grid)
    set!(bf, -Lz)
    bf
end

landmask = convert(Array, interior(bottom_height))[:, :, 1] .> -1e-4

# Shallow-copy arrays out of the model for saving:
T_final = convert(Array, interior(model.tracers.T))
u_final = convert(Array, interior(model.velocities.u))
v_final = convert(Array, interior(model.velocities.v))
ssh     = convert(Array, interior(model.free_surface.η))

dT             = convert(Array, interior(dTᵢ))
dS             = convert(Array, interior(dSᵢ))
du_ws          = convert(Array, interior(du_wind_stress))
dv_ws          = convert(Array, interior(dv_wind_stress))
dT_flux_arr    = convert(Array, interior(dT_flux))

jldsave(joinpath(output_dir, "channel_results.jld2");
        Nx, Ny, Nz,
        zonal_transport,
        T_final, u_final, v_final, ssh,
        dT, dS, du_ws, dv_ws, dT_flux_arr,
        landmask,
        xc = collect(xc), yc = collect(yc), zc = collect(zc),
        xu = collect(xu), yu = collect(yu),
        xv = collect(xv), yv = collect(yv),
        zw = collect(zw))

@info "Results saved to $(output_dir)/channel_results.jld2"
nothing #hide

# ## Visualization
#
# Load the saved data (allows rerunning this section independently):

data = jldopen(joinpath(output_dir, "channel_results.jld2"), "r")

Nx′  = data["Nx"];   Ny′ = data["Ny"];   Nz′ = data["Nz"]
xc′  = data["xc"];   yc′ = data["yc"]
xu′  = data["xu"];   yu′ = data["yu"]
xv′  = data["xv"];   yv′ = data["yv"]
zc′  = data["zc"];   zw′ = data["zw"]
lm   = data["landmask"]

T_f  = data["T_final"]
u_f  = data["u_final"]
ssh′ = data["ssh"]
dT′  = data["dT"]
du′  = data["du_ws"]
dv′  = data["dv_ws"]

J    = data["zonal_transport"]
close(data)
nothing #hide

# Helper: mask land cells with NaN before plotting:
function apply_mask(field, mask)
    masked = copy(field)
    masked[mask] .= NaN
    return masked
end

# ### Surface temperature and SSH after spinup
#
# A quick sanity check: the temperature field should show a stratified channel with
# the topographic wall visible as a blank strip:

k_surf = Nz′   # top grid cell index

lm_u = lm
lm_v = falses(Nx′, Ny′ + 1)
lm_v[:, 2:Ny′] = lm[:, 1:Ny′-1] .| lm[:, 2:Ny′]
lm_v[:, 1]     = lm[:, 1]
lm_v[:, Ny′+1] = lm[:, Ny′]

fig1, ax1, hm1 = heatmap(xc′ .* 1e-3, yc′ .* 1e-3,
                          apply_mask(T_f[:, :, k_surf], lm);
                          colormap = :thermal,
                          nan_color = :gray70,
                          axis = (xlabel = "x (km)", ylabel = "y (km)",
                                  title = @sprintf("Surface temperature  [°C]  (J = %.3f Sv)", J)))
Colorbar(fig1[1, 2], hm1, label = "T [°C]")

save(joinpath(output_dir, "surface_temperature.png"), fig1)
nothing #hide

# ![](differentiable_channel_output/surface_temperature.png)

# ### Adjoint sensitivity: ``\partial J / \partial T``
#
# Each panel is a horizontal slice of the temperature adjoint at the surface and at
# mid-depth. A positive value means "warming the water at this location would increase
# the ACC transport"; negative means the opposite.

k_mid = max(1, Nz′ ÷ 2)

function symmetric_colorrange(arr, mask; scale = 0.8)
    v = abs.(arr[.!mask])
    isempty(v) && return (-1.0, 1.0)
    m = scale * maximum(v)
    m == 0 && return (-1.0, 1.0)
    return (-m, m)
end

fig2 = Figure(size = (1200, 550))

ax2a = Axis(fig2[1, 1], xlabel = "x (km)", ylabel = "y (km)",
            title = @sprintf("∂J/∂T(z ≈ %.0f m)  [Sv °C⁻¹]", zc′[k_surf]))
cr_surf = symmetric_colorrange(dT′[:, :, k_surf], lm)
hm2a = heatmap!(ax2a, xc′ .* 1e-3, yc′ .* 1e-3,
                apply_mask(dT′[:, :, k_surf], lm);
                colormap = :balance, colorrange = cr_surf, nan_color = :gray70)
Colorbar(fig2[1, 2], hm2a, label = "Sv °C⁻¹")

ax2b = Axis(fig2[1, 3], xlabel = "x (km)", ylabel = "y (km)",
            title = @sprintf("∂J/∂T(z ≈ %.0f m)  [Sv °C⁻¹]", zc′[k_mid]))
cr_mid = symmetric_colorrange(dT′[:, :, k_mid], lm)
hm2b = heatmap!(ax2b, xc′ .* 1e-3, yc′ .* 1e-3,
                apply_mask(dT′[:, :, k_mid], lm);
                colormap = :balance, colorrange = cr_mid, nan_color = :gray70)
Colorbar(fig2[1, 4], hm2b, label = "Sv °C⁻¹")

save(joinpath(output_dir, "adj_dT.png"), fig2)
nothing #hide

# ![](differentiable_channel_output/adj_dT.png)

# ### Adjoint sensitivity: wind stress

fig3 = Figure(size = (1200, 550))

ax3a = Axis(fig3[1, 1], xlabel = "x (km)", ylabel = "y (km)",
            title = "∂J/∂τₓ  [Sv (m² s⁻²)⁻¹]")
cr_u = symmetric_colorrange(du′[:, :, 1], lm_u)
hm3a = heatmap!(ax3a, xu′ .* 1e-3, yu′ .* 1e-3,
                apply_mask(du′[:, :, 1], lm_u);
                colormap = :balance, colorrange = cr_u, nan_color = :gray70)
Colorbar(fig3[1, 2], hm3a, label = "Sv / (m² s⁻²)")

ax3b = Axis(fig3[1, 3], xlabel = "x (km)", ylabel = "y (km)",
            title = "∂J/∂τᵧ  [Sv (m² s⁻²)⁻¹]")
cr_v = symmetric_colorrange(dv′[:, :, 1], lm_v)
hm3b = heatmap!(ax3b, xv′ .* 1e-3, yv′ .* 1e-3,
                apply_mask(dv′[:, :, 1], lm_v);
                colormap = :balance, colorrange = cr_v, nan_color = :gray70)
Colorbar(fig3[1, 4], hm3b, label = "Sv / (m² s⁻²)")

save(joinpath(output_dir, "adj_windstress.png"), fig3)
nothing #hide

# ![](differentiable_channel_output/adj_windstress.png)

# ### Summary figure
#
# A single four-panel figure — model state and all three adjoint fields — is saved as
# `differentiable_esms.png` for the workshop run-status dashboard:

fig_summary = Figure(size = (1400, 1100))

ax_T  = Axis(fig_summary[1, 1], xlabel = "x (km)", ylabel = "y (km)",
             title = @sprintf("Surface T [°C]  (J = %.3f Sv)", J))
heatmap!(ax_T, xc′ .* 1e-3, yc′ .* 1e-3,
         apply_mask(T_f[:, :, k_surf], lm);
         colormap = :thermal, nan_color = :gray70)

ax_dT = Axis(fig_summary[1, 3], xlabel = "x (km)", ylabel = "y (km)",
             title = @sprintf("∂J/∂T(z ≈ %.0f m)  [Sv °C⁻¹]", zc′[k_surf]))
heatmap!(ax_dT, xc′ .* 1e-3, yc′ .* 1e-3,
         apply_mask(dT′[:, :, k_surf], lm);
         colormap = :balance, colorrange = cr_surf, nan_color = :gray70)

ax_du = Axis(fig_summary[2, 1], xlabel = "x (km)", ylabel = "y (km)",
             title = "∂J/∂τₓ  [Sv (m² s⁻²)⁻¹]")
heatmap!(ax_du, xu′ .* 1e-3, yu′ .* 1e-3,
         apply_mask(du′[:, :, 1], lm_u);
         colormap = :balance, colorrange = cr_u, nan_color = :gray70)

ax_dv = Axis(fig_summary[2, 3], xlabel = "x (km)", ylabel = "y (km)",
             title = "∂J/∂τᵧ  [Sv (m² s⁻²)⁻¹]")
heatmap!(ax_dv, xv′ .* 1e-3, yv′ .* 1e-3,
         apply_mask(dv′[:, :, 1], lm_v);
         colormap = :balance, colorrange = cr_v, nan_color = :gray70)

save(joinpath(output_dir, "differentiable_esms.png"), fig_summary)
nothing #hide

# ## Interpretation
#
# The temperature adjoint ``\partial J / \partial T`` is the **adjoint forcing** in
# the language of data assimilation: it tells you, linearized around the current
# trajectory, which observations of temperature would most constrain a prediction of
# the ACC transport. Regions with large sensitivity are prime targets for ARGO float
# deployment or mooring placement. In the Southern Ocean, the largest signals tend to
# cluster near the Drake Passage gap — the dynamical "bottleneck" through which all
# the ACC transport must pass.
#
# ## Things to try
#
# !!! tip "Resolution"
#     Increase `Nx`, `Ny`, `Nz` (and `Ntimesteps`, `Nspinup`) to approach the
#     original Abernathey et al. (2011) resolution (80 × 160 × 32). On a GPU this
#     takes minutes rather than hours.
#
# !!! tip "Different cost functions"
#     Replace `sum(zonal_transport)` with the transport squared (a measure of
#     variance), or with the mean temperature at a particular depth. The adjoint
#     pattern changes qualitatively — a useful exercise in understanding what "sensitivity"
#     means for different objectives.
#
# !!! tip "Gradient-based optimization"
#     Use `dT′` as a search direction and nudge `Tᵢ` by a small step in the gradient
#     direction. Re-run the AD pass and check that the transport increased. You have
#     just performed one step of gradient descent on the ocean initial conditions —
#     the core of 4D-Var data assimilation.

nothing #hide
