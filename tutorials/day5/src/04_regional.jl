# # The Barents Sea with Biogeochemistry: Nutrients, Detritus, and Carbon
#
# *Extending the regional Barents Sea simulation with an
# implicit-productivity biogeochemical model.*
#
# Having established a physical foundation for the **Barents Sea** with
# coupled ocean–sea ice dynamics, we now turn to its thriving and rapidly
# shifting ecosystem. 
#
# In this tutorial, we extend our realistic regional setup by layering a
# biogeochemical (BGC) model atop the circulation. Following the classical,
# lightweight design found in the MITgcm `global_oce_biogeo` package, we
# implement an *implicit productivity* scheme. Rather than explicitly
# simulating phytoplankton and zooplankton populations, biological production
# is parameterized as a function of available light, temperature, and
# nutrients. This allows us to frugally track the down-gradient lifecycle of
# ecosystems using just three essential tracer families:
# 1. **Nutrients** (PO₄, NO₃, Fe), which constrain biological uptake in the euphotic zone.
# 2. **Detritus** (POP, DOP), which captures the vertical export, gravitational sinking,
# and remineralization of organic matter.
# 3. **Carbon Tracers** (DIC and Alkalinity), enabling us to diagnose air–sea
# $CO_2$ fluxes and regional acidification trends.
#
# By leveraging [NumericalEarth.jl](https://github.com/NumericalEarth/NumericalEarth.jl),
# adding these biogeochemical tracers requires no heavy architectural
# rewrites. The BGC equations are injected directly into the `ocean_simulation`
#
# !!! warning "Computational overhead"
#     While implicit productivity avoids the stiff equations and computational
#     weight of explicit multi-trophic ecosystems, tracking multiple passive
#     tracers (NO₃, PO₄, Fe, DOP, POP, DIC, ALK) significantly increases the cost
#     (mainly because of advection)
using Pkg; Pkg.activate("..")
using NumericalEarth, Oceananigans, Oceananigans.Units
using Oceananigans.BoundaryConditions: Radiation, FlatherBoundaryCondition, NormalFlowBoundaryCondition
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.Units: Time
using Dates, CUDA, Printf
using CopernicusMarine   # enables the GLORYS download extension
using OceanBioME
using GlobalOceanBioME
using CairoMakie

arch = GPU()

# Global/regional runs of OceanBioME are still in their infancy so we have to include some
# hacky thing that will eventually find their way into NumericalEarth or OceanBioME
include("../src/hacks.jl")

const λ₁, λ₂ =  5, 60
const φ₁, φ₂ = 63, 78

Nx = 8 * (λ₂ - λ₁)
Ny = 8 * (φ₂ - φ₁)
Nz = 40

depth = 4000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        longitude = (λ₁, λ₂),
                                        latitude = (φ₁, φ₂),
                                        z,
                                        halo = (7, 7, 7))

dir_kw = haskey(ENV, "DATA_DIR") ? (; dir = ENV["DATA_DIR"]) : (;)
bathymetry = Metadatum(:bottom_height; dataset = ETOPO2022(), dir_kw...)
bottom_height = regrid_bathymetry(underlying_grid, bathymetry;
                                  minimum_depth = 15,
                                  interpolation_passes = 25,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height); active_cells_map = true)

atmosphere    = JRA55PrescribedAtmosphere(arch; dir_kw...)
radiation     = JRA55PrescribedRadiation(arch; dir_kw...)
land          = JRA55PrescribedLand(arch; dir_kw...)

# Here we have to load external fields for all of the biogeochemical tracers which currently come 
# from ECCO rather than GLORYS, but that is in the works...
dates   = DateTime(1993, 1, 1) : Day(1) : DateTime(1993, 3, 1)
dataset = GLORYSDaily()
region  = BoundingBox(longitude=(0, 80), latitude=(55, 85))

bgc_dataset = ECCO4DarwinMonthly()
bgc_dates   = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 3, 1)

bgc_kwargs = (dataset = bgc_dataset, dates = bgc_dates, region)

Tᵉˣᵗ   = FieldTimeSeries(Metadata(:temperature;  dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
Sᵉˣᵗ   = FieldTimeSeries(Metadata(:salinity;     dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
uᵉˣᵗ   = FieldTimeSeries(Metadata(:u_velocity;   dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
vᵉˣᵗ   = FieldTimeSeries(Metadata(:v_velocity;   dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))
ηᵉˣᵗ   = FieldTimeSeries(Metadata(:free_surface; dates, dataset, region, dir_kw...), grid, inpainting=100, time_indices_in_memory=length(dates))

NO₃ᵉˣᵗ = FieldTimeSeries(Metadata(:nitrate;  bgc_kwargs..., dir_kw...), grid, inpainting=100, time_indices_in_memory=length(bgc_dates))
PO₄ᵉˣᵗ = FieldTimeSeries(Metadata(:phosphate;  bgc_kwargs..., dir_kw...), grid, inpainting=100, time_indices_in_memory=length(bgc_dates))
Feᵉˣᵗ  = FieldTimeSeries(Metadata(:dissolved_iron;  bgc_kwargs..., dir_kw...), grid, inpainting=100, time_indices_in_memory=length(bgc_dates))
DICᵉˣᵗ = FieldTimeSeries(Metadata(:dissolved_inorganic_carbon;  bgc_kwargs..., dir_kw...), grid, inpainting=100, time_indices_in_memory=length(bgc_dates))
Alkᵉˣᵗ = FieldTimeSeries(Metadata(:alkalinity;  bgc_kwargs..., dir_kw...), grid, inpainting=100, time_indices_in_memory=length(bgc_dates))

@inline  west_obc(j, k, grid, clock, fields, φ) = @inbounds φ[1,           j, k, Time(clock.time)]
@inline  east_obc(j, k, grid, clock, fields, φ) = @inbounds φ[grid.Nx,     j, k, Time(clock.time)]
@inline north_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny,     k, Time(clock.time)]

@inline  east_u_obc(j, k, grid, clock, fields, φ) = @inbounds φ[grid.Nx+1, j, k, Time(clock.time)]
@inline north_v_obc(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny+1, k, Time(clock.time)]

u_obcs = FieldBoundaryConditions(
    west = NormalFlowBoundaryCondition(west_obc,   discrete_form = true, parameters = uᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east = NormalFlowBoundaryCondition(east_u_obc, discrete_form = true, parameters = uᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

v_obcs = FieldBoundaryConditions(
    north = NormalFlowBoundaryCondition(north_v_obc, discrete_form = true, parameters = vᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

T_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Tᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

S_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Sᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

zero_grad = GradientBoundaryCondition(0)

NO₃_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = NO₃ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = NO₃ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = NO₃ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

Fe_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Feᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Feᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Feᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

PO₄_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = PO₄ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = PO₄ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = PO₄ᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

# Here is our first difference from the physics setup, we need to setup the air-sea exchange
# carbon dioxide into the DIC pool. We follow the typical parametrisation:
# ```math
# F = k(u_{10}, T)(pCO_{2, w} - pCO_{2, a}),
# ```
# and OceanBioME takes care of solving the carbon chemistry to find $pCO_{2, w}$.
# When we have sea ice we need to modify this to:
# ```math
# F = (1-η)k(u_{10}, T)(pCO_{2, w} - pCO_{2, a}),
# ```
# where $\eta$ is the sea ice concentration. This is currently in the GlobalOceanBioME
# repo but will find its way into NumericlaEarth when I figure out a nice way todo it.
# So for now we construct the normal exchange, then mask it.
underlying_CO₂_flux = CarbonDioxideGasExchangeBoundaryCondition(; air_concentration = 357.21,
                                                                  wind_speed = wind_from_atmosphere(atmosphere))

surface_CO₂_exchange = GlobalOceanBioME.IceMaskedGasExchangeBoundaryCondition(underlying_CO₂_flux, Field{Center, Center, Nothing}(grid))

DIC_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = DICᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = DICᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = DICᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    top   = surface_CO₂_exchange)

Alk_obcs = FieldBoundaryConditions(
    west  = ValueBoundaryCondition(west_obc,  discrete_form = true, parameters = Alkᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    east  = ValueBoundaryCondition(east_obc,  discrete_form = true, parameters = Alkᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)),
    north = ValueBoundaryCondition(north_obc, discrete_form = true, parameters = Alkᵉˣᵗ, scheme = Radiation(inflow_timescale = 1days)))

@inline wetcell(i, j, k, grid, ℓx, ℓy, ℓz) =
    !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

@inline function vertical_integral(i, j, grid, u, t, Δz, ℓx, ℓy, ℓz)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = wetcell(i, j, k, grid, ℓx, ℓy, ℓz)
        U += ifelse(wet, u[i, j, k, t] * Δz(i, j, k, grid), zero(U))
    end
    return U
end

@inline function west_U_obc(j, k, grid, clock, fields, p)
    t = Time(clock.time)
    U = vertical_integral(1, j, grid, p.u, t, Δzᶠᶜᶜ, Face(), Center(), Center())
    return (U, @inbounds p.η[1, j, 1, t])
end

@inline function east_U_obc(j, k, grid, clock, fields, p)
    i = grid.Nx+1
    t = Time(clock.time)
    U = vertical_integral(i, j, grid, p.u, t, Δzᶠᶜᶜ, Face(), Center(), Center())
    return (U, @inbounds p.η[grid.Nx, j, 1, t])
end

@inline function north_V_obc(i, k, grid, clock, fields, p)
    j = grid.Ny+1
    t = Time(clock.time)
    V = vertical_integral(i, j, grid, p.v, t, Δzᶜᶠᶜ, Center(), Face(), Center())
    return (V, @inbounds p.η[i, j, 1, t])
end

U_obcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
    west = FlatherBoundaryCondition(west_U_obc, discrete_form = true, parameters = (u = uᵉˣᵗ, η = ηᵉˣᵗ)),
    east = FlatherBoundaryCondition(east_U_obc, discrete_form = true, parameters = (u = uᵉˣᵗ, η = ηᵉˣᵗ)))

V_obcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
    north = FlatherBoundaryCondition(north_V_obc, discrete_form = true, parameters = (v = vᵉˣᵗ, η = ηᵉˣᵗ)))

@inline rim(ξ, edge, width) = exp(-(ξ - edge)^2 / 2width^2)
@inline sponge_mask(λ, φ, z, t) = max(rim(λ, λ₁, 2), rim(λ, λ₂, 2), rim(φ, φ₂, 1))

# Here we setup the relaxation as before, but have to tell `DatasetRestoring` what we call
# the tracer, this interface will change soon though.
Fu = DatasetRestoring(Metadata(:u_velocity;  dates, dataset, region, dir_kw...), grid; rate = 1/20minutes, mask = sponge_mask, inpainting=100)
Fv = DatasetRestoring(Metadata(:v_velocity;  dates, dataset, region, dir_kw...), grid; rate = 1/20minutes, mask = sponge_mask, inpainting=100)
FT = DatasetRestoring(Metadata(:temperature; dates, dataset, region, dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
FS = DatasetRestoring(Metadata(:salinity;    dates, dataset, region, dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
FNO₃ = DatasetRestoring(Metadata(:nitrate;                    bgc_kwargs..., dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100, 
                        field_name = SingleNitrogen())
FFe  = DatasetRestoring(Metadata(:dissolved_iron;             bgc_kwargs..., dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
FPO₄ = DatasetRestoring(Metadata(:phosphate;                  bgc_kwargs..., dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
FDIC = DatasetRestoring(Metadata(:dissolved_inorganic_carbon; bgc_kwargs..., dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100)
FAlk = DatasetRestoring(Metadata(:alkalinity;                 bgc_kwargs..., dir_kw...), grid; rate = 1/1days,     mask = sponge_mask, inpainting=100, 
                        field_name = Alk_alinity())



closure = (CATKEVerticalDiffusivity(minimum_tke=1e-7), 
	       HorizontalScalarBiharmonicDiffusivity(ν = 5e8))
time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5)

# We construct the light model, because we don't have phytoplankton we don't need 
# to integrate like normal so just have a fixed coefficient (again, I will put this in 
# OceanBioME soon). This just sets:
# ```math
# PAR(z) = PAR(z=0)exp(kz),
# ```
# and then we mask with the ice again.
underlying_light_attenuation = fixed_attenuation_par_from_radiation(grid, radiation; PAR_fraction = 0.43, attenuation_coefficient = 0.3)
light_attenuation = GlobalOceanBioME.IceMaskedLightAttenuation(; underlying_light_attenuation, 
                                                                 ice_thickness = Field{Center, Center, Nothing}(grid), 
                                                                 ice_concentration = Field{Center, Center, Nothing}(grid))

# Finally we built the biogeochemistry and put it into the `ocean_simulation`:
biogeochemistry = ImplicitBiology(grid;
                                  light_attenuation,
                                  scale_negatives = true)

ocean = ocean_simulation(grid;
                         biogeochemistry,
                         free_surface = SplitExplicitFreeSurface(grid; substeps=80),
                         momentum_advection = WENOVectorInvariant(; order=5, time_discretization),
			             tracer_advection = WENO(; order = 5, time_discretization),
                         closure,
                         forcing = (T = FT, S = FS, u = Fu, v = Fv, N = FNO₃, Fe = FFe, PO₄ = FPO₄, DIC = FDIC, Alk = FAlk),
                         boundary_conditions = (u = u_obcs, v = v_obcs,
                                                T = T_obcs, S = S_obcs,
                                                U = U_obcs, V = V_obcs,
                                                N = NO₃_obcs, PO₄ = PO₄_obcs, 
                                                Fe = Fe_obcs, 
                                                DIC = DIC_obcs, Alk = Alk_obcs))

sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)

set!(ocean.model, T = Tᵉˣᵗ[1], S = Sᵉˣᵗ[1], 
                  N = NO₃ᵉˣᵗ[1], PO₄ = PO₄ᵉˣᵗ[1], Fe = Feᵉˣᵗ[1], 
                  DIC = DICᵉˣᵗ[1], Alk = Alkᵉˣᵗ[1])

set!(sea_ice.model, h = Metadata(:sea_ice_thickness;     dates, dataset=ECCO4Monthly(), dir_kw...)[1],
                    ℵ = Metadata(:sea_ice_concentration; dates, dataset=ECCO4Monthly(), dir_kw...)[1])


coupled_model = EarthSystemModel(; ocean, sea_ice, land, atmosphere, radiation)

simulation = Simulation(coupled_model; Δt = 6minutes, stop_time = 60days)

# we have to add this temporary hack so the bgc knows about the seaice...
add_callback!(simulation, pass_sea_ice_to_bgc!)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    sea_ice = sim.model.sea_ice
    T = ocean.model.tracers.T
    S = ocean.model.tracers.S
    h = sea_ice.model.ice_thickness
    DIC = ocean.model.tracers.DIC
    msg = @sprintf("time: %s, iter: %d, extrema(T, S): (%.1f, %.1f) °C (%.1f, %.1f) psu, max(h): %.2f m, DIC ∈ [%.1f, %.1f], wall: %s",
                   prettytime(sim), iteration(sim),
                   extrema(T)..., extrema(S)..., maximum(h),
                   extrema(DIC)...,
                   prettytime(1e-9 * (time_ns() - wall_time[])))
    @info msg
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# ## Output
#
# Daily surface fields from both components:

u, v, w = ocean.model.velocities
h = sea_ice.model.ice_thickness
ℵ = sea_ice.model.ice_concentration
𝒱 = @at((Center, Center, Center), sqrt(u^2 + v^2))
he = h * ℵ

# To diagnose the air-sea CO₂ exchange we can record the boundary condition usign `BoundaryConditionOperation`
CO₂_flux = BoundaryConditionOperation(ocean.model.tracers.DIC, :top, ocean.model)

ocean_outputs = merge(ocean.model.tracers, (; 𝒱, PAR = underlying_light_attenuation.field, CO₂_flux))

sea_ice_outputs = (; he)

suffix = "bgc"

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            filename = "barents_ocean_surface_$(suffix).jld2",
                                            indices = (:, :, grid.Nz-2),
                                            schedule = TimeInterval(0.5days),
                                            overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              filename = "barents_sea_ice_surface_$(suffix).jld2",
                                              schedule = TimeInterval(1days),
                                              overwrite_existing = true)

run!(simulation)

N    = FieldTimeSeries("barents_ocean_surface.jld2", "N")
Fe   = FieldTimeSeries("barents_ocean_surface.jld2", "Fe")
DIC  = FieldTimeSeries("barents_ocean_surface.jld2", "DIC")
qCO₂ = FieldTimeSeries("barents_ocean_surface.jld2", "CO₂_flux")

times = To.times
n = Observable(length(times))

title = @lift "Barents Sea — day " * string(round(Int, times[$n] / days))

   Nₙ = @lift(N[$n])
  Feₙ = @lift(Fe[$n])
 DICₙ = @lift(DIC[$n])
qCO₂ₙ = @lift(qCO₂[$n])

fig = Figure(size = (1200, 650))
fig[0, 1:4] = Label(fig, title, fontsize = 20, tellwidth = false)

ax = Axis(fig[1, 1], ylabel = "latitude [°N]")
hm_N = heatmap!(ax, Nₙ, colormap = Reverse(:bamako), colorrange = (0, 13), nan_color = :gray80)
ax = Axis(fig[1, 3])
hm_F = heatmap!(ax, Feₙ, colormap = Reverse(:lajolla), colorrange = (0, 0.001), nan_color = :gray80)
ax = Axis(fig[2, 1], xlabel = "longitude [°E]", ylabel = "latitude [°N]")
hm_D = heatmap!(ax, DICₙ, colormap = Reverse(:lipari), colorrange = (2050, 2250), nan_color = :gray80)
ax = Axis(fig[2, 3], xlabel = "longitude [°E]")
hm_q = heatmap!(ax, qCO₂ₙ, colormap = :balance, colorrange = (-0.0003, 0.0003), nan_color = :gray80)
Colorbar(fig[1, 2], hm_N, label = "Nitrogen [mmolN/m³]")
Colorbar(fig[1, 4], hm_F, label = "Iron [mmolFe/m³]")
Colorbar(fig[2, 2], hm_D, label = "Dissolved inorganic carbon [mmolC/m³]")
Colorbar(fig[2, 4], hm_q, label = "Air-sea CO₂ exchange [gC/m²/year]")

CairoMakie.record(fig, "barents_sea_bgc.mp4", 1:length(times), framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](barents_sea_bgc.mp4)

