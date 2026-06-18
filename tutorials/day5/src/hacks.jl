using Oceananigans.BoundaryConditions: getbc

import Base: getindex

struct SingleNitrogen end
struct Alk_alinity end

@inline Base.getindex(fields, i, j, k, ::SingleNitrogen) = @inbounds   fields.N[i, j, k]
@inline Base.getindex(fields, i, j, k, ::Alk_alinity)    = @inbounds fields.Alk[i, j, k]

import Oceananigans.Biogeochemistry: 
    biogeochemical_auxiliary_fields,
    update_biogeochemical_state!

using Adapt

import Adapt: adapt_structure

struct FixedAttenuationPAR{K, F}
      kfo :: K
    field :: F
end

Adapt.adapt_structure(to, PAR::FixedAttenuationPAR) = 
    FixedAttenuationPAR(nothing, 
                        adapt(to, PAR.field))

struct InterpolateTheFTS{S, F}
         source :: S
          field :: F
end

Adapt.adapt_structure(to, PAR::InterpolateTheFTS) = 
    (; field = adapt(to, PAR.field))


update_biogeochemical_state!(model, par::InterpolateTheFTS) =
    Oceananigans.Fields.interpolate!(par.field, par.source[Time(model.clock.time)])


@inline function fixed_attenuation_par(i, j, k, grid, surface_PAR, α, K)
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    PAR₀ = surface_PAR.field[i, j, grid.Nz]

    return α * PAR₀ * exp(z * K)
end

biogeochemical_auxiliary_fields(par::FixedAttenuationPAR) = (PAR = par.field, )

function fixed_attenuation_par_from_radiation(grid, radiation; PAR_fraction = 0.43, attenuation_coefficient = 0.3)
    surface_PAR = InterpolateTheFTS(radiation.downwelling_shortwave, Field{Center, Center, Nothing}(grid))

    kfo = KernelFunctionOperation{Center, Center, Center}(fixed_attenuation_par, grid, surface_PAR, PAR_fraction, attenuation_coefficient)

    field = CenterField(grid)

    set!(field, kfo)

    return FixedAttenuationPAR(kfo, field)
end

function Oceananigans.Biogeochemistry.update_biogeochemical_state!(model, PAR::FixedAttenuationPAR)
    update_biogeochemical_state!(model, PAR.kfo.arguments[2])
    set!(PAR.field, PAR.kfo)

    return nothing
end

function pass_sea_ice_to_bgc!(simulation)
    ocean      = simulation.model.ocean
    sea_ice    = simulation.model.sea_ice

    set!(ocean.model.tracers.DIC.boundary_conditions.top.condition.func.ice_concentration,
         sea_ice.model.ice_concentration)
    set!(ocean.model.biogeochemistry.light_attenuation.ice_concentration,
         sea_ice.model.ice_concentration)
    set!(ocean.model.biogeochemistry.light_attenuation.ice_thickness,
         sea_ice.model.ice_thickness)
         
    return nothing
end

nothing