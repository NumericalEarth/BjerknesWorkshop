
using NetCDF, Dates, Statistics

function load_ocean_color(path)
    possible_files = readdir(path)

    modis_files = [fname for fname in possible_files if ((length(fname) == 40) && (fname[1:10] == "AQUA_MODIS"))]

    dates = map(fname -> Date(parse(Int, fname[12:15]), 
                              parse(Int, fname[16:17]), 
                              parse(Int, fname[18:19])),
                modis_files)

    add_offset = ncgetatt(path * modis_files[1], "par", "add_offset")
    scale_factor = ncgetatt(path * modis_files[1], "par", "scale_factor")

    par = zeros(length(dates))

    for (n, fname) in enumerate(modis_files)
        raw_par = ncread(path * fname, "par")
        raw_par = Float64[par for par in raw_par if par != -32767]
        raw_par .*= scale_factor
        raw_par .+= add_offset

        par[n] = mean(raw_par) 
    end

    # mol m^-2 day^-1 -> Watts
    par ./= 0.394	

    return par, dates
end

import OceanBioME.Particles: 
    required_particle_fields,
    required_tracers,
    coupled_tracers,
    advect_particles!

@kwdef struct ReleaseIron{FT}
    rate :: FT = 1.0
end

required_particle_fields(::ReleaseIron) = ()
required_tracers(::ReleaseIron) = ()
coupled_tracers(::ReleaseIron) = (:Fe, )

@inline (release::ReleaseIron)(::Val{:Fe}, t) = release.rate

using JLD2

function load_particle_dataset(fname)
    f = jldopen(fname)

    its = keys(f["timeseries/t"])

    fields = nothing
end


@kwdef struct Whaleish{FT}
             grazing_rate :: FT = 1e-3
  grazing_half_saturation :: FT = 0.1
           excretion_rate :: FT = 1e-4
end

required_particle_fields(::Whaleish) = (:biomass, )
required_tracers(::Whaleish) = (:N, :Fe, :Z)
coupled_tracers(::Whaleish) = (:N, :Fe, :Z)

@inline function (whale::Whaleish)(::Val{:biomass}, t, biomass, N, Fe, Z)
    g = whale.grazing_rate
    K = whale.grazing_half_saturation
    ν = whale.excretion_rate

    grazing = g * Z / (Z + K)

    excretion = ν

    return grazing - excretion
end


@inline function (whale::Whaleish)(::Val{:Z}, t, biomass, N, Fe, Z)
    g = whale.grazing_rate
    K = whale.grazing_half_saturation

    grazing = g * Z / (Z + K)

    return -grazing 
end

@inline function (whale::Whaleish)(::Val{:N}, t, biomass, N, Fe, Z)
    ν = whale.excretion_rate

    excretion = ν

    return excretion
end

@inline function (whale::Whaleish)(::Val{:Fe}, t, biomass, N, Fe, Z)
    ν = whale.excretion_rate

    excretion = ν

    return excretion * 4.6375e-5
end

@kwdef struct SwimmingUpAndDown{FT}
           cycle_time :: FT = 90minutes
           dive_depth :: FT = 900.0
    horizontal_radius :: FT = 500000
end

function advect_particles!(swimming::SwimmingUpAndDown, particles, model, Δt)
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    arch = Oceananigans.Architectures.architecture(model)

    # Advect particles
    advect_particles_kernel! = _swim_particles!(Oceananigans.Architectures.device(arch), workgroup, worksize)

    advect_particles_kernel!(swimming, particles, model.grid, model.clock, Δt)

    return nothing
end

using KernelAbstractions: @index, @kernel

@kernel function _swim_particles!(swimming, particles, grid, clock, Δt)
    n = @index(Global)

    t = clock.time

    @inbounds begin
        particles.x[n] = grid.Lx/2 + swimming.horizontal_radius * cos(2π * t / 10days)
        particles.y[n] = swimming.horizontal_radius * sin(2π * t / 10days)
        particles.z[n] = min(0, swimming.dive_depth * sin(2π * t / swimming.cycle_time))
    end
end