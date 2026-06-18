using NetCDF
using Interpolations: linear_interpolation, Periodic # as of last checking this won't work on GPU

include("../src/00_tools.jl")


struct DownwellingShortWave{IR, LA, IN} <: Function
    incident_radiation :: IR
              latitude :: LA
    initial_day_number ::IN
end

@inline function (dsw::DownwellingShortWave)(t)
    L  = dsw.latitude
    N₀ = dsw.initial_day_number
    I₀ = dsw.incident_radiation(t)

    N = N₀ + floor(Int, t / 24hours)

    δ = 23.45 * sind(360 * (284 + N) / 365)

    h = t * 15 / hours

    α = 90 - asind(sind(L) * sind(δ) + cosd(L) * cosd(δ))#* cosd(h))

    αᵣ = asind(sind(α) / 1.431)

    ρ = 1/2 * ((sind(α-αᵣ)/sind(α+αᵣ))^2 / 2 + (tand(α-αᵣ)/tand(α+αᵣ))^2)

    return max(0, (1 - ρ)) * I₀
end

@inline (dsw::DownwellingShortWave)(x, y, t) = dsw(t)

data_path = ENV["DATA_DIR"]

# This could all be done with NumericalEarth now, but I already had these files etc.
data_times = ncread(data_path * "nitrate.nc", "time")[1:end-1] # seconds since 01/01/1970
data_times .-= data_times[1]
NO₃_data = reverse(ncread(data_path * "nitrate.nc", "no3")[1, 1, end-3:end-3, 1:end-1], dims = 1) # mmolN/m³
NO₃_plotting_data = reverse(ncread(data_path * "nitrate.nc", "no3")[1, 1, 1:end-3, 1:end-1], dims = 1) # mmolN/m³
DIC_data = reverse(ncread(data_path * "carbon.nc", "dissic")[1, 1, end-3:end-3, 1:end], dims = 1) * 1000 # mmolC/m³
Alk_data = reverse(ncread(data_path * "carbon.nc", "talk")[1, 1, end-3:end-3, 1:end], dims = 1) * 1000 # meq/m³
T_data = reverse(ncread(data_path * "temp.nc", "thetao")[1, 1, end-3:end-3, 1:end-1], dims = 1) # mmolN/m³
wind_data = reverse(ncread(data_path * "wind.nc", "se_model_speed")[1, 1, 1:end], dims = 1) # mmolN/m³
mld_data = -ncread(data_path * "mld.nc", "mlotst")[1, 1, 1:end-1]

# some NaNs in the wind observations
wind_times = data_times[isfinite.(wind_data)]
wind_data = wind_data[isfinite.(wind_data)]

NO₃_itp = linear_interpolation(data_times, mean(NO₃_data, dims = 1)[1, :], extrapolation_bc = Periodic())
DIC_itp = linear_interpolation(data_times, mean(DIC_data, dims = 1)[1, :], extrapolation_bc = Periodic())
Alk_itp = linear_interpolation(data_times, mean(Alk_data, dims = 1)[1, :], extrapolation_bc = Periodic())
T_itp = linear_interpolation(data_times, mean(T_data, dims = 1)[1, :], extrapolation_bc = Periodic())
wind_itp = linear_interpolation(wind_times, wind_data, extrapolation_bc = Periodic())
mld_itp = linear_interpolation(data_times, mld_data, extrapolation_bc = Periodic())

PAR, PAR_dates = load_ocean_color(data_path * "light/")
PAR_times = map(n->(PAR_dates[n] .- PAR_dates[1]).value * days, 1:length(PAR_dates))
PAR_itp = linear_interpolation(PAR_times, PAR, extrapolation_bc = Periodic())

P_obs = reverse(ncread(data_path*"phyto.nc", "phyc")[1, 1, 1:end-2, 1:end-1], dims = 1) ./ 6.56 # mmolC/m³ to mmolN/m³
P_obs_dt = DateTime(1970, 1, 1) .+ Second.(ncread(data_path*"/phyto.nc", "time")[1:end-1])

qCO₂_obs = ncread(data_path*"qco2.nc", "fgco2")[1, 1, :] # mol C / m² / yr -> kg CO₂ (eq) / m² / year
qCO₂_time = ncread(data_path*"qco2.nc", "time") 

qCO₂_time = qCO₂_time[isfinite.(qCO₂_obs)] 
qCO₂_obs = qCO₂_obs[isfinite.(qCO₂_obs)] 
qCO₂_obs_dt = DateTime(1970, 1, 1) .+ Second.(qCO₂_time)

plotting = (; NO₃_data = NO₃_plotting_data,
              data_times,
              P_obs,
              P_obs_dt,
              qCO₂_obs,
              qCO₂_obs_dt)

faeroe_data = (; surface_PAR = DownwellingShortWave(PAR_itp, 62, 151), 
                 NO₃_itp, DIC_itp, Alk_itp, T_itp, wind_itp, mld_itp,
                 plotting)