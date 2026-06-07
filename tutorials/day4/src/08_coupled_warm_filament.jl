# # A warm filament writes a cloud street: 3D coupled air–sea LES
#
# *Boundary heterogeneity writes turbulence into the fluid — now the boundary is a
# two-way-coupled, living ocean feature.*
#
# This is the **flagship coupled case**. A warm sea-surface-temperature (SST)
# filament — a ribbon of warm water a couple of kilometers wide, the kind shed by
# ocean fronts and eddies everywhere from the Gulf Stream to the Kuroshio — sits
# beneath a cooler marine atmosphere. The warm water drives an upward sensible and
# latent heat flux concentrated over the filament; the moistened, buoyant air rises,
# the mean wind shears the rising thermals into a **band of convection aligned with
# the wind — a cloud street** — and the condensing water vapor paints that band as a
# line of cloud directly above the warm water. Meanwhile the *same* fluxes act back
# on the ocean: surface cooling and the coupled wind stress erode the filament and
# deepen the oceanic mixed layer beneath it. Nothing about the surface forcing is
# prescribed — it is **computed every step from the instantaneous state of both
# fluids**.
#
# ## The science: an SST front imprints on the atmosphere
#
# Satellite and reanalysis studies have shown that mid-latitude SST fronts leave a
# clear fingerprint *in the atmosphere above them*: surface winds, cloud, and even
# rainfall organize along the warm flank of the front. Minobe et al. (2008,
# *Nature*) showed the Gulf Stream's SST front anchors a band of surface wind
# convergence, upward motion, and precipitation that reaches into the free
# troposphere — the ocean is steering the atmosphere, not merely responding to it.
# Small et al. (2008) review the mechanisms by which SST fronts drive air–sea
# interaction: the warm side destabilizes the marine boundary layer, increasing
# turbulent mixing, surface wind, and convection directly over the warm water. Our
# filament is a clean, idealized instance of exactly this: a warm anomaly that
# **organizes** the overlying turbulence into a wind-aligned convective band.
#
# **Cloud streets** are the visible signature: longitudinal convective rolls (and,
# over a localized heat source, a wind-aligned convective line) that form when
# buoyant surface heating combines with mean wind shear. Over a warm filament the
# heat source is the filament itself, so the cloud band traces the warm water
# downwind. This ties the Thursday theme together: the boundary writes turbulence
# into the fluid — and here the boundary is *alive*, evolving under the very fluxes
# it generates.
#
# This is a **3D, two-way-coupled** LES built on the **`EarthSystemModel`** interface
# from NumericalEarth: a Breeze `AtmosphereModel` stacked above an Oceananigans
# nonhydrostatic ocean, sharing one sea surface. The mean wind runs along `x` so the
# convective rolls / cloud street align with it; the filament is a warm band in `y`,
# centered in the domain, so the warm water lies *along* the wind and the cloud
# street forms over it.
#
# !!! warning "Untested coupling path"
#     `AtmosphereOceanModel` with a **nonhydrostatic** ocean (LES) is not yet
#     exercised by the upstream NumericalEarth test suite — the tested path uses a
#     `SlabOcean`. The construction below follows the patterns in
#     `NumericalEarth.jl/test/test_breeze_coupling.jl` and the 2D precursor
#     (`07_intro_coupled_convection.jl`) as closely as possible; treat the
#     nonhydrostatic-ocean coupling as experimental and verify on GPU.

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(1994)

config = RunConfig("08_coupled_warm_filament")
arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float64  # coupled: ESM clock is Float64, component grids must match
FT = Float64
nothing #hide

# ## Two grids that share a sea surface
#
# The atmosphere lives in `z ∈ [0, Lz_a]` above the interface; the ocean lives in
# `z ∈ [-Lz_o, 0]` below it. Both share the *same* horizontal extent (`Lx × Ly`) and
# the *same* horizontal cell count (`Nx × Ny`), so the coupler maps surface columns
# one-to-one across the interface. Both are doubly periodic in the horizontal and
# `Bounded` in `z`.
#
# The mean wind is along `x`; the warm filament is a band centered in `y`. A domain
# longer in `x` than `y` gives the wind-aligned cloud street room to develop
# downwind. This is a **modest validation grid** — 192 × 96 × 96 atmosphere cells and
# a 48-level ocean — sized to construct and step quickly. For a production rendering
# refine to e.g. 768 × 384 × 192 (atmosphere) / 768 × 384 × 96 (ocean) at
# ≈ 15–30 m horizontal resolution.

const Lx   = 12kilometers   # along-wind (cloud-street axis)
const Ly   = 6kilometers    # cross-wind (across the filament)
const Lz_a = 3kilometers    # atmosphere depth
const Lz_o = 120meters      # ocean depth

const Nx   = 192
const Ny   = 96
const Nz_a = 96             # atmosphere vertical cells
const Nz_o = 48             # ocean vertical cells

memory_report(Nx, Ny, Nz_a; FT, nfields = 8)

atmos_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz_a), halo = (5, 5, 5),
                             x = (0, Lx), y = (0, Ly), z = (0, Lz_a),
                             topology = (Periodic, Periodic, Bounded))

ocean_grid = RectilinearGrid(arch; size = (Nx, Ny, Nz_o), halo = (5, 5, 5),
                             x = (0, Lx), y = (0, Ly), z = (-Lz_o, 0),
                             topology = (Periodic, Periodic, Bounded))

# ## The atmosphere component
#
# `atmosphere_simulation` builds a Breeze `AtmosphereModel` wrapped in a `Simulation`,
# **pre-wired for coupling**: its bottom boundary conditions on momentum, energy, and
# moisture are blank 2D fields that the coupler fills each step. We must *not* add our
# own surface flux BCs — the coupler owns them.
#
# The atmosphere is moist by default (warm-phase saturation-adjustment microphysics),
# so the rising air can saturate and form the cloud street. We give it a reference
# potential temperature of 288 K (a mild mid-latitude marine airmass — explicitly
# **non-polar**), a Smagorinsky–Lilly LES closure, and an `FPlane` at mid-latitude
# `f = 1e-4 s⁻¹`.

atmosphere = atmosphere_simulation(atmos_grid;
                                   potential_temperature = FT(288),
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(f = FT(1e-4)))

# ### Initial atmospheric state
#
# We start the atmosphere at its reference potential-temperature profile with a mean
# wind `U₀ ≈ 5 m s⁻¹` along `x` and a modest, near-surface humidity so a cloud street
# can form once the warm filament moistens the boundary layer. The humidity decays
# with height (confined to the boundary layer). Small random perturbations near the
# surface seed turbulence.
#
# `set!` acts on the wrapped `atmosphere.model`. `qᵗ` is the total-water mass fraction
# (the moist prognostic variable).

const U₀  = FT(5)        # m s⁻¹ mean wind, along x (the cloud-street axis)
const θ₀  = FT(288)      # K, reference / surface potential temperature
const qᵗ₀ = FT(6e-3)     # kg/kg near-surface total water (modest marine humidity)
const zq  = FT(800)      # m, humidity scale height (boundary-layer moisture)
const zδ  = FT(600)      # m, perturbation seeding depth
const δθ  = FT(0.05)     # K, thermal perturbation amplitude
const δu  = FT(0.05)     # m s⁻¹, velocity perturbation amplitude

ϵ() = rand(FT) - FT(0.5)

θᵢ(x, y, z) = θ₀ + δθ * ϵ() * (z < zδ)
uᵢ(x, y, z) = U₀ + δu * ϵ() * (z < zδ)
vᵢ(x, y, z) = δu * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀ * exp(-z / zq)

set!(atmosphere.model, θ = θᵢ, u = uᵢ, v = vᵢ, qᵗ = qᵢ)

# ## The ocean component
#
# `ocean_simulation(grid; model = :nonhydrostatic)` builds an Oceananigans
# `NonhydrostaticModel` (full 3D pressure, LES-ready) wrapped in a `Simulation`, with
# `(T, S)` tracers and a TEOS-10 seawater equation of state. Its *top* boundary
# conditions are blank coupling fields the coupler fills — we do not set surface
# fluxes by hand. We add a mid-latitude `FPlane` so the ocean response feels rotation.

ocean = ocean_simulation(ocean_grid; model = :nonhydrostatic,
                         coriolis = FPlane(f = FT(1e-4)))

# ### Initialize the WARM SST FILAMENT — the ocean's living boundary
#
# The filament is a warm band centered in `y`, uniform along the wind (`x`), riding on
# a mixed layer over a stratified interior. The surface temperature is
#
# ```math
# T(x, y, z=0) = T_\mathrm{cold} + \Delta T\,\exp\!\Big(-\frac{(y - L_y/2)^2}{2\sigma^2}\Big)
# ```
#
# with `ΔT ≈ 3 K` and `σ ≈ 1 km` (a ≈ 2 km-wide warm filament). Below a mixed-layer
# depth `h` the water is stratified at `N²`, so the warm anomaly sits in a
# well-mixed surface layer that the coupled fluxes can erode and deepen. Salinity is a
# uniform 35 g kg⁻¹. The warm filament is the **boundary feature** the atmosphere
# reads and the cloud street traces.
#
# We add a tiny random thermal perturbation in the mixed layer to seed ocean
# convection once surface cooling begins.

const T_cold = FT(290)      # K, background SST (≈ 17 °C, mid-latitude)
const ΔT     = FT(3)        # K, filament warm anomaly
const σ      = FT(1000)     # m, filament half-width scale (≈ 2 km full width)
const h      = FT(40)       # m, initial ocean mixed-layer depth
const N²     = FT(1e-4)     # s⁻², interior stratification (buoyancy frequency²)
const S₀     = FT(35)       # g/kg, uniform salinity

## Convert an interior buoyancy frequency to a temperature lapse rate via a constant
## thermal expansion coefficient α ≈ 2e-4 K⁻¹ and g ≈ 9.81 m s⁻²: dT/dz = N²/(g α).
const α_T    = FT(2e-4)     # K⁻¹, thermal expansion (for the initial T profile only)
const g_oce  = FT(9.81)     # m s⁻²
const dTdz   = N² / (g_oce * α_T)   # K/m, interior temperature gradient

const y_c    = FT(Ly / 2)   # filament center in y
const δT_o   = FT(0.01)     # K, ocean mixed-layer perturbation amplitude

## Warm filament at the surface, mixed to depth h, stratified below.
function Tᵢ(x, y, z)
    filament = ΔT * exp(-(y - y_c)^2 / (2 * σ^2))
    mixed    = T_cold + filament
    ## Below the mixed layer (z < -h) decay toward a stratified interior.
    stratified = mixed + dTdz * (z + h)   # z + h ≤ 0 below the mixed layer
    T = z > -h ? mixed : stratified
    return T + δT_o * (rand(FT) - FT(0.5)) * (z > -h)
end

set!(ocean.model, T = Tᵢ, S = S₀)

# ## Couple them
#
# `AtmosphereOceanModel(atmosphere, ocean)` returns an `EarthSystemModel`: it builds
# the `atmosphere_ocean_interface` bookkeeping and wires the similarity-theory flux
# computation between the components. The *coupled* `Simulation` wrapped around it
# owns the time step `Δt`; the inner atmosphere and ocean simulations defer to it.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Simulation
#
# The coupled model is stepped by a single outer `Simulation`. A fixed 5 s step is
# comfortable for this grid (the atmosphere's ≈ 31 m vertical surface cell and
# ≈ 62 m horizontal cells, with U₀ = 5 m s⁻¹ and convective velocities of order
# 1 m s⁻¹, give CFL well below 1). We integrate **1 hour** for first validation —
# long enough to spin up a recognizable cloud street and start eroding the filament.
# A production run would integrate several hours (e.g. `stop_time = 6hours`) to watch
# the filament measurably erode and the ocean mixed layer deepen.

simulation = Simulation(model; Δt = 2, stop_time = 1hour)

wall_clock = Ref(time_ns())
function progress(sim)
    cm = sim.model
    wa = maximum(abs, cm.atmosphere.model.velocities.w)
    wo = maximum(abs, cm.ocean.model.velocities.w)
    Q  = cm.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat
    Qmax = maximum(abs, Q)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, wall %s, max|w_atm| %.2e, max|w_oce| %.2e m/s, max|Qsens| %.1f W/m²",
                   cm.clock.iteration, prettytime(cm.clock.time),
                   prettytime(elapsed), wa, wo, Qmax)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# ## Outputs
#
# The story is told in four views:
#
# 1. **The cloud street** — near-surface cloud liquid `qˡ` and vertical velocity `w`
#    in an `x–y` plane just above the surface. This is the band of convection the
#    warm filament writes into the atmosphere.
# 2. **An across-filament atmospheric transect** (`x` fixed, the `y–z` plane, or an
#    `x–z` plane along the wind) showing the rising moist plume and condensate.
# 3. **The ocean SST** (`x–y` at the surface) and a vertical `w` transect of the
#    oceanic response — cooling-driven sinking that erodes the filament and deepens
#    the mixed layer.
# 4. **The air–sea fluxes** — sensible and latent heat at the interface, computed by
#    similarity theory, localized over the warm filament.
#
# Output writers attach to the *coupled* `simulation.output_writers`, but each
# `JLD2Writer` is built around the relevant **component model** (`atmosphere.model`
# or `ocean.model`) whose fields it samples. We slice in `z` near the surface and at a
# representative `x` for the transect.

k_a_surface = 2                  # near-surface atmosphere level
k_o_surface = Nz_o               # top ocean level (z = 0⁻)
imid        = Nx ÷ 2 + 1         # mid-domain x for the across-filament transect

## Atmosphere-side fields.
w_a  = atmosphere.model.velocities.w
u_a  = atmosphere.model.velocities.u
θ_a  = liquid_ice_potential_temperature(atmosphere.model)
qˡ_a = atmosphere.model.microphysical_fields.qˡ   # cloud liquid (the visible cloud)
qᵛ_a = atmosphere.model.microphysical_fields.qᵛ   # water vapor

## Near-surface horizontal slices (the cloud-street top view) and an across-filament
## y–z transect at mid-domain x.
atmos_outputs = (
    w_xy  = view(w_a,  :, :, k_a_surface),
    qˡ_xy = view(qˡ_a, :, :, k_a_surface),
    θ_xy  = view(θ_a,  :, :, k_a_surface),
    w_yz  = view(w_a,  imid, :, :),
    qˡ_yz = view(qˡ_a, imid, :, :),
    θ_yz  = view(θ_a,  imid, :, :),
    qᵛ_yz = view(qᵛ_a, imid, :, :),
)

## Ocean-side fields.
w_o = ocean.model.velocities.w
T_o = ocean.model.tracers.T

ocean_outputs = (
    T_xy = view(T_o, :, :, k_o_surface),   # SST (the warm filament, eroding)
    w_yz = view(w_o, imid, :, :),          # across-filament ocean overturning
    T_yz = view(T_o, imid, :, :),
)

## Interface fluxes — 2D (x, y) fields along the sea surface, computed by similarity
## theory. Sensible and latent heat both live on the interface fluxes named tuple.
ao_fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
Q_sensible = ao_fluxes.sensible_heat
Q_latent   = ao_fluxes.latent_heat
flux_outputs = (; Q_sensible, Q_latent)

simulation.output_writers[:atmosphere] = JLD2Writer(atmosphere.model, atmos_outputs;
    filename = output_name(config, "atmosphere"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = output_name(config, "ocean"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = output_name(config, "fluxes"), schedule = TimeInterval(30seconds),
    overwrite_existing = true)

# ## Go time
run!(simulation)

# ## Visualization
#
# A multi-panel figure tells the coupled story in one frame:
#
# - top-left: the **cloud street** — near-surface cloud liquid `qˡ(x, y)` (the band of
#   cloud over the warm filament, leaning downwind);
# - top-right: near-surface vertical velocity `w(x, y)` (the convective updrafts);
# - middle-left: **SST** `T(x, y)` (the warm filament, being eroded);
# - middle-right: the **air–sea sensible heat flux** `Q(x, y)` (localized over the
#   filament);
# - bottom-left: across-filament atmospheric transect `w(y, z)` (the rising plume);
# - bottom-right: across-filament ocean transect `w(y, z)` (the sinking response).
#
# We build a movie from the time series and save the final frame.

using CairoMakie
_safelim(x, fallback) = (m = maximum(abs, x); isfinite(m) && m > 0 ? m : fallback)

atmos_file = output_name(config, "atmosphere")
ocean_file = output_name(config, "ocean")
flux_file  = output_name(config, "fluxes")

if isfile(atmos_file) && isfile(ocean_file)
    qˡxy = FieldTimeSeries(atmos_file, "qˡ_xy")
    wxy  = FieldTimeSeries(atmos_file, "w_xy")
    wyz  = FieldTimeSeries(atmos_file, "w_yz")
    Txy  = FieldTimeSeries(ocean_file, "T_xy")
    woyz = FieldTimeSeries(ocean_file, "w_yz")

    times = qˡxy.times
    Nt = length(times)

    xa, ya, _  = nodes(qˡxy)
    _,  yaz, za = nodes(wyz)
    xo, yo, _  = nodes(Txy)
    _,  yoz, zo = nodes(woyz)

    xkm  = xa  ./ 1e3
    ykm  = ya  ./ 1e3
    yazkm = yaz ./ 1e3
    xokm = xo  ./ 1e3
    yokm = yo  ./ 1e3
    yozkm = yoz ./ 1e3

    Qfields = isfile(flux_file)
    if Qfields
        Qts = FieldTimeSeries(flux_file, "Q_sensible")
        xq, yq, _ = nodes(Qts)
        xqkm = xq ./ 1e3
        yqkm = yq ./ 1e3
    end

    n = Observable(Nt)
    qln = @lift interior(qˡxy[$n], :, :, 1) .* 1e3   # g/kg
    wn  = @lift interior(wxy[$n], :, :, 1)
    Tn  = @lift interior(Txy[$n], :, :, 1)
    wyzn = @lift interior(wyz[$n], 1, :, :)
    woyzn = @lift interior(woyz[$n], 1, :, :)
    title = @lift "A warm filament writes a cloud street — t = " * prettytime(times[$n])

    ## Color limits from the final frame.
    qlmax = _safelim(interior(qˡxy[Nt]) .* 1e3, 1.0)
    wlim  = _safelim(interior(wxy[Nt]), 1e-3)
    wyzlim = _safelim(interior(wyz[Nt]), 1e-3)
    woyzlim = _safelim(interior(woyz[Nt]), 1e-5)
    Tmin = minimum(interior(Txy[Nt]))
    Tmax = (m=maximum(interior(Txy[Nt])); isfinite(m) ? m : 1.0)

    fig = Figure(size = (1300, 1100))
    Label(fig[0, 1:4], title, fontsize = 18, tellwidth = false)

    axq = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)",
               title = "cloud liquid qˡ (g kg⁻¹) — the cloud street")
    axw = Axis(fig[1, 3], xlabel = "x (km)", ylabel = "y (km)",
               title = "near-surface w (m s⁻¹)")
    axT = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "y (km)",
               title = "SST T (K) — the warm filament")
    axF = Axis(fig[2, 3], xlabel = "x (km)", ylabel = "y (km)",
               title = "sensible heat flux Q (W m⁻²)")
    axwa = Axis(fig[3, 1], xlabel = "y (km)", ylabel = "z (m)",
                title = "atmosphere transect w (m s⁻¹)")
    axwo = Axis(fig[3, 3], xlabel = "y (km)", ylabel = "z (m)",
                title = "ocean transect w (m s⁻¹)")

    hmq = heatmap!(axq, xkm, ykm, qln, colormap = :dense, colorrange = (0, qlmax))
    hmw = heatmap!(axw, xkm, ykm, wn, colormap = :balance, colorrange = (-wlim, wlim))
    hmT = heatmap!(axT, xokm, yokm, Tn, colormap = :thermal, colorrange = (Tmin, Tmax))
    hmwa = heatmap!(axwa, yazkm, za, wyzn, colormap = :balance, colorrange = (-wyzlim, wyzlim))
    hmwo = heatmap!(axwo, yozkm, zo, woyzn, colormap = :balance, colorrange = (-woyzlim, woyzlim))

    Colorbar(fig[1, 2], hmq)
    Colorbar(fig[1, 4], hmw)
    Colorbar(fig[2, 2], hmT)
    Colorbar(fig[3, 2], hmwa)
    Colorbar(fig[3, 4], hmwo)

    if Qfields
        Qn = @lift interior(Qts[$n], :, :, 1)
        Qmax = _safelim(interior(Qts[Nt]), 1.0)
        hmF = heatmap!(axF, xqkm, yqkm, Qn, colormap = :balance, colorrange = (-Qmax, Qmax))
        Colorbar(fig[2, 4], hmF)
    end

    save(figure_name(config, "coupled_warm_filament_final"), fig)

    if Nt > 1
        record(fig, movie_name(config, "coupled_warm_filament"), 1:Nt; framerate = 12) do i
            n[] = i
        end
        @info "Wrote movie" movie_name(config, "coupled_warm_filament")
    end
end

@info "Case 08 complete" run_stamp(config)...
nothing #hide

# ## References
#
# - **Minobe, S., Kuwano-Yoshida, A., Komori, N., Xie, S.-P., Small, R. J. (2008).**
#   Influence of the Gulf Stream on the troposphere. *Nature*, 452, 206–209.
#   <https://doi.org/10.1038/nature06690> — the SST front anchors surface wind
#   convergence, upward motion, and precipitation reaching into the free troposphere:
#   the ocean steers the atmosphere.
# - **Small, R. J., et al. (2008).** Air–sea interaction over ocean fronts and
#   eddies. *Dynamics of Atmospheres and Oceans*, 45, 274–319.
#   <https://doi.org/10.1016/j.dynatmoce.2008.01.001> — review of the mechanisms by
#   which SST fronts drive marine-boundary-layer response, surface wind, and
#   convection over the warm side of the front.
# - **Monin, A. S., Obukhov, A. M. (1954).** Basic laws of turbulent mixing in the
#   surface layer of the atmosphere. *Tr. Akad. Nauk SSSR Geofiz. Inst.*, 24, 163–187.
#   — the similarity theory underlying the bulk air–sea flux formulae the coupler uses.
# - **Large, W. G., Yeager, S. G. (2009).** The global climatology of an
#   interannually varying air–sea flux data set. *Climate Dynamics*, 33, 341–364.
#   <https://doi.org/10.1007/s00382-008-0441-3> — bulk transfer coefficients for the
#   air–sea turbulent fluxes.
# - **Atkinson, B. W., Zhang, J. W. (1996).** Mesoscale shallow convection in the
#   atmosphere. *Reviews of Geophysics*, 34, 403–431.
#   <https://doi.org/10.1029/96RG02623> — formation and dynamics of cloud streets /
#   longitudinal convective rolls under combined buoyancy and wind shear.
#
# !!! note "Idealizations"
#     This is a modest-resolution, 1-hour validation run with a shallow 120 m ocean
#     and a warm-phase (liquid-only) cloud scheme. The cloud street's structure,
#     orientation, and the localization of fluxes over the filament are
#     representative; the precise cloud water content, filament erosion rate, and
#     mixed-layer deepening are not quantitative until the grid is refined and the run
#     extended (see the production grid/time notes above). The point is the **two-way
#     coupling**: an evolving ocean feature writing organized convection into the
#     atmosphere, while the atmosphere erodes the feature.
