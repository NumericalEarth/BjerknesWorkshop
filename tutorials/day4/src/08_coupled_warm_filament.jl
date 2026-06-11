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

Lx   = 12kilometers   # along-wind (cloud-street axis)
Ly   = 6kilometers    # cross-wind (across the filament)
Lz_a = 3kilometers    # atmosphere depth
Lz_o = 120meters      # ocean depth

Nx   = 192
Ny   = 96
Nz_a = 96             # atmosphere vertical cells
Nz_o = 48             # ocean vertical cells

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
                                   potential_temperature = 288,
                                   closure = SmagorinskyLilly(),
                                   coriolis = FPlane(f = 1e-4))

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

U₀    = 5       # m s⁻¹ mean wind, along x (the cloud-street axis)
θ_bl  = 288     # K, well-mixed boundary-layer potential temperature (= reference)
zᵢ    = 1000    # m, initial inversion height (boundary-layer top)
Δθᵢ   = 6       # K, capping-inversion strength (θ jump at zᵢ)
Γθ    = 4e-3    # K m⁻¹, stable free-tropospheric θ lapse above the inversion
qᵗ_bl = 9e-3    # kg/kg, moist mixed-layer total water (RH ≈ 85% → cloud base in the BL)
qᵗ_ft = 2e-3    # kg/kg, dry free troposphere above the inversion
zδ    = 600     # m, perturbation seeding depth
δθ    = 0.05    # K, thermal perturbation amplitude
δu    = 0.05    # m s⁻¹, velocity perturbation amplitude

ϵ() = rand() - 0.5

## A well-mixed, moist boundary layer capped by an inversion at zᵢ, with a stable,
## dry free troposphere above. The warm filament heats and moistens this layer; thermals
## rise to their lifting condensation level (a few hundred metres up), saturate into
## cloud, and are capped near zᵢ — organizing into wind-aligned cloud streets.
θᵢ(x, y, z) = (z < zᵢ ? θ_bl : θ_bl + Δθᵢ + Γθ * (z - zᵢ)) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = z < zᵢ ? qᵗ_bl : qᵗ_ft
uᵢ(x, y, z) = U₀ + δu * ϵ() * (z < zδ)
vᵢ(x, y, z) = δu * ϵ() * (z < zδ)

set!(atmosphere.model, θ = θᵢ, u = uᵢ, v = vᵢ, qᵗ = qᵢ)

# ## The ocean component
#
# `ocean_simulation(grid; model = :nonhydrostatic)` builds an Oceananigans
# `NonhydrostaticModel` (full 3D pressure, LES-ready) wrapped in a `Simulation`, with
# `(T, S)` tracers and a TEOS-10 seawater equation of state. Its *top* boundary
# conditions are blank coupling fields the coupler fills — we do not set surface
# fluxes by hand. We add a mid-latitude `FPlane` so the ocean response feels rotation.

ocean = ocean_simulation(ocean_grid; model = :nonhydrostatic,
                         coriolis = FPlane(f = 1e-4))

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
#
# !!! warning "Units: the ocean is in °C, the atmosphere is in K"
#     The ocean uses the TEOS-10 equation of state, whose conservative temperature
#     is in **degrees Celsius**, and the air–sea coupler defaults to
#     `ocean_temperature_units = DegreesCelsius()`. So the ocean temperature here is
#     `17–20 °C`, *not* Kelvin. The Breeze atmosphere, by contrast, is in Kelvin
#     (`potential_temperature = 288 K`). Setting the ocean in Kelvin silently breaks
#     the flux calculation: the coupler adds 273.15, evaluates the saturation
#     humidity at ~563 K — where the saturation vapour pressure exceeds the ambient
#     pressure — and returns a *negative* interface humidity, which drives a runaway
#     spurious-condensation instability. Keep the ocean in °C.

T_cold = 17       # °C, background SST (mid-latitude)
ΔT     = 3        # K = °C, filament warm anomaly
σ      = 1000     # m, filament half-width scale (≈ 2 km full width)
h      = 40       # m, initial ocean mixed-layer depth
N²     = 1e-4     # s⁻², interior stratification (buoyancy frequency²)
S₀     = 35       # g/kg, uniform salinity

## Convert an interior buoyancy frequency to a temperature lapse rate via a constant
## thermal expansion coefficient α ≈ 2e-4 K⁻¹ and g ≈ 9.81 m s⁻²: dT/dz = N²/(g α).
α_T    = 2e-4     # K⁻¹, thermal expansion (for the initial T profile only)
g_oce  = 9.81     # m s⁻²
dTdz   = N² / (g_oce * α_T)   # K/m, interior temperature gradient

y_c    = Ly / 2   # filament center in y
δT_o   = 0.01     # K, ocean mixed-layer perturbation amplitude

## Warm filament at the surface, mixed to depth h, stratified below.
function Tᵢ(x, y, z)
    filament = ΔT * exp(-(y - y_c)^2 / (2 * σ^2))
    mixed    = T_cold + filament
    ## Below the mixed layer (z < -h) decay toward a stratified interior.
    stratified = mixed + dTdz * (z + h)   # z + h ≤ 0 below the mixed layer
    T = z > -h ? mixed : stratified
    return T + δT_o * (rand() - 0.5) * (z > -h)
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

## Cloud base sits a few hundred metres up (at the lifting condensation level), so a
## near-surface slice would miss the cloud entirely. We view the cloud street on a
## horizontal level inside the cloud layer (≈ 800 m, between cloud base and the
## inversion), where the warm-filament-organized cloud pattern is clearest.
k_a_cloud = max(1, round(Int, 800 / (Lz_a / Nz_a)))   # atmosphere level nearest z ≈ 800 m

## Near-surface horizontal slices (surface convergence + SST) and an across-filament
## y–z transect at mid-domain x; the cloud is shown on an in-cloud horizontal level.
atmos_outputs = (
    w_xy  = view(w_a,  :, :, k_a_surface),
    qˡ_xy = view(qˡ_a, :, :, k_a_cloud),
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
    filename = "warm_filament_atmosphere.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:ocean] = JLD2Writer(ocean.model, ocean_outputs;
    filename = "warm_filament_ocean.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

simulation.output_writers[:fluxes] = JLD2Writer(atmosphere.model, flux_outputs;
    filename = "warm_filament_fluxes.jld2", schedule = TimeInterval(30seconds),
    overwrite_existing = true)

# ## Go time
run!(simulation)

@info "Case 08 complete"
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

