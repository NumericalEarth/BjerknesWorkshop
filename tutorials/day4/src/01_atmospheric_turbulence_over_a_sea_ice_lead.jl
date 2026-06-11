# # A crack in the ice: atmospheric turbulence over a sea-ice lead
#
# *Boundary heterogeneity writes turbulence into the fluid — case 1 of 3.*
#
# A **sea-ice lead** is a narrow ribbon of open water — meters to several
# kilometers wide — torn open in an otherwise frozen surface as the pack ice
# diverges. In the polar winter the exposed ocean sits near its freezing point
# (≈ −2 °C) while the overlying air, chilled by the surrounding ice, can be
# 20–40 °C colder. That contrast drives a vigorous upward sensible heat flux:
# aircraft measurements over Fram Strait leads (Tetzlaff et al. 2015, STABLE
# campaign) found near-surface sensible fluxes of 15–180 W m⁻², while the
# open-water patch itself can force several hundred W m⁻² locally (Glendening
# 1994; Gryschka et al. 2023). The lead injects heat — and, because open water is
# a different aerodynamic surface than ridged ice, momentum — into a cold, stably
# stratified atmospheric boundary layer (ABL).
#
# The atmospheric response is not a smooth chimney but a **turbulent convective
# plume**: a cluster of quasi-random buoyant thermals that lean downwind, converge
# along the lead axis (a small "lead breeze"), and rise until a capping inversion
# arrests them. For the narrow-to-moderate leads typical of winter the plume is
# *fetch-limited and inversion-capped*: it penetrates only ≈ 150–300 m (Zulauf &
# Krueger 2003 report ≈ 180, 220, 300 m for 200, 400, 800 m leads) rather than
# ventilating the whole ABL, and the warmed, moistened air — often carrying a thin
# ice-crystal fog — spreads laterally beneath the inversion and advects tens of
# kilometers over the downwind ice. Plume vigor grows with lead width (roughly
# +1 m s⁻¹ in peak updraft per doubling of width); leads narrower than ≈ 4 km
# produce a single merged plume, while wider leads develop edge plumes and
# interior convective cells that amplify the effective turbulent exchange
# (Esau 2007).
#
# This is a **Breeze atmosphere-only large-eddy simulation** with *prescribed*
# bottom fluxes. There is no prognostic sea ice and no ocean model: the lead is a
# smooth surface-flux mask. The run is **moist**: the lead supplies a latent heat
# flux comparable to its sensible flux (Tetzlaff et al. 2015), and warm-phase
# saturation-adjustment microphysics lets the moistened plume condense into the
# visible **lead fog / steam plume** that is the open water's signature — a thin
# cloud that rises with the thermals and advects downwind over the ice. (Real lead
# fog is largely ice crystals; the warm-phase scheme produces the supercooled-liquid
# analogue — see the microphysics caveat in the References.) We pick a single 1 km
# lead — the canonical width that yields one clean plume and
# matches the historical 1 km lead-parametrization baseline (Michaelis et al.
# 2020) — and size the run to develop a recognizable plume on one H100 in roughly
# fifteen minutes.
#
# Coordinate orientation:
#
# ```text
# x = across-lead / mean-wind direction
# y = along-lead direction
# z = vertical
# ```
#
# Periodic horizontal boundaries represent an infinite periodic array of leads
# separated by the domain width. With U₀ = 8 m s⁻¹ over 40 min the air travels
# ≈ 19 km, comfortably less than the 40 km across-lead domain, so the downwind
# plume does not wrap around and contaminate the upwind ice before statistics are
# taken — watch this for stronger winds or longer runs.

using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux, PolynomialCoefficient, FilteredSurfaceVelocities
using Oceananigans
using Oceananigans: Oceananigans
using Oceananigans.Units
using Printf
using Random

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

Random.seed!(1994)

config = RunConfig("01_lead_atmosphere")
arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float32
FT = Float32
nothing #hide

# ## Domain and grid
#
# A 40 km × 12 km × 3 km domain at 50 m horizontal resolution (≈12 m near the
# surface, coarsening aloft): 800 × 240 × 96 ≈ 18 million cells. We run at half the
# 25 m "production" resolution but for a long **3 hours** of simulated time: the
# convective plume needs ~10 turnovers to reach a quasi-steady downwind structure
# and the boundary layer to deepen, so a longer, coarser run shows far more than a
# short ultra-fine one. Refine to 1600×480×192 @ 25 m for a production rendering.
#
# !!! note "Resolution"
#     At Δx = 50 m about 20 cells span the 1 km lead — plume-permitting (the
#     published lead-LES range is 1–25 m; Glendening 1994; Esau 2007; Gryschka et
#     al. 2023). Treat the plume structure as qualitative; refine for flux convergence.

const Lx = 40kilometers   # across-lead / mean wind
const Ly = 12kilometers   # along-lead
const Lz = 3kilometers    # vertical

const Nx = 800
const Ny = 240
const Nz = 96

# A smooth exponential vertical stretch: fine near the surface where the plume is
# generated, coarsening toward the top.

function stretched_z_faces(Lz, Nz; stretching = 0.9)
    σ(k) = (k - 1) / Nz
    return [Lz * expm1(stretching * σ(k)) / expm1(stretching) for k in 1:Nz+1]
end

z_faces = stretched_z_faces(Lz, Nz)
@info "Vertical grid" Δz_surface = z_faces[2] - z_faces[1] Δz_top = z_faces[end] - z_faces[end-1]

memory_report(Nx, Ny, Nz; FT, nfields = 7)

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and anelastic dynamics
#
# A cold Arctic boundary layer: surface potential temperature θ₀ = 260 K. We use
# the anelastic formulation with a dry adiabatic reference state.

const p₀ = FT(1e5)   # Pa
const θ₀ = FT(260)   # K

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)
dynamics = AnelasticDynamics(reference_state)

q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
const ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)
const cₚ = constants.dry_air.heat_capacity
nothing #hide

# A capping inversion stabilizes the boundary layer so the lead plume has
# something to work against: neutral below `zᵢ`, then a strong jump, then a weak
# free-tropospheric lapse rate. The inversion is the *antagonist* of the plume.
#
# We place the inversion base at `zᵢ ≈ 300 m`, the canonical depth used in the
# lead-parametrization literature (Michaelis et al. 2020 use idealized
# lead-perpendicular near-neutral inflow capped by a strong inversion at
# ≈ 250–350 m). At this depth the cap is an *active, visible* control on the plume
# top — the pedagogical point. (A 600 m inversion is a deliberately deep,
# hard-to-penetrate variant: there the plume fills only the lower half of a deep
# neutral layer and the cap is barely engaged.)
#
# The inversion strength `Δθᵢ = 6 K` over a ≈ 100 m layer gives an interfacial
# buoyancy frequency N ≈ 0.04–0.06 s⁻¹, strong enough to arrest a w⋆ ≈ 1–1.7 m s⁻¹
# plume near the inversion base — the observed capped, laterally spreading
# behavior. Winter Arctic capping inversions are genuinely this strong (several K
# over tens to ≈ 100 m); 4–10 K is the realistic range. The free-tropospheric
# lapse rate `Γᵗᵒᵖ = 4 K km⁻¹` is a standard polar-winter value (3–5 K km⁻¹) and
# supports the gravity waves the plume radiates into the stable layer.

const zᵢ = FT(300)        # m, inversion base (lead-study canonical 250–350 m)
const Δθᵢ = FT(6)         # K, inversion strength (realistic range 4–10 K)
const δᵢ = FT(100)        # m, inversion thickness
const Γᵗᵒᵖ = FT(0.004)    # K/m, free-tropospheric lapse rate (polar winter 3–5 K/km)

θᵣ(z) = θ₀ + Δθᵢ * smooth_step(z - zᵢ, δᵢ) + Γᵗᵒᵖ * max(z - zᵢ, zero(z))
nothing #hide

# ## The lead: boundary heterogeneity as a smooth mask
#
# A single top-hat mask `χ(x)` — open water inside a band of width `Wˡᵉᵃᵈ`, ice
# outside, with a smooth `δˡᵉᵃᵈ` transition — defines the geometry. Here it sets the
# *surface temperature* (warm water over the lead, cold ice outside); the bulk
# formulae below then turn that contrast into heat, moisture and momentum fluxes.
#
# **Lead width.** Real leads span meters to several kilometers. Penetration and
# vigor scale with width (Glendening 1994; Zulauf & Krueger 2003: ≈ 180/220/300 m
# penetration for 200/400/800 m leads, +≈ 1 m s⁻¹ peak updraft per width doubling).
# Leads narrower than ≈ 4 km force a *single* merged plume onto the lead axis; wider
# leads (4–10 km) develop edge plumes plus interior convective cells (Esau 2007). We
# use the 1 km canonical width — narrow (0.1–0.5), moderate (1–2), wide (4–10 km)
# are the regimes you can sweep.
#
# The resulting lead-averaged sensible flux (now *diagnosed*, not prescribed) lands
# near 100–300 W m⁻² for this ≈ 26 K air–sea contrast — consistent with Tetzlaff et
# al. (2015) aircraft data (15–180 W m⁻² near-surface) and lead LES/2D studies
# (Glendening 1994; Zulauf & Krueger 2003; Esau 2007; Gryschka et al. 2023), with a
# comparable latent flux from the open water.

const Wˡᵉᵃᵈ = 1kilometer    # narrow 0.1–0.5, moderate 1–2, wide 4–10 km
const δˡᵉᵃᵈ = 100meters

# ### Boundary heterogeneity as a *surface state*, not a prescribed flux
#
# Instead of prescribing the fluxes, we prescribe the **surface temperature** — cold
# ice outside the lead, near-freezing open water over it — and let Breeze compute the
# turbulent fluxes from bulk aerodynamic formulae,
#
# ```math
# J_φ = -C_φ\,|ΔU|\,(φ_a - φ_0), \qquad |ΔU| = \sqrt{(u-u_0)^2 + (v-v_0)^2 + U_g^2}
# ```
#
# for momentum (`Cᴰ`), sensible heat (`Cᵀ`) and moisture (`Cᵛ`). This is more physical
# than a fixed flux — the lead's exchange responds to the evolving wind and stability —
# and is the same machinery coupled air–sea runs use. The ≈ 26 K air–sea temperature
# difference (ice-chilled air over near-freezing water) drives the plume.
const T_ice   = FT(245)     # K, cold ice/snow surface (≈ -28 °C)
const T_water = FT(271.35)  # K, open water near the seawater freezing point (≈ -1.8 °C)

# Surface temperature as a 2-D field: the lead is the warm top-hat.
Tˢ = Field{Center, Center, Nothing}(grid)
set!(Tˢ, (x, y) -> T_ice + top_hat(x; center = 0, width = Wˡᵉᵃᵈ, edge = δˡᵉᵃᵈ) * (T_water - T_ice))

# Wind- and stability-dependent exchange coefficients (Large & Yeager 2009) via a
# polynomial bulk coefficient; `gustiness` floors |ΔU| so calm air still exchanges.
const Uᵍ = FT(0.5)          # m s⁻¹ gustiness
coefficient = PolynomialCoefficient(roughness_length = 1.5e-4)

# ### The filtered surface state — and how it changes the flux
#
# At LES resolution the *instantaneous* near-surface wind carries the full turbulent
# fluctuation spectrum. Feeding it straight into the quadratic bulk formula aliases
# those fluctuations into the mean flux and makes the surface exchange
# resolution-dependent — a spurious `u★`–`u′` correlation (Nishizawa & Kitamura 2018;
# Shin, Yang & Howland 2025). We instead drive the bulk fluxes with a **temporally
# filtered** surface velocity — an exponential filter `ū ← (ū + ϵ u)/(1 + ϵ)`,
# `ϵ = Δt/τ`, with `τ = 10 min` — which tracks the evolving mean wind while smoothing
# the fastest eddies. The flux is then computed from `ū`, not the instantaneous `u`:
# a small change that markedly cleans up the surface-layer flux statistics and removes
# their grid dependence. Pass `filtered_velocities = nothing` to recover the raw
# instantaneous-flux behavior and compare.
filtered_velocities = FilteredSurfaceVelocities(grid; filter_timescale = 10minutes)

# Bulk momentum (drag), sensible-heat and moisture fluxes — all sharing the filter and
# the lead surface temperature. Sensible heat is a potential-temperature (`ρθ`) density
# flux; moisture a `ρqᵉ` flux; drag acts on `ρu`/`ρv`.
ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(; coefficient, gustiness = Uᵍ, surface_temperature = Tˢ, filtered_velocities))
ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(; coefficient, gustiness = Uᵍ, surface_temperature = Tˢ, filtered_velocities))
ρθ_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(; coefficient, gustiness = Uᵍ, surface_temperature = Tˢ, filtered_velocities))
ρqᵉ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(; coefficient, gustiness = Uᵍ, surface_temperature = Tˢ, filtered_velocities))

# ## Sponge layer and large-scale forcing
#
# A Gaussian sponge near the top relaxes potential temperature to the
# free-tropospheric profile and damps vertical motion, preventing reflection of
# plume-generated gravity waves off the lid. Coriolis plus a geostrophic wind
# drive a mean flow of U₀ across the lead.
#
# **What sets the plume scale.** The relevant velocity scale is the Deardorff
# (1970) convective velocity built on the lead buoyancy flux and the ABL depth,
#
# ```text
# w⋆ = ( (g/θ₀) · (w′θ′)_sfc · zᵢ )^(1/3)
# ```
#
# With the lead kinematic heat flux ≈ 0.15 K m s⁻¹ (Qʰ = 200 W m⁻²), θ₀ = 260 K
# and a capped depth zᵢ ≈ 300 m, w⋆ ≈ 1.1 m s⁻¹ (≈ 1.2–1.7 m s⁻¹ for
# Qʰ = 200–300 W m⁻² and zᵢ = 300–600 m), so updrafts of order 1 m s⁻¹ and
# turnover times zᵢ/w⋆ ≈ 3–8 min are expected — which is why a 40-min integration
# captures several convective turnovers (the first ≈ 10–15 min are spin-up;
# compute statistics from the latter part of the run).
#
# **Why the wind matters.** The mean wind advects each thermal a horizontal
# distance U₀ · (zᵢ/w⋆) ≈ 1–3 km while it rises, giving the plume its downwind lean
# and its long downwind warm/fog tail (Zulauf & Krueger 2003 see ice cloud 50+ km
# downwind). Across-lead winds in lead studies are a few to ≈ 10 m s⁻¹; at
# `U₀ = 8 m s⁻¹` the plume leans strongly downwind — the realistic, instructive
# regime. Low wind (2–3 m s⁻¹) gives an upright, near-symmetric plume; above
# ≈ 10 m s⁻¹ the shear flattens the plume into a near-surface internal boundary
# layer and suppresses penetration. Whether organization appears as a single
# tilted plume or as longitudinal convective rolls depends on the wind-to-w⋆ ratio
# and the over-water fetch: a 1 km lead at 8 m s⁻¹ sits firmly in the single-plume
# regime; roll structures need the much longer over-water fetch of a marine
# cold-air outbreak. Note that with f = 1.4×10⁻⁴ s⁻¹, Ekman turning will develop
# an along-lead (v) component over the run — the wind is only purely across-lead
# initially.

const U₀ = FT(8)   # m s⁻¹ mean wind across the lead (range 3–10; low wind ⇒ upright plume)
coriolis = FPlane(f = 1.4e-4)
geostrophic = geostrophic_forcings(U₀, 0)

sponge_width = FT(400)
sponge_rate = FT(0.01)
sponge_mask = GaussianMask{:z}(center = Lz, width = sponge_width)

ρθᵣ = Field{Nothing, Nothing, Center}(grid)
set!(ρθᵣ, z -> θᵣ(z))
set!(ρθᵣ, reference_state.density * ρθᵣ)
ρθᵣ_data = interior(ρθᵣ, 1, 1, :)

@inline function ρθ_sponge_fun(i, j, k, grid, clock, model_fields, p)
    zₖ = znode(k, grid, Center())
    return @inbounds p.rate * p.mask(0, 0, zₖ) * (p.target[k] - model_fields.ρθ[i, j, k])
end

ρθ_sponge = Forcing(ρθ_sponge_fun; discrete_form = true,
                    parameters = (rate = sponge_rate, mask = sponge_mask, target = ρθᵣ_data))
ρw_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask)

forcing = (; u = geostrophic.u, v = geostrophic.v, ρw = ρw_sponge, ρθ = ρθ_sponge)

# ## Model
#
# 9th-order WENO advection, a Smagorinsky–Lilly LES closure, and warm-phase
# saturation-adjustment microphysics so the moistened plume can condense into fog.

advection = WENO(order = 9)
closure = SmagorinskyLilly()
microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

model = AtmosphereModel(grid; dynamics, coriolis, advection, closure, microphysics, forcing,
                        boundary_conditions = (; ρθ = ρθ_bcs, ρqᵉ = ρqᵉ_bcs, ρu = ρu_bcs, ρv = ρv_bcs))

# ## Initial conditions
#
# Mean wind U₀ across the lead, the inversion-capped θ profile, and small
# perturbations below `zδ` (within the neutral sub-inversion layer) to seed
# turbulence.

const δu = FT(0.1)   # m s⁻¹
const δθ = FT(0.1)   # K
const zδ = FT(300)   # m
const qᵗ₀ = FT(1.1e-3) # kg/kg, near-saturated sub-inversion background (≈ 80 % RH
                       # at 260 K, where qˢᵃᵗ ≈ 1.4e-3); cold near-saturated air over
                       # the warm lead is the sea-smoke / steam-fog setup.

ϵ() = rand(FT) - FT(0.5)
uᵢ(x, y, z) =  U₀  + δu * ϵ() * (z < zδ)
vᵢ(x, y, z) = δu * ϵ() * (z < zδ)
θᵢ(x, y, z) = θᵣ(z) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀ * (z < zᵢ)

set!(model, θ = θᵢ, u = uᵢ, v = vᵢ, qᵗ = qᵢ)

# ## Simulation
#
# Adaptive time-stepping at CFL 0.7, run for 40 minutes of simulated time so the
# plume develops through several convective turnovers (zᵢ/w⋆ ≈ 3–8 min). The first
# ≈ 10–15 min are spin-up; diagnose fluxes and profiles from the latter part.

simulation = Simulation(model; Δt = 0.5, stop_time = 3hours)
conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 5.0)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

wall_clock = Ref(time_ns())
function progress(sim)
    wmax = maximum(abs, sim.model.velocities.w)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    @info @sprintf("Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   prettytime(elapsed), wmax)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(100))

# ## Outputs
#
# The lead mask and surface heat-flux line as static fields; a vertical (`x–z`)
# transect through the lead and a near-surface horizontal slice at high cadence
# for animation; full 3D fields sparingly; along-lead-averaged profiles.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)
T = model.temperature
qᵛ = model.microphysical_fields.qᵛ   # water-vapor specific humidity
qˡ = model.microphysical_fields.qˡ   # cloud-liquid (fog) specific humidity

χ_field = Field{Center, Center, Nothing}(grid)
set!(χ_field, (x, y) -> top_hat(x; center = 0, width = Wˡᵉᵃᵈ, edge = δˡᵉᵃᵈ))


jmid = Ny ÷ 2 + 1
k_surface = 2

base_3d = (; u, v, w, θ, T, qᵛ, qˡ)

slice_outputs = (
    w_xz = view(w, :, jmid, :),
    θ_xz = view(θ, :, jmid, :),
    qˡ_xz = view(qˡ, :, jmid, :),
    qᵛ_xz = view(qᵛ, :, jmid, :),
    w_xy = view(w, :, :, k_surface),
    θ_xy = view(θ, :, :, k_surface),
    qˡ_xy = view(qˡ, :, :, k_surface),
)

along_lead = NamedTuple(name => Average(@at((Center, Center, Center), base_3d[name]), dims = 2)
                        for name in keys(base_3d))

simulation.output_writers[:statics] = JLD2Writer(model, (; χ = χ_field, Tˢ = Tˢ);
    filename = output_name(config, "statics"), schedule = IterationInterval(typemax(Int)),
    overwrite_existing = true)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
    filename = slice_name(config), schedule = TimeInterval(15seconds), overwrite_existing = true)

simulation.output_writers[:profiles] = JLD2Writer(model, along_lead;
    filename = output_name(config, "profiles"), schedule = TimeInterval(30seconds), overwrite_existing = true)

## (No full-3D field output: the visualization and gallery use the slices and
## along-lead profiles. Add a sparse 3D writer here if you need volumetric data.)

write_once!(simulation.output_writers[:statics], model)

# ## Go time
run!(simulation)

@info "Case 1 complete" run_stamp(config)...
nothing #hide

# ## References
#
# - **Tetzlaff, A., Lüpkes, C., Hartmann, J. (2015).** Aircraft-based observations
#   of atmospheric boundary-layer modification over Arctic leads. *Q. J. R.
#   Meteorol. Soc.*, 141, 2839–2856 (STABLE campaign, Fram Strait, March 2013).
#   <https://epic.awi.de/38065/> — primary observational anchor: near-surface
#   sensible fluxes 15–180 W m⁻², warming up to 3.2 °C, humidity +0.2 g kg⁻¹, flux
#   maximum in the plume core, entrainment fluxes > 30 % of the surface flux.
# - **Gryschka, M., et al. (2023).** Turbulent Heat Exchange Over Polar Leads
#   Revisited: A Large Eddy Simulation Study. *JGR Atmospheres*, 128(12),
#   e2022JD038236. <https://doi.org/10.1029/2022JD038236> — LES survey across lead
#   widths; lead-averaged surface heat flux depends non-monotonically on width.
# - **Michaelis, J., Lüpkes, C., Zhou, X., Gryschka, M., Gryanik, V. M. (2020).**
#   Influence of Lead Width on the Turbulent Flow Over Sea Ice Leads. *JGR
#   Atmospheres*, 125, e2019JD031996. <https://doi.org/10.1029/2019JD031996> —
#   idealized lead-perpendicular inflow capped by a strong inversion at ≈ 250–350 m;
#   nonlocal lead-width-dependent w⋆ framework. Basis for zᵢ ≈ 300 m.
# - **Michaelis, J., Lüpkes, C. (2021).** Modelling and parametrization of the
#   convective flow over leads in sea ice and comparison with airborne
#   observations. *Q. J. R. Meteorol. Soc.*, 147, 914–943.
#   <https://doi.org/10.1002/qj.3953> — links the width parametrization to STABLE
#   aircraft data; validates plume structure, inversion height and flux profiles.
# - **Esau, I. N. (2007).** Amplification of turbulent exchange over wide Arctic
#   leads: Large-eddy simulation study. *JGR Atmospheres*, 112, D08109.
#   <https://doi.org/10.1029/2006JD007225> — organized convection amplifies
#   effective exchange over wide leads; informs the wide-lead / roll discussion.
# - **Zulauf, M. A., Krueger, S. K. (2003).** Two-dimensional numerical
#   simulations of Arctic leads: Plume penetration height. *JGR Oceans*, 108(C2),
#   3041. <https://doi.org/10.1029/2000JC000495> — penetration ≈ 180/220/300 m for
#   200/400/800 m leads; +≈ 1 m s⁻¹ per width doubling; single plume below ≈ 4 km;
#   downwind ice cloud 50+ km.
# - **Glendening, J. W. (1994); Glendening & Burk (1992).** LES of lead-induced
#   convection. *Boundary-Layer Meteorology*.
#   <https://link.springer.com/article/10.1007/BF02215457> — foundational
#   single-plume / lead-breeze convergence picture; O(100) W m⁻² fluxes in the
#   plume core; transition to multiple plumes over wide leads.
# - **Lüpkes, C., et al. (2008).** Modeling convection over arctic leads with LES
#   and a non-eddy-resolving microscale model. *JGR Oceans*, 113, C09028.
#   <https://doi.org/10.1029/2007JC004099> — bridges LES and parametrization;
#   plume structure, fetch dependence, convective scaling.
# - **Deardorff, J. W. (1970).** Convective velocity and temperature scales for
#   the unstable planetary boundary layer. *J. Atmos. Sci.*, 27, 1211–1213.
#   <https://doi.org/10.1175/1520-0469(1970)027%3C1211:CVATSF%3E2.0.CO;2> —
#   defines w⋆, the convective velocity scale used throughout.
#
# !!! note "Microphysics caveat (warm-phase vs. ice fog)"
#     This run includes a latent heat flux and warm-phase (liquid-only) saturation
#     adjustment, so the moistened plume condenses into a cloud-liquid `qˡ` fog and
#     releases condensational latent heat. A *real* winter lead fog at ≈ 260 K is
#     dominated by **ice crystals** (sea smoke / frost smoke); the warm-phase scheme
#     produces the supercooled-liquid analogue and omits ice-phase microphysics and
#     ice-cloud radiative effects. The fog's location, timing, and downwind advection
#     are representative; its phase and exact water content are not quantitative.

