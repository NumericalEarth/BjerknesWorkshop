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
# A 40 km × 12 km × 3 km domain at 62.5 m horizontal resolution with a stretched
# vertical (≈12 m near the surface, coarsening aloft): 640 × 192 × 128 ≈ 15.7
# million cells.
#
# !!! note "Marginal LES — a plume-permitting teaching run"
#     At Δx = 62.5 m only ≈ 16 cells span the 1 km lead and the near-surface
#     energy-containing eddies (tens of meters) are barely resolved. Published
#     lead LES uses 1–25 m grids (Glendening 1994; Esau 2007; Gryschka et al.
#     2023). Treat this as a *plume-permitting* run that develops a recognizable,
#     citable lead plume on one H100 in ≈ 15 min — **do not claim quantitative
#     flux convergence**. For a quantitative comparison, refine to 10–25 m
#     horizontally over a smaller domain.

const Lx = 40kilometers   # across-lead / mean wind
const Ly = 12kilometers   # along-lead
const Lz = 3kilometers    # vertical

const Nx = 640
const Ny = 192
const Nz = 128

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
# A single top-hat mask `χ(x)` — open water inside a band of width `Wˡᵉᵃᵈ`,
# ice outside, with a smooth `δˡᵉᵃᵈ` transition — defines the geometry of *every*
# surface flux.
#
# **Lead width.** Real leads span meters to several kilometers. The penetration
# and vigor scale with width (Glendening 1994; Zulauf & Krueger 2003: ≈ 180/220/
# 300 m penetration for 200/400/800 m leads, +≈ 1 m s⁻¹ peak updraft per width
# doubling). Leads narrower than ≈ 4 km force a *single* merged plume onto the
# lead axis; wider leads (4–10 km) develop edge plumes plus interior convective
# cells (Esau 2007). We use the 1 km canonical width: narrow (0.1–0.5 km),
# moderate (1–2 km), and wide (4–10 km) are the regimes you can sweep.
#
# **Lead sensible heat flux.** We default to `Qʰ_lead = 200 W m⁻²`, a
# representative lead-averaged winter value; the strong-but-credible upper case is
# 300 W m⁻² and the lower end ≈ 100 W m⁻². Tetzlaff et al. (2015) measured
# near-surface sensible fluxes of 15–180 W m⁻²; LES/2D lead studies (Glendening
# 1994; Zulauf & Krueger 2003; Esau 2007; Gryschka et al. 2023) impose/diagnose
# several-hundred W m⁻² over the open-water patch for ice–water ΔT of 20–40 K.
# Because this dry run omits the (comparable, in reality) latent heat flux, even
# 200 W m⁻² of *pure* sensible heating is vigorous enough for a clear 15-min plume.
#
# **Ice sensible heat flux.** We set `Qʰ_ice = 0`, a clean idealization that makes
# the lead the only heat source and keeps the contrast unambiguous. The real
# stable ABL over thick ice has a weak *downward* sensible flux of a few to
# ≈ 20 W m⁻² (i.e. ≈ −10 W m⁻²); zero is not literally observed but sharpens the
# pedagogy.

const Wˡᵉᵃᵈ = 1kilometer    # narrow 0.1–0.5, moderate 1–2, wide 4–10 km
const δˡᵉᵃᵈ = 100meters

const Qʰ_ice  = FT(0)      # W m⁻², sensible heat flux (real ice ≈ −10; 0 idealizes)
const Qʰ_lead = FT(200)    # W m⁻², lead-averaged (range 100–300; 300 = strong case)
const τx_ice  = FT(0.01)   # N m⁻², surface stress magnitude
const τx_lead = FT(0.05)

# **Latent heat / moisture flux.** Over a winter lead the latent heat flux is
# comparable to the sensible flux (Tetzlaff et al. 2015). We prescribe a lead
# evaporation rate `E_lead = 1e-4 kg m⁻² s⁻¹`, i.e. a latent heat flux
# `Lᵥ·E ≈ 2.5e6 × 1e-4 ≈ 250 W m⁻²` — comparable to the 200 W m⁻² sensible flux.
# The ice surface supplies no moisture (`E_ice = 0`).
const E_ice   = FT(0)      # kg m⁻² s⁻¹, moisture (water-vapor) flux
const E_lead  = FT(3e-4)   # strong lead evaporation (≈ 750 W m⁻² latent): the
                           # latent-dominated "sea smoke" regime that saturates the
                           # cold near-surface air and forms fog.

# ### Sensible heat flux on ρθ
#
# A kinematic heat flux `w′θ′ = Qʰ / (ρ cₚ)` corresponds to a flux of the
# prognostic `ρθ` equal to `Qʰ / cₚ`. A positive (upward) flux warms the
# atmosphere above.

@inline function ρθ_flux(x, y, t, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    Qʰ = p.Q_ice + χ * (p.Q_lead - p.Q_ice)
    return Qʰ / p.cₚ
end

ρθ_flux_parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, Q_ice = Qʰ_ice, Q_lead = Qʰ_lead, cₚ)
ρθ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρθ_flux; parameters = ρθ_flux_parameters))

# ### Moisture flux on ρqᵉ
#
# `E` is already a mass flux (kg m⁻² s⁻¹), exactly the flux of the moisture
# prognostic `ρqᵉ`. A positive (upward) flux moistens the air above (matching
# `bomex.jl`, where a positive `w′qᵗ′` moistens).

@inline function ρqᵉ_flux(x, y, t, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    return p.E_ice + χ * (p.E_lead - p.E_ice)
end

ρqᵉ_flux_parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, E_ice, E_lead)
ρqᵉ_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρqᵉ_flux; parameters = ρqᵉ_flux_parameters))

# ### Momentum flux (drag)
#
# We prescribe the stress through a spatially varying friction velocity, `τ = ρ u★²`,
# so here the open water drags the flow more than the ice. Writing it as a drag —
# proportional to and opposing the local velocity — makes the sign automatic.
# With ρ ≈ 1.3 kg m⁻³, `τx_lead = 0.05` and `τx_ice = 0.01 N m⁻²` give friction
# velocities u★ ≈ 0.20 and 0.09 m s⁻¹.
#
# !!! note "Lead-vs-ice drag ratio is an idealization"
#     A smooth open-water bulk estimate gives τ ≈ ρ C_DN U² ≈ 0.13 N m⁻² at 8 m s⁻¹,
#     so `τx_lead = 0.05` is on the low side — a defensible *smooth-young-ice* lead.
#     More importantly, the chosen `τx_lead > τx_ice` is **not universal**:
#     ridged/snow-covered pack ice and floe edges (form drag) are often
#     aerodynamically *rougher* than smooth open water, so in reality `τx_ice` can
#     exceed `τx_lead`. The qualitative point — heterogeneous surface drag — holds
#     either way.
#
# !!! note "Sign convention (verified)"
#     A bottom flux is *added* to the tendency in Breeze
#     (`BoundaryConditions/compute_flux_bcs.jl`), so a positive `ρθ` flux warms the
#     air (matching `bomex.jl` where `w′θ′ = +8e-3` heats) and a *negative* `ρu`
#     flux removes momentum — exactly the drag form used in `bomex.jl` and
#     `neutral_atmospheric_boundary_layer.jl`. The signs here are correct.

@inline function ρu_drag(x, y, t, ρu, ρv, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    u★² = p.u★²_ice + χ * (p.u★²_lead - p.u★²_ice)
    return - p.ρ₀ * u★² * ρu / max(sqrt(ρu^2 + ρv^2), p.ϵ)
end

@inline function ρv_drag(x, y, t, ρu, ρv, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    u★² = p.u★²_ice + χ * (p.u★²_lead - p.u★²_ice)
    return - p.ρ₀ * u★² * ρv / max(sqrt(ρu^2 + ρv^2), p.ϵ)
end

drag_parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, ρ₀,
                     u★²_ice = τx_ice / ρ₀, u★²_lead = τx_lead / ρ₀, ϵ = FT(1e-6))
ρu_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρu_drag; field_dependencies = (:ρu, :ρv), parameters = drag_parameters))
ρv_bcs = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρv_drag; field_dependencies = (:ρu, :ρv), parameters = drag_parameters))

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

simulation = Simulation(model; Δt = 0.5, stop_time = 40minutes)
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
Qʰ_field = Field{Center, Center, Nothing}(grid)
set!(Qʰ_field, (x, y) -> Qʰ_ice + top_hat(x; center = 0, width = Wˡᵉᵃᵈ, edge = δˡᵉᵃᵈ) * (Qʰ_lead - Qʰ_ice))

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

simulation.output_writers[:statics] = JLD2Writer(model, (; χ = χ_field, Qʰ = Qʰ_field);
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

# ## Visualization
#
# A vertical transect of vertical velocity `w(x, z, t)` shows the plume rising
# over the lead until the inversion caps it and it spreads downwind; the
# potential-temperature transect shows the warm anomaly leaning and advecting over
# the downwind ice. We build a movie from the high-cadence slices and a
# final-frame figure.

using CairoMakie

if isfile(slice_name(config))
    w_xz = FieldTimeSeries(slice_name(config), "w_xz")
    θ_xz = FieldTimeSeries(slice_name(config), "θ_xz")
    qˡ_xz = FieldTimeSeries(slice_name(config), "qˡ_xz")
    times = w_xz.times
    Nt = length(times)

    xw, _, zw = nodes(w_xz)
    xkm = xw ./ 1e3
    zkm = zw ./ 1e3

    n = Observable(Nt)
    wn = @lift interior(w_xz[$n], :, 1, :)
    θn = @lift interior(θ_xz[$n], :, 1, :)
    qln = @lift interior(qˡ_xz[$n], :, 1, :) .* 1e3   # g/kg
    title = @lift "Sea-ice lead plume — t = " * prettytime(times[$n])

    fig = Figure(size = (1100, 950))
    Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
    axw = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = "w (m s⁻¹)")
    axθ = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "θ (K)")
    axq = Axis(fig[3, 1], xlabel = "x (km)", ylabel = "z (km)", title = "cloud liquid qˡ (g kg⁻¹) — the lead fog")

    wlim = max(1e-3, maximum(abs, interior(w_xz[Nt])))
    qlmax = max(1e-4, maximum(interior(qˡ_xz[Nt])) * 1e3)
    hmw = heatmap!(axw, xkm, zkm, wn, colormap = :balance, colorrange = (-wlim, wlim))
    hmθ = heatmap!(axθ, xkm, zkm, θn, colormap = :thermal)
    hmq = heatmap!(axq, xkm, zkm, qln, colormap = :dense, colorrange = (0, qlmax))
    Colorbar(fig[1, 2], hmw)
    Colorbar(fig[2, 2], hmθ)
    Colorbar(fig[3, 2], hmq)

    save(figure_name(config, "atmosphere_lead_final_slice"), fig)

    if Nt > 1
        record(fig, movie_name(config, "lead_atmosphere_plume"), 1:Nt; framerate = 12) do i
            n[] = i
        end
        @info "Wrote movie" movie_name(config, "lead_atmosphere_plume")
    end
end

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
