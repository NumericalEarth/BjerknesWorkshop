# # Beneath the crack: ocean turbulence, brine rejection, and wave-driven Langmuir structures
#
# *Boundary heterogeneity writes turbulence into the fluid — case 2 of 3.*
#
# Directly below the sea-ice lead of case 1, the ocean feels the same crack from
# the other side: localized surface cooling, **brine rejection** as the exposed
# water freezes and salt is expelled into the liquid below, wind stress, and
# **surface-wave** forcing. The waves enter through a prescribed Stokes-drift
# profile that interacts with the flow vorticity (the Craik–Leibovich mechanism)
# and organizes turbulence into Langmuir-like roll structures.
#
# The physical setting follows the LES literature on convection and Langmuir
# turbulence beneath leads and in the marginal ice zone. Brine-driven plume
# convection under a refreezing lead is described by Skyllingstad & Denbo (2001)
# and Smith (2002); the combination of Langmuir circulation with convection goes
# back to Skyllingstad & Denbo (1995). The Craik–Leibovich vortex-force mechanism
# that organizes Langmuir cells derives from Craik & Leibovich (1976), and the
# canonical LES of Langmuir turbulence and the turbulent Langmuir number `Laₜ` is
# McWilliams, Sullivan & Moeng (1997). Belcher et al. (2012) place a given forcing
# in the global wind/wave/convection regime diagram via `Laₜ`, and Harcourt &
# D'Asaro (2008) refine this with the surface-layer Langmuir number `La_{SL}`.
# Pearson, Grant, Polton & Belcher (2015) is the reference for Langmuir turbulence
# acting *under a surface buoyancy flux* — exactly the convection-plus-waves
# competition simulated here — and Tavri, Horvat, Pearson et al. (2026) apply this
# `Laₜ`-regime framework specifically to leads and the marginal ice zone.
#
# This is an **Oceananigans nonhydrostatic ocean-only LES**. Surface waves are
# *wave-averaged*: we do not resolve the free surface. Oceananigans solves the
# Craik–Leibovich equations using the Lagrangian-mean velocity as the prognostic
# momentum, so passing a `StokesDrift` is all that is required — the vortex force
# is added automatically.
#
# The whole point of this case is the comparison, so the script runs **both** the
# no-waves control and the waves case, one after the other, and writes a separate
# set of outputs for each. The gallery (`04_...`) loads both.
#
# Coordinate orientation:
#
# ```text
# x = across-lead
# y = along-lead / wind / wave direction
# z = vertical, z = 0 at the surface, negative downward
# ```
#
# !!! note "References"
#     - Pearson, B. C., Grant, A. L. M., Polton, J. A. & Belcher, S. E. (2015).
#       Langmuir turbulence and surface heating. *J. Phys. Oceanogr.* **45**,
#       2897–2911. https://doi.org/10.1175/JPO-D-15-0018.1
#     - Tavri, A., Horvat, C., Pearson, B. et al. (2026). *The Cryosphere* **20**,
#       3073–3089. https://doi.org/10.5194/tc-20-3073-2026
#     - McWilliams, J. C., Sullivan, P. P. & Moeng, C.-H. (1997). *J. Fluid Mech.*
#       **334**, 1–30. https://doi.org/10.1017/S0022112096004375
#     - Craik, A. D. D. & Leibovich, S. (1976). *J. Fluid Mech.* **73**, 401–426.
#       https://doi.org/10.1017/S0022112076001420
#     - Belcher, S. E. et al. (2012). *Geophys. Res. Lett.* **39**, L18605.
#       https://doi.org/10.1029/2012GL052932
#     - Skyllingstad, E. D. & Denbo, D. W. (2001). *J. Geophys. Res.* **106**,
#       2477–2497. https://doi.org/10.1029/1999JC000091
#     - Skyllingstad, E. D. & Denbo, D. W. (1995). *J. Geophys. Res.* **100**,
#       8501–8522. https://doi.org/10.1029/94JC03202
#     - Smith, D. C. (2002). *J. Geophys. Res.* **107**, 3022.
#       https://doi.org/10.1029/2001JC000822
#     - Harcourt, R. R. & D'Asaro, E. A. (2008). *J. Phys. Oceanogr.* **38**,
#       1542–1562. https://doi.org/10.1175/2007JPO3842.1

using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Printf
using Random
using CairoMakie

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

arch = choose_architecture()
gpu_report()
Oceananigans.defaults.FloatType = Float32
const FT = Float32

# ## Domain, grid, and fixed parameters
#
# A 2 km × 1 km × 160 m domain at 6.25 m horizontal resolution with a stretched
# vertical refined near the surface: 320 × 160 × 128 ≈ 6.6 million cells per run.

const Lx = 2kilometers   # across-lead
const Ly = 1kilometer    # along-lead / wind / waves
const Lz = 160meters     # depth

const Nx = 320
const Ny = 160
const Nz = 128

const refinement = 1.2   # higher → finer near surface
const stretching = 8.0   # higher → faster coarsening at depth
hᵏ(k) = (k - 1) / Nz
ζ₀(k) = 1 + (hᵏ(k) - 1) / refinement
Σ(k)  = (1 - exp(-stretching * hᵏ(k))) / (1 - exp(-stretching))
z_faces(k) = Lz * (ζ₀(k) * Σ(k) - 1)

# Buoyancy: temperature and salinity with the TEOS-10 equation of state.
const ρₒ = FT(1026)   # kg m⁻³, reference surface density

# The lead: the same top-hat mask as case 1 (narrower here), centered on the lead.
const Wˡᵉᵃᵈ = 500meters
const δˡᵉᵃᵈ = 50meters

const cᴾ = FT(3991)        # J K⁻¹ kg⁻¹, seawater heat capacity
const Qᵀ_lead = FT(200)    # W m⁻², surface heat loss (cooling) over the lead.
                           #   Conservative refreezing-lead heat loss, Skyllingstad & Denbo (2001).
const Fˢ_lead = FT(2e-5)   # g kg⁻¹ m s⁻¹, brine-rejection salt input over the lead.
                           #   Self-consistent with Qᵀ via the freezing rate (Skyllingstad & Denbo 2001).
const τʸ_lead = FT(-1e-4)  # m² s⁻², along-lead kinematic wind stress over the lead.
                           #   u★ ≈ √|τ| ≈ 0.010 m s⁻¹ (≈ 7–8 m s⁻¹ wind); keeps the
                           #   forcing Langmuir-dominated (low Laₜ, see banner below).
const dTdz = FT(0.005)     # K m⁻¹, interior thermal stratification.
const dSdz = FT(0.02)      # g kg⁻¹ m⁻¹, interior haline stratification (halocline).
                           #   Near the freezing point buoyancy is dominated by salinity,
                           #   so a stable halocline below the fresh mixed layer sets the
                           #   restratification that the plumes and Langmuir cells work against.

# Surface waves: a monochromatic deep-water wave field traveling along `y`. Its
# Stokes drift decays as `exp(2k z)`; we pass the vertical derivative `∂z vˢ`,
# and Oceananigans builds the vortex force internally. `const` lets the Stokes
# functions compile on the GPU.
const wave_amplitude = FT(0.8)               # m, swell-like amplitude (a local wind sea would be ≈ 0.4–0.6 m)
const wavelength     = FT(60)                # m, swell long enough to penetrate the whole mixed layer
const wavenumber     = FT(2π) / wavelength   # m⁻¹
const g_earth        = FT(9.81)
const wave_frequency = sqrt(g_earth * wavenumber)              # deep-water dispersion ω = √(g k)
const Uˢ             = wave_amplitude^2 * wavenumber * wave_frequency  # surface Stokes drift ≈ 0.068 m s⁻¹
# Stokes-drift e-folding depth 1/(2k) ≈ 4.8 m. A longer swell (80–150 m) would
# shear the full mixed layer and deepen the active Langmuir region.

@inline ∂z_vˢ(z, t) = 2 * wavenumber * Uˢ * exp(2 * wavenumber * z)

# ### Surface flux functions
#
# A positive surface temperature flux is upward (cooling); `Jᵀ = Qᵀ/(ρ cᴾ)`.
# Brine rejection adds salt into the ocean (a negative upward flux). Wind stress
# acts on `v` (along the lead). All localized by the lead mask.
#
# !!! note "Sign convention (verified)"
#     A positive *top* tracer flux fluxes the quantity upward, out of the ocean
#     (Oceananigans `BoundaryConditions`; cf. the `ocean_wind_mixing_and_convection.jl`
#     comment "a positive temperature flux at the surface implies cooling"). So a
#     positive `Jᵀ` cools, and brine rejection — which *adds* salt — is a negative
#     `Jˢ`. The wind stress on `v` is negative (like the example's negative `τx`),
#     which accelerates the along-lead current. These signs are correct.

@inline function Jᵀ(x, y, t, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    return χ * p.Qᵀ / (p.ρₒ * p.cᴾ)
end

@inline function Jˢ(x, y, t, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    return - χ * p.Fˢ
end

@inline function τy(x, y, t, p)
    χ = top_hat(x; center = 0, width = p.W, edge = p.δ)
    return χ * p.τ
end

# ## One run (no waves or waves)
#
# Everything for a single simulation: grid, boundary conditions, model, initial
# conditions, time stepping, output, and visualization. Called twice below.

function run_ocean_case(waves::Bool)
    label = waves ? "waves" : "nowaves"
    config = RunConfig(string("02_ocean_lead_", label))
    @info "=== Ocean below lead: $(label) ==="

    grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (-Lx/2, Lx/2), y = (-Ly/2, Ly/2), z = z_faces,
                           topology = (Periodic, Periodic, Bounded))
    @info "Vertical grid" Δz_surface = minimum(zspacings(grid, Center())) Δz_max = maximum(zspacings(grid, Center()))
    memory_report(Nx, Ny, Nz; FT, nfields = 6)

    equation_of_state = TEOS10EquationOfState(reference_density = ρₒ)
    seawater_buoyancy = SeawaterBuoyancy(FT; equation_of_state)

    T_top = FluxBoundaryCondition(Jᵀ; parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, Qᵀ = Qᵀ_lead, ρₒ, cᴾ))
    T_bcs = FieldBoundaryConditions(top = T_top, bottom = GradientBoundaryCondition(dTdz))
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jˢ; parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, Fˢ = Fˢ_lead)),
                                    bottom = GradientBoundaryCondition(dSdz))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τy; parameters = (; W = Wˡᵉᵃᵈ, δ = δˡᵉᵃᵈ, τ = τʸ_lead)))

    stokes_drift = waves ? UniformStokesDrift(∂z_vˢ = ∂z_vˢ) : nothing

    model = NonhydrostaticModel(grid;
                                advection = WENO(order = 9),
                                tracers = (:T, :S),
                                buoyancy = seawater_buoyancy,
                                coriolis = FPlane(f = 1.4e-4),
                                closure = AnisotropicMinimumDissipation(),
                                stokes_drift,
                                boundary_conditions = (; T = T_bcs, S = S_bcs, v = v_bcs))

    ## Initial conditions: a shallow mixed layer over weak stratification, uniform
    ## salinity, and velocity noise scaled by the friction velocity.
    Random.seed!(2718)
    initial_mixed_layer_depth = FT(20)   # shallow fresh mixed layer; deepen only with a longer swell
    T_surface = FT(-1.7)                 # ≈ surface freezing point at S₀, so brine rejection is the
                                         #   active forcing (not mere cooling of warm water)
    S₀ = FT(33)                          # g kg⁻¹, fresh near-surface salinity
    Ξ(z) = randn(FT) * exp(z / 8)
    Tᵢ(x, y, z) = T_surface + dTdz * min(z + initial_mixed_layer_depth, zero(z)) + 1e-3 * dTdz * Lz * Ξ(z)
    ## Mixed layer of uniform S₀ over a stable halocline (haline restratification).
    Sᵢ(x, y, z) = S₀ - dSdz * min(z + initial_mixed_layer_depth, zero(z))
    u★ = sqrt(abs(τʸ_lead))
    uᵢ(x, y, z) = u★ * 1e-1 * Ξ(z)
    set!(model, T = Tᵢ, S = Sᵢ, u = uᵢ, v = uᵢ, w = uᵢ)

    ## Turbulent Langmuir number Laₜ = √(u★ / Uˢ) (McWilliams, Sullivan & Moeng 1997).
    ## Laₜ ≲ 0.3–0.5 → Langmuir-dominated; ≈ 1 → shear-dominated. With waves on,
    ## Uˢ ≈ 0.068 m s⁻¹ and u★ ≈ 0.010 m s⁻¹ give Laₜ ≈ 0.38, firmly Langmuir-dominated.
    ## Belcher et al. (2012) / Pearson et al. (2015) place this regime where waves
    ## reorganize the convective + wind-driven turbulence into Langmuir cells.
    if waves
        Laₜ = sqrt(u★ / Uˢ)
        @info @sprintf("[%s] Langmuir diagnostics: u★ = %.4f m/s, Uˢ = %.4f m/s, Laₜ = %.2f (Langmuir-dominated)",
                       label, u★, Uˢ, Laₜ)
    else
        @info @sprintf("[%s] No waves: shear/convection only (Uˢ = 0, Laₜ → ∞)", label)
    end

    simulation = Simulation(model; Δt = 1.0, stop_time = 30minutes)
    conjure_time_step_wizard!(simulation, cfl = 0.7, max_Δt = 30.0)

    wall_clock = Ref(time_ns())
    function progress(sim)
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("[%s] Iter %d, t %s, Δt %s, wall %s, max|w| %.2e m/s",
                       label, iteration(sim), prettytime(sim), prettytime(sim.Δt),
                       prettytime(elapsed), maximum(abs, sim.model.velocities.w))
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(100))

    ## Outputs: velocity, tracers, vorticity, TKE proxy; the lead mask; horizontal
    ## w slices at several depths; a vertical transect; horizontally averaged profiles.
    u, v, w = model.velocities
    T, S = model.tracers.T, model.tracers.S
    e = (u^2 + v^2 + w^2) / 2

    χ_field = Field{Center, Center, Nothing}(grid)
    set!(χ_field, (x, y) -> top_hat(x; center = 0, width = Wˡᵉᵃᵈ, edge = δˡᵉᵃᵈ))

    jmid = Ny ÷ 2 + 1
    zc = Array(znodes(grid, Center()))
    depths = (5, 20, 40)
    k_at(d) = clamp(searchsortedfirst(zc, -float(d)), 1, length(zc))

    ## Slice outputs must be views of Fields, not AbstractOperations.
    slice_outputs = (
        w_xz = view(w, :, jmid, :),
        T_xz = view(T, :, jmid, :),
        S_xz = view(S, :, jmid, :),
        w_xy_5  = view(w, :, :, k_at(depths[1])),
        w_xy_20 = view(w, :, :, k_at(depths[2])),
        w_xy_40 = view(w, :, :, k_at(depths[3])),
    )

    averages = (
        U  = Average(u, dims = (1, 2)),
        V  = Average(v, dims = (1, 2)),
        T̄  = Average(T, dims = (1, 2)),
        S̄  = Average(S, dims = (1, 2)),
        w² = Average(w^2, dims = (1, 2)),
        E  = Average(e, dims = (1, 2)),
    )

    simulation.output_writers[:statics] = JLD2Writer(model, (; χ = χ_field);
        filename = output_name(config, "statics"), schedule = IterationInterval(typemax(Int)),
        overwrite_existing = true)
    simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
        filename = slice_name(config), schedule = TimeInterval(15seconds), overwrite_existing = true)
    simulation.output_writers[:profiles] = JLD2Writer(model, averages;
        filename = output_name(config, "profiles"),
        schedule = AveragedTimeInterval(30seconds, window = 30seconds), overwrite_existing = true)
    ## (No full-3D field output — slices and profiles drive the visualization.)

    write_once!(simulation.output_writers[:statics], model)
    run!(simulation)

    ## A vertical transect of w(x, z, t) across the lead, plus a final-frame figure.
    if isfile(slice_name(config))
        w_xz = FieldTimeSeries(slice_name(config), "w_xz")
        times = w_xz.times
        Nt = length(times)
        xw, _, zw = nodes(w_xz)

        n = Observable(Nt)
        wn = @lift interior(w_xz[$n], :, 1, :)
        title = @lift string(waves ? "Ocean below lead (waves)" : "Ocean below lead (no waves)",
                             " — t = ", prettytime(times[$n]))

        fig = Figure(size = (1100, 450))
        Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
        ax = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "w (m s⁻¹)")
        wlim = max(1e-5, maximum(abs, interior(w_xz[Nt])))
        hm = heatmap!(ax, xw, zw, wn, colormap = :balance, colorrange = (-wlim, wlim))
        Colorbar(fig[1, 2], hm)
        save(figure_name(config, string("ocean_lead_", label, "_final_slice")), fig)
        if Nt > 1
            record(fig, movie_name(config, string("ocean_lead_", label)), 1:Nt; framerate = 12) do i
                n[] = i
            end
        end
    end

    @info "Ocean case complete" run_stamp(config)...
    return nothing
end

# ## Run both cases
#
# No-waves control first, then the waves case. The two write distinct outputs so
# the gallery can place them side by side.

run_ocean_case(false)
run_ocean_case(true)

@info "Case 2 complete (no-waves control + waves)."
nothing #hide
