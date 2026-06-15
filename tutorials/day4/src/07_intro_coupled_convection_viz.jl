# # Visualizing: a warm filament drives 2D coupled convection
#
# *This is the **visualization** half of the case. The simulation ran on a GPU before
# this page was built and cached its output; everything here executes live during the
# docs build, reading that cached output to draw the figures and record the animation —
# so these are the genuine production-resolution results.*

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics

atmos_file = "coupled_convection_atmosphere.jld2"
ocean_file = "coupled_convection_ocean.jld2"
flux_file  = "coupled_convection_fluxes.jld2"

wa_t = FieldTimeSeries(atmos_file, "w_a")
wo_t = FieldTimeSeries(ocean_file, "w_o")
To_t = FieldTimeSeries(ocean_file, "T_o")
Qs_t = FieldTimeSeries(flux_file,  "Q_sensible")
Ql_t = FieldTimeSeries(flux_file,  "Q_latent")
τx_t = FieldTimeSeries(flux_file,  "τx")
times = wa_t.times
Nt = length(times)
println("Loaded ", Nt, " frames spanning ", prettytime(times[1]), " – ", prettytime(times[end]))

xa, _, za = nodes(wa_t)
xo, _, zo = nodes(wo_t)
xa_km = xa ./ 1e3
xo_km = xo ./ 1e3

# ## The initial condition: a surface-trapped warm ocean filament
#
# Before the model steps, the ocean carries a warm Gaussian filament centered in the
# domain, trapped in the upper ~15 m of a 25 m mixed layer, over a weakly stratified
# interior. This is the boundary heterogeneity the coupled system responds to — the
# warm band is where the air–sea contrast is largest, so it is where the heat loss,
# the convection, and the mixed-layer deepening will concentrate.

T_ic = interior(To_t[1], :, 1, :)
@printf("Initial ocean temperature: background ≈ %.1f °C, filament peak ≈ %.1f °C\n",
        minimum(T_ic), maximum(T_ic))

fig_ic = Figure(size = (900, 360))
ax_ic = Axis(fig_ic[1, 1], xlabel = "x (km)", ylabel = "z (m)",
             title = "initial ocean temperature (°C) — the warm filament")
hm_ic = heatmap!(ax_ic, xo_km, zo, T_ic, colormap = :thermal)
Colorbar(fig_ic[1, 2], hm_ic, label = "T (°C)")
save("coupled_convection_initial_condition.png", fig_ic)
fig_ic

# ## Coupled convection
#
# A two-panel figure stacks the two fluids across the interface at `z = 0`: atmospheric
# vertical velocity `w(x, z, t)` on top (thermals rising over the warm filament) and
# ocean vertical velocity `w(x, z, t)` on the bottom (cooled water sinking beneath it).
# The shared blue–red colormap makes the *symmetry* of convection above and below the
# interface visible, and both are concentrated over the filament. Note the very
# different *speeds*: atmospheric thermals reach metres per second, while ocean
# convection — driven by the same heat flux but in water that is ~1000× denser and has
# ~4× the heat capacity — is an order of magnitude slower. Each panel is therefore
# scaled to its own fluid.

n = Observable(Nt)
wan = @lift interior(wa_t[$n], :, 1, :)
won = @lift interior(wo_t[$n], :, 1, :)
title = @lift "Coupled convection over a warm filament — t = " * prettytime(times[$n])

## Fixed, symmetric color limits per fluid. The atmosphere grows monotonically, so its
## series max is fine. The ocean spikes briefly during spin-up, so scaling to that max
## washes out the developed plumes — instead use a high percentile of |w| over the
## developed period (after the first quarter of the run) so the slow plumes are visible.
wa_lim = max(1e-3, maximum(abs, wa_t))
wo_dev = reduce(vcat, (abs.(vec(interior(wo_t[i], :, 1, :))) for i in (Nt ÷ 4):Nt))
wo_lim = max(1e-5, quantile(wo_dev, 0.99))

fig = Figure(size = (1100, 800))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
axa = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (m)",
           title = "atmosphere: vertical velocity w (m s⁻¹)")
axo = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (m)",
           title = "ocean: vertical velocity w (m s⁻¹)")

hma = heatmap!(axa, xa_km, za, wan, colormap = :balance, colorrange = (-wa_lim, wa_lim))
hmo = heatmap!(axo, xo_km, zo, won, colormap = :balance, colorrange = (-wo_lim, wo_lim))
Colorbar(fig[1, 2], hma)
Colorbar(fig[2, 2], hmo)

save("coupled_convection.png", fig)
fig

# ## Animation

record(fig, "coupled_convection.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="coupled_convection.mp4" style="max-width:100%"></video>
# ```

# ## Analysis of the air–sea fluxes
#
# The surface fluxes are *computed* by similarity theory at the interface — never
# prescribed. Two views tie them to the filament: (left) the final-time flux profiles
# along `x` — sensible, latent, and total heat flux — showing the exchange concentrated
# over the warm water; (right) a Hovmöller (time–`x`) of the sensible heat flux, showing
# the filament's imprint switch on and persist through the run.

## Each flux is a 1D line along x; stack the frames into an (x, t) matrix for the Hovmöller.
Qs_xt = reduce(hcat, (interior(Qs_t[i], :, 1, 1) for i in 1:Nt))   # Nx × Nt
Qsf = interior(Qs_t[Nt], :, 1, 1)
Qlf = interior(Ql_t[Nt], :, 1, 1)
τxf = interior(τx_t[Nt], :, 1, 1)

ic = length(xa) ÷ 2 + 1   # filament-center column
@printf("Final fluxes at the filament center: 𝒬ᵀ ≈ %.0f W/m², 𝒬ᵛ ≈ %.0f W/m², τˣ ≈ %.3f kg m⁻¹ s⁻²\n",
        Qsf[ic], Qlf[ic], τxf[ic])

fig_flux = Figure(size = (1250, 430))

ax_prof = Axis(fig_flux[1, 1], xlabel = "x (km)", ylabel = "heat flux (W m⁻²)",
               title = "surface heat flux at t = $(prettytime(times[Nt]))")
lines!(ax_prof, xa_km, Qsf, color = :firebrick,  linewidth = 2, label = "sensible 𝒬ᵀ")
lines!(ax_prof, xa_km, Qlf, color = :dodgerblue, linewidth = 2, label = "latent 𝒬ᵛ")
lines!(ax_prof, xa_km, Qsf .+ Qlf, color = :black, linewidth = 3, label = "total")
axislegend(ax_prof, position = :rt)

ax_hov = Axis(fig_flux[1, 2], xlabel = "x (km)", ylabel = "time (hours)",
              title = "sensible heat flux 𝒬ᵀ (W m⁻²)")
Qlim = max(1e-3, maximum(abs, Qs_xt))
hm_hov = heatmap!(ax_hov, xa_km, times ./ 3600, Qs_xt, colormap = :balance, colorrange = (-Qlim, Qlim))
Colorbar(fig_flux[1, 3], hm_hov, label = "𝒬ᵀ (W m⁻²)")

save("coupled_convection_fluxes.png", fig_flux)
fig_flux
