# # Visualizing the five acts
#
# *This is the **visualization** half of the intro atmosphere case. The five
# simulations ran on a GPU before this page was built and cached their output;
# everything below executes live during the docs build, reading that cached
# output — one quick, single-variable movie per act (until the finale, where we
# allow ourselves four panels).*

using Breeze   # so the terrain-following grids in the output files reconstruct natively
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

# Shared bits: the background profile (to form perturbations) and the terrain
# shapes (to outline on the panels). Every movie plays every saved frame at
# 24 fps, with H.264 compression keeping the files small.

θ₀, N², g = 290, 1e-4, 9.81
θ̄(z) = θ₀ * exp(N² * z / g)

h₀, a, σᵐ = 600, 1e3, 1.5e3
agnesi(x) = h₀ / (1 + (x / a)^2)
mountain(x, y) = h₀ * exp(-(x^2 + y^2) / (2σᵐ^2))
nothing #hide

# ## Act I — the thermal bubble
#
# One variable: the potential temperature *perturbation* `θ′ = θ - θ̄(z)`. Watch
# the bubble roll up into the mushroom vortex pair, overshoot its neutral level,
# and ring the stratification with gravity waves.

θt = FieldTimeSeries("thermal_bubble.jld2", "θ")
times = θt.times
Nt = length(times)
println("Act I: ", Nt, " frames spanning ", prettytime(times[end]))

xθ, _, zθ = nodes(θt)
θ̄row = reshape(θ̄.(zθ), 1, :)

n = Observable(Nt)
θ′ = @lift interior(θt[$n], :, 1, :) .- θ̄row
title = @lift "Act I — thermal bubble: θ′ (K), t = " * prettytime(times[$n])

fig = Figure(size = (900, 420))
ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = title)
hm = heatmap!(ax, xθ ./ 1e3, zθ ./ 1e3, θ′, colormap = :balance, colorrange = (-5, 5))
Colorbar(fig[1, 2], hm, label = "θ′ (K)")
ylims!(ax, 0, 6)

save("thermal_bubble.png", fig)

record(fig, "thermal_bubble.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="thermal_bubble.mp4" style="max-width:100%"></video>
# ```

# ## Acts II & III — free convection, two formulations
#
# One variable: vertical velocity `w`. Top: anelastic. Bottom: split-explicit
# compressible. Both runs start from bit-identical noise, so early frames match
# almost perfectly; once the convection is fully turbulent the two fields
# decorrelate frame-by-frame (chaos!) while remaining statistical twins — same
# boundary-layer depth, same thermal spacing, same `max |w|`.

w_ane = FieldTimeSeries("free_convection_anelastic.jld2", "w")
w_aco = FieldTimeSeries("free_convection_acoustic.jld2", "w")
times = w_ane.times
Nt = min(length(times), length(w_aco.times))
println("Acts II & III: ", Nt, " frames spanning ", prettytime(times[Nt]))

xw, _, zw = nodes(w_ane)
wlim = max(maximum(abs, w_ane[Nt]), maximum(abs, w_aco[Nt]))

n = Observable(Nt)
wa = @lift interior(w_ane[$n], :, 1, :)
wc = @lift interior(w_aco[$n], :, 1, :)
title = @lift "Acts II & III — free convection: w (m s⁻¹), t = " * prettytime(times[$n])

fig = Figure(size = (900, 620))
Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
ax1 = Axis(fig[1, 1], ylabel = "z (km)", title = "anelastic (CFL 0.7)")
ax2 = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)",
           title = "split-explicit compressible (CFL 1.0)")
hm = heatmap!(ax1, xw ./ 1e3, zw ./ 1e3, wa, colormap = :balance, colorrange = (-wlim, wlim))
heatmap!(ax2, xw ./ 1e3, zw ./ 1e3, wc, colormap = :balance, colorrange = (-wlim, wlim))
Colorbar(fig[1:2, 2], hm, label = "w (m s⁻¹)")
ylims!.((ax1, ax2), 0, 4)

save("free_convection.png", fig)

record(fig, "free_convection.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="free_convection.mp4" style="max-width:100%"></video>
# ```

# ## Act IV — lee waves over the Witch of Agnesi
#
# Still one variable: `w`. The standing wave over the crest tilts upstream with
# height — the signature of upward energy radiation — while the convective
# boundary layer keeps bubbling underneath and a turbulent wake churns in the
# lee. (The heatmap uses the nominal vertical coordinate; terrain-following
# levels sit slightly higher over the hill, so we draw the hill on top.)

w_lee = FieldTimeSeries("agnesi_lee_waves.jld2", "w")
times = w_lee.times
Nt = length(times)
println("Act IV: ", Nt, " frames spanning ", prettytime(times[Nt]))

xw, _, zw = nodes(w_lee)
hill_km = [agnesi(x) / 1e3 for x in xw]
wlim = 0.8 * maximum(abs, w_lee[Nt])

n = Observable(Nt)
wn = @lift interior(w_lee[$n], :, 1, :)
title = @lift "Act IV — lee waves: w (m s⁻¹), t = " * prettytime(times[$n])

fig = Figure(size = (900, 420))
ax = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = title)
hm = heatmap!(ax, xw ./ 1e3, zw ./ 1e3, wn, colormap = :balance, colorrange = (-wlim, wlim))
band!(ax, xw ./ 1e3, zero(hill_km), hill_km, color = :grey25)
Colorbar(fig[1, 2], hm, label = "w (m s⁻¹)")
ylims!(ax, 0, 6)

save("lee_waves.png", fig)

record(fig, "lee_waves.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="lee_waves.mp4" style="max-width:100%"></video>
# ```

# ## Act V — clouds and drizzle on the mountain
#
# The finale earns a four-panel view. Left column, the along-wind slice through
# the summit: vertical velocity, and cloud water with the drizzle shaft drawn in
# rain-rate contours. Right column, the view from above: the cloud field at
# ≈ 1.5 km, and the rain that reaches the lowest model level — the drizzle map.

wxz   = FieldTimeSeries("mountain_clouds.jld2", "wxz")
qᶜˡxz = FieldTimeSeries("mountain_clouds.jld2", "qᶜˡxz")
qʳxz  = FieldTimeSeries("mountain_clouds.jld2", "qʳxz")
qᶜˡxy = FieldTimeSeries("mountain_clouds.jld2", "qᶜˡxy")
qʳxy  = FieldTimeSeries("mountain_clouds.jld2", "qʳxy")
times = wxz.times
Nt = length(times)
println("Act V: ", Nt, " frames spanning ", prettytime(times[Nt]))

x, _, z = nodes(wxz)      # w lives on z-faces…
_, _, zq = nodes(qᶜˡxz)   # …the moisture fields on z-centers
xc, yc, _ = nodes(qᶜˡxy)
xkm, zkm, zqkm, xckm, yckm = x ./ 1e3, z ./ 1e3, zq ./ 1e3, xc ./ 1e3, yc ./ 1e3
ridge_km = [mountain(x, 0) / 1e3 for x in x]

g2kg(q) = 1e3 .* q   # plot moist fields in g kg⁻¹

wlim  = max(0.8 * maximum(abs, wxz[Nt]), 0.1)
qᶜmax = max(0.8e3 * maximum(qᶜˡxz), 1e-2)   # g kg⁻¹; floors guard cloud-free smoke runs
qʳmax = max(0.8e3 * maximum(qʳxy), 1e-3)

n = Observable(Nt)
w_n  = @lift interior(wxz[$n],   :, 1, :)
qc_n = @lift g2kg(interior(qᶜˡxz[$n], :, 1, :))
qr_n = @lift g2kg(interior(qʳxz[$n],  :, 1, :))
cl_n = @lift g2kg(interior(qᶜˡxy[$n], :, :, 1))
rn_n = @lift g2kg(interior(qʳxy[$n],  :, :, 1))
title = @lift "Act V — mountain clouds and drizzle, t = " * prettytime(times[$n])

fig = Figure(size = (1300, 700))
Label(fig[0, 1:4], title, fontsize = 18, tellwidth = false)

axw = Axis(fig[1, 1], ylabel = "z (km)", title = "w along the wind (m s⁻¹)")
axq = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)",
           title = "cloud water (g kg⁻¹) + rain contours")
axc = Axis(fig[1, 3], ylabel = "y (km)", title = "cloud water at ≈ 1.5 km (g kg⁻¹)")
axr = Axis(fig[2, 3], xlabel = "x (km)", ylabel = "y (km)", title = "drizzle at the surface (g kg⁻¹)")

hmw = heatmap!(axw, xkm, zkm, w_n, colormap = :balance, colorrange = (-wlim, wlim))
hmq = heatmap!(axq, xkm, zqkm, qc_n, colormap = :Blues, colorrange = (0, qᶜmax))
contour!(axq, xkm, zqkm, qr_n, levels = [0.01, 0.05, 0.2], color = :purple)
hmc = heatmap!(axc, xckm, yckm, cl_n, colormap = :Blues, colorrange = (0, qᶜmax))
hmr = heatmap!(axr, xckm, yckm, rn_n, colormap = :Purples, colorrange = (0, qʳmax))

for ax in (axw, axq)
    band!(ax, xkm, zero(ridge_km), ridge_km, color = :grey25)
    ylims!(ax, 0, 6)
end
for ax in (axc, axr)
    contour!(ax, xckm, yckm, [mountain(x, y) for x in xc, y in yc],
             levels = [150, 350, 550], color = :grey30)
end

Colorbar(fig[1, 2], hmw, label = "w (m s⁻¹)")
Colorbar(fig[2, 2], hmq, label = "qᶜˡ (g kg⁻¹)")
Colorbar(fig[1, 4], hmc, label = "qᶜˡ (g kg⁻¹)")
Colorbar(fig[2, 4], hmr, label = "qʳ (g kg⁻¹)")

save("mountain_clouds.png", fig)

record(fig, "mountain_clouds.mp4", 1:Nt; framerate = 24, compression = 28) do i
    n[] = i
end
nothing #hide

# ```@raw html
# <video autoplay loop muted playsinline controls src="mountain_clouds.mp4" style="max-width:100%"></video>
# ```
