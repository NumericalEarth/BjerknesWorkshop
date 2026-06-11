# # Thursday gallery and discussion
#
# *Boundary heterogeneity writes turbulence into the fluid.*
#
# This source loads the outputs of the three case studies and assembles the
# connected Thursday story. It does **not** launch expensive runs — it reads the
# JLD2 output written by `01`, `02`, and `03` and plots. Run those first (or point
# `RUN_CLASS` at whichever class you produced).
#
# The arc:
#
# 1. **A crack in the ice** — a narrow warm/moist strip writes a turbulent plume
#    into a cold boundary layer.
# 2. **Beneath the crack** — the same strip cools and salts the ocean while waves
#    organize turbulence below.
# 3. **Fjords as boundary conditions** — a real coastline and mountains impose
#    geometry and flux heterogeneity on the boundary layer.
#
# > The boundary is not a passive wall. It is an active script that writes
# > structure into the fluid.

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

# Helper: load the last frame of a slice field from a cached output file, if present.
function last_slice(file, field)
    isfile(file) || return nothing
    try
        ts = FieldTimeSeries(file, field)
        return (; ts, frame = ts[length(ts.times)], nodes = nodes(ts), time = ts.times[end])
    catch err
        @warn "Could not load slice" file field exception = err
        return nothing
    end
end

# ## Panel 1 — the atmospheric lead plume

fig = Figure(size = (1300, 1100), fontsize = 14)
Label(fig[0, 1:2], "Thursday: boundary heterogeneity writes turbulence into the fluid",
      fontsize = 20, tellwidth = false)

p1 = last_slice("lead_atmosphere_slices.jld2", "w_xz")
ax1 = Axis(fig[1, 1:2], title = "1. A crack in the ice — atmospheric w(x,z)",
           xlabel = "x (km)", ylabel = "z (km)")
if !isnothing(p1)
    x, _, z = p1.nodes
    data = interior(p1.frame, :, 1, :)
    wlim = max(1e-3, maximum(abs, data))
    hm = heatmap!(ax1, x ./ 1e3, z ./ 1e3, data, colormap = :balance, colorrange = (-wlim, wlim))
    Colorbar(fig[1, 3], hm, label = "w (m s⁻¹)")
else
    text!(ax1, 0.5, 0.5, text = "run 01 first", align = (:center, :center), space = :relative)
end

# ## Panel 2 — ocean below the lead: no waves vs. waves
#
# The signature comparison: how surface waves reorganize the turbulence beneath
# the same crack.

ax2a = Axis(fig[2, 1], title = "2a. Ocean below lead — NO waves, w(x,z)", xlabel = "x (m)", ylabel = "z (m)")
ax2b = Axis(fig[2, 2], title = "2b. Ocean below lead — waves, w(x,z)", xlabel = "x (m)", ylabel = "z (m)")

p2a = last_slice("ocean_lead_nowaves_slices.jld2", "w_xz")
p2b = last_slice("ocean_lead_waves_slices.jld2", "w_xz")
for (ax, p) in ((ax2a, p2a), (ax2b, p2b))
    if !isnothing(p)
        x, _, z = p.nodes
        data = interior(p.frame, :, 1, :)
        wlim = max(1e-5, maximum(abs, data))
        heatmap!(ax, x, z, data, colormap = :balance, colorrange = (-wlim, wlim))
    else
        text!(ax, 0.5, 0.5, text = "run 02 first", align = (:center, :center), space = :relative)
    end
end

# ## Panel 3 — Norway terrain

ax3 = Axis(fig[3, 1:2], title = "3. Fjords as boundary conditions — near-surface wind",
           xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
pu = last_slice("norway_slices.jld2", "u_xy")
pv = last_slice("norway_slices.jld2", "v_xy")
if !isnothing(pu) && !isnothing(pv)
    x, y, _ = pu.nodes
    speed = sqrt.(interior(pu.frame, :, :, 1).^2 .+ interior(pv.frame, :, :, 1).^2)
    hm = heatmap!(ax3, x ./ 1e3, y ./ 1e3, speed, colormap = :speed)
    Colorbar(fig[3, 3], hm, label = "|u| (m s⁻¹)")
else
    text!(ax3, 0.5, 0.5, text = "run 03 first (needs terrain artifact from 03a)",
          align = (:center, :center), space = :relative)
end

save("thursday_gallery.png", fig)
@info "Saved gallery figure thursday_gallery.png"
fig

# ## Ocean waves-vs-no-waves vertical profiles
#
# A quantitative companion to the panels above: the horizontally averaged vertical
# velocity variance ⟨w²⟩(z), the clearest single signature of wave-organized
# turbulence.

function load_profile(file, field)
    isfile(file) || return nothing
    try
        return FieldTimeSeries(file, field)
    catch err
        @warn "Could not load profile" file field exception = err
        return nothing
    end
end

w²_nw = load_profile("ocean_lead_nowaves_profiles.jld2", "w²")
w²_w  = load_profile("ocean_lead_waves_profiles.jld2", "w²")
if !isnothing(w²_nw) || !isnothing(w²_w)
    fig2 = Figure(size = (500, 600))
    ax = Axis(fig2[1, 1], xlabel = "⟨w²⟩ (m² s⁻²)", ylabel = "z (m)", title = "Wave organization of TKE")
    ## ⟨w²⟩ lives at z-Faces (w is a Face field), so use each field's own z-nodes.
    profile(ts) = (interior(ts[length(ts.times)], 1, 1, :), znodes(ts[length(ts.times)]))
    if !isnothing(w²_nw); p, z = profile(w²_nw); lines!(ax, p, z, label = "no waves"); end
    if !isnothing(w²_w);  p, z = profile(w²_w);  lines!(ax, p, z, label = "waves");    end
    axislegend(ax, position = :rb)
    save("ocean_waves_vs_nowaves_profiles.png", fig2)
    @info "Saved profile comparison ocean_waves_vs_nowaves_profiles.png"
end

@info "Gallery complete."
nothing #hide
