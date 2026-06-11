# # Visualizing: ocean turbulence below the lead — waves vs no waves
#
# *This is the **visualization** half of the case. The two simulations (no-waves
# control and the Stokes-drift / Langmuir waves case) ran on a GPU before this page
# was built and cached their output; everything here executes live during the build,
# reading that cached output to draw the transects and record the animations — the
# genuine production-resolution results.*

using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

include(joinpath(@__DIR__, "00_common.jl"))
using .ThursdayLES

# A small helper: load a case's `w(x,z,t)` transect, draw the final frame, and record
# the animation. Both the no-waves control and the waves case go through it.

function visualize_case(label)
    config = RunConfig("02_ocean_lead_$(label)")
    w_xz = FieldTimeSeries(slice_name(config), "w_xz")
    times = w_xz.times
    Nt = length(times)
    println(label, ": loaded ", Nt, " frames, max|w| = ",
            round(maximum(abs, interior(w_xz[Nt])); sigdigits = 3), " m/s")
    xw, _, zw = nodes(w_xz)

    n = Observable(Nt)
    wn = @lift interior(w_xz[$n], :, 1, :)
    pretty = label == "waves" ? "waves (Langmuir)" : "no waves (control)"
    title = @lift "Ocean below the lead — $(pretty) — t = " * prettytime(times[$n])

    fig = Figure(size = (1100, 450))
    Label(fig[0, 1:2], title, fontsize = 18, tellwidth = false)
    ax = Axis(fig[1, 1], xlabel = "x (m)", ylabel = "z (m)", title = "vertical velocity w (m s⁻¹)")
    wlim = max(1e-5, maximum(abs, interior(w_xz[Nt])))
    hm = heatmap!(ax, xw, zw, wn, colormap = :balance, colorrange = (-wlim, wlim))
    Colorbar(fig[1, 2], hm)

    record(fig, movie_name(config, "ocean_lead_$(label)"), 1:Nt; framerate = 12) do i
        n[] = i
    end
    return fig
end

# ## No-waves control
#
# Brine-rejection convection under the lead with no surface-wave forcing: plumes sink
# from the cooled, salted surface in roughly cellular, shear/convection-driven
# overturning.

fig_nowaves = visualize_case("nowaves")

# ```@raw html
# <video autoplay loop muted playsinline controls src="ocean_lead_nowaves.mp4" style="max-width:100%"></video>
# ```

# ## With surface waves (Langmuir turbulence)
#
# Adding the localized Stokes drift over the open lead (`Laₜ ≈ 0.30`, the
# wave-favorable regime) organizes the overturning into deeper, wind-aligned Langmuir
# cells — the wave-driven enhancement of vertical transport the case is built to show.

fig_waves = visualize_case("waves")

# ```@raw html
# <video autoplay loop muted playsinline controls src="ocean_lead_waves.mp4" style="max-width:100%"></video>
# ```
