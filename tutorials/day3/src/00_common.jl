# # Wednesday shared infrastructure (placeholder)
#
# *Hybrid physics + machine learning, and differentiable Earth-system models.*
#
# Day 3 (Wednesday) is, for now, a lightweight pair of placeholder tutorials whose
# only job is to exercise the deployment pipeline cheaply while the real content
# is written. This shared file collects the trivial helpers the two placeholder
# cases reuse.
#
# Like the day4 sources, anything written here honors `CASE_OUTPUT_DIR` when the
# workflow sets it, and otherwise falls back to a local `output` directory so the
# script stays runnable standalone.

module Day3Common

export case_output_dir

# Resolve the directory artifacts should be written to.
function case_output_dir()
    dir = get(ENV, "CASE_OUTPUT_DIR", joinpath("output", "day3_standalone"))
    mkpath(dir)
    return dir
end

end # module Day3Common

nothing #hide
