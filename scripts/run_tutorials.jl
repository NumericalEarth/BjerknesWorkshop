#!/usr/bin/env julia
# # run_tutorials.jl
#
# Drive the tutorial deployment: generate scripts, run the selected cases
# resiliently (each in its own Julia subprocess), record status, and refresh the
# documentation status pages. This is the orchestrator the batch jobs call.
#
# Flow:
#   1. Generate Literate scripts for the selected cases (delegates to
#      `generate_literate_outputs.jl`, so generation scope == run scope).
#   2. For each selected case, skip it if `outputs_are_current` (unless
#      `FORCE_RERUN`); otherwise `run_case_resilient!` it in a subprocess.
#   3. Continue past failures when `ALLOW_CASE_FAILURES` is set; otherwise stop
#      at the first failure of a `critical` case.
#   4. `write_summary_pages!` and print a successes/skips/failures/stale table.
#   5. `exit(2)` iff `STRICT_CASES` is set and any selected case failed.
#
# Environment:
#   RUN_CASES / RUN_DAYS    which cases to run (see TutorialWorkflow)
#   FORCE_RERUN=1           run even if outputs_are_current
#   ALLOW_CASE_FAILURES=1   keep going after a (critical) failure
#   STRICT_CASES=1          exit(2) if any selected case failed
#   GENERATE_NOTEBOOKS=1    also emit notebooks in the generate step
#   SIMULATE_CASE_FAILURE   force-fail slugs (passed through to the runner)
#
# Usage:
#   julia --project=. scripts/run_tutorials.jl

using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "TutorialWorkflow.jl"))
using .TutorialWorkflow

_truthy(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")
_flag(envvar::AbstractString, default::Bool = false) =
    haskey(ENV, envvar) ? _truthy(ENV[envvar]) : default

# ---------------------------------------------------------------------------
# Step 1: generate scripts (same selection scope as the run).
# ---------------------------------------------------------------------------

function generate_scripts()
    gen = joinpath(ROOT, "scripts", "generate_literate_outputs.jl")
    @info "Generating Literate scripts" gen
    cmd = setenv(`$(Base.julia_cmd()) --project=$(ROOT) $(gen)`, ENV; dir = ROOT)
    proc = run(ignorestatus(cmd))
    if proc.exitcode != 0
        @warn "Script generation exited nonzero; continuing (missing scripts will fail individually)" exitcode = proc.exitcode
    end
    return proc.exitcode
end

# ---------------------------------------------------------------------------
# Step 2-3: run the selected cases.
# ---------------------------------------------------------------------------

@enum CaseOutcome SUCCEEDED SKIPPED FAILED

function run_selected(cases)
    force_rerun = _flag("FORCE_RERUN")
    allow_failures = _flag("ALLOW_CASE_FAILURES")

    results = Tuple{TutorialCase,CaseOutcome}[]
    aborted = false

    for case in cases
        if !force_rerun && outputs_are_current(case; root = ROOT)
            @info "Skipping (outputs current)" slug = case.slug
            push!(results, (case, SKIPPED))
            continue
        end

        @info "Running case" slug = case.slug day = case.day critical = case.critical
        ok = run_case_resilient!(case; root = ROOT)
        push!(results, (case, ok ? SUCCEEDED : FAILED))

        if !ok && case.critical && !allow_failures
            @error "Critical case failed and ALLOW_CASE_FAILURES not set; aborting remaining cases" slug = case.slug
            aborted = true
            break
        elseif !ok
            @warn "Case failed; continuing" slug = case.slug allow_failures critical = case.critical
        end
    end

    # Cases never reached because we aborted: record them as skipped-due-to-abort.
    if aborted
        reached = Set(c.slug for (c, _) in results)
        for case in cases
            case.slug in reached || push!(results, (case, SKIPPED))
        end
    end

    return results, aborted
end

# ---------------------------------------------------------------------------
# Step 4: summary table.
# ---------------------------------------------------------------------------

function print_summary(results, aborted)
    successes = [c for (c, o) in results if o == SUCCEEDED]
    skips     = [c for (c, o) in results if o == SKIPPED]
    failures  = [c for (c, o) in results if o == FAILED]
    # "stale" = selected but its outputs are not current after the run (failed or
    # never produced fresh artifacts) — a heads-up that docs may show old/no data.
    stale = TutorialCase[]
    for (c, _) in results
        outputs_are_current(c; root = ROOT) || push!(stale, c)
    end

    println()
    println("=" ^ 78)
    println("Tutorial run summary")
    println("=" ^ 78)
    @printf("  %-22s %s\n", "selected cases:", length(results))
    @printf("  %-22s %s\n", "succeeded:", length(successes))
    @printf("  %-22s %s\n", "skipped (current):", length(skips))
    @printf("  %-22s %s\n", "failed:", length(failures))
    @printf("  %-22s %s\n", "stale (not current):", length(stale))
    aborted && println("  NOTE: run aborted early on a critical failure.")
    println("-" ^ 78)
    @printf("  %-26s %-6s %-10s %s\n", "case", "day", "outcome", "current?")
    println("-" ^ 78)
    for (c, o) in results
        outcome = o == SUCCEEDED ? "success" : o == SKIPPED ? "skip" : "FAIL"
        cur = outputs_are_current(c; root = ROOT) ? "yes" : "no"
        @printf("  %-26s %-6d %-10s %s\n", c.slug, c.day, outcome, cur)
    end
    println("=" ^ 78)
    println()

    return (; successes, skips, failures, stale)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    all_cases = case_registry(ROOT)
    selected = selected_cases(all_cases)

    if isempty(selected)
        @info "No cases selected (RUN_CASES/RUN_DAYS); nothing to run."
        # Still refresh status pages so docs reflect the (unchanged) state.
        write_summary_pages!(all_cases; root = ROOT)
        return 0
    end

    @info "Selected cases" slugs = [c.slug for c in selected]

    generate_scripts()

    results, aborted = run_selected(selected)

    # Always refresh the documentation status pages from the on-disk JSON.
    try
        pages = write_summary_pages!(all_cases; root = ROOT)
        @info "Wrote status pages" pages
    catch err
        @warn "Failed to write summary pages" exception = (err, catch_backtrace())
    end

    summary = print_summary(results, aborted)

    if _flag("STRICT_CASES") && !isempty(summary.failures)
        @error "STRICT_CASES set and cases failed; exiting 2" failed = [c.slug for c in summary.failures]
        return 2
    end

    return 0
end

exit(main())
