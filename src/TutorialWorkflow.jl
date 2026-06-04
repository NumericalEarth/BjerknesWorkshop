"""
    TutorialWorkflow

Core of the manual tutorial deployment workflow for the Bjerknes workshop.

This module is the contract other agents build against. It defines the
[`TutorialCase`](@ref) description of a tutorial, a [`case_registry`](@ref) of
every day-3 and day-4 case, environment-driven selection of which days/cases to
run and document, a per-run output layout under `output/dayN/<slug>/runs/...`,
JSON status pointers (`latest_success.json` / `latest_attempt.json`), a
resilient subprocess runner ([`run_case_resilient!`](@ref)) that never throws,
and currency checks ([`outputs_are_current`](@ref)) based on source / parameter /
manifest hashes.

Only the Julia standard library is used (`Dates`, `SHA`, `TOML`, `UUIDs`,
`Printf`). JSON is hand-rolled (see [`atomic_write`](@ref) and the `_json_*`
helpers) to avoid external dependencies.
"""
module TutorialWorkflow

using Dates
using SHA
using TOML
using UUIDs
using Printf

export TutorialCase, RunInfo,
       case_registry,
       selected_days, selected_cases, selected_doc_days,
       case_output_root, new_run_dir,
       latest_success, latest_attempt,
       safe_latest_success, safe_artifact,
       outputs_are_current,
       run_case_resilient!,
       write_status!, write_summary_pages!,
       file_hash, parameter_hash, manifest_hash, atomic_write

# ============================================================================
# TutorialCase
# ============================================================================

"""
    TutorialCase

Canonical description of one tutorial case. The `source` is the Literate `.jl`
file (the canonical artifact); `generated_script` is the plain script produced
from it by `Literate.script` and is what the runner actually executes. All path
fields are stored *relative to the repository root* and are resolved against the
root passed to the workflow functions.
"""
Base.@kwdef struct TutorialCase
    day::Int
    name::String
    slug::String
    source::String              # tutorials/dayN/src/XX.jl  (Literate source, canonical)
    generated_script::String    # tutorials/dayN/scripts/XX.jl (Literate.script output)
    output_root::String         # output/dayN/<slug>
    required_outputs::Vector{String}
    optional_outputs::Vector{String} = String[]
    parameters::NamedTuple = NamedTuple()
    critical::Bool = false
    description::String = ""
end

"""
    RunInfo

Parsed contents of a `latest_success.json` / `latest_attempt.json` pointer (and
the per-run `status.json` they mirror). `run_dir` is an absolute path to the run
directory; `artifacts_dir` is `<run_dir>/artifacts`. `extra` carries any
additional JSON fields not promoted to named fields.
"""
Base.@kwdef struct RunInfo
    run_id::String              = ""
    status::String              = ""   # "success" | "failure" | "running" | "simulated_failure"
    slug::String                = ""
    day::Int                    = 0
    run_dir::String             = ""
    artifacts_dir::String       = ""
    run_class::String           = ""
    parameter_hash::String      = ""
    source_hash::String         = ""
    manifest_hash::String       = ""
    required_present::Bool      = false
    missing_outputs::Vector{String} = String[]
    started::String             = ""
    finished::String            = ""
    duration_s::Float64         = 0.0
    exit_code::Int              = -1
    message::String             = ""
    extra::Dict{String,Any}     = Dict{String,Any}()
end

# ============================================================================
# Minimal JSON (hand-rolled, stdlib-only)
# ============================================================================

_json_escape(s::AbstractString) = sprint() do io
    for c in s
        if c == '"';      print(io, "\\\"")
        elseif c == '\\'; print(io, "\\\\")
        elseif c == '\n'; print(io, "\\n")
        elseif c == '\r'; print(io, "\\r")
        elseif c == '\t'; print(io, "\\t")
        elseif c < '\x20'; print(io, "\\u", lpad(string(UInt16(c), base = 16), 4, '0'))
        else; print(io, c)
        end
    end
end

_json_value(io, v::AbstractString)   = print(io, '"', _json_escape(v), '"')
_json_value(io, v::Symbol)           = _json_value(io, String(v))
_json_value(io, v::Bool)             = print(io, v ? "true" : "false")
_json_value(io, v::Integer)          = print(io, v)
_json_value(io, ::Nothing)           = print(io, "null")
function _json_value(io, v::AbstractFloat)
    if isfinite(v); print(io, v) else print(io, "null") end
end
function _json_value(io, v::AbstractVector)
    print(io, '[')
    for (i, x) in enumerate(v)
        i > 1 && print(io, ',')
        _json_value(io, x)
    end
    print(io, ']')
end
function _json_value(io, v::AbstractDict)
    print(io, '{')
    first = true
    for (k, x) in v
        first || print(io, ',')
        first = false
        print(io, '"', _json_escape(string(k)), '"', ':')
        _json_value(io, x)
    end
    print(io, '}')
end
_json_value(io, v) = _json_value(io, string(v))   # fallback

"""
    json_string(dict) -> String

Serialize an (ordered) collection of pairs / a Dict to a JSON object string.
Accepts a `Vector{Pair}` to preserve key order, or any `AbstractDict`.
"""
function json_string(pairs::AbstractVector{<:Pair})
    io = IOBuffer()
    print(io, '{')
    for (i, (k, v)) in enumerate(pairs)
        i > 1 && print(io, ',')
        print(io, '"', _json_escape(string(k)), '"', ':')
        _json_value(io, v)
    end
    print(io, '}')
    return String(take!(io))
end
json_string(d::AbstractDict) = (io = IOBuffer(); _json_value(io, d); String(take!(io)))

# --- a tiny recursive-descent JSON parser (objects, arrays, strings, numbers,
#     bools, null). Sufficient for the flat status pointers we write. ---

mutable struct _JParser
    s::String
    i::Int
end

function _jskip_ws!(p::_JParser)
    n = lastindex(p.s)
    while p.i <= n
        c = p.s[p.i]
        (c == ' ' || c == '\n' || c == '\r' || c == '\t') || break
        p.i = nextind(p.s, p.i)
    end
end

function _jparse_value(p::_JParser)
    _jskip_ws!(p)
    c = p.s[p.i]
    if c == '{';      return _jparse_object(p)
    elseif c == '[';  return _jparse_array(p)
    elseif c == '"';  return _jparse_string(p)
    elseif c == 't';  p.i += 4; return true
    elseif c == 'f';  p.i += 5; return false
    elseif c == 'n';  p.i += 4; return nothing
    else;             return _jparse_number(p)
    end
end

function _jparse_string(p::_JParser)
    p.i = nextind(p.s, p.i)  # opening quote
    io = IOBuffer()
    n = lastindex(p.s)
    while p.i <= n
        c = p.s[p.i]
        if c == '"'
            p.i = nextind(p.s, p.i)
            return String(take!(io))
        elseif c == '\\'
            p.i = nextind(p.s, p.i)
            e = p.s[p.i]
            if e == 'n'; print(io, '\n')
            elseif e == 't'; print(io, '\t')
            elseif e == 'r'; print(io, '\r')
            elseif e == 'u'
                hex = p.s[p.i+1:p.i+4]
                print(io, Char(parse(UInt16, hex, base = 16)))
                p.i += 4
            else; print(io, e)
            end
            p.i = nextind(p.s, p.i)
        else
            print(io, c)
            p.i = nextind(p.s, p.i)
        end
    end
    return String(take!(io))
end

function _jparse_number(p::_JParser)
    n = lastindex(p.s)
    start = p.i
    while p.i <= n && (isdigit(p.s[p.i]) || p.s[p.i] in ('-', '+', '.', 'e', 'E'))
        p.i = nextind(p.s, p.i)
    end
    tok = p.s[start:prevind(p.s, p.i)]
    v = tryparse(Int, tok)
    return v === nothing ? parse(Float64, tok) : v
end

function _jparse_array(p::_JParser)
    p.i = nextind(p.s, p.i)  # [
    arr = Any[]
    _jskip_ws!(p)
    if p.s[p.i] == ']'; p.i = nextind(p.s, p.i); return arr; end
    while true
        push!(arr, _jparse_value(p))
        _jskip_ws!(p)
        c = p.s[p.i]; p.i = nextind(p.s, p.i)
        c == ']' && break
    end
    return arr
end

function _jparse_object(p::_JParser)
    p.i = nextind(p.s, p.i)  # {
    obj = Dict{String,Any}()
    _jskip_ws!(p)
    if p.s[p.i] == '}'; p.i = nextind(p.s, p.i); return obj; end
    while true
        _jskip_ws!(p)
        key = _jparse_string(p)
        _jskip_ws!(p)
        p.i = nextind(p.s, p.i)  # :
        obj[key] = _jparse_value(p)
        _jskip_ws!(p)
        c = p.s[p.i]; p.i = nextind(p.s, p.i)
        c == '}' && break
    end
    return obj
end

"""
    parse_json(str) -> Any

Parse a JSON document into Julia `Dict{String,Any}` / `Vector{Any}` / scalars.
"""
parse_json(str::AbstractString) = _jparse_value(_JParser(String(str), 1))

# ============================================================================
# atomic write & hashing helpers
# ============================================================================

"""
    atomic_write(path, str)

Write `str` to `path` atomically: write to a temporary file in the same
directory, then `mv` it over the destination. Creates parent directories.
"""
function atomic_write(path::AbstractString, str::AbstractString)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    tmp = string(path, ".tmp.", string(uuid4())[1:8])
    open(tmp, "w") do io
        write(io, str)
    end
    mv(tmp, path; force = true)
    return path
end

"""
    file_hash(path) -> String

SHA-256 hex digest of a file's contents, or `""` if the file does not exist.
"""
function file_hash(path::AbstractString)
    isfile(path) || return ""
    return bytes2hex(open(sha256, path))
end

"""
    parameter_hash(nt::NamedTuple) -> String

Stable SHA-256 hex digest of a NamedTuple of parameters. Keys are sorted so the
hash is order-independent.
"""
function parameter_hash(nt::NamedTuple)
    isempty(nt) && return bytes2hex(sha256(""))
    ks = sort!(collect(keys(nt)))
    io = IOBuffer()
    for k in ks
        print(io, k, '=', repr(getfield(nt, k)), ';')
    end
    return bytes2hex(sha256(String(take!(io))))
end

"""
    manifest_hash(root=pwd()) -> String

SHA-256 hex digest of the project `Manifest.toml` at `root` (the resolved
dependency set). Returns `""` if no manifest is present.
"""
function manifest_hash(root::AbstractString = pwd())
    path = joinpath(root, "Manifest.toml")
    return file_hash(path)
end

# ============================================================================
# Registry
# ============================================================================

const DAY4_SRC = "tutorials/day4/src"
const DAY4_SCRIPTS = "tutorials/day4/scripts"
const DAY3_SRC = "tutorials/day3/src"
const DAY3_SCRIPTS = "tutorials/day3/scripts"

"""
    case_registry(root=pwd()) -> Vector{TutorialCase}

Every day-3 and day-4 tutorial case, in document order. Day-4 holds the three
GPU science cases (`lead_atmosphere`, `lead_ocean_waves`, `norway_100m`) plus a
tiny `smoke_case`; day-3 holds two lightweight placeholder cases. `root` is
accepted for symmetry but the returned paths are repo-relative.
"""
function case_registry(root::AbstractString = pwd())
    cases = TutorialCase[]

    # ---- Day 3 (lightweight placeholders) ----
    push!(cases, TutorialCase(
        day = 3, name = "Hybrid physics + ML", slug = "hybrid_physics_ml",
        source = joinpath(DAY3_SRC, "01_hybrid_physics_ml.jl"),
        generated_script = joinpath(DAY3_SCRIPTS, "01_hybrid_physics_ml.jl"),
        output_root = joinpath("output", "day3", "hybrid_physics_ml"),
        required_outputs = ["hybrid_physics_ml.png"],
        critical = false,
        description = "Placeholder: a physical core with a learned closure correction.",
    ))
    push!(cases, TutorialCase(
        day = 3, name = "Differentiable ESMs", slug = "differentiable_esms",
        source = joinpath(DAY3_SRC, "02_differentiable_esms.jl"),
        generated_script = joinpath(DAY3_SCRIPTS, "02_differentiable_esms.jl"),
        output_root = joinpath("output", "day3", "differentiable_esms"),
        required_outputs = ["differentiable_esms.png"],
        critical = false,
        description = "Placeholder: gradient-based calibration of an Earth-system component.",
    ))

    # ---- Day 4 (the working Thursday science cases) ----
    push!(cases, TutorialCase(
        day = 4, name = "Atmospheric turbulence over a sea-ice lead",
        slug = "lead_atmosphere",
        source = joinpath(DAY4_SRC, "01_atmospheric_turbulence_over_a_sea_ice_lead.jl"),
        generated_script = joinpath(DAY4_SCRIPTS, "01_atmospheric_turbulence_over_a_sea_ice_lead.jl"),
        output_root = joinpath("output", "day4", "lead_atmosphere"),
        required_outputs = [
            "01_lead_atmosphere_statics.jld2",
            "01_lead_atmosphere_slices.jld2",
            "01_lead_atmosphere_profiles.jld2",
        ],
        optional_outputs = [
            "atmosphere_lead_final_slice.png",
            "lead_atmosphere_plume.mp4",
        ],
        parameters = (Lx = 40_000, Ly = 12_000, Lz = 3_000, Nx = 640, Ny = 192, Nz = 128,
                      stop_minutes = 40, U0 = 8, Q_lead = 300),
        critical = true,
        description = "Breeze atmosphere-only LES of a convective plume over a sea-ice lead.",
    ))
    push!(cases, TutorialCase(
        day = 4, name = "Ocean turbulence below a lead with surface waves",
        slug = "lead_ocean_waves",
        source = joinpath(DAY4_SRC, "02_ocean_turbulence_below_a_lead_with_surface_waves.jl"),
        generated_script = joinpath(DAY4_SCRIPTS, "02_ocean_turbulence_below_a_lead_with_surface_waves.jl"),
        output_root = joinpath("output", "day4", "lead_ocean_waves"),
        required_outputs = [
            "02_ocean_lead_nowaves_statics.jld2",
            "02_ocean_lead_nowaves_slices.jld2",
            "02_ocean_lead_nowaves_profiles.jld2",
            "02_ocean_lead_waves_statics.jld2",
            "02_ocean_lead_waves_slices.jld2",
            "02_ocean_lead_waves_profiles.jld2",
        ],
        optional_outputs = [
            "ocean_lead_nowaves_final_slice.png",
            "ocean_lead_waves_final_slice.png",
            "ocean_lead_nowaves.mp4",
            "ocean_lead_waves.mp4",
        ],
        parameters = (Lx = 2_000, Ly = 1_000, Lz = 160, Nx = 320, Ny = 160, Nz = 128,
                      stop_minutes = 30, wavelength = 60, wave_amplitude = 0.8),
        critical = true,
        description = "Oceananigans nonhydrostatic LES below the lead: no-waves control vs. Craik-Leibovich waves.",
    ))
    push!(cases, TutorialCase(
        day = 4, name = "Norway 100 m prescribed fluxes",
        slug = "norway_100m",
        source = joinpath(DAY4_SRC, "03_norway_100m_prescribed_fluxes.jl"),
        generated_script = joinpath(DAY4_SCRIPTS, "03_norway_100m_prescribed_fluxes.jl"),
        output_root = joinpath("output", "day4", "norway_100m"),
        required_outputs = [
            "03_norway_100m_statics.jld2",
            "03_norway_100m_slices.jld2",
        ],
        optional_outputs = [
            "norway_final_w_slice.png",
            "norway_100m_prescribed_fluxes.mp4",
        ],
        parameters = (Lx = 100_000, Ly = 100_000, Lz = 12_000, Nx = 256, Ny = 256, Nz = 64,
                      stop_minutes = 15, U0 = 10),
        critical = false,
        description = "Breeze terrain-following LES over coastal Norway (Lofoten) with prescribed land/ocean fluxes.",
    ))
    push!(cases, TutorialCase(
        day = 4, name = "Smoke case", slug = "smoke_case",
        source = joinpath(DAY4_SRC, "99_smoke_case.jl"),
        generated_script = joinpath(DAY4_SCRIPTS, "99_smoke_case.jl"),
        output_root = joinpath("output", "day4", "smoke_case"),
        required_outputs = ["fields.jld2"],
        optional_outputs = ["summary.png"],
        critical = false,
        description = "Trivial CPU smoke test exercising the full deploy pipeline end-to-end.",
    ))

    return cases
end

# ============================================================================
# Selection (RUN_DAYS / RUN_CASES / DOC_DAYS)
# ============================================================================

function _parse_days(value::AbstractString, all_days::Vector{Int})
    v = strip(lowercase(value))
    if v == "all" || isempty(v)
        return sort!(unique(all_days))
    elseif v == "none"
        return Int[]
    else
        days = Int[]
        for tok in split(v, ',')
            t = strip(tok)
            isempty(t) && continue
            d = tryparse(Int, t)
            d === nothing || push!(days, d)
        end
        return sort!(unique(days))
    end
end

"""
    selected_days(envvar) -> Vector{Int}

Days selected by the environment variable named `envvar` (e.g. `"RUN_DAYS"`).
Accepts `all` (default when unset/empty), `none`, or a comma list like `3,4`.
Filtered to the days that actually exist in the registry.
"""
function selected_days(envvar::AbstractString)
    all_days = sort!(unique([c.day for c in case_registry()]))
    raw = get(ENV, envvar, "all")
    requested = _parse_days(raw, all_days)
    return filter(d -> d in all_days, requested)
end

"""
    selected_cases(all::Vector{TutorialCase}) -> Vector{TutorialCase}

Subset of `all` to run, honoring `RUN_CASES` then `RUN_DAYS`. `RUN_CASES=all`
(default) defers to `RUN_DAYS`; otherwise `RUN_CASES` is a comma list of slugs
and selects exactly those (intersected with `all`).
"""
function selected_cases(all::Vector{TutorialCase})
    raw_cases = strip(lowercase(get(ENV, "RUN_CASES", "all")))
    if raw_cases != "all" && !isempty(raw_cases)
        wanted = Set(strip(s) for s in split(raw_cases, ',') if !isempty(strip(s)))
        return filter(c -> lowercase(c.slug) in wanted, all)
    end
    days = Set(selected_days("RUN_DAYS"))
    return filter(c -> c.day in days, all)
end

"""
    selected_doc_days() -> Vector{Int}

Days to include when building documentation, honoring `DOC_DAYS` (same grammar
as `RUN_DAYS`). Defaults to all registry days.
"""
selected_doc_days() = selected_days("DOC_DAYS")

# ============================================================================
# Output layout & run directories
# ============================================================================

"""
    case_output_root(case; root=pwd()) -> String

Absolute path to the case's output root (`output/dayN/<slug>`).
"""
case_output_root(case::TutorialCase; root::AbstractString = pwd()) =
    abspath(joinpath(root, case.output_root))

_short_hash(s::AbstractString) = bytes2hex(sha256(s))[1:8]

"""
    new_run_dir(case; root=pwd()) -> String

Create and return a fresh run directory `output/dayN/<slug>/runs/<UTC-ts>_<shorthash>/`
with `artifacts/` and `logs/` subdirectories. The short hash is derived from the
timestamp and a UUID so concurrent runs do not collide.
"""
function new_run_dir(case::TutorialCase; root::AbstractString = pwd())
    ts = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSSsss")
    sh = _short_hash(string(ts, "-", uuid4()))
    rundir = joinpath(case_output_root(case; root), "runs", string(ts, "_", sh))
    mkpath(joinpath(rundir, "artifacts"))
    mkpath(joinpath(rundir, "logs"))
    return rundir
end

_pointer_path(case::TutorialCase, name::AbstractString; root::AbstractString = pwd()) =
    joinpath(case_output_root(case; root), name)

# ============================================================================
# Status (status.json + pointers) — read & write
# ============================================================================

"""
    write_status!(rundir; kwargs...) -> String

Atomically write `<rundir>/status.json` from the given keyword fields and return
its path. Vector and Dict values are serialized as JSON arrays/objects.
"""
function write_status!(rundir::AbstractString; kwargs...)
    pairs = Pair{String,Any}[]
    for (k, v) in kwargs
        push!(pairs, string(k) => v)
    end
    path = joinpath(rundir, "status.json")
    atomic_write(path, json_string(pairs))
    return path
end

# Build a RunInfo from a parsed JSON dict.
function _runinfo_from_dict(d::AbstractDict)
    get_s(k, default = "") = haskey(d, k) ? string(d[k]) : default
    get_i(k, default = 0)  = haskey(d, k) ? (d[k] isa Integer ? Int(d[k]) : something(tryparse(Int, string(d[k])), default)) : default
    get_f(k, default = 0.0) = haskey(d, k) ? Float64(d[k] isa Number ? d[k] : something(tryparse(Float64, string(d[k])), default)) : default
    get_b(k, default = false) = haskey(d, k) ? (d[k] === true) : default
    get_v(k) = haskey(d, k) && d[k] isa AbstractVector ? String[string(x) for x in d[k]] : String[]

    known = Set(["run_id","status","slug","day","run_dir","artifacts_dir","run_class",
                 "parameter_hash","source_hash","manifest_hash","required_present",
                 "missing_outputs","started","finished","duration_s","exit_code","message"])
    extra = Dict{String,Any}(k => v for (k, v) in d if !(k in known))

    return RunInfo(
        run_id = get_s("run_id"),
        status = get_s("status"),
        slug = get_s("slug"),
        day = get_i("day"),
        run_dir = get_s("run_dir"),
        artifacts_dir = get_s("artifacts_dir"),
        run_class = get_s("run_class"),
        parameter_hash = get_s("parameter_hash"),
        source_hash = get_s("source_hash"),
        manifest_hash = get_s("manifest_hash"),
        required_present = get_b("required_present"),
        missing_outputs = get_v("missing_outputs"),
        started = get_s("started"),
        finished = get_s("finished"),
        duration_s = get_f("duration_s"),
        exit_code = get_i("exit_code", -1),
        message = get_s("message"),
        extra = extra,
    )
end

function _read_pointer(path::AbstractString)
    isfile(path) || return nothing
    d = parse_json(read(path, String))
    d isa AbstractDict || return nothing
    return _runinfo_from_dict(d)
end

"""
    latest_success(case; root=pwd()) -> Union{RunInfo,Nothing}

Read `output/dayN/<slug>/latest_success.json`, or `nothing` if absent. May throw
on a corrupt file; use [`safe_latest_success`](@ref) for the non-throwing form.
"""
latest_success(case::TutorialCase; root::AbstractString = pwd()) =
    _read_pointer(_pointer_path(case, "latest_success.json"; root))

"""
    latest_attempt(case; root=pwd()) -> Union{RunInfo,Nothing}

Read `output/dayN/<slug>/latest_attempt.json`, or `nothing` if absent.
"""
latest_attempt(case::TutorialCase; root::AbstractString = pwd()) =
    _read_pointer(_pointer_path(case, "latest_attempt.json"; root))

"""
    safe_latest_success(case; root=pwd()) -> Union{RunInfo,Nothing}

Like [`latest_success`](@ref) but never throws; returns `nothing` on any error.
"""
function safe_latest_success(case::TutorialCase; root::AbstractString = pwd())
    try
        return latest_success(case; root)
    catch
        return nothing
    end
end

"""
    safe_artifact(case, name; root=pwd()) -> Union{String,Nothing}

Absolute path to artifact `name` from the latest successful run if that run
exists and the file is present on disk; otherwise `nothing`. Never throws.
"""
function safe_artifact(case::TutorialCase, name::AbstractString; root::AbstractString = pwd())
    try
        info = latest_success(case; root)
        info === nothing && return nothing
        adir = isempty(info.artifacts_dir) ? joinpath(info.run_dir, "artifacts") : info.artifacts_dir
        path = joinpath(adir, name)
        return isfile(path) ? path : nothing
    catch
        return nothing
    end
end

# ============================================================================
# Currency check
# ============================================================================

const _RUN_CLASS_DEFAULT = "production"

current_run_class() = get(ENV, "RUN_CLASS", _RUN_CLASS_DEFAULT)

"""
    outputs_are_current(case; root=pwd()) -> Bool

`true` iff the latest successful run is reusable: it exists with
`status == "success"`, all `required_outputs` are present on disk, and its
recorded `run_class`, `parameter_hash`, and `source_hash` match the current case
(source-hash and manifest-hash checks are skipped when `IGNORE_SOURCE_HASH` /
`IGNORE_MANIFEST_HASH` are set in the environment).
"""
function outputs_are_current(case::TutorialCase; root::AbstractString = pwd())
    info = safe_latest_success(case; root)
    info === nothing && return false
    info.status == "success" || return false

    # required artifacts present on disk
    adir = isempty(info.artifacts_dir) ? joinpath(info.run_dir, "artifacts") : info.artifacts_dir
    for req in case.required_outputs
        isfile(joinpath(adir, req)) || return false
    end

    # run-class match
    info.run_class == current_run_class() || return false

    # parameter hash
    info.parameter_hash == parameter_hash(case.parameters) || return false

    # source hash (unless ignored)
    if !haskey(ENV, "IGNORE_SOURCE_HASH")
        info.source_hash == file_hash(joinpath(root, case.source)) || return false
    end

    # manifest hash (unless ignored)
    if !haskey(ENV, "IGNORE_MANIFEST_HASH")
        info.manifest_hash == manifest_hash(root) || return false
    end

    return true
end

# ============================================================================
# Resilient runner
# ============================================================================

"""
    run_case_resilient!(case; root=pwd()) -> Bool

Run `case.generated_script` in a *separate* Julia process and record the result.
Never throws: returns `true` on success, `false` on any failure.

The child is launched as `julia --project=<root> <generated_script>` from `root`
with environment `CASE_OUTPUT_DIR=<run_dir>/artifacts`, `RUN_CLASS`, and
`DOCS_PHASE=run`. stdout/stderr are captured to `<run_dir>/logs/{stdout,stderr}.log`.
A per-run `status.json` is written atomically; `latest_attempt.json` is always
updated, and `latest_success.json` is updated only when the process exits 0 *and*
every `required_outputs` file is present.

Honors `SIMULATE_CASE_FAILURE` (a comma list of slugs, or `all`) to force a
failure without running. Honors `ALLOW_CASE_FAILURES`: when unset/false and a
`critical` case fails, the failure still returns `false` (callers decide whether
to abort) — the flag is recorded in the status for downstream tooling.
"""
function run_case_resilient!(case::TutorialCase; root::AbstractString = pwd())
    rundir = ""
    try
        rundir = new_run_dir(case; root)
    catch err
        @warn "Could not create run directory" slug = case.slug exception = (err, catch_backtrace())
        return false
    end

    artifacts = joinpath(rundir, "artifacts")
    logdir = joinpath(rundir, "logs")
    run_id = basename(rundir)
    run_class = current_run_class()
    src_hash = file_hash(joinpath(root, case.source))
    par_hash = parameter_hash(case.parameters)
    man_hash = manifest_hash(root)
    started = now(UTC)

    allow_failures = _truthy(get(ENV, "ALLOW_CASE_FAILURES", "false"))

    # Mark this attempt as running before we start.
    _write_pointer!(case, "latest_attempt.json", root;
        run_id, status = "running", run_dir = rundir, artifacts_dir = artifacts,
        run_class, parameter_hash = par_hash, source_hash = src_hash,
        manifest_hash = man_hash, started = string(started))

    # Simulated failure: skip the subprocess entirely.
    if _simulate_failure(case.slug)
        finished = now(UTC)
        dur = (finished - started).value / 1000
        _finish!(case, root, rundir, run_id; status = "simulated_failure",
            run_class, par_hash, src_hash, man_hash, started, finished, dur,
            exit_code = -1, required_present = false, missing = case.required_outputs,
            message = "SIMULATE_CASE_FAILURE active for this slug.",
            success = false, allow_failures)
        @warn "Simulated case failure" slug = case.slug
        return false
    end

    script = joinpath(root, case.generated_script)
    exit_code = -1
    proc_ok = false
    if !isfile(script)
        finished = now(UTC)
        dur = (finished - started).value / 1000
        _finish!(case, root, rundir, run_id; status = "failure",
            run_class, par_hash, src_hash, man_hash, started, finished, dur,
            exit_code = -1, required_present = false, missing = case.required_outputs,
            message = "Generated script not found: $script (run generate step first).",
            success = false, allow_failures)
        @warn "Generated script missing" slug = case.slug script
        return false
    end

    stdout_log = joinpath(logdir, "stdout.log")
    stderr_log = joinpath(logdir, "stderr.log")

    try
        env = copy(ENV)
        env["CASE_OUTPUT_DIR"] = artifacts
        env["RUN_CLASS"] = run_class
        env["DOCS_PHASE"] = "run"
        cmd = `$(Base.julia_cmd()) --project=$(root) $(script)`
        cmd = setenv(cmd, env; dir = root)
        open(stdout_log, "w") do out
            open(stderr_log, "w") do err
                proc = run(pipeline(cmd; stdout = out, stderr = err); wait = false)
                wait(proc)
                exit_code = proc.exitcode
            end
        end
        proc_ok = exit_code == 0
    catch err
        proc_ok = false
        exit_code = exit_code == 0 ? 1 : exit_code
        try
            open(stderr_log, "a") do io
                println(io, "\n[run_case_resilient! caught exception]")
                showerror(io, err)
                println(io)
            end
        catch
        end
    end

    # Check required artifacts.
    missing = String[req for req in case.required_outputs if !isfile(joinpath(artifacts, req))]
    required_present = isempty(missing)
    success = proc_ok && required_present

    finished = now(UTC)
    dur = (finished - started).value / 1000
    status = success ? "success" :
             (proc_ok ? "failure" : "failure")
    message = success ? "ok" :
              (!proc_ok ? "child process exited $exit_code" :
               "required outputs missing: $(join(missing, ", "))")

    _finish!(case, root, rundir, run_id; status,
        run_class, par_hash, src_hash, man_hash, started, finished, dur,
        exit_code, required_present, missing, message, success, allow_failures)

    if success
        @info "Case succeeded" slug = case.slug run_id duration_s = round(dur, digits = 1)
    else
        @warn "Case failed" slug = case.slug exit_code missing
    end
    return success
end

_truthy(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function _simulate_failure(slug::AbstractString)
    raw = strip(lowercase(get(ENV, "SIMULATE_CASE_FAILURE", "")))
    isempty(raw) && return false
    raw == "all" && return true
    return lowercase(slug) in Set(strip(s) for s in split(raw, ',') if !isempty(strip(s)))
end

# Common status-record fields for a finished attempt.
function _status_pairs(case, rundir, run_id; status, run_class, par_hash, src_hash,
                       man_hash, started, finished, dur, exit_code, required_present,
                       missing, message, allow_failures)
    return Pair{String,Any}[
        "run_id" => run_id,
        "status" => status,
        "slug" => case.slug,
        "day" => case.day,
        "name" => case.name,
        "critical" => case.critical,
        "run_dir" => rundir,
        "artifacts_dir" => joinpath(rundir, "artifacts"),
        "run_class" => run_class,
        "parameter_hash" => par_hash,
        "source_hash" => src_hash,
        "manifest_hash" => man_hash,
        "required_present" => required_present,
        "missing_outputs" => missing,
        "required_outputs" => case.required_outputs,
        "optional_outputs" => case.optional_outputs,
        "started" => string(started),
        "finished" => string(finished),
        "duration_s" => dur,
        "exit_code" => exit_code,
        "allow_case_failures" => allow_failures,
        "message" => message,
    ]
end

# Write status.json + latest_attempt.json (+ latest_success.json on success).
function _finish!(case, root, rundir, run_id; status, run_class, par_hash, src_hash,
                  man_hash, started, finished, dur, exit_code, required_present,
                  missing, message, success, allow_failures)
    pairs = _status_pairs(case, rundir, run_id; status, run_class, par_hash, src_hash,
                          man_hash, started, finished, dur, exit_code, required_present,
                          missing, message, allow_failures)
    js = json_string(pairs)
    try
        atomic_write(joinpath(rundir, "status.json"), js)
    catch err
        @warn "Failed to write status.json" exception = err
    end
    try
        atomic_write(_pointer_path(case, "latest_attempt.json"; root), js)
    catch err
        @warn "Failed to write latest_attempt.json" exception = err
    end
    if success
        try
            atomic_write(_pointer_path(case, "latest_success.json"; root), js)
        catch err
            @warn "Failed to write latest_success.json" exception = err
        end
    end
    return nothing
end

# Lightweight pointer write (used for the initial "running" mark).
function _write_pointer!(case, name, root; kwargs...)
    pairs = Pair{String,Any}[string(k) => v for (k, v) in kwargs]
    try
        atomic_write(_pointer_path(case, name; root), json_string(pairs))
    catch err
        @warn "Failed to write pointer" name exception = err
    end
    return nothing
end

# ============================================================================
# Status summary pages
# ============================================================================

_status_badge(s::AbstractString) =
    s == "success" ? "✅ success" :
    s == "running" ? "🟡 running" :
    s == "simulated_failure" ? "🟠 simulated failure" :
    s == "failure" ? "❌ failure" : "⚪ $(isempty(s) ? "no run" : s)"

function _case_row(case::TutorialCase; root)
    attempt = safe_latest_attempt(case; root)
    success = safe_latest_success(case; root)
    status = attempt === nothing ? "" : attempt.status
    run_id = attempt === nothing ? "—" : attempt.run_id
    when = attempt === nothing ? "—" : (isempty(attempt.finished) ? attempt.started : attempt.finished)
    dur = attempt === nothing ? "—" : @sprintf("%.1f s", attempt.duration_s)
    cur = outputs_are_current(case; root) ? "yes" : "no"
    return (; case, status, run_id, when, dur, cur, has_success = success !== nothing)
end

function safe_latest_attempt(case::TutorialCase; root::AbstractString = pwd())
    try
        return latest_attempt(case; root)
    catch
        return nothing
    end
end

function _write_day_page(path, day, cases; root)
    io = IOBuffer()
    println(io, "# Day $day status\n")
    println(io, "_Generated $(now(UTC)) UTC._\n")
    println(io, "| Case | Slug | Status | Current | Latest run | When | Duration |")
    println(io, "|------|------|--------|---------|-----------|------|----------|")
    for c in cases
        r = _case_row(c; root)
        println(io, "| $(c.name) | `$(c.slug)` | $(_status_badge(r.status)) | $(r.cur) | `$(r.run_id)` | $(r.when) | $(r.dur) |")
    end
    println(io)
    atomic_write(path, String(take!(io)))
    return path
end

"""
    write_summary_pages!(all; root=pwd()) -> Vector{String}

Write the documentation status pages from the current status JSON pointers:
`docs/src/status/{index,day3,day4}.md`. `all` is the full case registry. Returns
the paths written.
"""
function write_summary_pages!(all::Vector{TutorialCase}; root::AbstractString = pwd())
    statusdir = joinpath(root, "docs", "src", "status")
    mkpath(statusdir)
    written = String[]

    days = sort!(unique([c.day for c in all]))

    # index
    io = IOBuffer()
    println(io, "# Tutorial run status\n")
    println(io, "_Generated $(now(UTC)) UTC._\n")
    println(io, "Per-day status pages:\n")
    for d in days
        println(io, "- [Day $d](day$d.md)")
    end
    println(io)
    println(io, "## Overview\n")
    println(io, "| Day | Case | Status | Current |")
    println(io, "|-----|------|--------|---------|")
    for c in all
        r = _case_row(c; root)
        println(io, "| $(c.day) | $(c.name) | $(_status_badge(r.status)) | $(r.cur) |")
    end
    println(io)
    idx = joinpath(statusdir, "index.md")
    atomic_write(idx, String(take!(io)))
    push!(written, idx)

    for d in days
        cases = filter(c -> c.day == d, all)
        push!(written, _write_day_page(joinpath(statusdir, "day$d.md"), d, cases; root))
    end

    return written
end

end # module TutorialWorkflow
