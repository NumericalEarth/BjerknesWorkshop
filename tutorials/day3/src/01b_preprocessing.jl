# # Preprocessing: global NetCDF fields -> column-wise Zarr store prepared for dataloaders
#
# You can safely ignore this file / notebook. This is just here for completeness of the example.
# We already prepared and preprocessed the data for you. The raw ERA5 files are not saved on the 
# server. Just skip ahead to the next notebook. 
#
 
# ## Preprocessing pipeline
#
#   1. Read the requested variables from one big multi-year NetCDF file, loading only
#      a single year's time-slice into RAM at a time.
#   2. Flatten every field column-wise: each grid point (at each time step) becomes
#      one sample / column.
#   3. Optionally restrict to land via a static land-sea mask (ERA5 `lsm`) — the ocean
#      here is filled with values, not left missing, so it must be masked explicitly.
#   4. Append each chunk's selected columns to a per-variable Zarr array, asserting
#      they hold no NaN/missing rather than silently filtering invalid samples.

using NCDatasets, Zarr, Dates

"""
    NetCDFVar(; path, name, level=nothing)

Points at one variable `name` inside the NetCDF file at `path`.

`level` selects along the vertical dimension of a 4D (lat, lon, level, time) field:
  * `nothing`  – keep all levels (each level becomes a separate feature),
                 or no-op for a 3D field that has no level dimension,
  * `Integer`  – take a single level, e.g. a surface or bottom-soil index,
  * a range / vector of indices – keep a subset of levels.

`name` doubles as the key the variable is stored under in the Zarr group.
"""
Base.@kwdef struct NetCDFVar{L}
    path::String
    name::String
    level::L = nothing
end

"""
    load_var_slice(ds, name, tidx; level=nothing) -> Array{Float32}

Read variable `name` from the already-open dataset `ds`, restricted to time indices
`tidx` along the last dimension (`Colon()` loads all times). Only the requested
slices are read from disk. missings -> `NaN`; an optional `level` is selected.
"""
function load_var_slice(ds, name::AbstractString, tidx; level = nothing)
    var = ds[name]
    arr = ndims(var) == 3 ? var[:, :, tidx] :
          ndims(var) == 4 ? var[:, :, :, tidx] :
          error("expected a 3D (lat,lon,time) or 4D (lat,lon,level,time) variable '$(name)', got $(ndims(var))D")
    arr = to_float32_nan(arr)
    if level !== nothing && ndims(arr) == 4
        arr = arr[:, :, level, :]
    end
    return arr
end

"""
    load_landseamask(v::NetCDFVar, threshold) -> Vector{Bool}

Read a static land-sea mask field (e.g. ERA5 `lsm`, a 0–1 land fraction on the same
grid) and return a boolean vector flattened in the same column-major order as
`flatten_pointwise` (first spatial dim fastest), `true` where the field is
`≥ threshold` (land). A trailing time dimension, if present, is reduced to its first
slice — a land-sea mask is static.
"""
function load_landseamask(v::NetCDFVar, threshold::Real)
    a = NCDataset(v.path) do ds
        haskey(ds, v.name) || error("land mask variable '$(v.name)' not found in $(v.path)")
        var = ds[v.name]
        ndims(var) == 2 ? var[:, :] :
        ndims(var) == 3 ? var[:, :, 1] :
        error("expected a 2D (lon,lat) or 3D (lon,lat,time) land mask '$(v.name)', got $(ndims(var))D")
    end
    return vec(to_float32_nan(a)) .>= Float32(threshold)
end

"""
    subsample_time(idx, stride) -> indices

Keep every `stride`-th time index (`stride ≤ 1` keeps all). Works on both ranges
(the whole-file case) and `Vector{Int}` (a single year's indices), preserving the
type so contiguous reads stay contiguous.
"""
subsample_time(idx, stride::Integer) =
    stride <= 1 ? idx : idx[firstindex(idx):stride:lastindex(idx)]

"Map each calendar year to the time indices belonging to it (from the CF-decoded time coordinate)."
function year_indices(ds, time_name::AbstractString)
    haskey(ds, time_name) || error("time coordinate '$(time_name)' not found in dataset")
    times = ds[time_name][:]
    eltype(times) <: Dates.TimeType ||
        error("time coordinate '$(time_name)' is not CF-decoded to dates (eltype $(eltype(times))); cannot split by year")
    idx = Dict{Int, Vector{Int}}()
    for (i, t) in enumerate(times)
        push!(get!(idx, Int(Dates.year(t)), Int[]), i)
    end
    return idx
end

"Convert any array (possibly holding `missing`) to a dense `Float32` array with `NaN` for missings."
function to_float32_nan(arr::AbstractArray)
    out = Array{Float32}(undef, size(arr))
    @inbounds for i in eachindex(arr, out)
        x = arr[i]
        out[i] = ismissing(x) ? NaN32 : Float32(x)
    end
    return out
end

"""
    flatten_pointwise(arr) -> Matrix (features, samples)

Flatten a global field so each column is one grid point at one time step.

  * 3D `(lat, lon, time)`         -> `(1, lat*lon*time)`
  * 4D `(lat, lon, level, time)`  -> `(level, lat*lon*time)`

The sample ordering (lat fastest, then lon, then time) is the same for both cases,
so different variables and the target line up column-for-column.
"""
function flatten_pointwise(arr::AbstractArray)
    if ndims(arr) == 3
        nlat, nlon, nt = size(arr)
        return reshape(arr, 1, nlat * nlon * nt)
    elseif ndims(arr) == 4
        nlat, nlon, nlev, nt = size(arr)
        permuted = permutedims(arr, (3, 1, 2, 4))        # (level, lat, lon, time)
        return reshape(permuted, nlev, nlat * nlon * nt)
    else
        error("expected a 3D (lat,lon,time) or 4D (lat,lon,level,time) array, got $(ndims(arr))D")
    end
end

"""
    preprocess_to_zarr(inputs, target, zarr_path; years=nothing, time_name="valid_time",
                       time_stride=61, chunk_samples=2^22, landmask=nothing,
                       landmask_threshold=0.5, overwrite=true, compressor=..., verbose=true)

Build the column-wise dataset from a big multi-year NetCDF file and persist it as a
Zarr group, loading only one year's time-slice into RAM at a time.

Arguments
  * `inputs::Vector{NetCDFVar}` – the predictor variables (land variables).
  * `target::NetCDFVar`         – the target variable (e.g. surface roughness).
  * `zarr_path::AbstractString` – directory store to create.

Memory model
  * `years` is the collection of calendar years to process (e.g. `2000:2020`). The
    CF-decoded `time_name` coordinate is used to find each year's time indices, and
    every variable is read sliced to those indices — so only one year of data is in
    memory at once. With `years=nothing` the whole file is processed in one go.
  * Each year is handled independently in a single pass: load all variables, select
    the land columns (via `landmask`, or all columns if none is given), and append
    them to each variable's Zarr array. Selected columns are asserted to be free of
    NaN/missing before writing — invalid values raise an error rather than being
    silently dropped. Different chunks may contribute different numbers of columns,
    which is fine since samples are stored column-wise.
  * `time_stride` thins the time axis before flattening (keep every `stride`-th
    step; `1` keeps all). The surface fields here are highly redundant in time
    (orography/geopotential is constant, vegetation cover is near-constant, surface
    roughness is quasi-static), so a large stride cuts the sample count — and hence
    the on-disk chunk count — by that factor with little loss. For 3-hourly data:
    `8`≈daily, `56`≈weekly, `240`≈monthly.
  * `landmask` optionally restricts the output to land. This file's variables are
    *filled* (not missing) over the ocean — soil temperature carries SST, vegetation
    and soil moisture are 0, etc. — so the NaN filter alone never drops sea points,
    and no in-file variable cleanly separates land from sea (e.g. deserts look like
    ocean in vegetation/soil moisture). Pass a `NetCDFVar` pointing at a static
    land-sea mask on the same grid (e.g. ERA5 `lsm`, a 0–1 land fraction); columns
    with mask `< landmask_threshold` (default `0.5`) are dropped. The mask is loaded
    once and reused for every timestep.

`chunk_samples` is the chunk width along the sample axis; the Zarr directory store
writes one file per chunk, so the file count is roughly
`(total_samples / chunk_samples) × n_variables`. Keep it large (default `2^22`) and
use `time_stride` to keep `total_samples` modest, or the store explodes into
millions of tiny files. The default `2^22` (~16 MiB chunks) keeps this file at a
few thousand chunk files for typical strides.

Output
  * One array per variable, keyed by `NetCDFVar.name`, of shape `(features,
    n_samples)`, grown year by year, chunked along the sample dimension and
    compressed with `compressor`. Group attributes record `input_names` and
    `target_name`; `n_samples` is simply each array's width.

`compressor` controls on-disk compression (Blosc + zstd by default); pass `nothing`
to store uncompressed. `verbose` logs per-(variable, year) progress and timings.

Assumes all variables share the same `time` axis (same timestamps in the same
order — the usual case for one file).

Returns `(; path, input_names, target_name, n_samples, n_dropped, years)`.
"""
function preprocess_to_zarr(inputs::AbstractVector{<:NetCDFVar}, target::NetCDFVar,
        zarr_path::AbstractString; years = nothing, time_name::AbstractString = "valid_time",
        time_stride::Int = 61, chunk_samples::Int = 1 << 22,
        landmask::Union{Nothing, NetCDFVar} = nothing, landmask_threshold::Real = 0.5,
        overwrite::Bool = true,
        compressor = Zarr.BloscCompressor(cname = "zstd", clevel = 5, shuffle = 1),
        verbose::Bool = true)

    vars  = NetCDFVar[inputs..., target]            # inputs first, target last
    names = String[v.name for v in vars]
    allunique(names) || error("variable names must be unique (they key the Zarr arrays); got $(names)")

    ## open each distinct file once and keep the handles for the whole run
    datasets = Dict(p => NCDataset(p) for p in unique(v.path for v in vars))
    try
        ## chunks = list of (label, path -> time indices), subsampled by `time_stride`.
        ## `nothing` years -> the whole file as one chunk.
        chunks = if isnothing(years)
            [("all", Dict(p => subsample_time(1:length(datasets[p][time_name]), time_stride)
                          for p in keys(datasets)))]
        else
            file_year_idx = Dict(p => year_indices(datasets[p], time_name) for p in keys(datasets))
            [(string(y), Dict(p => subsample_time(get(file_year_idx[p], y, Int[]), time_stride)
                              for p in keys(datasets))) for y in years]
        end

        verbose && @info "Preprocessing → $(zarr_path)" variables=length(vars) years=(isnothing(years) ? "all" : length(chunks))

        ## create the group; per-variable arrays are created lazily on first write,
        ## taking their feature-row count from the loaded slice.
        if overwrite && ispath(zarr_path)
            rm(zarr_path; recursive = true, force = true)
        end
        g = zgroup(zarr_path; attrs = Dict(
            "input_names" => String[v.name for v in inputs],
            "target_name" => target.name,
        ))
        arrays = Dict{String, Any}()

        ## optional static land-sea mask: a per-grid-point boolean reused every timestep.
        ## This file's variables are filled (not missing) over ocean, so this mask — not
        ## a NaN filter — is what restricts the output to land.
        landvec = isnothing(landmask) ? nothing : load_landseamask(landmask, landmask_threshold)
        verbose && !isnothing(landvec) &&
            @info "  land-sea mask" land_points=count(landvec) grid_points=length(landvec) threshold=landmask_threshold

        total_kept = total_seen = 0
        for (label, idxmap) in chunks
            ## skip a requested year that is absent from any file
            if any(idxmap[v.path] isa AbstractVector && isempty(idxmap[v.path]) for v in vars)
                verbose && @warn "  year $(label): no data in at least one file, skipping"
                continue
            end

            ## load and column-flatten every variable for this chunk
            fields = Dict{String, Any}()
            for v in vars
                t = time()
                fields[v.name] = flatten_pointwise(load_var_slice(datasets[v.path], v.name, idxmap[v.path]; level = v.level))
                verbose && @info "  $(label) · loaded $(v.name)" features=size(fields[v.name], 1) samples=size(fields[v.name], 2) seconds=round(time() - t; digits = 2)
            end

            ## column selector: the land points (tiled across this chunk's timesteps,
            ## since samples run space-fastest then time), or all columns if no mask.
            nseen = size(fields[vars[1].name], 2)
            cols = if isnothing(landvec)
                Colon()
            else
                nsp = length(landvec)
                nseen % nsp == 0 ||
                    error("land mask has $(nsp) points but year $(label) has $(nseen) samples; grid mismatch?")
                repeat(landvec, nseen ÷ nsp)
            end
            k = cols isa Colon ? nseen : count(cols)
            total_seen += nseen
            total_kept += k

            ## append the selected columns; assert they are clean rather than filtering.
            if k > 0
                for v in vars
                    f = fields[v.name][:, cols]
                    nbad = count(isnan, f)
                    nbad == 0 ||
                        error("year $(label): '$(v.name)' has $(nbad) NaN/missing value(s) among the $(k) kept samples")
                    z = get!(arrays, v.name) do
                        zcreate(Float32, g, v.name, size(f, 1), 0;
                                chunks = (size(f, 1), chunk_samples), compressor = compressor)
                    end
                    old = size(z, 2)
                    resize!(z, size(z, 1), old + k)
                    z[:, old+1:old+k] = f
                end
            end
            verbose && @info "  $(label) · appended" kept=k of=nseen
        end

        total_kept > 0 || error("no samples to write (empty after applying the land mask?)")
        verbose && @info "Preprocessing done" path=zarr_path n_samples=total_kept

        return (; path = zarr_path,
                  input_names = String[v.name for v in inputs],
                  target_name = target.name,
                  n_samples = total_kept,
                  n_dropped = total_seen - total_kept,
                  years = isnothing(years) ? nothing : collect(years))
    finally
        foreach(close, values(datasets))
    end
end