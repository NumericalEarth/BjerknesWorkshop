# # Offline training of the learned surface roughness
#
# Now, that we have code for the dataloaders, we can train the column-wise MLP of 
# `LearnedSurfaceRoughness`. We first actually load the data, then we set up the model
# and train it offline. 

using Lux, Optimisers, Zygote, ADTypes, Random, Printf, JLD2, CUDA, cuDNN
using MLDataDevices: cpu_device, gpu_device
using Lux: Training

include(joinpath(@__DIR__, "01c_dataloaders.jl"))

# ## Data configuration 
# 
# Here, we set everything needed for our dataloaders and some general configuration

zarr_path  = "/cluster/projects/nn9984k/speedy-data/era5-roughness.zarr" # preprocessed store
#zarr_path  = "/p/projects/ou/labs/ai/max/era5-roughness.zarr"  # preprocessed store
batchsize  = 4096
split      = 0.9        # fraction of samples used for training
seed       = 0          # RNG seed for reproducible init & shuffles
use_gpu    = true       # true -> train on the GPU
device = use_gpu ? gpu_device(; force = true) : cpu_device()
rng = Random.Xoshiro(seed)

# Then, actually load the data 

train_loader, val_loader, stats =
    pointwise_dataloaders(zarr_path; batchsize, split, target_transform = log, rng)

# ## Model 
#
# Now, we construct our model, a standard MLP. 

model = Lux.Chain(
    Lux.Dense(7 => 32, Lux.leakyrelu),
    Lux.Dense(32 => 64, Lux.leakyrelu),
    Lux.Dropout(0.2),
    Lux.Dense(64 => 64, Lux.leakyrelu),
    Lux.Dropout(0.1),
    Lux.Dense(64 => 32, Lux.leakyrelu),
    Lux.Dense(32 => 1),
)

# ## Training
# 
# First, we define our training loop. The loop includes an early stopping based 
# on the error on the validation set and a basic monitoring of the loss on the 
# training and validation set that is saved to CSV file for later plotting. 
#
# We also define a few utilities for loading and saving the model. 

"""
    train_model(model, train_loader, val_loader=nothing; kwargs...)

Train `model` on the column-wise data with AdamW + (by default) MSE loss, in
whatever space the target was prepared in by the dataloader (e.g. normalised
`log` surface roughness).

Keywords
  * `epochs`, `learning_rate` – max number of epochs and AdamW learning rate.
  * `lossfn`           – any Lux loss with the `(model, ps, st, (x,y))` signature.
  * `ad`               – AD backend, `AutoZygote()` by default.
  * `device`           – `cpu_device()` (default) or `gpu_device()`.
  * `patience`         – early-stopping patience. Training stops once the
                         validation loss has not improved (by more than
                         `min_delta`) for `patience` consecutive epochs, and the
                         best-validation parameters/states are restored. Active
                         only when a `val_loader` is given and `patience > 0`.
  * `min_delta`        – minimum decrease in validation loss that counts as an
                         improvement.
  * `history_path`     – if given, the per-epoch train/validation losses are
                         appended to this CSV file (columns `epoch,train,val`)
                         and flushed every epoch. Robust to crashes/early
                         stopping and ready to `tail -f` or plot *during*
                         training. `nothing` (default) disables the CSV.
  * `rng`, `verbose`.

Returns `(; train_state, ps, st, history, best_epoch, best_val)` where `ps`/`st`
are the trained parameters and the **test-mode** states (the best-validation ones
when early stopping is active, ready to drop into a Lux inference call or the
`LearnedSurfaceRoughness` scheme), and `history` holds per-epoch train/validation
losses.
"""
function train_model(model, train_loader, val_loader = nothing;
        epochs::Int = 50, learning_rate = 1.0f-3, lossfn = MSELoss(), ad = AutoZygote(),
        device = cpu_device(), patience::Int = 10, min_delta::Real = 0,
        history_path::Union{Nothing, AbstractString} = nothing,
        rng::AbstractRNG = Random.default_rng(), verbose = true)

    ps, st = device(Lux.setup(rng, model))
    train_state = Training.TrainState(model, ps, st, AdamW(learning_rate))

    history = (; train = Float64[], val = Float64[])

    early_stop = patience > 0 && !isnothing(val_loader)   # need validation to monitor
    best_val, best_epoch, stale = Inf, 0, 0               # early-stopping bookkeeping
    best_ps = best_st = nothing                           # snapshot of best weights

    ## per-epoch CSV log of the losses, flushed every epoch so it survives a crash
    ## or early stop and can be plotted / `tail -f`'d while training is still running
    io = isnothing(history_path) ? nothing : open(history_path, "w")
    if !isnothing(io)
        println(io, "epoch,train,val")
        flush(io)
    end

    try
    for epoch in 1:epochs
        ## --- one pass over the training data ---
        running, nbatches = 0.0, 0
        for (x, y) in train_loader
            x, y = device(x), device(y)
            _, loss, _, train_state =
                Training.single_train_step!(ad, lossfn, (x, y), train_state)
            running += loss
            nbatches += 1
        end
        train_loss = running / max(nbatches, 1)
        push!(history.train, train_loss)

        ## --- validation (no parameter update) ---
        val_loss = isnothing(val_loader) ? NaN :
            evaluate(model, train_state.parameters, train_state.states, val_loader; lossfn, device)
        push!(history.val, val_loss)

        verbose && @printf("epoch %4d   train %.5f   val %.5f\n", epoch, train_loss, val_loss)

        ## --- persist the losses for later plotting ---
        if !isnothing(io)
            @printf(io, "%d,%.8f,%.8f\n", epoch, train_loss, val_loss)
            flush(io)
        end

        ## --- early stopping: track the best epoch and stop when patience runs out ---
        if early_stop
            if val_loss < best_val - min_delta
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_ps = deepcopy(train_state.parameters)
                best_st = deepcopy(train_state.states)
            else
                stale += 1
            end
            if stale >= patience
                verbose && @info "early stopping" stopped_at = epoch best_epoch best_val
                break
            end
        end
    end
    finally
        isnothing(io) || close(io)
    end

    ## restore the best-validation weights when early stopping was active
    out_ps = isnothing(best_ps) ? train_state.parameters : best_ps
    out_st = isnothing(best_st) ? train_state.states : best_st

    return (; train_state,
              ps = out_ps,
              st = Lux.testmode(out_st),
              history, best_epoch, best_val)
end

"""
    evaluate(model, ps, st, loader; lossfn=MSELoss(), device=cpu_device())

Mean loss over `loader` without updating parameters. States are put in test mode
so layers like `Dropout` behave deterministically.
"""
function evaluate(model, ps, st, loader; lossfn = MSELoss(), device = cpu_device())
    st = Lux.testmode(st)
    running, nbatches = 0.0, 0
    for (x, y) in loader
        x, y = device(x), device(y)
        loss, _, _ = lossfn(model, ps, st, (x, y))
        running += loss
        nbatches += 1
    end
    return running / max(nbatches, 1)
end

"""
    save_model(path, model, ps, st; stats=nothing, history=nothing)

Save a trained model to `path` (a `.jld2` file) with JLD2. Parameters and states
are moved to the CPU first so the file is portable across CPU/GPU runs. Pass the
dataloader `stats` (input/target mean & std) to bundle the normalisation
constants needed at inference, and the `history` returned by [`train_model`](@ref)
to keep the per-epoch train/validation loss curves alongside the weights for
later plotting. Returns `path`.
"""
function save_model(path::AbstractString, model, ps, st; stats = nothing, history = nothing)
    cdev = cpu_device()
    jldsave(path; model, parameters = cdev(ps), states = cdev(st), stats, history)
    return path
end

"""
    load_model(path) -> (; model, parameters, states, stats, history)

Load a model saved with [`save_model`](@ref). The same packages used to build the
model (e.g. Lux) must be loaded for deserialisation to succeed. `history` is the
train/validation loss record (or `nothing` for models saved without it).
"""
function load_model(path::AbstractString)
    data = JLD2.load(path)
    return (; model = data["model"],
              parameters = data["parameters"],
              states = data["states"],
              stats = data["stats"],
              history = get(data, "history", nothing))
end

# ## Now actually train it! 
# 
# Now, we actually train our model, save it and plot the training log. 
# First some hyperparameters: 

epochs = 100 # we have a lot of data, we won't need many
patience = 10 # early stopping patience
learning_rate = 1.0f-3
csv_path = "training_results.csv"
model_path = "trained_model.jld2"
plot_path = "learning_curve.png"

# Then, the actual training: 

result = train_model(model, train_loader, val_loader;
    epochs, learning_rate, patience, device, history_path = csv_path, rng)

@info "training finished" best_epoch = result.best_epoch best_val = result.best_val

(; ps, st, history) =  result 
save_model(model_path, model, ps, st; stats, history)

# and a  plot of the loss curve: 

using CairoMakie

PLOT = false 
if PLOT # hide it away in case we execute this on a compute node without graphics
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "epoch", ylabel = "loss", yscale = log10, title = "training loss")
    lines!(ax, result.history.train; label = "train")
    any(!isnan, result.history.val) && lines!(ax, result.history.val; label = "val")
    axislegend(ax)
    save(plot_path, fig)
end 

# The plot shows the train and validation loss per epoch, so convergence,
# overfitting and the early-stopping point are easy to spot.
#
# Next, we'll integrate it in SpeedyWeather.jl and see if it works there as well!
