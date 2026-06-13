# Day 3 — Differentiable Oceananigans (adjoint ACC)

This folder holds the Wednesday tutorial: adjoint sensitivities of the Antarctic
Circumpolar Current, differentiating zonal transport backwards through a 3-D
Oceananigans model with Reactant + Enzyme.

- **Canonical source:** [`src/02_differentiable_oceananigans.jl`](src/02_differentiable_oceananigans.jl) (Literate format).
- **Notebook:** [`02_differentiable_oceananigans.ipynb`](02_differentiable_oceananigans.ipynb), regenerated from the source.
- **Pinned environment:** [`Project.toml`](Project.toml) / [`Manifest.toml`](Manifest.toml).

> ⚠️ **Known issue (open).** The notebook currently throws a `StackOverflowError`
> at the **grid-build cell** (`grid = make_grid(ReactantState(), …)`) **inside the
> Jupyter kernel only**. The *exact same code, same environment, same Julia* runs
> fine as a plain script. See [the issue section below](#known-issue-cell-stackoverflow-in-the-jupyter-kernel-only).

## Pinned environment

Validated end-to-end as a script on the GH200 (aarch64) node.

| Package | Version |
| --- | --- |
| Julia | **1.11.7** |
| Oceananigans | 0.101.2 |
| Reactant | 0.2.264 |
| Enzyme | 0.13.157 |
| ReactantCore | 0.1.20 |
| CUDA | 5.11.3 |
| IJulia | 1.34.4 |

## How to start the Open OnDemand (OOD) Jupyter session

The notebook can be run on a GPU node through the cluster's Open OnDemand Jupyter
portal. This is how you can bring the session up:

```bash
module load JupyterNotebook/7.4.7-GCCcore-14.3.0
module load NRIS/GPU
module load Julia/1.11
export JULIA_DEPOT_PATH=/cluster/work/projects/nn9984k/$USER/.julia

julia -e 'import Pkg; Pkg.add("IJulia")'
julia -e 'using IJulia; installkernel("Julia", "-O0")'
```

Then in Jupyter open `02_differentiable_oceananigans.ipynb` with the **"Julia
1.11"** kernel.

Two details that matter:

- **Stay on Julia 1.11.** Use `module load Julia/1.11` because Reactant currently does not work on Julia 1.12**
- **`-O0`** (the second arg to `installkernel` above). Without it, compiling the
  Reactant/Enzyme time-stepping loop is *extremely* slow — at `-O2` the traced-loop
  compile was still going after ~19 minutes; at `-O0` it finishes in ~2.6 minutes.
  This bakes `-O0` into the kernel's `argv`; the environment-check cell reports the
  optimization level so you can confirm it took effect.

## Running as a plain script (this works)

```bash
module load Julia/1.11
export JULIA_DEPOT_PATH=/cluster/work/projects/nn9984k/$USER/.julia
cd ~/BjerknesWorkshop
julia -O0 --project=tutorials/day3 tutorials/day3/src/02_differentiable_oceananigans.jl
```

End-to-end this compiles (~2.6 min at `-O0`), spins up, runs the AD pass, prints
`J ≈ 2.45 Sv`, and writes the figures + `channel_results.jld2` into
`differentiable_channel_output/`.

## Known issue: cell StackOverflow in the Jupyter kernel only

When the **same code** is run cell-by-cell in the Jupyter kernel, the grid-build
cell throws:

```
StackOverflowError
 promote_to(::Type{…}, rhs::Field{…})
   @ Reactant …/src/TracedPromotion.jl:37
 promote_to(::Type{…}, rhs::Field{…}) (repeats ~80000 times)
```

What we have confirmed:

- The grid builds fine in a **fresh `julia` script process** — on the GPU, at both
  `-O0` and `-O2`, even with `using IJulia` loaded first. All OK.
- In the live kernel the active project, `pkgversion(Reactant)` (0.2.264), and the
  loaded `OceananigansReactantExt` are **identical** to the working script process,
  yet the cell still overflows after a fresh kernel restart.

So this looks like **kernel-session state** (world-age / per-cell top-level eval /
method invalidation), not an environment or version mismatch.

The committed notebook is executed up to (and including) the failing grid-build
cell so you can see exactly where it stops.
