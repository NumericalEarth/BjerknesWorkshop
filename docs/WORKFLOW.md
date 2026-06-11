# Workshop docs workflow — operator guide

How to generate the tutorial website, publish it to GitHub Pages, and regenerate
just part of it without rerunning everything.

**Live site:** <https://numericalearth.github.io/BjerknesWorkshop/>

---

## Mental model

1. **Literate sources are canonical.** Edit `tutorials/day*/src/*.jl` only. The
   generated scripts (`tutorials/*/scripts/`), Markdown (`docs/src/day*/*.md`),
   and built site (`docs/build/`) are all derived and git-ignored.
2. **Running simulations is separate from rendering docs.** Expensive cases run
   via the runner and write cached artifacts under `output/day*/<slug>/`. The
   docs build only *reads* those cached artifacts and embeds figures/movies
   inline (base64) — `makedocs` never launches a simulation.
3. **A content-hash cache makes partial updates safe.** A case is re-run only if
   its source / parameters / Manifest changed (or you force it). Re-running Day 3
   never touches Day 4.
4. **A failed case never breaks the site.** Its page shows a status card; the
   last successful output is preserved.
5. **Publishing is explicit.** The static site is pushed to the `gh-pages` branch
   only when you ask for it.

The registry of cases lives in `src/TutorialWorkflow.jl` (`case_registry`).

---

## Prerequisites

Two Julia environments:

```bash
# Simulation/runtime env (Breeze, Oceananigans, CUDA, CairoMakie, …)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Docs env (Documenter, Literate, CairoMakie, JLD2)
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

- The docs build is **CPU-only** and runs anywhere (login/head node).
- Running the Day 4 LES cases needs a **GPU** (see [Running on the GPU](#running-the-day-4-les-cases-on-the-gpu)).
- `ffmpeg` (via CairoMakie/`record`) is needed to (re)generate movies.

---

## Entry points

| Script | What it does | Where to run |
|---|---|---|
| `scripts/generate_literate_outputs.jl` | `Literate.script` the sources → `tutorials/*/scripts/` | anywhere |
| `scripts/run_tutorials.jl` | Generate scripts, run selected cases in isolated subprocesses, write status, refresh status pages | head node (cheap cases) / **GPU** (LES) |
| `docs/make.jl` | Render Literate→Markdown (`execute=false`), embed cached artifacts, `makedocs` → `docs/build/` | head node (CPU) |
| `scripts/full_build_publish.sh` | One-stage: run → build docs → (optional) publish | head node, or GPU if running LES |
| `scripts/publish_docs.sh` | Push `docs/build/` to `gh-pages` via a git worktree | head node |
| `scripts/run_day4_gpu.sbatch` | Slurm job: run the Day 4 cases on `gpu-prod` (H100) | `sbatch` |
| `scripts/test_*.sh` | Acceptance / robustness / partial-update tests | head node |

---

## Environment variables

**Selection**

| Var | Values | Meaning |
|---|---|---|
| `RUN_DAYS` | `all` \| `3` \| `4` \| `3,4` \| `none` | Which days the runner runs |
| `RUN_CASES` | `all` \| `lead_atmosphere,norway_100m` … | Specific case slugs (overrides `RUN_DAYS`) |
| `DOC_DAYS` | `all` \| `3` \| `4` \| `3,4` | Which day pages the docs build renders |
| `RUN_CLASS` | `smoke` \| `calibration` \| `production` | Tags the run + cache key (Day 4 LES are hardcoded "developed" regardless; mainly affects cache identity and the cheap cases) |

Case slugs: `lead_atmosphere`, `lead_ocean_waves`, `norway_100m`, `smoke_case`
(day 4); `hybrid_physics_ml`, `differentiable_esms` (day 3).

**Run behavior**

| Var | Default | Meaning |
|---|---|---|
| `FORCE_RERUN` | `0` | Re-run selected cases even if the cache says current |
| `ALLOW_CASE_FAILURES` | `1` (in `full_build_publish.sh`) | Continue past a failed case |
| `STRICT_CASES` | `0` | After finishing, exit nonzero if any selected case failed |
| `ABORT_ON_RUN_FAILURE` | `0` | Abort the whole pipeline if the run step exits nonzero |
| `IGNORE_SOURCE_HASH` | `0` | Treat outputs as current even if the source changed |
| `IGNORE_MANIFEST_HASH` | `0` | Treat outputs as current even if `Manifest.toml` changed |
| `SIMULATE_CASE_FAILURE` | — | Comma list of slugs (or `all`) to force-fail *without running* — for testing resilience |

**Publishing**

| Var | Default | Meaning |
|---|---|---|
| `PUBLISH_DOCS` | `false` | In `full_build_publish.sh`, run `publish_docs.sh` at the end |
| `PUBLISH_DRY_RUN` | `false` | Commit to the worktree but do not `git push` |
| `PUBLISH_SUBDIR` | — | Publish under a subdir, e.g. `staging/2026-06-04` (leaves the site root intact) |
| `PUBLISH_BRANCH` | `gh-pages` | Target branch |
| `PUBLISH_REMOTE` | `origin` | Target remote |

---

## Generate the docs

### Build & view locally (no simulations)

```bash
DOC_DAYS=all julia --project=docs docs/make.jl
# → docs/build/index.html  (open it in a browser, or scp it locally)
```

Cases that have not been run show graceful "No run yet" / "unavailable" cards.
This is exactly what CI-style smoke validation does:

```bash
RUN_DAYS=none DOC_DAYS=all PUBLISH_DOCS=false bash scripts/full_build_publish.sh
```

### One-stage: run cheap cases, build, (optionally) publish

```bash
# Build the whole site from whatever is cached, run only the cheap day-3 + smoke
RUN_DAYS=3 DOC_DAYS=all PUBLISH_DOCS=false bash scripts/full_build_publish.sh
```

---

## Running the Day 4 LES cases on the GPU

The Day 4 cases (`lead_atmosphere`, `lead_ocean_waves`, `norway_100m`) need an
H100. They are driven through the runner so artifacts + status land in
`output/day4/<slug>/runs/<run_id>/`:

```bash
sbatch scripts/run_day4_gpu.sbatch          # runs all three on gpu-prod
# watch:  squeue -u $USER ; tail -f logs/day4_run_<jobid>.out
```

To run a subset, edit `RUN_CASES` in that sbatch file, or submit ad hoc:

```bash
RUN_CASES=lead_atmosphere RUN_CLASS=production ALLOW_CASE_FAILURES=1 \
  julia --project=. scripts/run_tutorials.jl        # ONLY on a GPU node
```

> ⚠️ Do **not** run the Day 4 LES on the head node — no GPU, so it falls back to
> CPU and is unusably slow. Build/publish docs on the head node; run LES via Slurm.
>
> ⚠️ `norway_100m` (compressible terrain-following) has a slow startup that stalls
> at large grids; it is kept at a modest grid and may time out. If it has no
> successful run, its page shows a card — by design.

After the GPU run finishes, rebuild + publish from the head node:

```bash
DOC_DAYS=all PUBLISH_DOCS=true bash scripts/full_build_publish.sh
```

---

## Publishing to GitHub Pages

The site is served from the `gh-pages` branch, `/` root (configured once; already
enabled for this repo).

### Publish the current build

```bash
DOC_DAYS=all julia --project=docs docs/make.jl     # ensure docs/build is fresh
bash scripts/publish_docs.sh                        # push docs/build → gh-pages
```

`publish_docs.sh`:
- refuses to publish unless `docs/build/index.html` exists,
- uses a git worktree at `.gh-pages-worktree`,
- writes `.nojekyll`,
- on push failure, leaves the build and worktree intact and prints a retry command.

### Dry run / staging

```bash
PUBLISH_DRY_RUN=true bash scripts/publish_docs.sh                 # commit, don't push
PUBLISH_SUBDIR=staging/$(date -u +%Y%m%dT%H%M%SZ) bash scripts/publish_docs.sh
#   → https://numericalearth.github.io/BjerknesWorkshop/staging/<id>/
```

### One-shot run + build + publish

```bash
RUN_DAYS=all DOC_DAYS=all PUBLISH_DOCS=true bash scripts/full_build_publish.sh
```

### (Re)configuring Pages, if ever needed

```bash
gh api -X POST repos/NumericalEarth/BjerknesWorkshop/pages \
  -f 'source[branch]=gh-pages' -f 'source[path]=/'
gh api repos/NumericalEarth/BjerknesWorkshop/pages -q '.html_url,.status'
```

---

## Regenerating just part of the docs

The cache key for each case is `source hash + parameter hash + Manifest hash +
RUN_CLASS`. The runner skips any case whose cache is still current, so you only
recompute what changed.

### Re-run a single case

```bash
# Forced (ignore the cache) — typical after fixing a case:
RUN_CASES=norway_100m FORCE_RERUN=1 DOC_DAYS=all PUBLISH_DOCS=true \
  bash scripts/full_build_publish.sh
# (on a GPU node, or via sbatch for the LES cases)
```

### Re-run one day, leave the others untouched

```bash
RUN_DAYS=3 DOC_DAYS=all PUBLISH_DOCS=true bash scripts/full_build_publish.sh
# Day 4 outputs and their latest_success pointers are NOT modified.
```

This is idempotent: editing a Day 3 source bumps only that case's source hash, so
only it re-runs; Day 4 is served from its existing cached outputs.

### Update only the narrative/prose (no simulation rerun)

If you edited Literate `#` comments but the science output is unchanged and you
just want fresh prose on the site:

```bash
DOC_DAYS=all julia --project=docs docs/make.jl   # re-renders Markdown from sources
bash scripts/publish_docs.sh
```

Note the next *run* of that case will re-run it (its source hash changed); the
docs render above does not, because rendering never runs simulations.

### Rebuild docs for one day only

```bash
DOC_DAYS=4 julia --project=docs docs/make.jl
```

`DOC_DAYS` controls which day pages are (re)rendered. To publish a *complete*
site (correct nav/index), build with `DOC_DAYS=all` — unchanged days are cheap
because they just re-embed already-cached artifacts.

### Force a case "current" without rerunning

If you bumped the Manifest or tweaked a source comment but the cached output is
still scientifically valid:

```bash
IGNORE_MANIFEST_HASH=1 IGNORE_SOURCE_HASH=1 RUN_DAYS=4 \
  julia --project=. scripts/run_tutorials.jl   # cases report "skip (current)"
```

---

## How outputs are stored

```text
output/day4/lead_atmosphere/
  latest_success.json      # pointer to the most recent successful run
  latest_attempt.json      # pointer to the most recent attempt
  runs/<UTCts>_<hash>/
    status.json
    logs/{stdout.log,stderr.log}
    artifacts/             # *.jld2, *.png, *.mp4  (what the page embeds)
```

`latest_success.json` is advanced **only** after a run exits cleanly and all
`required_outputs` exist — a failed attempt never destroys the last good site.
Raw outputs stay on the cluster (git-ignored); only the static, media-embedded
HTML is published.

---

## Troubleshooting

- **A page shows "No run yet" / "unavailable".** That case has no successful
  cached run. Run it (GPU for LES), then rebuild + publish.
- **Figures/movies don't appear.** They are embedded as inline base64 in
  `@raw html` blocks (generated in `docs/make.jl`). Do **not** switch them to
  markdown `![](data:...)` — Documenter treats the data URI as a file path and
  the build fails.
- **`makedocs` tried to run a simulation.** Literate emits `@example` fences that
  Documenter executes; `docs/make.jl`'s `demote_example_blocks` rewrites every
  `@example`/`@repl`/`@eval` fence (including Literate's 4-backtick fences) to
  plain `julia`. Keep that post-processor.
- **Job logs look empty while running.** Julia block-buffers stderr to a file;
  output appears at exit. Watch `output/day4/<slug>/latest_success.json` (or the
  per-run `logs/`) for progress instead.
- **Push to `gh-pages` failed.** Nothing was lost; rerun `scripts/publish_docs.sh`
  (it prints the exact retry command).
