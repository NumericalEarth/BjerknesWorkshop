#!/usr/bin/env bash
#
# full_build_publish.sh — end-to-end tutorial build + docs + (optional) publish.
#
# Pipeline:
#   1. scripts/run_tutorials.jl   — generate Literate scripts, run the selected
#      cases resiliently (each in its own Julia subprocess), record status, and
#      refresh the docs status pages. By default we CONTINUE past case failures
#      (ALLOW_CASE_FAILURES=1) so a single broken case never blocks the build.
#   2. docs/make.jl               — render the docs and build the HTML site.
#      This step NEVER runs a simulation (CPU only).
#   3. scripts/publish_docs.sh    — publish docs/build to gh-pages, but ONLY
#      when PUBLISH_DOCS=true.
#
# Robustness:
#   - The tutorial run continues after case failures by default. Set
#     STRICT_CASES=1 (passed through) to make run_tutorials.jl exit nonzero on
#     any case failure; even then we keep going to the docs build unless
#     ABORT_ON_RUN_FAILURE=1.
#   - The docs build is REQUIRED: if make.jl fails we abort before publishing.
#
# Environment:
#   PUBLISH_DOCS=true        run publish_docs.sh at the end (default: false).
#   ALLOW_CASE_FAILURES      default 1 here; set 0 to stop at a critical failure.
#   ABORT_ON_RUN_FAILURE=1   abort the whole pipeline if run_tutorials.jl exits
#                            nonzero (default: continue to the docs build).
#   RUN_CASES / RUN_DAYS     case selection (see TutorialWorkflow).
#   DOC_DAYS                 doc-day selection (see docs/make.jl).
#   PUBLISH_DRY_RUN / PUBLISH_SUBDIR  forwarded to publish_docs.sh.
#   JULIA                    julia binary (default: julia).
#
# Usage:
#   scripts/full_build_publish.sh
#   PUBLISH_DOCS=true scripts/full_build_publish.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

JULIA="${JULIA:-julia}"
# Continue past case failures by default.
export ALLOW_CASE_FAILURES="${ALLOW_CASE_FAILURES:-1}"
PUBLISH_DOCS="${PUBLISH_DOCS:-false}"

log() { printf '[full_build] %s\n' "$*"; }
die() { printf '[full_build] ERROR: %s\n' "$*" >&2; exit 1; }

_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# --------------------------------------------------------------------------
# Step 1: run the tutorials (resilient; continues past failures by default).
# --------------------------------------------------------------------------
log "Step 1/3: running tutorials (run_tutorials.jl) ..."
RUN_RC=0
"$JULIA" --project="$REPO_ROOT" "$REPO_ROOT/scripts/run_tutorials.jl" || RUN_RC=$?

if [[ "$RUN_RC" -ne 0 ]]; then
    if _truthy "${ABORT_ON_RUN_FAILURE:-0}"; then
        die "run_tutorials.jl exited $RUN_RC and ABORT_ON_RUN_FAILURE is set; aborting."
    fi
    log "WARNING: run_tutorials.jl exited $RUN_RC; continuing to docs build (cases may be stale)."
else
    log "Tutorial run completed."
fi

# --------------------------------------------------------------------------
# Step 2: build the docs (required).
# --------------------------------------------------------------------------
log "Step 2/3: building docs (docs/make.jl) ..."
"$JULIA" --project="$REPO_ROOT/docs" "$REPO_ROOT/docs/make.jl" \
    || die "docs build failed (docs/make.jl); not publishing."

if [[ ! -f "$REPO_ROOT/docs/build/index.html" ]]; then
    die "docs build produced no index.html; not publishing."
fi
log "Docs built: docs/build/index.html"

# --------------------------------------------------------------------------
# Step 3: publish (optional).
# --------------------------------------------------------------------------
if _truthy "$PUBLISH_DOCS"; then
    log "Step 3/3: publishing docs (PUBLISH_DOCS=$PUBLISH_DOCS) ..."
    "$REPO_ROOT/scripts/publish_docs.sh" || die "publish_docs.sh failed."
    log "Publish complete."
else
    log "Step 3/3: skipping publish (PUBLISH_DOCS=$PUBLISH_DOCS). Site is in docs/build."
fi

log "Done."
