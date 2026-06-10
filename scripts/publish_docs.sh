#!/usr/bin/env bash
#
# publish_docs.sh — publish the built Documenter site to the `gh-pages` branch.
#
# Strategy (worktree-based, idempotent, safe-by-default):
#   - Locate the repo root via `git rev-parse --show-toplevel`.
#   - REFUSE to run unless docs/build/index.html exists (nothing to publish).
#   - Check out the `gh-pages` branch into a dedicated worktree at
#     .gh-pages-worktree/ (orphan-create it on first publish; otherwise reuse
#     the remote branch). Never `worktree add` a branch already checked out
#     elsewhere — we always `worktree remove --force` at the end.
#   - Clear the worktree contents (except .git) and copy in docs/build.
#       * If PUBLISH_SUBDIR is set, the build is staged UNDER that subdir
#         (e.g. PUBLISH_SUBDIR=preview -> served at <site>/preview/), and only
#         that subdir is cleared — other published content is preserved.
#   - (Re)create .nojekyll so GitHub Pages does not strip underscore dirs.
#   - Commit and push to origin gh-pages.
#
# Safety:
#   - PUBLISH_DRY_RUN=1 does everything except the final `git push` (and prints
#     the diff/status it would have pushed).
#   - On push failure we DO NOT delete docs/build or the worktree; instead we
#     print the exact command to retry the push by hand.
#
# Environment:
#   PUBLISH_DRY_RUN=1   stage + commit locally but skip the push.
#   PUBLISH_SUBDIR=foo  publish under <site>/foo/ instead of the site root.
#   PUBLISH_BRANCH      gh-pages branch name (default: gh-pages).
#   PUBLISH_REMOTE      remote name (default: origin).
#
# Usage:
#   scripts/publish_docs.sh

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration / locate the repo.
# --------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="${PUBLISH_BRANCH:-gh-pages}"
REMOTE="${PUBLISH_REMOTE:-origin}"
WORKTREE="$REPO_ROOT/.gh-pages-worktree"
BUILD_DIR="$REPO_ROOT/docs/build"
DRY_RUN="${PUBLISH_DRY_RUN:-0}"
SUBDIR="${PUBLISH_SUBDIR:-}"

log() { printf '[publish_docs] %s\n' "$*"; }
die() { printf '[publish_docs] ERROR: %s\n' "$*" >&2; exit 1; }

_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# --------------------------------------------------------------------------
# Refuse to publish without a built site.
# --------------------------------------------------------------------------
if [[ ! -f "$BUILD_DIR/index.html" ]]; then
    die "no built site: $BUILD_DIR/index.html is missing. Run docs/make.jl first."
fi
log "Found built site: $BUILD_DIR/index.html"

# --------------------------------------------------------------------------
# Prepare the gh-pages worktree.
# --------------------------------------------------------------------------
# A stale worktree from an aborted earlier run would block `worktree add`.
if [[ -d "$WORKTREE" ]]; then
    log "Removing stale worktree at $WORKTREE"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi
git worktree prune

# Make sure we have the freshest view of the remote branch (best effort).
git fetch "$REMOTE" "$BRANCH" 2>/dev/null || true

if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
    log "Adding worktree tracking $REMOTE/$BRANCH"
    git worktree add "$WORKTREE" -B "$BRANCH" "$REMOTE/$BRANCH"
elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    log "Adding worktree from local branch $BRANCH"
    git worktree add "$WORKTREE" "$BRANCH"
else
    log "No existing $BRANCH; creating an orphan branch in the worktree"
    git worktree add --detach "$WORKTREE"
    git -C "$WORKTREE" checkout --orphan "$BRANCH"
    git -C "$WORKTREE" reset --hard
    git -C "$WORKTREE" clean -fdx
fi

# Always remove the worktree on exit (success OR failure) — never leave a
# checked-out branch lingering. We deliberately do NOT delete docs/build.
cleanup() {
    log "Cleaning up worktree $WORKTREE"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
    git worktree prune 2>/dev/null || true
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# Stage the build into the worktree.
# --------------------------------------------------------------------------
if [[ -n "$SUBDIR" ]]; then
    DEST="$WORKTREE/$SUBDIR"
    log "Publishing under subdir: $SUBDIR (preserving other content)"
    rm -rf "$DEST"
    mkdir -p "$DEST"
    cp -a "$BUILD_DIR/." "$DEST/"
else
    log "Publishing to site root (clearing existing content except .git)"
    # Remove everything except the .git file/dir that ties the worktree back.
    find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
    cp -a "$BUILD_DIR/." "$WORKTREE/"
fi

# nojekyll is mandatory at the site root so Pages serves underscore dirs.
touch "$WORKTREE/.nojekyll"

# --------------------------------------------------------------------------
# Commit.
# --------------------------------------------------------------------------
git -C "$WORKTREE" add -A

if git -C "$WORKTREE" diff --cached --quiet; then
    log "No changes to publish; site is already up to date."
    exit 0
fi

COMMIT_MSG="Deploy docs $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git -C "$WORKTREE" commit -q -m "$COMMIT_MSG"
log "Committed: $COMMIT_MSG"

# --------------------------------------------------------------------------
# Push (unless dry-run).
# --------------------------------------------------------------------------
RETRY_CMD="git -C \"$WORKTREE\" push \"$REMOTE\" \"HEAD:$BRANCH\""

if _truthy "$DRY_RUN"; then
    log "PUBLISH_DRY_RUN set — NOT pushing. Staged commit summary:"
    git -C "$WORKTREE" --no-pager show --stat --oneline HEAD | sed 's/^/[publish_docs]   /'
    log "To push manually: $RETRY_CMD"
    exit 0
fi

log "Pushing to $REMOTE $BRANCH ..."
if git -C "$WORKTREE" push "$REMOTE" "HEAD:$BRANCH"; then
    log "Published to $REMOTE/$BRANCH."
else
    # Disarm the cleanup trap so the worktree (with the committed deploy) is
    # preserved for a manual retry; also keep docs/build intact.
    trap - EXIT
    printf '[publish_docs] ERROR: push failed. Build and worktree PRESERVED.\n' >&2
    printf '[publish_docs] Retry the push with:\n    %s\n' "$RETRY_CMD" >&2
    printf '[publish_docs] Then clean up:\n    git worktree remove --force "%s"\n' "$WORKTREE" >&2
    exit 1
fi
