#!/usr/bin/env bash
#
# test_full_flow.sh — Test 0 + Test 1 of the deployment test plan.
#
#   Test 0 (local, no-publish smoke build): run the orchestrator scoped to the
#     single trivial CPU `smoke_case`, then build the docs locally (no GPU, no
#     publish). Assert the case produced its required artifact, recorded a
#     success, and that the rendered HTML site exists and contains the case.
#
#   Test 1 (dry run): with RUN_DAYS=none, assert the orchestrator runs zero
#     cases, leaves the smoke artifacts from Test 0 untouched, still refreshes
#     the status pages, and the docs still build.
#
# This script NEVER publishes (no git push, no gh-pages). It only builds the
# static site under docs/build/ locally.
#
# Usage:  bash scripts/test_full_flow.sh
#
# Env (optional):
#   JULIA           julia binary to use (default: julia)
#   SKIP_DOCS=1     skip the (slow) Documenter build; only exercise the runner
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JULIA="${JULIA:-julia}"
SMOKE_SLUG="smoke_case"
SMOKE_OUT="$ROOT/output/day4/$SMOKE_SLUG"
SMOKE_REQ="fields.jld2"
SITE="$ROOT/docs/build"

log()  { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*" >&2; exit 1; }

assert_file()    { [[ -f "$1" ]] || fail "expected file missing: $1"; pass "file present: ${1#$ROOT/}"; }
assert_dir()     { [[ -d "$1" ]] || fail "expected dir missing: $1"; pass "dir present: ${1#$ROOT/}"; }
assert_grep()    { grep -rqI -- "$2" "$1" || fail "expected to find '$2' under ${1#$ROOT/}"; pass "found '$2' under ${1#$ROOT/}"; }

# Resolve the artifacts dir of the latest successful smoke run from its pointer.
smoke_required_artifact() {
  "$JULIA" --project="$ROOT" -e '
    include(joinpath(pwd(), "src", "TutorialWorkflow.jl")); using .TutorialWorkflow
    reg = case_registry(pwd())
    c = reg[findfirst(x -> x.slug == "'"$SMOKE_SLUG"'", reg)]
    p = safe_artifact(c, "'"$SMOKE_REQ"'"; root = pwd())
    print(p === nothing ? "" : p)
  '
}

# ---------------------------------------------------------------------------
log "Test 0: local no-publish smoke build (RUN_CASES=$SMOKE_SLUG)"
# ---------------------------------------------------------------------------

RUN_CASES="$SMOKE_SLUG" FORCE_RERUN=1 \
  "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"

# The runner must have produced the required artifact and recorded a success.
art="$(smoke_required_artifact)"
[[ -n "$art" ]] || fail "smoke_case has no successful run / required artifact ($SMOKE_REQ)"
assert_file "$art"
assert_file "$SMOKE_OUT/latest_success.json"
assert_file "$SMOKE_OUT/latest_attempt.json"
assert_grep "$SMOKE_OUT/latest_success.json" '"status"'
assert_grep "$SMOKE_OUT/latest_success.json" 'success'
pass "smoke_case ran and recorded success"

# Status pages refreshed by the runner.
assert_file "$ROOT/docs/src/status/index.md"
assert_file "$ROOT/docs/src/status/day4.md"
assert_grep "$ROOT/docs/src/status/day4.md" "$SMOKE_SLUG"

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  log "Test 0: build docs locally (DOC_DAYS=4, no publish)"
  DOC_DAYS=4 "$JULIA" --project="$ROOT/docs" "$ROOT/docs/make.jl"
  assert_dir "$SITE"
  assert_file "$SITE/index.html"
  # The smoke case page and a status page must be present in the rendered site.
  assert_grep "$SITE" "$SMOKE_SLUG"
  pass "static site built locally (no publish performed)"
else
  log "SKIP_DOCS=1 set; skipping Documenter build"
fi

# Snapshot the smoke required artifact's hash so Test 1 can prove it is untouched.
hash_before="$( "$JULIA" --project="$ROOT" -e '
  include(joinpath(pwd(), "src", "TutorialWorkflow.jl")); using .TutorialWorkflow
  print(file_hash("'"$art"'"))
' )"
[[ -n "$hash_before" ]] || fail "could not hash $art"

# ---------------------------------------------------------------------------
log "Test 1: dry run (RUN_DAYS=none) runs nothing and changes nothing"
# ---------------------------------------------------------------------------

RUN_DAYS=none "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"

hash_after="$( "$JULIA" --project="$ROOT" -e '
  include(joinpath(pwd(), "src", "TutorialWorkflow.jl")); using .TutorialWorkflow
  print(file_hash("'"$art"'"))
' )"
[[ "$hash_before" == "$hash_after" ]] \
  || fail "dry run mutated smoke artifact ($art): $hash_before -> $hash_after"
pass "smoke artifact unchanged by dry run"

# Status pages still exist (dry run refreshes them from on-disk JSON).
assert_file "$ROOT/docs/src/status/index.md"
assert_grep "$ROOT/docs/src/status/day4.md" "$SMOKE_SLUG"

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  log "Test 1: docs still build after dry run"
  DOC_DAYS=4 "$JULIA" --project="$ROOT/docs" "$ROOT/docs/make.jl"
  assert_file "$SITE/index.html"
  assert_grep "$SITE" "$SMOKE_SLUG"
fi

log "test_full_flow.sh: ALL CHECKS PASSED"
