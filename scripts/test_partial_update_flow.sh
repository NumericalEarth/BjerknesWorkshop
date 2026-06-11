#!/usr/bin/env bash
#
# test_partial_update_flow.sh — Test 5 + Test 6 of the deployment test plan.
#
# Scenario: a day-4 run already exists. We then deploy ONLY day 3
# (RUN_DAYS=3) and assert the partial update is genuinely partial:
#
#   Test 5 (day-4 outputs untouched): record the day-4 cases' latest_success
#     run_ids and the hashes of their required artifacts BEFORE the day-3 run,
#     run RUN_DAYS=3, then assert every day-4 pointer + artifact is byte-for-byte
#     identical afterward (the day-3 deploy never re-ran or clobbered day 4).
#
#   Test 6 (site shows both days): build the docs with DOC_DAYS=all and assert
#     the rendered site contains BOTH a day-3 page and a day-4 page, and that the
#     status overview lists day-3 and day-4 cases — i.e. a partial run still
#     produces a complete site that carries the prior day's cached results.
#
# To keep this cheap and GPU-free, the "existing day-4 run" is seeded from the
# trivial CPU smoke_case (day 4); the heavy LES cases are not required to have
# run — the invariant under test (RUN_DAYS=3 must not touch day-4 state) holds
# for any day-4 case that has state on disk.
#
# Usage:  bash scripts/test_partial_update_flow.sh
#
# Env (optional):
#   JULIA        julia binary (default: julia)
#   SKIP_DOCS=1  skip the Documenter build (only exercise the runner invariants)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JULIA="${JULIA:-julia}"
SITE="$ROOT/docs/build"
STATE="$(mktemp -d)"
trap 'rm -rf "$STATE"' EXIT

log()  { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*" >&2; exit 1; }

assert_file() { [[ -f "$1" ]] || fail "expected file missing: $1"; }
assert_grep() { grep -rqI -- "$2" "$1" || fail "expected to find '$2' under ${1#$ROOT/}"; pass "found '$2' under ${1#$ROOT/}"; }

# Emit, one per line, "<slug>\t<run_id>\t<required-artifact-hash-concat>" for
# every day-4 case that currently has a latest_success pointer on disk.
day4_state() {
  "$JULIA" --project="$ROOT" -e '
    include(joinpath(pwd(), "src", "TutorialWorkflow.jl")); using .TutorialWorkflow
    for c in case_registry(pwd())
        c.day == 4 || continue
        info = safe_latest_success(c; root = pwd())
        info === nothing && continue
        h = IOBuffer()
        for name in c.required_outputs
            p = safe_artifact(c, name; root = pwd())
            print(h, name, "=", p === nothing ? "MISSING" : file_hash(p), ";")
        end
        println(string(c.slug, "\t", info.run_id, "\t", String(take!(h))))
    end
  '
}

# ---------------------------------------------------------------------------
log "Seed: ensure at least one day-4 case has a successful run on disk"
# ---------------------------------------------------------------------------
RUN_CASES=smoke_case FORCE_RERUN=1 \
  "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"

day4_state > "$STATE/before.txt"
[[ -s "$STATE/before.txt" ]] || fail "no day-4 case has a successful run to protect"
printf '  recorded day-4 state:\n'; sed 's/^/    /' "$STATE/before.txt"

# ---------------------------------------------------------------------------
log "Test 5: deploy ONLY day 3 (RUN_DAYS=3) and assert day-4 state is untouched"
# ---------------------------------------------------------------------------
# ALLOW_CASE_FAILURES so a placeholder day-3 case that can't render a figure
# does not abort the run; we only care that day-4 state is not mutated.
RUN_DAYS=3 ALLOW_CASE_FAILURES=1 \
  "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"

day4_state > "$STATE/after.txt"

if diff -u "$STATE/before.txt" "$STATE/after.txt"; then
  pass "day-4 latest_success run_ids and required-artifact hashes unchanged"
else
  fail "RUN_DAYS=3 mutated day-4 state (run_id or artifact hash changed; see diff above)"
fi

# ---------------------------------------------------------------------------
log "Test 6: site carries BOTH days"
# ---------------------------------------------------------------------------
# Status pages are always refreshed; they must list both day-3 and day-4 cases.
assert_file "$ROOT/docs/src/status/index.md"
assert_grep "$ROOT/docs/src/status/index.md" "smoke_case"          # a day-4 case
assert_grep "$ROOT/docs/src/status/index.md" "hybrid_physics_ml"   # a day-3 case
assert_file "$ROOT/docs/src/status/day3.md"
assert_file "$ROOT/docs/src/status/day4.md"

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  log "Test 6: build full site (DOC_DAYS=all) and assert both day pages render"
  DOC_DAYS=all "$JULIA" --project="$ROOT/docs" "$ROOT/docs/make.jl"
  assert_file "$SITE/index.html"
  [[ -d "$SITE/day3" ]] || fail "rendered site missing day3/ pages"
  [[ -d "$SITE/day4" ]] || fail "rendered site missing day4/ pages"
  pass "rendered site contains both day3/ and day4/"
  assert_grep "$SITE" "smoke_case"
  assert_grep "$SITE" "hybrid_physics_ml"
else
  log "SKIP_DOCS=1 set; skipping Documenter build"
fi

log "test_partial_update_flow.sh: ALL CHECKS PASSED"
