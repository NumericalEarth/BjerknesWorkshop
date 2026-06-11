#!/usr/bin/env bash
#
# test_failure_robustness.sh — Test 3 + Test 4 of the deployment test plan.
#
#   Test 3 (resilient deploy survives a case failure): force the
#     `lead_ocean_waves` case to fail via SIMULATE_CASE_FAILURE, run the
#     orchestrator with ALLOW_CASE_FAILURES=1 (so the critical failure does not
#     abort the run), then build the docs and assert (a) the site still builds,
#     and (b) it surfaces the failure — the status page shows the simulated
#     failure badge and the case's Results section shows a "no successful run"
#     style banner instead of throwing.
#
#   Test 4 (strict mode fails the build): the SAME simulated failure under
#     STRICT_CASES=1 must make the orchestrator exit nonzero (exit 2). This is
#     the gate a CI/publish step would use to refuse to publish a broken deploy.
#
# No GPU is used: SIMULATE_CASE_FAILURE short-circuits the subprocess, so the
# real LES never launches. The smoke_case is also selected so there is at least
# one *successful* case alongside the failing one, proving the site builds with
# a mix of success and failure.
#
# Usage:  bash scripts/test_failure_robustness.sh
#
# Env (optional):
#   JULIA        julia binary (default: julia)
#   SKIP_DOCS=1  skip the Documenter build (only exercise the runner exit codes)

# NOTE: deliberately NOT using `set -e` because Test 4 expects a nonzero exit
# from the orchestrator and must capture it without killing this script.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JULIA="${JULIA:-julia}"
FAIL_SLUG="lead_ocean_waves"
SITE="$ROOT/docs/build"
STATUS_DAY4="$ROOT/docs/src/status/day4.md"

log()  { printf '\n=== %s ===\n' "$*"; }
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*" >&2; exit 1; }

assert_file() { [[ -f "$1" ]] || fail "expected file missing: $1"; }
assert_grep() { grep -rqI -- "$2" "$1" || fail "expected to find '$2' under ${1#$ROOT/}"; pass "found '$2' under ${1#$ROOT/}"; }

# ---------------------------------------------------------------------------
log "Test 3: resilient deploy — simulate $FAIL_SLUG failure, keep going"
# ---------------------------------------------------------------------------
# Run the failing case plus the trivial smoke_case so the site has a success too.
# ALLOW_CASE_FAILURES=1 means the critical failure does NOT abort the run, and
# (without STRICT_CASES) the orchestrator returns 0.
SIMULATE_CASE_FAILURE="$FAIL_SLUG" \
RUN_CASES="$FAIL_SLUG,smoke_case" \
ALLOW_CASE_FAILURES=1 FORCE_RERUN=1 \
  "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"
rc=$?
[[ $rc -eq 0 ]] || fail "resilient run (ALLOW_CASE_FAILURES=1, no STRICT) should exit 0, got $rc"
pass "orchestrator exited 0 despite a simulated critical-case failure"

# The failure must be recorded in the status pointers / pages.
assert_file "$ROOT/output/day4/$FAIL_SLUG/latest_attempt.json"
assert_grep "$ROOT/output/day4/$FAIL_SLUG/latest_attempt.json" "simulated_failure"
assert_file "$STATUS_DAY4"
assert_grep "$STATUS_DAY4" "simulated failure"
pass "status page shows the simulated-failure badge"

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  log "Test 3: docs still build and surface the failure"
  DOC_DAYS=4 "$JULIA" --project="$ROOT/docs" "$ROOT/docs/make.jl"
  drc=$?
  [[ $drc -eq 0 ]] || fail "docs build should succeed even with a failed case, got $drc"
  assert_file "$SITE/index.html"
  # The failing case's page rendered (did not throw), and the status overview
  # carries the failure banner text.
  assert_grep "$SITE" "$FAIL_SLUG"
  assert_grep "$SITE/status" "simulated failure"
  pass "site built and surfaces the failure banner"
else
  log "SKIP_DOCS=1 set; skipping Documenter build"
fi

# ---------------------------------------------------------------------------
log "Test 4: strict mode — same failure must make the orchestrator exit nonzero"
# ---------------------------------------------------------------------------
SIMULATE_CASE_FAILURE="$FAIL_SLUG" \
RUN_CASES="$FAIL_SLUG,smoke_case" \
ALLOW_CASE_FAILURES=1 STRICT_CASES=1 FORCE_RERUN=1 \
  "$JULIA" --project="$ROOT" "$ROOT/scripts/run_tutorials.jl"
src=$?
if [[ $src -eq 0 ]]; then
  fail "STRICT_CASES=1 with a failing case should exit nonzero, but got 0"
fi
pass "strict mode exited nonzero ($src) as required"

log "test_failure_robustness.sh: ALL CHECKS PASSED"
