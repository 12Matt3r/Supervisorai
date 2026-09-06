# lib/replanning.sh — Improvement #24: Mid-Plan Replanning
# On task failure, branch to a fallback sub-plan (max_wait_s).
set -euo pipefail
LIB_PREFIX="[replanning]"

replanning_branch() {
  local task_id="$1" branch="$2" max_wait_s="${3:-60}"
  echo "[replanning] task=$task_id -> branch=$branch (≤ ${max_wait_s}s)"
  if [[ -f $branch ]]; then
    # Spawn sub-plan from the branch file.
    plan_run_branch "$branch" "$max_wait_s"
  else
    echo "[replanning] branch file missing: $branch"
    return 1
  fi
}

# Stub; implemented alongside subplan.sh.
plan_run_branch() { echo "[branch] spawning sub-plan: $1"; }
