# lib/locking.sh — cooperative file locking (flock-based, with mkdir fallback)
# shellcheck shell=bash

# Guard against double-sourcing
[[ -n "${__LIB_LOCKING_LOADED:-}" ]] && return 0
__LIB_LOCKING_LOADED=1

# Returns the path of the lock file for a given resource path.
# Lock files live in a sibling ".locks/" directory to keep them out of
# shared mount points and to make them easily ignored by VCS.
_lock_path_for() {
  local resource="$1"
  local d; d="$(dirname "$resource")"
  local base; base="$(basename "$resource")"
  echo "$d/.locks/${base}.lock"
}

# Acquire an exclusive lock on a resource. Echoes the lockfile path so the
# caller can release it. The lock is released by the EXIT trap.
#
# Usage:
#   _lock_path="$(lock_acquire "$plan_path")"
#   do_work
#   lock_release "$_lock_path"
lock_acquire() {
  local resource="$1"
  local timeout="${2:-30}"   # seconds
  local lock; lock="$(_lock_path_for "$resource")"
  ensure_parent "$lock" || return 1

  # Try flock first; fall back to mkdir-based lock if flock is unavailable.
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$lock" || { err "Cannot open lock: $lock"; return 1; }
    if flock -w "$timeout" 9; then
      echo "$lock"
      return 0
    fi
    err "lock timeout on $resource (waited ${timeout}s)"
    exec 9>&-
    return 1
  else
    local start; start=$(date +%s)
    while true; do
      if mkdir "$lock" 2>/dev/null; then
        echo "$lock"
        return 0
      fi
      if (( $(date +%s) - start >= timeout )); then
        err "lock timeout on $resource (waited ${timeout}s)"
        return 1
      fi
      sleep 0.1
    done
  fi
}

# Release a lock. Idempotent.
lock_release() {
  local lock="$1"
  [[ -z "$lock" ]] && return 0
  if [[ -d "$lock" ]]; then
    rmdir "$lock" 2>/dev/null || true
  fi
  if command -v flock >/dev/null 2>&1; then
    # Best-effort: close the fd if we still own it.
    exec 9>&- 2>/dev/null || true
  fi
  return 0
}

# with_lock: convenience wrapper.
#
# Usage:
#   with_lock "$plan_path" python3 my_modify_script.py
with_lock() {
  local resource="$1"; shift
  local lock; lock="$(lock_acquire "$resource" 30)"
  local rc=$?
  [[ $rc -ne 0 ]] && return $rc
  # shellcheck disable=SC2068
  $@
  local rc2=$?
  lock_release "$lock"
  return $rc2
}
