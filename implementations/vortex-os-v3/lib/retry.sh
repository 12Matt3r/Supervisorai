# lib/retry.sh — exponential-backoff retry policy for orchestration
# shellcheck shell=bash

[[ -n "${__LIB_RETRY_LOADED:-}" ]] && return 0
__LIB_RETRY_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"
source "$SKILL_SCRIPT_DIR/lib/plan_state.sh"

# Decide whether to retry a failed task.
#
# Inputs (as env vars):
#   RETRY_ATTEMPTS                current attempt count
#   RETRY_MAX_ATTEMPTS            policy ceiling
#   RETRY_BASE_DELAY_S            policy base delay (seconds)
#   RETRY_MAX_DELAY_S             policy max delay (seconds)
#   RETRY_JITTER                  "1"/"0"
#   RETRY_ERROR_CLASS             current failure tag (e.g. timeout)
#
# Outputs (to stdout):  either "wait:<N>"  or  "giveup".
retry_decide() {
  python3 - <<PY
import json, random, sys
attempts = ${RETRY_ATTEMPTS:-0}
max_a    = ${RETRY_MAX_ATTEMPTS:-2}
base     = ${RETRY_BASE_DELAY_S:-2.0}
maxd     = ${RETRY_MAX_DELAY_S:-60.0}
jitter   = "${RETRY_JITTER:-1}" == "1"
err_cls  = "${RETRY_ERROR_CLASS:-}"
if attempts >= max_a:
    print("giveup"); sys.exit(0)
delay = min(base * (2 ** attempts), maxd)
if jitter:
    delay *= random.uniform(0.7, 1.3)
print(f"wait:{delay:.1f}")
PY
}

# Apply a retry policy to a dispatched task: increments attempts and may
# decide to wait or give up. The wait is performed with sleep.
retry_apply() {
  local task_id="$1"
  python3 - "$task_id" <<'PY'
import json, os, time
import importlib.util as iu
spec = iu.spec_from_file_location("core", os.environ.get("SKILL_SCRIPT_DIR",".")+"/lib/core.sh")
PY
  # Re-implementation in shell: read task, set attempts, decide.
  local pp; pp=$(plan_path)
  ensure_plan "$(gen_plan_id)" || return 1
  with_locked_plan "
t = next((t for t in p['tasks'] if t['id'] == '$task_id'), None)
if not t:
    print(json.dumps({'error':'not_found','task_id':'$task_id'}))
else:
    pol = t.get('retry_policy') or {}
    print(json.dumps({
        'attempts': t.get('attempts', 0),
        'policy': pol,
        'task_id': t['id']
    }))
" >/tmp/.retry_pre.json

  local attempts max_a base maxd jitter
  attempts=$(python3 -c "import json;print(json.load(open('/tmp/.retry_pre.json'))['attempts'])")
  max_a=$(python3 -c "import json;p=json.load(open('/tmp/.retry_pre.json'))['policy'];print(p.get('max_attempts',2))")
  base=$(python3 -c "import json;p=json.load(open('/tmp/.retry_pre.json'))['policy'];print(p.get('base_delay_s',2.0))")
  maxd=$(python3 -c "import json;p=json.load(open('/tmp/.retry_pre.json'))['policy'];print(p.get('max_delay_s',60.0))")
  jitter=$(python3 -c "import json;p=json.load(open('/tmp/.retry_pre.json'))['policy'];print('1' if p.get('jitter',True) else '0')")

  RETRY_ATTEMPTS="$attempts" \
  RETRY_MAX_ATTEMPTS="$max_a" \
  RETRY_BASE_DELAY_S="$base" \
  RETRY_MAX_DELAY_S="$maxd" \
  RETRY_JITTER="$jitter" \
  RETRY_ERROR_CLASS="" \
    retry_decide
  rm -f /tmp/.retry_pre.json
}
