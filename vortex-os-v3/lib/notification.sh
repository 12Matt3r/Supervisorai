# lib/notification.sh — Pass 5 notifications
# shellcheck shell=bash

[[ -n "${__LIB_NOTIFY_LOADED:-}" ]] && return 0
__LIB_NOTIFY_LOADED=1

source "$SKILL_SCRIPT_DIR/lib/core.sh"

# Send a notification. Supports multiple channels (mailto, slack webhook, generic webhook, file drop).
# Channels are configured in config.json under notifications.
notification_fire() {
  local event="$1"
  local pp; pp=$(plan_path)
  local plan_id; plan_id=$(json_get "$pp" "plan_id")
  local goal; goal=$(json_get "$pp" "goal")

  python3 - "$CONFIG" "$event" "$plan_id" "$goal" <<'PY'
import json, os, subprocess, sys, urllib.request, urllib.error, time
cfg_path, event, plan_id, goal = sys.argv[1:5]
cfg = json.load(open(cfg_path))
notif = cfg.get("notifications", {}) or {}
chans = notif.get("channels", []) or []
if not chans:
    sys.exit(0)
fmt = notif.get("message_format", "Plan {plan_id} {event}: {goal}")
body = fmt.format(plan_id=plan_id, goal=goal, event=event, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
print(f"NOTIFY: {body}")
for c in chans:
    kind = c.get("type","log")
    if kind == "log":
        print(f"[{c.get('name','notify')}] {body}")
    elif kind == "file":
        with open(c["path"], "a") as f:
            f.write(body + "\n")
    elif kind == "webhook":
        url = c.get("url")
        if not url: continue
        try:
            req = urllib.request.Request(url, data=body.encode("utf-8"),
                                         headers={"Content-Type": c.get("content_type","text/plain")},
                                         method=c.get("method","POST"))
            urllib.request.urlopen(req, timeout=5).read()
            print(f"webhook {url} ok")
        except urllib.error.URLError as e:
            print(f"webhook {url} FAIL: {e}", file=sys.stderr)
    elif kind == "mail":
        cmd = c.get("command","mail -s 'orchestrator' nobody")
        try:
            subprocess.run(cmd, shell=True, input=body, text=True, timeout=5)
        except Exception as e:
            print(f"mail FAIL: {e}", file=sys.stderr)
    elif kind == "slack":
        url = c.get("url")
        if not url: continue
        payload = json.dumps({"text": body}).encode()
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            print(f"slack FAIL: {e}", file=sys.stderr)
PY
}
