#!/usr/bin/env python3
"""
migrate_plan.py — Convert a v1 plan.json to v2.

Usage:
  python3 migrate_plan.py old.json new.json

Adds defaults that v2 expects and is idempotent (running on a v2 plan
is a no-op).
"""
import json, sys, time

if len(sys.argv) != 3:
    print("usage: migrate_plan.py OLD.json NEW.json", file=sys.stderr); sys.exit(2)

src, dst = sys.argv[1], sys.argv[2]
p = json.load(open(src))
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# Top-level fields v2 expects.
p.setdefault("sealed", bool(p.get("sealed_at")))
p.setdefault("created_at", now)
p.setdefault("sub_plan_ids", [])

# Per-task fields v2 expects.
for t in p.get("tasks", []):
    t.setdefault("attempts", 0)
    t.setdefault("notes", "")
    if not isinstance(t["notes"], str):
        t["notes"] = " | ".join(str(x) for x in t["notes"])
    t.setdefault("status", "pending")
    if t["status"] not in {"pending","ready","dispatched","running","complete","failed","sealed","skipped","escalated"}:
        t["status"] = "pending"
    t.setdefault("handoff", "bidirectional")
    t.setdefault("depends_on", [])
    t.setdefault("agent_role", "?")

json.dump(p, open(dst, "w"), indent=2)
print(f"Migrated: {dst}  ({len(p.get('tasks',[]))} tasks)")
