#!/usr/bin/env python3
"""
mcp_server.py — A minimal MCP server exposing orchestrator_skill.

Exposes tools that wrap the orchestrator's commands. This lets any MCP-
aware client drive a plan via JSON-RPC instead of shell.

Tools:
  - plan.show
  - plan.add_task
  - plan.dispatch_ready
  - plan.mark_done
  - plan.mark_failed
  - plan.status
  - plan.seal
  - plan.export

Run with stdio transport (default):
  python3 mcp_server.py --config /path/to/config.json

Or HTTP (via 'mcp' over FastAPI - optional):
  python3 mcp_server.py --http --port 8765
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
SKILL_BIN = os.path.join(SKILL_DIR, "skill.sh")

def _run_skill(*args: str) -> Dict[str, Any]:
    """Invoke skill.sh and return a parsed JSON-ish dict if possible."""
    proc = subprocess.run(
        ["bash", SKILL_BIN, *args],
        capture_output=True, text=True
    )
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    try:
        return {"ok": proc.returncode == 0, "result": json.loads(out), "stderr": err, "raw_stdout": out}
    except Exception:
        return {"ok": proc.returncode == 0, "result": out, "stderr": err, "raw_stdout": out}

# -- MCP plumbing -------------------------------------------------------------
TOOLS: Dict[str, Any] = {}

def tool(name: str, description: str, schema: Dict[str, Any]):
    def deco(fn):
        TOOLS[name] = {"description": description, "input_schema": schema, "fn": fn}
        return fn
    return deco

@tool("plan.show",       "Show the current plan.",        {"type":"object","properties":{}})
def t_show(_):        return _run_skill("--plan")

@tool("plan.status",     "Show orchestration status.",    {"type":"object","properties":{}})
def t_status(_):      return _run_skill("--status")

@tool("plan.add_task",
      "Append a task to the plan.",
      {"type":"object",
       "properties":{
         "id": {"type":"string"},
         "description": {"type":"string"},
         "agent_role": {"type":"string"},
         "depends_on": {"type":"array","items":{"type":"string"}},
         "output_path": {"type":"string"}
       },
       "required": ["id","description","agent_role"]})
def t_add(args):      return _run_skill("--add-task", json.dumps(args))

@tool("plan.dispatch_ready", "Dispatch all ready tasks.",
      {"type":"object","properties":{}})
def t_dispatch(_):    return _run_skill("--dispatch-ready")

@tool("plan.mark_done",
      "Mark a task complete.",
      {"type":"object",
       "properties":{
         "id": {"type":"string"},
         "output_path": {"type":"string"}
       },"required":["id","output_path"]})
def t_done(args):     return _run_skill("--mark-done", args["id"], args["output_path"])

@tool("plan.mark_failed",
      "Mark a task failed with a reason.",
      {"type":"object",
       "properties":{
         "id": {"type":"string"},
         "reason": {"type":"string"}
       },"required":["id","reason"]})
def t_fail(args):     return _run_skill("--mark-failed", args["id"], args["reason"])

@tool("plan.seal",
      "Seal the plan (sets sealed=true).",
      {"type":"object","properties":{"force":{"type":"boolean"}}})
def t_seal(args):     return _run_skill("--seal")

@tool("plan.export",
      "Export plan + state + audit to a single JSON bundle.",
      {"type":"object",
       "properties":{"path":{"type":"string"}},"required":["path"]})
def t_export(args):   return _run_skill("--export", args["path"])

# -- JSON-RPC 2.0 server (stdio) ----------------------------------------------
def handle_rpc(req: Dict[str, Any]) -> Dict[str, Any]:
    method = req.get("method","")
    if method == "initialize":
        return {"jsonrpc":"2.0","id":req["id"],
                "result":{
                  "protocolVersion":"2024-11-05",
                  "capabilities":{"tools":{"listChanged":False}},
                  "serverInfo":{"name":"orchestrator_skill","version":"2.0.0"},
                }}
    if method == "notifications/initialized":
        return None  # ack only
    if method == "tools/list":
        return {"jsonrpc":"2.0","id":req["id"],
                "result":{"tools":[
                  {"name":n, "description":m["description"], "inputSchema":m["input_schema"]}
                  for n,m in TOOLS.items()
                ]}}
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name","")
        args = params.get("arguments", {}) or {}
        if name not in TOOLS:
            return {"jsonrpc":"2.0","id":req["id"],
                    "error":{"code":-32601,"message":f"unknown tool {name}"}}
        result = TOOLS[name]["fn"](args)
        return {"jsonrpc":"2.0","id":req["id"],
                "result":{"content":[{"type":"text","text":json.dumps(result, indent=2)}],
                          "isError": not result.get("ok",True)}}
    return {"jsonrpc":"2.0","id":req.get("id"),
            "error":{"code":-32601,"message":f"unknown method {method}"}}

def serve_stdio(config_path: str | None = None) -> None:
    if config_path:
        os.environ["SKILL_CONFIG"] = config_path
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            req = json.loads(line)
        except Exception:
            print(json.dumps({"jsonrpc":"2.0","error":{"code":-32700,"message":"bad json"}}), flush=True)
            continue
        resp = handle_rpc(req)
        if resp is not None:
            print(json.dumps(resp), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--http",   action="store_true", help="Run a tiny HTTP server on --port (optional).")
    ap.add_argument("--port",   type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        # Optional HTTP transport — uses Python stdlib only.
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length","0"))
                body = self.rfile.read(n) if n else b""
                try: req = json.loads(body)
                except Exception:
                    self.send_response(400); self.end_headers(); self.wfile.write(b"bad json"); return
                resp = handle_rpc(req)
                if resp is None: self.send_response(204); self.end_headers(); return
                data = json.dumps(resp).encode()
                self.send_response(200); self.send_header("Content-Type","application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers()
                self.wfile.write(data)
            def log_message(self, *a, **k): pass
        HTTPServer(("0.0.0.0", args.port), H).serve_forever()
    else:
        serve_stdio(args.config)
