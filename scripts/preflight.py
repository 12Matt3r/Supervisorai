#!/usr/bin/env python3
"""
SupervisorAI preflight — validate configuration before you run anything.

Addresses the biggest onboarding footgun: the client falls back to a
*deterministic offline mock* when no key (or a wrong base URL) is configured, so
a misconfigured deployment can run silently on the mock without anyone noticing.
This script fails **loudly** instead.

It checks, in order:
  1. Environment / .env — GMI_API_KEY, GMI_BASE_URL, SUPERVISOR_MODEL.
  2. JSON config files parse (config/weights.json, config.json, and the bash
     implementations' config.minimax.json).
  3. A live auth probe — one tiny request to the configured endpoint to confirm
     the key actually works, and that you are on **live MiniMax-M3**, not the mock.

Run it:
    python scripts/preflight.py            # human-readable report
    python scripts/preflight.py --require-live   # exit non-zero unless live M3

Exit codes: 0 = ready (live, or mock explicitly acknowledged);
            1 = misconfigured (bad JSON, or a key that fails to authenticate);
            2 = running on the offline MOCK and --require-live was set.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GREEN, YELLOW, RED, DIM, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
OK, WARN, BAD = f"{GREEN}✓{RESET}", f"{YELLOW}⚠{RESET}", f"{RED}✗{RESET}"


def load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def check_json(rel: str, required: bool) -> bool:
    p = ROOT / rel
    if not p.exists():
        print(f"  {WARN if not required else BAD} {rel} — not found")
        return not required
    try:
        json.loads(p.read_text())
        print(f"  {OK} {rel} — valid JSON")
        return True
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {BAD} {rel} — INVALID: {e}")
        return False


async def main() -> int:
    require_live = "--require-live" in sys.argv
    load_dotenv()
    from llm.client import LLMClient  # after sys.path + dotenv

    print(f"\n{BOLD}SupervisorAI · preflight{RESET}")
    print("=" * 52)
    problems = 0

    # 1. Environment ---------------------------------------------------------
    print(f"\n{BOLD}1. Environment{RESET}")
    client = LLMClient()
    key = os.environ.get("GMI_API_KEY", "")
    if client.is_configured:
        print(f"  {OK} GMI_API_KEY — set ({len(key)} chars)")
    else:
        print(f"  {WARN} GMI_API_KEY — NOT set → the system will run on the OFFLINE MOCK")
    print(f"  {OK} GMI_BASE_URL — {client.base_url}")
    print(f"  {OK} SUPERVISOR_MODEL — {client.model}")
    budget = os.environ.get("SUPERVISOR_TOKEN_BUDGET")
    print(f"  {DIM}·{RESET} SUPERVISOR_TOKEN_BUDGET — {budget or 'unset (no cap)'}")

    # 2. Config files --------------------------------------------------------
    print(f"\n{BOLD}2. Config files{RESET}")
    if not check_json("config/weights.json", required=True):
        problems += 1
    check_json("config.json", required=False)
    check_json("implementations/vortex-os-v3/config.minimax.json", required=False)

    # 3. Live auth probe -----------------------------------------------------
    print(f"\n{BOLD}3. Live auth probe{RESET}")
    live = False
    if not client.is_configured:
        print(f"  {WARN} skipped — no key configured; responses would be MOCKED.")
    else:
        print(f"  {DIM}·{RESET} pinging {client.api_url} …")
        env = await client.complete("Reply with the single word: OK", max_tokens=5)
        if env.get("mock"):
            print(f"  {BAD} response came back as a MOCK despite a key being set — check GMI_BASE_URL.")
            problems += 1
        elif env.get("ok"):
            print(f"  {OK} authenticated — live MiniMax-M3 responded "
                  f"({env.get('model')}). You are NOT on the mock.")
            live = True
        else:
            print(f"  {BAD} auth/probe FAILED: {env.get('error', 'unknown error')}")
            print(f"      → the key is set but the endpoint rejected it. Fix the key or GMI_BASE_URL.")
            problems += 1

    # Verdict ----------------------------------------------------------------
    print("\n" + "=" * 52)
    if problems:
        print(f"{RED}{BOLD}PREFLIGHT FAILED{RESET} — {problems} problem(s) above. Fix them before deploying.")
        return 1
    if live:
        print(f"{GREEN}{BOLD}READY — live MiniMax-M3.{RESET} Everything checks out.")
        return 0
    # No hard problems, but running on the mock.
    if require_live:
        print(f"{RED}{BOLD}NOT LIVE{RESET} — running on the offline mock, but --require-live was set.")
        print("Set GMI_API_KEY (see .env.example) and re-run.")
        return 2
    print(f"{YELLOW}{BOLD}OK — offline mock mode.{RESET} Fine for a dry run; set GMI_API_KEY for live M3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
