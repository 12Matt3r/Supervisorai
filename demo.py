#!/usr/bin/env python3
"""
SupervisorAI — Turnkey demo for the MiniMax Week × GMI Cloud Hackathon (Track 1).

Runs the full supervisory reasoning loop end to end against MiniMax-M3 served by
GMI Cloud:

    PLAN  ->  DELEGATE  ->  AUDIT  ->  CORRECT  ->  SYNTHESIS

Scenario: an incoming bug report on a small mock repository. MiniMax-M3, acting
as the supervisor, plans the fix as a validated task DAG, delegates each step to
an M3-backed worker, audits every output against the plan (catching drift /
hallucination), self-heals failures, and only declares success once every
dependency has passed — finishing with a verified PR summary.

Run it:

    export GMI_API_KEY="<your GMI serving key>"
    python demo.py

With no key set the demo still runs end to end using a deterministic mock so the
trace is always legible; set GMI_API_KEY to see real MiniMax-M3 reasoning.
"""

from __future__ import annotations

import os
import sys
import asyncio
import textwrap
from pathlib import Path

# Make `src/` importable without requiring PYTHONPATH to be set by the caller.
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency) so the demo is turnkey."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Real environment always wins over the file.
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from llm.client import LLMClient  # noqa: E402
from supervisor_agent.core import SupervisorCore  # noqa: E402
from orchestrator.core import Orchestrator  # noqa: E402
from orchestrator.trace import ExecutionTrace  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock repository + issue for the scenario
# --------------------------------------------------------------------------- #
MOCK_REPO_FILE = "billing/discount.py"
MOCK_REPO_SOURCE = textwrap.dedent(
    '''\
    def apply_discount(price, percent):
        # BUG: does not validate percent; a percent > 100 yields a negative price,
        # and a None percent raises a TypeError at runtime.
        return price - (price * percent / 100)
    '''
)
MOCK_ISSUE = (
    "Bug report (issue #482): `apply_discount` in billing/discount.py returns a "
    "negative price when the discount percent is greater than 100, and crashes "
    "with a TypeError when percent is None. Fix the function to clamp the percent "
    "to the 0-100 range and treat a missing percent as 0, and add regression "
    "tests covering percent>100, percent=None, and a normal discount.\n\n"
    "IMPORTANT for the worker team: you are authoring artifacts (analysis, a code "
    "patch, a pytest test file, a static coverage review, and a PR summary) for a "
    "human to review. You do NOT have a code execution environment, so never "
    "claim to have run tests or report fabricated test output. The verification "
    "step must be a STATIC review that inspects the written tests and argues, from "
    "the code alone, that they cover every required edge case. Finish with a "
    "verified summary suitable for a pull request."
)


def build_orchestrator() -> Orchestrator:
    """Wire the supervisor, the M3 client, and the orchestrator together."""
    supervisor = SupervisorCore()
    llm_client = LLMClient()  # reads GMI_API_KEY / GMI_BASE_URL / SUPERVISOR_MODEL
    orch = Orchestrator(supervisor=supervisor, llm_client=llm_client)

    # Register the worker "team". Their capabilities inform M3's decomposition.
    orch.register_agent("w-analyst", "CodeAnalyst", ["code_analysis"])
    orch.register_agent("w-coder", "CodeEditor", ["code_edit"])
    orch.register_agent("w-tester", "TestWriter", ["test_execution"])
    return orch, llm_client


async def run() -> int:
    trace = ExecutionTrace(enabled=True)
    orch, llm_client = build_orchestrator()

    mode = "MiniMax-M3 (live via GMI Cloud)" if llm_client.is_configured else "deterministic mock (no GMI_API_KEY set)"
    trace.info(f"Model backend: {mode}")
    trace.info(f"Endpoint: {llm_client.api_url}   Model: {llm_client.model}")

    # Show the mock repo the supervisor is about to work on.
    trace.rule("MOCK REPO")
    trace.info(f"{MOCK_REPO_FILE}:")
    for ln in MOCK_REPO_SOURCE.rstrip().splitlines():
        trace.info(f"    {ln}")

    goal_name = "Fix issue #482: apply_discount"
    goal_description = f"{MOCK_ISSUE}\n\nRelevant file {MOCK_REPO_FILE}:\n{MOCK_REPO_SOURCE}"

    try:
        project, final = await orch.run_goal(
            goal_name=goal_name,
            goal_description=goal_description,
            trace=trace,
            max_corrections=2,
        )
    except Exception as e:
        trace.drift(f"Demo failed: {e}")
        return 1

    trace.rule("RESULT")
    if project.status == "COMPLETED":
        trace.passed("Verified PR summary produced. All dependencies passed audit.")
        trace.banner("VERIFIED PR SUMMARY", final[:2000] if final else "(mock synthesis)")
    else:
        trace.drift(f"Project ended with status: {project.status}")

    # Per-task audit ledger.
    trace.rule("AUDIT LEDGER")
    for task in project.tasks.values():
        verdict = (task.audit or {}).get("verdict", "n/a")
        conf = (task.audit or {}).get("confidence", 0.0)
        trace.info(f"  {task.status.value:10s} | {task.name:28s} | verdict={verdict} conf={conf:.2f} attempts={task.attempts}")

    # Explicit self-correction showcase: run the real audit path on a
    # deliberately incomplete worker output so the DRIFT -> RECOVERY -> PASS
    # cycle is always visible (this exercises orch._audit, not a canned string).
    await demonstrate_self_correction(orch, trace)

    trace.info(f"\nToken usage: {llm_client.usage_summary()}")
    return 0 if project.status == "COMPLETED" else 2


async def demonstrate_self_correction(orch: Orchestrator, trace: ExecutionTrace) -> None:
    """Show the supervisor catching a bad output and driving a correction."""
    from orchestrator.models import OrchestrationTask

    trace.rule("SELF-CORRECTION SHOWCASE")
    task = OrchestrationTask(
        task_id="showcase",
        name="Implement the fix",
        description="Clamp percent to 0-100 and treat a missing percent as 0.",
        required_capabilities=["code_edit"],
        validation_conditions=[
            "The output must include a concrete code change to apply_discount.",
            "The output must handle percent > 100 and percent is None.",
        ],
    )
    goal = "Fix apply_discount so it never returns a negative price."

    # First worker attempt: empty / non-answer -> audit must catch it.
    bad_output = ""
    trace.dispatch("worker → 'Implement the fix' [attempt 1] (returns an empty stub)")
    audit = await orch._audit(goal, task, bad_output)
    trace.audit(f"verdict={audit.get('verdict')} issues={audit.get('issues')}")
    if audit.get("verdict") == "fail" or audit.get("drift_detected"):
        trace.drift("Empty/insufficient output detected — plan not satisfied.")
        trace.recovery(f"Re-prompting worker with hint: {audit.get('correction_hint') or audit.get('issues')}")

    # Second attempt: a real fix -> audit should pass.
    good_output = (
        "def apply_discount(price, percent):\n"
        "    percent = 0 if percent is None else max(0, min(100, percent))\n"
        "    return price - (price * percent / 100)\n"
    )
    trace.dispatch("worker → 'Implement the fix' [attempt 2] (returns a real patch)")
    audit2 = await orch._audit(goal, task, good_output)
    if audit2.get("verdict") == "pass":
        trace.passed(f"Correction verified (confidence {audit2.get('confidence', 0):.2f}).")
    else:
        trace.audit(f"verdict={audit2.get('verdict')} — {audit2.get('reasoning')}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
