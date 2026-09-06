"""
Tests for the VORTEX-OS-derived features ported into SupervisorAI:
agent roster (personas), HITL approval gate, continuity enforcement, and
on-disk deliverables.
"""

import os
import sys
import asyncio
import tempfile
import unittest
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from orchestrator.core import (
    Orchestrator, _strip_code_fence, _safe_filename, _continuity_violation,
    _extract_deliverable,
)
from orchestrator.roster import AgentRoster
from orchestrator.hitl import HITLGate, PENDING, APPROVED, DENIED
from orchestrator.trace import ExecutionTrace
from supervisor_agent.core import SupervisorCore
from llm.client import LLMClient


class TestHelpers(unittest.TestCase):
    def test_strip_code_fence(self):
        self.assertEqual(_strip_code_fence("```html\n<x>\n```"), "<x>")
        self.assertEqual(_strip_code_fence("```\ncode\n```"), "code")
        self.assertEqual(_strip_code_fence("no fence here"), "no fence here")

    def test_safe_filename(self):
        self.assertEqual(_safe_filename("deliverables/index.html and notes"), "index.html")
        self.assertEqual(_safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(_safe_filename(""), "")

    def test_extract_deliverable(self):
        # prose + fenced code -> just the code
        self.assertEqual(
            _extract_deliverable("Here:\n```python\ndef f():\n    return 1\n```\ndone", "m.py"),
            "def f():\n    return 1")
        # JSON envelope -> the file body field
        self.assertIn("assert True", _extract_deliverable(
            '{"test_file_content": "def t():\\n    assert True\\n"}', "t.py"))
        # raw code untouched
        self.assertEqual(_extract_deliverable("def g():\n    return 2\n", "x.py"),
                         "def g():\n    return 2\n")
        # markdown prose untouched (not a code ext)
        self.assertEqual(_extract_deliverable("# Title\n\ntext", "notes.md"), "# Title\n\ntext")
        # a real json deliverable stays raw
        self.assertEqual(_extract_deliverable('{"a": 1}', "data.json"), '{"a": 1}')

    def test_continuity_violation(self):
        self.assertIn("smartphone", _continuity_violation("She pulled out a smartphone.", ["no smartphones"]))
        self.assertEqual(_continuity_violation("oil lamp glow", ["no smartphones"]), "")
        # A non-negative rule is not enforced deterministically.
        self.assertEqual(_continuity_violation("a smartphone", ["smartphones are fine"]), "")


class TestAgentRoster(unittest.TestCase):
    def test_load_repo_roster(self):
        roster = AgentRoster()
        self.assertGreaterEqual(len(roster), 1)
        self.assertIn("coder.python", roster.names())
        self.assertTrue(roster.persona_for("coder.python"))

    def test_resolve_by_capability(self):
        roster = AgentRoster()
        spec = roster.resolve(None, ["python"])
        self.assertIsNotNone(spec)
        # exact name wins
        spec2 = roster.resolve("writer.docs", ["python"])
        self.assertEqual(spec2.name, "writer.docs")

    def test_catalog_string(self):
        roster = AgentRoster()
        cat = roster.as_prompt_catalog()
        self.assertIn("coder.python", cat)


class TestHITLGate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.gate = HITLGate(self.dir)

    def test_request_and_approve(self):
        self.gate.request("task_a", "write file", "HIGH", "context")
        self.assertEqual(self.gate.status_of("task_a"), PENDING)
        self.assertEqual(len(self.gate.pending()), 1)
        self.assertTrue(self.gate.approve("task_a"))
        self.assertTrue(self.gate.is_approved("task_a"))
        self.assertEqual(self.gate.pending(), [])

    def test_deny(self):
        self.gate.request("task_b", "deploy", "CRITICAL")
        self.assertTrue(self.gate.deny("task_b"))
        self.assertEqual(self.gate.status_of("task_b"), DENIED)
        self.assertFalse(self.gate.is_approved("task_b"))

    def test_decide_unknown_task(self):
        self.assertFalse(self.gate.approve("nope"))

    def test_request_idempotent_keeps_decision(self):
        self.gate.request("t", "x")
        self.gate.approve("t")
        self.gate.request("t", "x")  # should not reset to pending
        self.assertEqual(self.gate.status_of("t"), APPROVED)


class TestRunGoalFeatures(unittest.TestCase):
    """End-to-end (mock LLM) check of roster + continuity + deliverables + HITL."""

    def _orch(self):
        data_dir = tempfile.mkdtemp()
        return Orchestrator(SupervisorCore(), LLMClient(), data_dir=data_dir)

    def test_run_goal_writes_deliverables_and_honors_hitl(self):
        orch = self._orch()
        deliverables_dir = tempfile.mkdtemp()
        approvals = []

        def approve(task):
            approvals.append(task.task_id)
            return True

        async def run():
            return await orch.run_goal(
                "Demo", "Fix a bug and add tests",
                trace=ExecutionTrace(enabled=False),
                max_corrections=1,
                continuity_rules=["no smartphones"],
                deliverables_dir=deliverables_dir,
                approval_callback=approve,
            )

        project, final = asyncio.run(run())
        self.assertEqual(project.status, "COMPLETED")
        # Deliverables were written for every task.
        files = os.listdir(deliverables_dir)
        self.assertEqual(len(files), len(project.tasks))

    def test_hitl_denial_fails_the_run(self):
        orch = self._orch()
        # Force every task high-stakes by post-processing the plan via a denial.
        deliverables_dir = tempfile.mkdtemp()

        # Mark tasks high-stakes by monkeypatching submit_goal result.
        real_submit = orch.submit_goal

        async def submit_with_high_stakes(name, desc):
            project = await real_submit(name, desc)
            for t in project.tasks.values():
                t.high_stakes = True
            return project

        orch.submit_goal = submit_with_high_stakes

        async def run():
            return await orch.run_goal(
                "Demo", "Ship something risky",
                trace=ExecutionTrace(enabled=False),
                deliverables_dir=deliverables_dir,
                approval_callback=lambda task: False,  # operator denies
            )

        project, final = asyncio.run(run())
        self.assertEqual(project.status, "FAILED")

    def test_hitl_halts_without_callback(self):
        orch = self._orch()
        deliverables_dir = tempfile.mkdtemp()
        real_submit = orch.submit_goal

        async def submit_with_high_stakes(name, desc):
            project = await real_submit(name, desc)
            for t in project.tasks.values():
                t.high_stakes = True
            return project

        orch.submit_goal = submit_with_high_stakes

        async def run():
            return await orch.run_goal(
                "Demo", "Ship something risky",
                trace=ExecutionTrace(enabled=False),
                deliverables_dir=deliverables_dir,
                approval_callback=None,  # no operator -> halt
            )

        project, final = asyncio.run(run())
        self.assertEqual(project.status, "PENDING_APPROVAL")
        self.assertGreaterEqual(len(orch.pending_approvals()), 1)


if __name__ == "__main__":
    unittest.main()
