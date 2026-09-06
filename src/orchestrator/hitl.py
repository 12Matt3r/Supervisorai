"""
Human-in-the-Loop (HITL) Deep-Sleep gate.

When a task is flagged high-stakes, the orchestrator suspends before finalizing
it, persists a pending-approval record to disk, and yields control. A human
operator approves or denies out-of-band (CLI / MCP tool); the run resumes only
after an explicit APPROVED decision.

Ported from the VORTEX-OS HITL gate and adapted to SupervisorAI. The core
invariant is preserved: **never auto-approve** — approval must come from a human
decision recorded here, not from the model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

PENDING = "PENDING_HUMAN"
APPROVED = "APPROVED"
DENIED = "DENIED"


class HITLGate:
    """File-backed approval queue for high-stakes actions."""

    def __init__(self, data_dir: str = "supervisor_data"):
        self.dir = Path(data_dir) / "pending_approvals"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        # Keep the filename filesystem-safe.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)
        return self.dir / f"{safe}.json"

    def status_of(self, task_id: str) -> Optional[str]:
        p = self._path(task_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text()).get("status")
        except (json.JSONDecodeError, OSError):
            return None

    def is_approved(self, task_id: str) -> bool:
        return self.status_of(task_id) == APPROVED

    def request(self, task_id: str, proposed_action: str, severity: str = "HIGH",
                context: str = "") -> Dict[str, Any]:
        """Record a pending approval (idempotent: keeps an existing decision)."""
        existing = self.status_of(task_id)
        if existing in (APPROVED, DENIED):
            return json.loads(self._path(task_id).read_text())
        record = {
            "task_id": task_id,
            "status": PENDING,
            "severity": severity,
            "proposed_action": proposed_action,
            "context": context,
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._path(task_id).write_text(json.dumps(record, indent=2))
        return record

    def approve(self, task_id: str, by: str = "operator") -> bool:
        return self._decide(task_id, APPROVED, by)

    def deny(self, task_id: str, by: str = "operator") -> bool:
        return self._decide(task_id, DENIED, by)

    def _decide(self, task_id: str, status: str, by: str) -> bool:
        p = self._path(task_id)
        if not p.exists():
            return False
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        rec["status"] = status
        rec["decided_by"] = by
        rec["decided_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p.write_text(json.dumps(rec, indent=2))
        return True

    def pending(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for f in sorted(self.dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("status") == PENDING:
                out.append(rec)
        return out

    def clear(self, task_id: str) -> None:
        p = self._path(task_id)
        if p.exists():
            p.unlink()
