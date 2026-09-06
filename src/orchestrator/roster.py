"""
Agent roster — named worker personas loaded from ``agents/*.json`` manifests.

Each manifest declares a worker agent's ``name``, ``capabilities``, a human
``description``, and a ``persona`` (the system prompt that specialises the M3
worker when it executes a task assigned to that agent). The roster is fed to M3
during decomposition (so it can assign the right specialist per task) and used
at execution time to give each worker its persona.

Ported from the VORTEX-OS agent-manifest concept and adapted to SupervisorAI.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentSpec:
    name: str
    capabilities: List[str] = field(default_factory=list)
    description: str = ""
    persona: str = ""


class AgentRoster:
    """Loads and serves agent manifests from a directory of JSON files."""

    def __init__(self, agents_dir: Optional[str] = None):
        # Default: <repo>/agents (three levels up from this file: src/orchestrator/roster.py)
        if agents_dir is None:
            agents_dir = str(Path(__file__).resolve().parents[2] / "agents")
        self.agents_dir = Path(agents_dir)
        self.agents: Dict[str, AgentSpec] = {}
        self.load()

    def load(self) -> None:
        self.agents.clear()
        if not self.agents_dir.is_dir():
            return
        for f in sorted(self.agents_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            name = data.get("name") or f.stem
            self.agents[name] = AgentSpec(
                name=name,
                capabilities=data.get("capabilities", []),
                description=data.get("description", ""),
                persona=data.get("persona", ""),
            )

    def __len__(self) -> int:
        return len(self.agents)

    def names(self) -> List[str]:
        return list(self.agents.keys())

    def get(self, name: str) -> Optional[AgentSpec]:
        return self.agents.get(name)

    def persona_for(self, name: str) -> Optional[str]:
        spec = self.agents.get(name)
        return spec.persona if spec else None

    def as_prompt_catalog(self) -> str:
        """A compact listing for the decomposition prompt."""
        if not self.agents:
            return "(no specialist agents registered)"
        return "\n".join(
            f"- {spec.name}: {spec.description} (capabilities: {', '.join(spec.capabilities)})"
            for spec in self.agents.values()
        )

    def resolve(self, name: Optional[str], required_capabilities: Optional[List[str]] = None) -> Optional[AgentSpec]:
        """Best-effort match: exact name, else first agent covering the capabilities."""
        if name and name in self.agents:
            return self.agents[name]
        if required_capabilities:
            for spec in self.agents.values():
                if any(cap in spec.capabilities for cap in required_capabilities):
                    return spec
        return None
