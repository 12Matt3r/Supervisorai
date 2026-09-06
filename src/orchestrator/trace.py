"""
Live execution trace for the SupervisorAI reasoning loop.

Emits tagged, colourised lines so a viewer can follow the Plan -> Delegate ->
Audit -> Correct cycle in real time:

    [SUPERVISOR - M3]   plan / reasoning produced by MiniMax-M3
    [DISPATCH]          a worker / MCP call being delegated
    [AUDIT]             the supervisor validating a worker's output
    [DRIFT DETECTED]    output diverged from the plan
    [RECOVERY]          self-heal / re-prompt / patch in progress

Uses `rich` when available and degrades to plain prints otherwise, so the demo
runs with zero hard dependencies.
"""

from __future__ import annotations

from typing import Optional

try:  # rich is optional; fall back to plain text if unavailable.
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
except Exception:  # pragma: no cover - exercised only without rich installed
    _RICH = False


_STYLES = {
    "SUPERVISOR - M3": "bold cyan",
    "DISPATCH": "bold yellow",
    "AUDIT": "bold magenta",
    "DRIFT DETECTED": "bold red",
    "RECOVERY": "bold red",
    "PASS": "bold green",
    "SYNTHESIS": "bold green",
    "INFO": "dim white",
}


class ExecutionTrace:
    """A thin, dependency-optional live tracer."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console() if (_RICH and enabled) else None

    def _emit(self, tag: str, message: str) -> None:
        if not self.enabled:
            return
        if self.console is not None:
            style = _STYLES.get(tag, "white")
            line = Text()
            line.append(f"[{tag}] ", style=style)
            line.append(message, style="white")
            self.console.print(line)
        else:
            print(f"[{tag}] {message}")

    # Convenience methods per stage ------------------------------------- #
    def supervisor(self, message: str) -> None:
        self._emit("SUPERVISOR - M3", message)

    def dispatch(self, message: str) -> None:
        self._emit("DISPATCH", message)

    def audit(self, message: str) -> None:
        self._emit("AUDIT", message)

    def drift(self, message: str) -> None:
        self._emit("DRIFT DETECTED", message)

    def recovery(self, message: str) -> None:
        self._emit("RECOVERY", message)

    def passed(self, message: str) -> None:
        self._emit("PASS", message)

    def synthesis(self, message: str) -> None:
        self._emit("SYNTHESIS", message)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if self.console is not None:
            body = Text(title, style="bold white")
            if subtitle:
                body.append(f"\n{subtitle}", style="dim white")
            self.console.print(Panel(body, border_style="cyan"))
        else:
            print("=" * 68)
            print(title)
            if subtitle:
                print(subtitle)
            print("=" * 68)

    def rule(self, message: str = "") -> None:
        if not self.enabled:
            return
        if self.console is not None:
            self.console.rule(message, style="dim cyan")
        else:
            print(f"\n--- {message} ---")
