"""
LLMClient — GMI Cloud / MiniMax-M3 adapter for SupervisorAI.

This is the single integration point every LLM-using component in the system
depends on (the orchestrator's task decomposer, the supervisor's LLM judge, and
the researcher's self-heal assistor). Retargeting this one class flips the whole
platform onto MiniMax-M3 served by GMI Cloud.

Endpoint (Anthropic Messages-compatible serving API, verified live):
    POST https://api.gmi-serving.com/v1/messages
    Auth: `x-api-key: <GMI_API_KEY>`  and/or  `Authorization: Bearer <GMI_API_KEY>`
    Model: MiniMaxAI/MiniMax-M3

Configuration is read from the environment so the demo is turnkey:
    GMI_API_KEY          - the GMI serving API key (JWT). Never commit it.
    GMI_BASE_URL         - default https://api.gmi-serving.com/v1
    SUPERVISOR_MODEL     - default MiniMaxAI/MiniMax-M3
    SUPERVISOR_TOKEN_BUDGET - optional int; soft cap on cumulative tokens.

When no key is configured the client degrades to a deterministic, context-aware
mock so the demo, the test suite, and the MCP server never hard-crash. Every
mock response carries a ``mock_response`` key (so downstream components can tell
they are running un-keyed) while still returning plausible structured data.
"""

from __future__ import annotations

import os
import json
import asyncio
import random
import re
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx

# Model ids GMI accepts for MiniMax-M3 (primary + fallback alias).
DEFAULT_MODEL = "MiniMaxAI/MiniMax-M3"
FALLBACK_MODEL = "minimax/minimax-m3"
DEFAULT_BASE_URL = "https://api.gmi-serving.com/v1"

_PLACEHOLDER_KEYS = {"", "YOUR_ANTHROPIC_API_KEY", "YOUR_GMI_API_KEY", "changeme"}


class LLMClient:
    """Async client for MiniMax-M3 served through GMI Cloud.

    The public :meth:`query` contract is preserved for backward compatibility:
    it returns a ``dict`` that is either the parsed JSON emitted by the model,
    ``{"text_response": ...}`` for free-form text, ``{"error": ...}`` on failure,
    or ``{"mock_response": ...}`` when no API key is configured.

    :meth:`complete` is the richer entry point used by the supervisory reasoning
    loop; it returns a structured envelope with text, parsed JSON, tool calls,
    usage and stop reason.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        timeout: float = 60.0,
        token_budget: Optional[int] = None,
    ):
        # Accept the legacy `api_key`/`model` positional args used across the
        # codebase, but source everything else from the environment.
        self.api_key = api_key or os.environ.get("GMI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        # Defend against the classic paste-into-a-secret-box bug: a trailing
        # newline or stray whitespace in the key makes an *illegal* HTTP header
        # value (httpx raises "Illegal header value"). Keys never have edge
        # whitespace, so stripping is always safe and prevents a broken deploy.
        if self.api_key:
            self.api_key = self.api_key.strip()
        # A model of "claude-3-*" is a legacy default passed by old call sites
        # (e.g. LLMJudge). Ignore it and use the configured supervisor model.
        if model and not model.lower().startswith("claude"):
            self.model = model
        else:
            self.model = os.environ.get("SUPERVISOR_MODEL", DEFAULT_MODEL)

        self.base_url = (base_url or os.environ.get("GMI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_url = f"{self.base_url}/messages"
        self.max_retries = max_retries
        self.timeout = timeout

        env_budget = os.environ.get("SUPERVISOR_TOKEN_BUDGET")
        self.token_budget = token_budget if token_budget is not None else (
            int(env_budget) if env_budget and env_budget.isdigit() else None
        )

        # Cumulative token accounting for budget management / reporting.
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if not self.is_configured:
            print(
                "Warning: GMI_API_KEY is not configured. LLM calls will be mocked "
                "(set GMI_API_KEY to run against MiniMax-M3)."
            )

    # ------------------------------------------------------------------ #
    # Properties / accounting
    # ------------------------------------------------------------------ #
    @property
    def is_configured(self) -> bool:
        """True when a usable API key is present."""
        return bool(self.api_key) and self.api_key not in _PLACEHOLDER_KEYS

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "token_budget": self.token_budget,
            "budget_remaining": (self.token_budget - self.total_tokens) if self.token_budget else None,
        }

    def _budget_exceeded(self) -> bool:
        return self.token_budget is not None and self.total_tokens >= self.token_budget

    def _headers(self) -> Dict[str, str]:
        # Send both auth styles: `x-api-key` (Anthropic style, verified working)
        # and `Authorization: Bearer` (OpenAI-compatible path). GMI accepts either.
        return {
            "x-api-key": self.api_key or "",
            "Authorization": f"Bearer {self.api_key or ''}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    # ------------------------------------------------------------------ #
    # Core request with retry + backoff
    # ------------------------------------------------------------------ #
    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to the Messages endpoint with exponential backoff.

        Retries on network errors, 429, and 5xx. Raises the last exception /
        returns the last error response otherwise.
        """
        last_error: Optional[Exception] = None
        async with httpx.AsyncClient() as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        self.api_url, headers=self._headers(), json=payload, timeout=self.timeout
                    )
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(
                            f"retryable status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else None
                    # Non-retryable client errors (except 429) fail fast.
                    if status is not None and status not in (429, 500, 502, 503, 504):
                        raise
                    last_error = e
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last_error = e

                if attempt < self.max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(backoff)

        if last_error:
            raise last_error
        raise RuntimeError("LLM request failed with no captured error")

    # ------------------------------------------------------------------ #
    # Rich completion entry point (used by the supervisory reasoning loop)
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send a prompt to MiniMax-M3 and return a structured envelope.

        Returns a dict with keys: ``ok`` (bool), ``text`` (str), ``json`` (parsed
        object or None), ``tool_calls`` (list), ``usage`` (dict), ``stop_reason``,
        ``model``, and ``mock`` (bool). Never raises; failures are reported in the
        envelope with ``ok=False`` and an ``error`` key.
        """
        if not self.is_configured:
            return self._mock_complete(prompt, tools)

        if self._budget_exceeded():
            return {
                "ok": False,
                "text": "",
                "json": None,
                "tool_calls": [],
                "usage": self.usage_summary(),
                "error": "token budget exceeded",
                "model": self.model,
                "mock": False,
            }

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        try:
            data = await self._post(payload)
        except httpx.HTTPStatusError as e:
            detail = e.response.text if e.response is not None else str(e)
            print(f"LLM Client API Error: {detail}")
            return {"ok": False, "text": "", "json": None, "tool_calls": [],
                    "usage": self.usage_summary(), "error": detail, "model": self.model, "mock": False}
        except Exception as e:  # network exhausted retries, etc.
            print(f"An unexpected error occurred with the LLM Client: {e}")
            return {"ok": False, "text": "", "json": None, "tool_calls": [],
                    "usage": self.usage_summary(), "error": str(e), "model": self.model, "mock": False}

        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse an Anthropic Messages-format response body into the envelope."""
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in data.get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input", {}),
                })
        text = "".join(text_parts).strip()

        usage = data.get("usage", {}) or {}
        self.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.total_output_tokens += int(usage.get("output_tokens", 0) or 0)

        return {
            "ok": True,
            "text": text,
            "json": extract_json(text),
            "tool_calls": tool_calls,
            "usage": {**usage, **self.usage_summary()},
            "stop_reason": data.get("stop_reason"),
            "model": data.get("model", self.model),
            "mock": False,
        }

    # ------------------------------------------------------------------ #
    # Legacy JSON-dict entry point (kept for backward compatibility)
    # ------------------------------------------------------------------ #
    async def query(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """Send a prompt and return parsed JSON (or a wrapped text/error dict).

        Backward-compatible contract used by the orchestrator, LLM judge, and
        researcher:
          * parsed JSON object emitted by the model, or
          * ``{"text_response": <text>}`` when the model returned prose, or
          * ``{"error": ..., "details": ...}`` on failure, or
          * ``{"mock_response": ...}`` when no API key is configured.
        """
        if not self.is_configured:
            mock = self._mock_complete(prompt, None)
            # Preserve the legacy shape while surfacing structured mock data.
            result: Dict[str, Any] = {"mock_response": mock["text"]}
            if isinstance(mock.get("json"), dict):
                result.update(mock["json"])
            return result

        envelope = await self.complete(prompt, max_tokens=max_tokens)
        if not envelope["ok"]:
            return {"error": "API error", "details": envelope.get("error", "unknown error")}

        if isinstance(envelope.get("json"), (dict, list)):
            parsed = envelope["json"]
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        return {"text_response": envelope["text"]}

    # ------------------------------------------------------------------ #
    # Streaming (used by the demo trace for a live token feel)
    # ------------------------------------------------------------------ #
    async def stream(self, prompt: str, max_tokens: int = 1024,
                     system: Optional[str] = None) -> AsyncIterator[str]:
        """Yield text deltas from a streaming completion.

        Falls back to yielding the mock text in chunks when unconfigured.
        """
        if not self.is_configured:
            text = self._mock_complete(prompt, None)["text"]
            for chunk in _chunk_text(text):
                await asyncio.sleep(0.01)
                yield chunk
            return

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.api_url, headers=self._headers(),
                                          json=payload, timeout=self.timeout) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if raw in ("", "[DONE]"):
                            continue
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
        except Exception as e:  # pragma: no cover - network dependent
            print(f"LLM Client streaming error: {e}")
            return

    # ------------------------------------------------------------------ #
    # Deterministic, context-aware mock fallback
    # ------------------------------------------------------------------ #
    def _mock_complete(self, prompt: str, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Produce a plausible response without an API key.

        The mock inspects the prompt so the demo tells a coherent story:
        decomposition prompts yield a task graph, audit prompts yield a verdict,
        everything else yields worker-style prose. Every response carries the
        ``mock_response`` marker text used by downstream components.
        """
        marker = "LLM client is not configured with an API key."
        lowered = prompt.lower()

        payload_json: Optional[Any] = None
        if "synthesize" in lowered or "final verified deliverable" in lowered:
            text = (
                "[mock synthesis] " + marker + " Final result: the discount function "
                "now clamps percent to 0-100 and treats a missing percent as 0, with "
                "regression tests covering percent>100, percent=None, and the normal "
                "case.\n\nVerification: every sub-task output is consistent and the "
                "goal (fix issue #482 with tests and a PR summary) is met."
            )
        elif "task plan" in lowered or "decompose" in lowered or '"tasks"' in prompt:
            payload_json = _MOCK_TASK_PLAN
            text = json.dumps(payload_json)
        elif "impartial ai supervisor" in lowered or "overall_score" in lowered:
            payload_json = {
                "overall_score": 0.86,
                "reasoning": f"[mock] {marker} Heuristic judgement only.",
                "is_safe": True,
            }
            text = json.dumps(payload_json)
        elif "verdict" in lowered or "audit" in lowered or "verification" in lowered:
            # Inspect the worker output embedded in the audit prompt so the mock
            # verdict actually reflects the output (empty/trivial output fails).
            worker_output = _extract_worker_output(prompt)
            if len(worker_output.strip()) < 15:
                payload_json = {
                    "verdict": "fail",
                    "confidence": 0.78,
                    "issues": ["output is empty or too short to satisfy the task"],
                    "drift_detected": True,
                    "correction_hint": "Produce a concrete, complete deliverable for the task.",
                    "reasoning": f"[mock] {marker} Output does not satisfy the plan.",
                }
            else:
                payload_json = {
                    "verdict": "pass",
                    "confidence": 0.82,
                    "issues": [],
                    "drift_detected": False,
                    "correction_hint": "",
                    "reasoning": f"[mock] {marker} Output looks consistent with the plan.",
                }
            text = json.dumps(payload_json)
        else:
            text = (
                f"[mock worker output] {marker} Produced a placeholder result for the "
                f"requested step. Provide GMI_API_KEY to run MiniMax-M3 for real output."
            )

        return {
            "ok": True,
            "text": text,
            "json": payload_json,
            "tool_calls": [],
            "usage": self.usage_summary(),
            "stop_reason": "end_turn",
            "model": f"{self.model} (mock)",
            "mock": True,
            "mock_response": marker,
        }


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON object/array from model text.

    Handles bare JSON, ```json fenced blocks, and JSON embedded in prose.
    Returns the parsed value, or None if no valid JSON is found.
    """
    if not text:
        return None
    # Strip markdown code fences.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _extract_worker_output(prompt: str) -> str:
    """Pull the fenced worker output out of an audit prompt (best-effort)."""
    marker = "Worker's output:"
    idx = prompt.find(marker)
    segment = prompt[idx + len(marker):] if idx != -1 else prompt
    fenced = re.search(r"```(.*?)```", segment, re.DOTALL)
    return fenced.group(1) if fenced else segment


def _chunk_text(text: str, size: int = 24) -> List[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


# A small, coherent mock plan for the demo's "fix a bug" scenario.
_MOCK_TASK_PLAN = {
    "tasks": {
        "analyze_issue": {
            "name": "Analyze the issue",
            "description": "Read the bug report and locate the offending code path.",
            "required_capabilities": ["code_analysis"],
            "dependencies": [],
        },
        "implement_fix": {
            "name": "Implement the fix",
            "description": "Edit the source to correct the defect described in the issue.",
            "required_capabilities": ["code_edit"],
            "dependencies": ["analyze_issue"],
        },
        "write_tests": {
            "name": "Write regression tests",
            "description": "Add tests that fail before the fix and pass after it.",
            "required_capabilities": ["test_execution"],
            "dependencies": ["implement_fix"],
        },
        "summarize_pr": {
            "name": "Summarize the pull request",
            "description": "Produce a verified summary of the change for review.",
            "required_capabilities": ["code_analysis"],
            "dependencies": ["write_tests"],
        },
    }
}
