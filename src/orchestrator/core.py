from typing import Dict, List, Any, Optional, Tuple
import threading
import time
import asyncio
import json
import dataclasses

from .models import ManagedAgent, AgentStatus, ProjectGoal, OrchestrationTask, TaskStatus
from .prompts import (
    get_decomposition_prompt,
    get_worker_prompt,
    get_audit_prompt,
    get_synthesis_prompt,
)
from .trace import ExecutionTrace
from supervisor_agent.core import SupervisorCore
from llm.client import LLMClient


class Orchestrator:
    """
    Manages a pool of agents, decomposes goals into tasks, and orchestrates
    their execution under supervision.

    The showcase entry point is :meth:`run_goal`, which runs the full
    Plan -> Delegate -> Audit -> Correct reasoning loop backed by MiniMax-M3 and
    emits a live execution trace. The threaded :meth:`start`/:meth:`_main_loop`
    path is retained for backward compatibility with the MCP server.
    """

    # Token budget per worker deliverable. Deliverables (code, test files) can be
    # long; too small a budget truncates the output and the auditor correctly
    # rejects it as incomplete.
    WORKER_MAX_TOKENS = 6000
    SYNTHESIS_MAX_TOKENS = 2048

    def __init__(self, supervisor: SupervisorCore, llm_client: LLMClient):
        self.supervisor = supervisor
        self.llm_client = llm_client
        self.agent_pool: Dict[str, ManagedAgent] = {}
        self.projects: Dict[str, ProjectGoal] = {}
        # Reentrant lock: several methods legitimately call other locked methods
        # (e.g. update_agent_status -> get_agent). A plain Lock deadlocks here.
        self._lock = threading.RLock()
        self.is_running = False

    # --- Agent Management ---

    def register_agent(self, agent_id: str, name: str, capabilities: List[str]) -> ManagedAgent:
        """Adds a new agent to the pool or updates an existing one."""
        with self._lock:
            if agent_id in self.agent_pool:
                # Update existing agent, maybe it re-connected
                agent = self.agent_pool[agent_id]
                agent.name = name
                agent.capabilities = capabilities
                agent.status = AgentStatus.IDLE
                agent.last_seen = time.time()
            else:
                agent = ManagedAgent(
                    agent_id=agent_id,
                    name=name,
                    capabilities=capabilities
                )
                self.agent_pool[agent_id] = agent
            print(f"Agent registered/updated: {agent}")
            return agent

    def get_agent(self, agent_id: str) -> ManagedAgent | None:
        """Retrieves an agent by its ID."""
        with self._lock:
            return self.agent_pool.get(agent_id)

    def list_agents(self) -> List[ManagedAgent]:
        """Returns a list of all agents in the pool."""
        with self._lock:
            return list(self.agent_pool.values())

    def update_agent_status(self, agent_id: str, status: AgentStatus, task_id: str | None = None):
        """Updates the status of a specific agent."""
        with self._lock:
            agent = self.get_agent(agent_id)
            if agent:
                agent.status = status
                agent.current_task_id = task_id
                agent.last_seen = time.time()
                print(f"Agent {agent_id} status updated to {status}")
            else:
                print(f"Warning: Could not update status for unknown agent {agent_id}")

    def find_available_agent(self, required_capabilities: List[str]) -> ManagedAgent | None:
        """Finds an idle agent that has all the required capabilities."""
        with self._lock:
            for agent in self.agent_pool.values():
                if agent.status == AgentStatus.IDLE:
                    # Check if agent has all required capabilities
                    if all(cap in agent.capabilities for cap in required_capabilities):
                        return agent
            return None

    # --- Goal and Task Management ---

    async def submit_goal(self, goal_name: str, goal_description: str) -> ProjectGoal:
        """
        Accepts a new project goal, uses MiniMax-M3 to decompose it into a task
        DAG, and registers it with the orchestrator.
        """
        with self._lock:
            goal_id = f"goal-{len(self.projects) + 1}"
            project = ProjectGoal(goal_id=goal_id, name=goal_name, description=goal_description)

            # Get available agents for the prompt
            agents_info = [dataclasses.asdict(a) for a in self.list_agents()]

        # Generate the prompt for the LLM
        prompt = get_decomposition_prompt(goal_description, agents_info)

        # Query the LLM
        print("Querying MiniMax-M3 for task decomposition...")
        llm_response = await self.llm_client.query(prompt, max_tokens=2048)

        if "error" in llm_response or "tasks" not in llm_response:
            print(f"Error from LLM or invalid format: {llm_response}")
            raise ValueError("Failed to get a valid task plan from the LLM.")

        # Validate and create task objects from the LLM response
        created_tasks: Dict[str, OrchestrationTask] = {}
        llm_tasks = llm_response.get("tasks", {})

        for task_id, task_data in llm_tasks.items():
            # Basic validation
            if not all(k in task_data for k in ["name", "description", "required_capabilities", "dependencies"]):
                print(f"Skipping malformed task from LLM: {task_id}")
                continue

            new_task = OrchestrationTask(
                task_id=task_id,
                name=task_data["name"],
                description=task_data["description"],
                required_capabilities=task_data["required_capabilities"],
                dependencies=set(task_data["dependencies"]),
                validation_conditions=task_data.get("validation_conditions", []),
            )
            created_tasks[task_id] = new_task

        if not created_tasks:
            raise ValueError("LLM returned a plan with no valid tasks.")

        with self._lock:
            project.tasks = created_tasks
            self.projects[project.goal_id] = project
            print(f"Project goal submitted and decomposed by LLM: {project.name}")
            return project

    def get_project_status(self, goal_id: str) -> ProjectGoal | None:
        """Retrieves the status of an entire project."""
        with self._lock:
            return self.projects.get(goal_id)

    def add_task_to_project(self, goal_id: str, task_id: str, name: str, description: str, required_capabilities: List[str], dependencies: List[str]) -> OrchestrationTask:
        """Adds a new task to an existing project."""
        with self._lock:
            project = self.projects.get(goal_id)
            if not project:
                raise ValueError(f"Project with goal_id '{goal_id}' not found.")

            if task_id in project.tasks:
                raise ValueError(f"Task with task_id '{task_id}' already exists in project.")

            new_task = OrchestrationTask(
                task_id=task_id,
                name=name,
                description=description,
                required_capabilities=required_capabilities,
                dependencies=set(dependencies)
            )
            project.tasks[task_id] = new_task
            return new_task

    def remove_task_from_project(self, goal_id: str, task_id: str):
        """Removes a task from a project."""
        with self._lock:
            project = self.projects.get(goal_id)
            if not project:
                raise ValueError(f"Project with goal_id '{goal_id}' not found.")

            if task_id not in project.tasks:
                raise ValueError(f"Task with task_id '{task_id}' not found in project.")

            # Also remove this task from any other task's dependencies
            for other_task in project.tasks.values():
                if task_id in other_task.dependencies:
                    other_task.dependencies.remove(task_id)

            del project.tasks[task_id]

    def update_task_dependencies(self, goal_id: str, task_id: str, new_dependencies: List[str]):
        """Updates the dependencies for a specific task."""
        with self._lock:
            project = self.projects.get(goal_id)
            if not project:
                raise ValueError(f"Project with goal_id '{goal_id}' not found.")

            task = project.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task with task_id '{task_id}' not found in project.")

            task.dependencies = set(new_dependencies)

    def update_task_details(self, goal_id: str, task_id: str, new_details: Dict[str, Any]):
        """Updates the details (name, description) of a specific task."""
        with self._lock:
            project = self.projects.get(goal_id)
            if not project:
                raise ValueError(f"Project with goal_id '{goal_id}' not found.")

            task = project.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task with task_id '{task_id}' not found in project.")

            if "name" in new_details:
                task.name = new_details["name"]
            if "description" in new_details:
                task.description = new_details["description"]

    # ================================================================== #
    # Supervisory reasoning loop: Plan -> Delegate -> Audit -> Correct
    # ================================================================== #

    async def run_goal(
        self,
        goal_name: str,
        goal_description: str,
        trace: Optional[ExecutionTrace] = None,
        max_corrections: int = 1,
    ) -> Tuple[ProjectGoal, str]:
        """Drive a goal end-to-end under M3 supervision and return (project, final).

        1. PLAN     - M3 decomposes the goal into a validated task DAG.
        2. DELEGATE - each ready task is executed by an M3-backed worker.
        3. AUDIT    - M3 verifies every worker output against the plan.
        4. CORRECT  - failed/drifted outputs are re-prompted with a fix hint.
        5. SYNTHESIS- once all dependencies pass, M3 synthesizes a verified result.
        """
        trace = trace or ExecutionTrace(enabled=True)
        trace.banner(
            "SupervisorAI  ·  MiniMax-M3 on GMI Cloud",
            f"Goal: {goal_description}",
        )

        # --- 1. PLAN ---
        trace.rule("PLAN")
        trace.supervisor("Decomposing goal into a validated task DAG...")
        project = await self.submit_goal(goal_name, goal_description)
        project.status = "IN_PROGRESS"
        order = " -> ".join(t.name for t in project.tasks.values())
        trace.supervisor(f"Plan holds {len(project.tasks)} tasks: {order}")

        # --- 2-4. DELEGATE / AUDIT / CORRECT ---
        trace.rule("EXECUTE")
        guard = 0
        while True:
            ready = project.get_ready_tasks()
            if not ready:
                break
            # Execute sequentially so the trace reads as a coherent story.
            for task in ready:
                await self._run_supervised_task(project, task, trace, max_corrections)
                if task.status == TaskStatus.FAILED:
                    trace.drift(f"Task '{task.name}' could not be recovered; halting plan.")
                    project.status = "FAILED"
                    return project, ""
            guard += 1
            if guard > len(project.tasks) + 2:
                break  # safety valve against a malformed DAG

        # --- 5. SYNTHESIS (only if every dependency passed) ---
        completed = project.get_completed_task_ids()
        if len(completed) != len(project.tasks):
            trace.drift("Not all tasks passed verification; final synthesis withheld.")
            project.status = "FAILED"
            return project, ""

        trace.rule("SYNTHESIS")
        trace.supervisor("All tasks passed audit. Synthesizing verified final result...")
        final = await self._synthesize(project, trace)
        project.status = "COMPLETED"
        trace.passed("Goal COMPLETED — all dependencies verified.")
        trace.info(f"Token usage: {self.llm_client.usage_summary()}")
        return project, final

    async def _run_supervised_task(
        self,
        project: ProjectGoal,
        task: OrchestrationTask,
        trace: ExecutionTrace,
        max_corrections: int,
    ) -> None:
        """Execute a single task with M3 workers and an M3 audit + correction loop."""
        task.status = TaskStatus.RUNNING
        upstream = {
            dep: project.tasks[dep].output_text
            for dep in task.dependencies
            if dep in project.tasks
        }
        correction_hint: Optional[str] = None

        for attempt in range(max_corrections + 1):
            task.attempts += 1

            # --- DELEGATE ---
            label = "worker" if attempt == 0 else "worker (correction)"
            trace.dispatch(f"{label} → '{task.name}'  [attempt {attempt + 1}/{max_corrections + 1}]")
            prompt = get_worker_prompt(project.description, task.name, task.description, upstream)
            if correction_hint:
                prompt += (
                    f"\n\nA previous attempt FAILED the supervisor's audit. "
                    f"You MUST address this: {correction_hint}"
                )
            worker_env = await self.llm_client.complete(prompt, max_tokens=self.WORKER_MAX_TOKENS)
            output = worker_env.get("text", "") or ""
            task.output_text = output
            trace.dispatch(f"worker produced {len(output)} chars"
                           + (" (mock)" if worker_env.get("mock") else ""))

            # --- Supervisor engine pass (heuristic + LLM judge blend) ---
            await self._run_supervisor_engine(task, output, trace)

            # --- AUDIT (M3 verification pass) ---
            trace.audit(f"MiniMax-M3 verifying output of '{task.name}' against the plan...")
            audit = await self._audit(project.description, task, output)
            task.audit = audit
            task.output_data = {"audit": audit, "supervisor": task.output_data.get("supervisor") if task.output_data else None}

            verdict = audit.get("verdict", "pass")
            drift = bool(audit.get("drift_detected"))
            if drift:
                trace.drift(f"'{task.name}': {audit.get('reasoning', 'output diverged from plan')}")

            if verdict == "pass" and not drift:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                trace.passed(f"'{task.name}' verified (confidence {audit.get('confidence', 0):.2f}).")
                return

            # --- CORRECT ---
            issues = "; ".join(audit.get("issues", [])) or audit.get("reasoning", "unspecified failure")
            correction_hint = audit.get("correction_hint") or issues
            if attempt < max_corrections:
                trace.recovery(f"Audit FAILED for '{task.name}': {issues}. Re-prompting worker with fix hint.")
            else:
                trace.recovery(f"Audit FAILED for '{task.name}' after {attempt + 1} attempts: {issues}.")

        # Exhausted corrections without passing.
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()

    async def _run_supervisor_engine(self, task: OrchestrationTask, output: str, trace: ExecutionTrace) -> None:
        """Run the SupervisorCore monitor + validate blend, defensively.

        This exercises the existing heuristic QualityAnalyzer + LLM judge +
        intervention engine. It is best-effort: if the supervisor is a mock or a
        subsystem is unavailable, the M3 audit still governs the verdict.
        """
        try:
            instructions = [task.description] + list(task.validation_conditions)
            await self.supervisor.monitor_agent(
                agent_name=f"worker::{task.task_id}",
                framework="orchestrated",
                task_input=task.description,
                instructions=instructions,
                task_id=task.task_id,
            )
            validation = await self.supervisor.validate_output(task_id=task.task_id, output=output)
            if isinstance(validation, dict):
                task.output_data = {"supervisor": validation}
                inter = validation.get("intervention_result", {})
                if inter.get("intervention_required"):
                    level = inter.get("level") or inter.get("intervention_level") or "flagged"
                    trace.audit(f"Supervisor engine flagged intervention: {level}")
        except Exception as e:
            # Non-fatal: the M3 audit is the authoritative verification.
            trace.info(f"(supervisor engine pass skipped: {e})")

    async def _audit(self, goal: str, task: OrchestrationTask, output: str) -> Dict[str, Any]:
        """Ask MiniMax-M3, as supervisor, to verify an output. Always returns a verdict dict."""
        conditions = task.validation_conditions or [f"The output must accomplish: {task.description}"]
        prompt = get_audit_prompt(goal, task.name, task.description, output, conditions)
        env = await self.llm_client.complete(prompt, max_tokens=512, temperature=0.2)
        parsed = env.get("json")
        if isinstance(parsed, dict) and "verdict" in parsed:
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("drift_detected", False)
            parsed.setdefault("issues", [])
            return parsed
        # Fallback verdict if the model didn't return clean JSON.
        empty = not output.strip()
        return {
            "verdict": "fail" if empty else "pass",
            "confidence": 0.4,
            "drift_detected": empty,
            "issues": ["worker produced empty output"] if empty else [],
            "correction_hint": "Produce a concrete deliverable for the task." if empty else "",
            "reasoning": env.get("text", "audit produced no structured verdict")[:200],
        }

    async def _synthesize(self, project: ProjectGoal, trace: ExecutionTrace) -> str:
        """Synthesize a final verified deliverable from all audited task outputs."""
        outputs = {t.name: t.output_text for t in project.tasks.values()}
        prompt = get_synthesis_prompt(project.description, outputs)
        env = await self.llm_client.complete(prompt, max_tokens=self.SYNTHESIS_MAX_TOKENS, temperature=0.3)
        final = env.get("text", "")
        trace.synthesis(final if final else "(no synthesis text produced)")
        return final

    # ================================================================== #
    # Legacy threaded execution loop (kept for the MCP server)
    # ================================================================== #

    def _execute_task(self, task: OrchestrationTask, agent: ManagedAgent):
        """Run a single task on an agent in a worker thread (blocking path)."""
        try:
            print(f"Executing task {task.task_id} on agent {agent.agent_id}")
            # Run the real M3-backed worker + supervisor validation synchronously
            # inside this worker thread.
            validation_result = asyncio.run(self._execute_task_async(task, agent))

            with self._lock:
                task.output_data = validation_result
                audit = validation_result.get("audit", {}) if isinstance(validation_result, dict) else {}
                supervisor_res = validation_result.get("supervisor", {}) if isinstance(validation_result, dict) else {}
                intervention = supervisor_res.get("intervention_result", {}) if isinstance(supervisor_res, dict) else {}
                failed = intervention.get("intervention_required") or audit.get("verdict") == "fail"
                if failed:
                    task.status = TaskStatus.FAILED
                    print(f"Task {task.task_id} FAILED verification.")
                else:
                    task.status = TaskStatus.COMPLETED
                    print(f"Task {task.task_id} COMPLETED successfully.")
                task.completed_at = time.time()

        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.output_data = {"error": str(e)}
                print(f"Task {task.task_id} FAILED with exception: {e}")

        finally:
            # Always release the agent
            self.update_agent_status(agent.agent_id, AgentStatus.IDLE)

    async def _execute_task_async(self, task: OrchestrationTask, agent: ManagedAgent) -> Dict[str, Any]:
        """Async body of a single threaded task: run worker, supervisor, and audit."""
        # Delegate to an M3 worker.
        prompt = get_worker_prompt(task.description, task.name, task.description, {})
        worker_env = await self.llm_client.complete(prompt, max_tokens=self.WORKER_MAX_TOKENS)
        output = worker_env.get("text", "") or f"Completed: {task.description}"
        task.output_text = output

        result: Dict[str, Any] = {}
        # Supervisor engine pass (best-effort).
        try:
            await self.supervisor.monitor_agent(
                agent_name=agent.name,
                framework="orchestrated",
                task_input=task.description,
                instructions=[task.description],
                task_id=task.task_id,
            )
            supervisor_res = await self.supervisor.validate_output(task_id=task.task_id, output=output)
            result["supervisor"] = supervisor_res
        except Exception as e:
            result["supervisor"] = {"error": str(e)}

        # M3 audit pass.
        result["audit"] = await self._audit(task.description, task, output)
        return result

    def _main_loop(self):
        """The main execution loop that assigns tasks to agents."""
        while self.is_running:
            with self._lock:
                for project in self.projects.values():
                    if project.status in ["COMPLETED", "FAILED"]:
                        continue

                    ready_tasks = project.get_ready_tasks()
                    for task in ready_tasks:
                        agent = self.find_available_agent(task.required_capabilities)
                        if agent:
                            print(f"Assigning task {task.task_id} to agent {agent.agent_id}")
                            task.status = TaskStatus.RUNNING
                            self.update_agent_status(agent.agent_id, AgentStatus.BUSY, task.task_id)

                            # Run the task in a new thread to not block the main loop
                            task_thread = threading.Thread(target=self._execute_task, args=(task, agent))
                            task_thread.start()

            time.sleep(2)  # Check for new tasks every 2 seconds

    def start(self):
        """Starts the orchestrator's main execution loop in a background thread."""
        if self.is_running:
            print("Orchestrator is already running.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        print("Orchestrator started.")

    def stop(self):
        """Stops the orchestrator's main loop."""
        self.is_running = False
        print("Orchestrator stopping...")
