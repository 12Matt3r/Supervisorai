from typing import Dict, List, Any
import threading
import time
import asyncio
import json
import dataclasses

from .models import ManagedAgent, AgentStatus, ProjectGoal, OrchestrationTask, TaskStatus
from .prompts import get_decomposition_prompt
from supervisor_agent.core import SupervisorCore
from llm.client import LLMClient

class Orchestrator:
    """
    Manages a pool of agents, decomposes goals into tasks,
    and orchestrates their execution.
    """

    def __init__(self, supervisor: SupervisorCore, llm_client: LLMClient):
        self.supervisor = supervisor
        self.llm_client = llm_client
        self.agent_pool: Dict[str, ManagedAgent] = {}
        self.projects: Dict[str, ProjectGoal] = {}
        self._lock = threading.Lock()
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
        Accepts a new project goal, uses an LLM to decompose it into tasks,
        and adds it to the orchestrator.
        """
        with self._lock:
            goal_id = f"goal-{len(self.projects) + 1}"
            project = ProjectGoal(goal_id=goal_id, name=goal_name, description=goal_description)

            # Get available agents for the prompt
            agents_info = [dataclasses.asdict(a) for a in self.list_agents()]

        # Generate the prompt for the LLM
        prompt = get_decomposition_prompt(goal_description, agents_info)

        # Query the LLM
        print("Querying LLM for task decomposition...")
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
                dependencies=set(task_data["dependencies"])
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

    # --- Execution Loop ---

    def _execute_task(self, task: OrchestrationTask, agent: ManagedAgent):
        """Wrapper to run a task in a separate thread and handle the result."""
        try:
            print(f"Executing task {task.task_id} on agent {agent.agent_id}")

            # This is a simplified execution. A real system would have a more robust
            # way to pass inputs and get outputs from the agent task.
            # We use the supervisor to monitor this "execution".
            asyncio.run(self.supervisor.monitor_agent(
                agent_name=agent.name,
                framework="orchestrated",
                task_input=task.description,
                instructions=[],
                task_id=task.task_id
            ))

            # Simulate work and get an output
            time.sleep(5) # Simulate the agent working on the task
            output = f"Completed: {task.description}"

            validation_result = asyncio.run(self.supervisor.validate_output(
                task_id=task.task_id,
                output=output
            ))

            # Update task based on supervision result
            with self._lock:
                task.output_data = validation_result
                if validation_result['intervention_result']['intervention_required']:
                    task.status = TaskStatus.FAILED
                    print(f"Task {task.task_id} FAILED due to required intervention.")
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

            time.sleep(2) # Check for new tasks every 2 seconds

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
