from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from enum import Enum
import time

class TaskStatus(Enum):
    """Represents the lifecycle status of a task."""
    PENDING = "PENDING"
    READY = "READY" # Dependencies are met, ready for assignment
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class AgentStatus(Enum):
    """Represents the current status of a managed agent."""
    IDLE = "IDLE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"

@dataclass
class ManagedAgent:
    """Represents an agent available to the orchestrator."""
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: str | None = None
    last_seen: float = field(default_factory=time.time)

@dataclass
class OrchestrationTask:
    """A specific, concrete task that can be assigned to a single agent."""
    task_id: str
    name: str
    description: str
    required_capabilities: List[str]
    dependencies: Set[str] = field(default_factory=set) # Set of other task_ids
    assigned_agent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def are_dependencies_met(self, completed_task_ids: Set[str]) -> bool:
        """Checks if all prerequisite tasks are complete."""
        return self.dependencies.issubset(completed_task_ids)

@dataclass
class ProjectGoal:
    """A high-level objective composed of multiple tasks."""
    goal_id: str
    name: str
    description: str
    tasks: Dict[str, OrchestrationTask] = field(default_factory=dict)
    status: str = "PENDING" # PENDING, IN_PROGRESS, COMPLETED, FAILED
    created_at: float = field(default_factory=time.time)

    def get_completed_task_ids(self) -> Set[str]:
        """Returns a set of all task IDs that are marked as COMPLETED."""
        return {task_id for task_id, task in self.tasks.items() if task.status == TaskStatus.COMPLETED}

    def get_ready_tasks(self) -> List[OrchestrationTask]:
        """Returns a list of tasks that are PENDING and whose dependencies are met."""
        completed_ids = self.get_completed_task_ids()
        ready_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.are_dependencies_met(completed_ids):
                ready_tasks.append(task)
        return ready_tasks
