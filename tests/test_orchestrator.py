import unittest
from unittest.mock import MagicMock
import time
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from orchestrator.core import Orchestrator
from orchestrator.models import ManagedAgent, AgentStatus, ProjectGoal, TaskStatus
from supervisor_agent.core import SupervisorCore

class TestOrchestrator(unittest.TestCase):
    """Test suite for the Orchestrator."""

    def setUp(self):
        """Set up a new Orchestrator and a mock Supervisor for each test."""
        # We need a mock SupervisorCore that has the methods the orchestrator calls.
        self.mock_supervisor = MagicMock(spec=SupervisorCore)
        # Mock the async methods on the supervisor
        self.mock_supervisor.monitor_agent = MagicMock()
        self.mock_supervisor.validate_output = MagicMock()

        self.orchestrator = Orchestrator(supervisor=self.mock_supervisor)

    def test_register_agent(self):
        """Test that an agent can be registered successfully."""
        self.orchestrator.register_agent("agent-1", "TestAgent", ["python"])
        self.assertEqual(len(self.orchestrator.list_agents()), 1)
        agent = self.orchestrator.get_agent("agent-1")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "TestAgent")

    def test_submit_goal_decomposition(self):
        """Test that a goal is decomposed into the correct tasks and dependencies."""
        # Use the predefined "scraping script" template
        project = self.orchestrator.submit_goal(
            "Scraping Project",
            "Create a Python script that can scrape a website."
        )

        self.assertEqual(len(project.tasks), 3)

        task1 = project.tasks[f"{project.goal_id}-t1"]
        task2 = project.tasks[f"{project.goal_id}-t2"]

        self.assertEqual(task1.name, "Write Scraper Code")
        self.assertEqual(len(task1.dependencies), 0)

        self.assertEqual(task2.name, "Write Unit Tests")
        self.assertIn(task1.task_id, task2.dependencies)

    def test_get_ready_tasks(self):
        """Test the logic for identifying tasks ready for execution."""
        project = self.orchestrator.submit_goal(
            "Test Project",
            "A test script project for scraping." # Triggers the template
        )

        # Initially, only the task with no dependencies should be ready
        ready_tasks = project.get_ready_tasks()
        self.assertEqual(len(ready_tasks), 1)
        self.assertEqual(ready_tasks[0].name, "Write Scraper Code")

        # Mark the first task as complete
        task1_id = ready_tasks[0].task_id
        project.tasks[task1_id].status = TaskStatus.COMPLETED

        # Now, the other two tasks (which depend on the first) should be ready
        ready_tasks_after_completion = project.get_ready_tasks()
        self.assertEqual(len(ready_tasks_after_completion), 2)
        task_names = {task.name for task in ready_tasks_after_completion}
        self.assertIn("Write Unit Tests", task_names)
        self.assertIn("Generate Report", task_names)

    def test_find_available_agent(self):
        """Test finding an agent with the right capabilities."""
        self.orchestrator.register_agent("agent-1", "PythonAgent", ["python", "file_io"])
        self.orchestrator.register_agent("agent-2", "TextAgent", ["text_analysis"])
        self.orchestrator.register_agent("agent-3", "BusyAgent", ["python"])
        self.orchestrator.update_agent_status("agent-3", AgentStatus.BUSY)

        # Find an agent for a python task
        found_agent = self.orchestrator.find_available_agent(["python"])
        self.assertIsNotNone(found_agent)
        self.assertEqual(found_agent.agent_id, "agent-1")

        # Find an agent for a task that no idle agent can do
        found_agent_none = self.orchestrator.find_available_agent(["test_execution"])
        self.assertIsNone(found_agent_none)

        # Find an agent for the text agent
        found_text_agent = self.orchestrator.find_available_agent(["text_analysis"])
        self.assertIsNotNone(found_text_agent)
        self.assertEqual(found_text_agent.agent_id, "agent-2")

    def test_full_execution_loop_simulation(self):
        """
        An integration-style test to simulate the orchestrator's main loop.
        """
        # --- Setup ---
        # Mock the supervisor's validate_output to return a successful result
        self.mock_supervisor.validate_output.return_value = {
            "intervention_result": {"intervention_required": False}
        }

        self.orchestrator.register_agent("agent-1", "MultiAgent", ["python", "file_io", "test_execution"])
        project = self.orchestrator.submit_goal("Test Project", "A test script project for scraping.")
        task1_id = f"{project.goal_id}-t1"
        task2_id = f"{project.goal_id}-t2"

        # --- Execution ---
        self.orchestrator.start()

        # Give the loop time to assign the first task
        time.sleep(0.1)

        # --- Assertions for Task 1 ---
        task1 = project.tasks[task1_id]
        agent = self.orchestrator.get_agent("agent-1")

        self.assertEqual(task1.status, TaskStatus.RUNNING)
        self.assertEqual(agent.status, AgentStatus.BUSY)
        self.assertEqual(agent.current_task_id, task1_id)

        # Let the task "finish" by sleeping past its simulated work time
        time.sleep(6)

        # --- Assertions for Task 2 ---
        # The _execute_task thread should have completed and updated the status
        self.assertEqual(task1.status, TaskStatus.COMPLETED)

        # Agent should be idle briefly before picking up the next task
        # We need to wait for the main loop to re-assign
        time.sleep(3)

        task2 = project.tasks[task2_id]
        self.assertEqual(task2.status, TaskStatus.RUNNING)
        self.assertEqual(agent.status, AgentStatus.BUSY)
        self.assertEqual(agent.current_task_id, task2_id)

        # --- Cleanup ---
        self.orchestrator.stop()


if __name__ == '__main__':
    unittest.main()
