import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from orchestrator.core import Orchestrator
from orchestrator.models import ProjectGoal, TaskStatus

class TestInteractiveEditing(unittest.TestCase):
    """Test suite for the interactive editing features of the Orchestrator."""

    def setUp(self):
        """Set up a new Orchestrator for each test."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        mock_supervisor = MagicMock()
        mock_llm_client = MagicMock()

        self.orchestrator = Orchestrator(
            supervisor=mock_supervisor,
            llm_client=mock_llm_client,
        )

        # Create a sample project for testing
        self.project = ProjectGoal(goal_id="proj_1", name="Test Project", description="A test project")
        self.orchestrator.projects["proj_1"] = self.project

    def tearDown(self):
        """Clean up the event loop after each test."""
        self.loop.close()

    def test_add_task_to_project(self):
        """Test that a task can be added to an existing project."""
        self.orchestrator.add_task_to_project(
            goal_id="proj_1",
            task_id="new_task_1",
            name="New Task",
            description="A new task to be done.",
            required_capabilities=["python"],
            dependencies=[]
        )
        self.assertIn("new_task_1", self.project.tasks)
        self.assertEqual(self.project.tasks["new_task_1"].name, "New Task")

    def test_remove_task_from_project(self):
        """Test that a task can be removed, and dependencies are cleaned up."""
        # Add tasks first
        self.orchestrator.add_task_to_project("proj_1", "task_a", "A", "...", [], [])
        self.orchestrator.add_task_to_project("proj_1", "task_b", "B", "...", [], ["task_a"])

        self.assertIn("task_a", self.project.tasks["task_b"].dependencies)

        # Remove task_a
        self.orchestrator.remove_task_from_project("proj_1", "task_a")

        self.assertNotIn("task_a", self.project.tasks)
        # Check that the dependency was also removed
        self.assertNotIn("task_a", self.project.tasks["task_b"].dependencies)

    def test_update_task_dependencies(self):
        """Test that a task's dependencies can be updated."""
        self.orchestrator.add_task_to_project("proj_1", "task_1", "1", "...", [], [])
        self.orchestrator.add_task_to_project("proj_1", "task_2", "2", "...", [], [])
        self.orchestrator.add_task_to_project("proj_1", "task_3", "3", "...", [], ["task_1"])

        self.assertEqual(self.project.tasks["task_3"].dependencies, {"task_1"})

        # Update dependencies
        self.orchestrator.update_task_dependencies("proj_1", "task_3", ["task_1", "task_2"])

        self.assertEqual(self.project.tasks["task_3"].dependencies, {"task_1", "task_2"})

    def test_update_task_details(self):
        """Test that a task's name and description can be updated."""
        self.orchestrator.add_task_to_project("proj_1", "task_x", "Old Name", "Old Desc", [], [])

        self.orchestrator.update_task_details("proj_1", "task_x", {
            "name": "New Name",
            "description": "New Desc"
        })

        task = self.project.tasks["task_x"]
        self.assertEqual(task.name, "New Name")
        self.assertEqual(task.description, "New Desc")

if __name__ == '__main__':
    unittest.main()
