import unittest
import asyncio
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from researcher.assistor import ResearchAssistor
from supervisor_agent import AgentTask, TaskStatus
from datetime import datetime

class TestResearchAssistor(unittest.TestCase):
    """Test suite for the ResearchAssistor."""

    def setUp(self):
        """Set up a new ResearchAssistor for each test."""
        self.assistor = ResearchAssistor()

    def test_research_and_suggest_returns_string(self):
        """
        Test that the research_and_suggest method returns a non-empty string.
        """
        # Create a mock task object that has the attributes the method expects
        mock_task = AgentTask(
            task_id="task-123",
            agent_name="TestAgent",
            framework="test",
            original_input="scrape a website",
            instructions=[],
            status=TaskStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        error_context = {"error_message": "site connection failed"}

        # Since the method is async, we need to run it in an event loop
        suggestion = asyncio.run(self.assistor.research_and_suggest(mock_task, error_context))

        self.assertIsInstance(suggestion, str)
        self.assertGreater(len(suggestion), 0)

if __name__ == '__main__':
    unittest.main()
