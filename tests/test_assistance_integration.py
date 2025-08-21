import unittest
import asyncio
import sys
import os
import tempfile
import shutil

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from supervisor_agent.core import SupervisorCore
from supervisor_agent import InterventionLevel

class TestAssistanceIntegration(unittest.TestCase):
    """
    Integration test for the stuck agent assistance feature.
    """

    def setUp(self):
        """Set up a temporary directory and a SupervisorCore instance."""
        self.temp_dir = tempfile.mkdtemp(prefix="supervisor_assist_test_")
        self.supervisor = SupervisorCore(data_dir=self.temp_dir)
        # We need to run the async setup for the supervisor if it has one
        # For now, we assume direct instantiation is sufficient.

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_stuck_agent_assistance_flow(self):
        """
        Test that a stuck agent correctly receives a proactive ASSISTANCE intervention.
        """
        # Since the methods are async, we need to run this test in an event loop
        async def run_test():
            task_id = await self.supervisor.monitor_agent(
                agent_name="stuck_agent",
                framework="test",
                task_input="a task that will cause repeated failures",
                instructions=["fail repeatedly"]
            )

            task = self.supervisor.active_tasks[task_id]
            low_quality_output = "this output is too short and will fail quality checks"

            # Fail 1
            result1 = await self.supervisor.validate_output(task_id, low_quality_output)
            self.assertTrue(result1['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 1)

            # Fail 2
            result2 = await self.supervisor.validate_output(task_id, low_quality_output)
            self.assertTrue(result2['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 2)

            # Fail 3 - This should trigger assistance
            result3 = await self.supervisor.validate_output(task_id, low_quality_output)
            self.assertTrue(result3['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 3)

            final_intervention = result3['intervention_result']
            self.assertEqual(final_intervention['level'], InterventionLevel.ASSISTANCE.value)
            self.assertIn("Based on research", final_intervention['reason'])

        # Run the async test
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
