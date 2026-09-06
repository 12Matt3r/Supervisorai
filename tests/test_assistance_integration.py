import unittest
import asyncio
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import AsyncMock, patch

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

        A genuinely low-quality output (which the heuristic QualityAnalyzer scores
        below the intervention threshold) is submitted repeatedly. The first two
        outputs trigger CORRECTION-level interventions; the third crosses the
        "stuck" threshold and triggers proactive research ASSISTANCE. The external
        research tools (web search / page fetch) and the research LLM are mocked at
        their boundary so the assistance suggestion is deterministic.
        """
        # A short, incoherent non-answer scores below the 0.4 intervention floor.
        low_quality_output = "idk"

        # Mock the external research boundary used by ResearchAssistor.
        mock_search = AsyncMock(return_value=json.dumps(
            [{"title": "Answer", "link": "http://stackoverflow.com/q/123"}]))
        mock_view = AsyncMock(return_value="The fix is to validate inputs before use.")
        # The research assistor synthesizes its suggestion via its own LLM client.
        self.supervisor.research_assistor.llm_client.query = AsyncMock(
            return_value={"text_response": "Based on research, validate inputs before use."})

        async def run_test():
            task_id = await self.supervisor.monitor_agent(
                agent_name="stuck_agent",
                framework="test",
                task_input="a task that will cause repeated failures",
                instructions=["produce a complete, correct answer"]
            )

            task = self.supervisor.active_tasks[task_id]

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

        with patch('researcher.assistor.google_search', mock_search), \
             patch('researcher.assistor.view_text_website', mock_view):
            asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
