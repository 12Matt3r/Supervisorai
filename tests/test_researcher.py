import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os
import json

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from researcher.assistor import ResearchAssistor
from supervisor_agent import AgentTask, TaskStatus
from datetime import datetime

class TestResearchAssistor(unittest.TestCase):
    """Test suite for the ResearchAssistor."""

    def setUp(self):
        """Set up a new ResearchAssistor for each test."""
        self.assistor = ResearchAssistor(api_key="test_key") # Use a dummy key

    @patch('llm.client.LLMClient.query', new_callable=AsyncMock)
    def test_full_research_flow(self, mock_llm_query):
        """
        Test the full research and suggestion flow with mocked external calls.
        """
        # --- Setup Mocks ---
        mock_google_search = AsyncMock()
        mock_view_website = AsyncMock()

        mock_search_results = json.dumps([
            {"title": "Some other result", "link": "http://example.com"},
            {"title": "Best Answer on StackOverflow", "link": "http://stackoverflow.com/q/123"}
        ])
        mock_google_search.return_value = mock_search_results

        mock_website_content = "The best way to fix this is to use `try-except` blocks."
        mock_view_website.return_value = mock_website_content

        mock_llm_suggestion = "Based on my research, you should use a try-except block."
        mock_llm_query.return_value = {"text_response": mock_llm_suggestion}

        # --- Test Data ---
        mock_task = AgentTask(
            task_id="task-123", agent_name="TestAgent", framework="test",
            original_input="scrape a website", instructions=[],
            status=TaskStatus.ACTIVE, created_at=datetime.now(), updated_at=datetime.now()
        )
        error_context = {"error_message": "site connection failed"}

        # --- Execute ---
        # Patch the tools directly in the module where they are looked up.
        with patch('researcher.assistor.google_search', mock_google_search), \
             patch('researcher.assistor.view_text_website', mock_view_website):
            suggestion = asyncio.run(self.assistor.research_and_suggest(mock_task, error_context))

        # --- Assertions ---
        # Assert that the external tools were called correctly
        mock_google_search.assert_called_once()
        self.assertIn("site connection failed", mock_google_search.call_args.kwargs['query'])
        mock_view_website.assert_called_once_with(url="http://stackoverflow.com/q/123")

        # Assert that the LLM was called with the content from the website
        mock_llm_query.assert_called_once()
        prompt_arg = mock_llm_query.call_args.args[0]
        self.assertIn(mock_website_content, prompt_arg)

        # Assert that the final suggestion is the one from the LLM
        self.assertEqual(suggestion, mock_llm_suggestion)


if __name__ == '__main__':
    unittest.main()
