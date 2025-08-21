import unittest
import asyncio
from unittest.mock import patch, MagicMock

import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from supervisor_agent.llm_judge import LLMJudge

class TestLLMJudge(unittest.TestCase):
    """Test suite for the LLMJudge class."""

    def setUp(self):
        """Set up a new LLMJudge for each test."""
        self.judge = LLMJudge(api_key="TEST_API_KEY")

    def test_prompt_creation(self):
        """Test that the prompt is created correctly."""
        output = "This is the agent's output."
        goals = ["Summarize the text.", "Be concise."]
        prompt = self.judge._create_prompt(output, goals)

        self.assertIn("impartial ai supervisor", prompt.lower().strip())
        self.assertIn("Original Goals:", prompt)
        self.assertIn("- Summarize the text.", prompt)
        self.assertIn("- Be concise.", prompt)
        self.assertIn("Agent's Output:", prompt)
        self.assertIn(output, prompt)
        self.assertIn('"overall_score"', prompt)
        self.assertIn('"reasoning"', prompt)

    @patch('httpx.AsyncClient.post', new_callable=unittest.mock.AsyncMock)
    async def test_evaluate_output_with_mock_api(self, mock_post):
        """Test the evaluate_output method with a mocked API call."""

        # Mock the successful API response
        mock_response_content = {
            "content": [
                {
                    "type": "text",
                    "text": '{"overall_score": 0.9, "reasoning": "The output is concise and relevant.", "is_safe": true}'
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_content
        mock_post.return_value = mock_response

        # Run the evaluation
        output = "This is a concise summary."
        goals = ["Summarize the text.", "Be concise."]

        result = await self.judge.evaluate_output(output, goals)

        # Assertions
        self.assertEqual(result["overall_score"], 0.9)
        self.assertEqual(result["reasoning"], "The output is concise and relevant.")
        self.assertTrue(result["is_safe"])
        mock_post.assert_called_once()

    def test_placeholder_response_no_api_key(self):
        """Test that a placeholder response is returned when no API key is provided."""
        judge_no_key = LLMJudge(api_key=None)
        output = "test"
        goals = ["test"]

        result = asyncio.run(judge_no_key.evaluate_output(output, goals))

        self.assertEqual(result["overall_score"], 0.85)
        self.assertIn("placeholder response", result["reasoning"])

if __name__ == '__main__':
    unittest.main()
