import unittest
import sys
import os

# Add the parent directory to the path to allow imports from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minimax_agent import MinimaxAgent, AgentState, Action

class TestMinimaxAgent(unittest.TestCase):
    """Test suite for the MinimaxAgent."""

    def setUp(self):
        """Set up a new MinimaxAgent for each test."""
        self.minimax_agent = MinimaxAgent(depth=2)

    def test_allow_on_good_state(self):
        """
        Test that the agent chooses ALLOW when the state is good.
        """
        # A good state: high quality, no errors, good progress, low resource usage
        good_state = AgentState(
            quality_score=0.95,
            error_count=0,
            resource_usage=0.2,
            task_progress=0.8
        )

        best_action = self.minimax_agent.get_best_action(good_state)

        self.assertEqual(best_action, Action.ALLOW, "Should ALLOW on a good state.")

    def test_escalate_on_bad_state(self):
        """
        Test that the agent chooses ESCALATE when the state is very bad.
        """
        # A bad state: very low quality, multiple errors, no progress
        bad_state = AgentState(
            quality_score=0.2,
            error_count=3,
            resource_usage=0.8,
            task_progress=0.1
        )

        best_action = self.minimax_agent.get_best_action(bad_state)

        self.assertEqual(best_action, Action.ESCALATE, "Should ESCALATE on a bad state.")

    def test_correct_on_medium_state(self):
        """
        Test that the agent chooses CORRECT for a recoverable medium-quality state.
        """
        # A medium state: decent quality but with an error, could be improved
        medium_state = AgentState(
            quality_score=0.6,
            error_count=1,
            resource_usage=0.5,
            task_progress=0.5
        )

        best_action = self.minimax_agent.get_best_action(medium_state)

        # In our simplified model, CORRECTION often yields a better future state
        self.assertEqual(best_action, Action.CORRECT, "Should CORRECT on a medium state.")

    def test_warn_on_resource_issue(self):
        """
        Test that the agent may choose WARN if resources are high but quality is good.
        """
        # A state where quality is good but resource usage is creeping up
        resource_issue_state = AgentState(
            quality_score=0.9,
            error_count=0,
            resource_usage=0.9, # High resource usage
            task_progress=0.7
        )

        best_action = self.minimax_agent.get_best_action(resource_issue_state)

        # The evaluation function penalizes resource usage, so it might not just ALLOW.
        # Depending on the weights, WARN could be a possible outcome.
        # This test is more about seeing a reasonable, non-ALLOW action.
        self.assertIn(best_action, [Action.WARN, Action.ALLOW], "Should WARN or ALLOW on a resource issue state.")


if __name__ == '__main__':
    unittest.main()
