import unittest
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from supervisor_agent.expectimax_agent import ExpectimaxAgent, AgentState, Action

class TestExpectimaxAgent(unittest.TestCase):
    """Test suite for the ExpectimaxAgent."""

    def setUp(self):
        """Set up a new ExpectimaxAgent for each test."""
        self.agent = ExpectimaxAgent(depth=2)

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

        decision = self.agent.get_best_action(good_state)
        best_action = decision["best_action"]

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

        decision = self.agent.get_best_action(bad_state)
        best_action = decision["best_action"]

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

        decision = self.agent.get_best_action(medium_state)
        best_action = decision["best_action"]

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

        decision = self.agent.get_best_action(resource_issue_state)
        best_action = decision["best_action"]

        self.assertIn(best_action, [Action.WARN, Action.ALLOW], "Should WARN or ALLOW on a resource issue state.")

    def test_get_best_action_with_trace_structure(self):
        """
        Test that get_best_action_with_trace returns the expected data structure.
        """
        state = AgentState(
            quality_score=0.7,
            error_count=1,
            resource_usage=0.4,
            task_progress=0.3,
            drift_score=0.2
        )

        result = self.agent.get_best_action_with_trace(state)

        # Check top-level keys
        self.assertIn("best_action", result)
        self.assertIn("best_score", result)
        self.assertIn("trace", result)
        self.assertIsInstance(result['best_action'], Action)
        self.assertIsInstance(result['best_score'], float)

        # Check root of the trace
        trace = result['trace']
        self.assertIn("name", trace)
        self.assertIn("state", trace)
        self.assertIn("evaluation", trace)
        self.assertIn("children", trace)
        self.assertIsInstance(trace['children'], list)

        # Check a first-level child (an action node)
        if len(trace['children']) > 0:
            action_node = trace['children'][0]
            self.assertIn("name", action_node)
            self.assertIn("action", action_node)
            self.assertIn("score", action_node)
            self.assertIn("children", action_node)
            self.assertIsInstance(action_node['children'], list)

            # Check a second-level child (an outcome node)
            if len(action_node['children']) > 0:
                outcome_node = action_node['children'][0]
                self.assertIn("name", outcome_node)
                self.assertIn("probability", outcome_node)
                self.assertIn("state", outcome_node)
                self.assertIn("children", outcome_node)
                self.assertIsInstance(outcome_node['children'], list)


if __name__ == '__main__':
    unittest.main()
