import unittest
import json
import os
import shutil
from typing import Dict, Any

from src.supervisor_agent.feedback_trainer import FeedbackTrainer

class TestFeedbackTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and sample data for tests."""
        self.test_dir = "temp_test_feedback_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.feedback_file = os.path.join(self.test_dir, "feedback.json")

        self.initial_weights = {
            "quality_score": 0.4,
            "inv_drift_score": 0.3,
            "inv_error_count": 0.2,
            "inv_resource_usage": 0.1,
        }

        # Sample feedback: The agent chose ALLOW, but the user corrected to ESCALATE.
        # This implies the score for ESCALATE should have been higher.
        # Heuristics for ESCALATE are generally higher (e.g., high error count -> low inv_error_count).
        # Let's craft the context to reflect a clear learning opportunity.
        self.sample_feedback = [
            {
                "event_id": "evt-123",
                "original_decision": "ALLOW",
                "corrected_action": "ESCALATE",
                "decision_context": {
                    "considered_actions": {
                        "ALLOW": {
                            "heuristics": {"quality_score": 0.9, "inv_drift_score": 0.8, "inv_error_count": 1.0, "inv_resource_usage": 0.9}
                        },
                        "WARN": {
                            "heuristics": {"quality_score": 0.8, "inv_drift_score": 0.7, "inv_error_count": 1.0, "inv_resource_usage": 0.8}
                        },
                        "CORRECT": {
                            "heuristics": {"quality_score": 0.6, "inv_drift_score": 0.5, "inv_error_count": 0.5, "inv_resource_usage": 0.6}
                        },
                        "ESCALATE": {
                            "heuristics": {"quality_score": 0.2, "inv_drift_score": 0.3, "inv_error_count": 0.1, "inv_resource_usage": 0.2}
                        }
                    }
                },
                "timestamp": "2023-01-01T12:00:00Z"
            }
        ]
        # In our transformed features, "higher is better".
        # The agent incorrectly chose ALLOW (all features high).
        # The user corrected to ESCALATE (all features low).
        # The trainer should therefore DECREASE the weights for all features.
        # Let's make a better example. Say the agent chose WARN but should have chosen CORRECT.
        self.sample_feedback_2 = [
             {
                "event_id": "evt-456",
                "original_decision": "WARN",
                "corrected_action": "CORRECT",
                "decision_context": {
                    "considered_actions": {
                        "WARN": { # High drift, so should be penalized
                            "heuristics": {"quality_score": 0.8, "inv_drift_score": 0.2, "inv_error_count": 1.0, "inv_resource_usage": 0.9}
                        },
                        "CORRECT": { # Low drift, should be rewarded
                            "heuristics": {"quality_score": 0.7, "inv_drift_score": 0.9, "inv_error_count": 1.0, "inv_resource_usage": 0.8}
                        }
                    }
                }
            }
        ]


    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_training_updates_weights_correctly(self):
        """Test that training adjusts weights in the expected direction."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.sample_feedback_2, f)

        trainer = FeedbackTrainer(self.feedback_file, self.initial_weights, learning_rate=0.1)
        new_weights = trainer.train_on_feedback()

        # The "corrected" action had much higher inv_drift_score.
        # The "incorrect" action had a much lower inv_drift_score.
        # Therefore, the weight for inv_drift_score should increase.
        # H_correct - H_incorrect for drift is (0.9 - 0.2) = 0.7 (positive)
        self.assertGreater(new_weights["inv_drift_score"], self.initial_weights["inv_drift_score"])

        # The "corrected" action had lower quality_score.
        # H_correct - H_incorrect for quality is (0.7 - 0.8) = -0.1 (negative)
        # Therefore, the weight for quality_score should decrease.
        self.assertLess(new_weights["quality_score"], self.initial_weights["quality_score"])

        # The other weights should also decrease slightly or stay similar after normalization.
        self.assertLess(new_weights["inv_resource_usage"], self.initial_weights["inv_resource_usage"])
        # The inv_error_count was the same for both actions, so its change should be minimal
        # and only due to normalization. We can assert it changed less than other weights.
        error_count_change = abs(new_weights["inv_error_count"] - self.initial_weights["inv_error_count"])
        drift_score_change = abs(new_weights["inv_drift_score"] - self.initial_weights["inv_drift_score"])
        self.assertLess(error_count_change, drift_score_change)


    def test_no_feedback_file(self):
        """Test that training with no feedback file returns original weights."""
        non_existent_file = os.path.join(self.test_dir, "non_existent.json")
        trainer = FeedbackTrainer(non_existent_file, self.initial_weights)
        new_weights = trainer.train_on_feedback()

        self.assertDictEqual(self.initial_weights, new_weights)

    def test_empty_feedback_data(self):
        """Test that training with an empty feedback list returns original weights."""
        with open(self.feedback_file, 'w') as f:
            json.dump([], f)

        trainer = FeedbackTrainer(self.feedback_file, self.initial_weights)
        new_weights = trainer.train_on_feedback()

        self.assertDictEqual(self.initial_weights, new_weights)

    def test_malformed_feedback_item(self):
        """Test that malformed feedback items are skipped without crashing."""
        malformed_feedback = [{"event_id": "bad-1"}] # Missing keys
        with open(self.feedback_file, 'w') as f:
            json.dump(malformed_feedback, f)

        trainer = FeedbackTrainer(self.feedback_file, self.initial_weights)
        new_weights = trainer.train_on_feedback()

        # No valid feedback was processed, so weights should be unchanged.
        self.assertDictEqual(self.initial_weights, new_weights)

    def test_weights_are_normalized_and_non_negative(self):
        """Test that weights are clamped and normalized after training."""
        # Use a high learning rate to force large changes
        with open(self.feedback_file, 'w') as f:
            json.dump(self.sample_feedback_2, f)

        trainer = FeedbackTrainer(self.feedback_file, self.initial_weights, learning_rate=1.0)
        new_weights = trainer.train_on_feedback()

        # Check for non-negativity
        for weight_value in new_weights.values():
            self.assertGreaterEqual(weight_value, 0.0)

        # Check for normalization
        self.assertAlmostEqual(sum(new_weights.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
