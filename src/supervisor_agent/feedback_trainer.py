import json
import os
from typing import Dict, List, Any

# A small learning rate is crucial to prevent chaotic oscillations in weights.
DEFAULT_LEARNING_RATE = 0.01

class FeedbackTrainer:
    """
    Analyzes feedback on supervisor decisions and suggests adjustments
    to the heuristic weights used by the Expectimax agent.
    """

    def __init__(self, feedback_file: str, current_weights: Dict[str, float], learning_rate: float = DEFAULT_LEARNING_RATE):
        """
        Initializes the trainer.

        Args:
            feedback_file: Path to the JSON file where feedback is stored.
            current_weights: The current set of heuristic weights.
            learning_rate: The step size for weight adjustments.
        """
        if not os.path.exists(feedback_file):
            # If the file doesn't exist, create it with an empty list.
            with open(feedback_file, 'w') as f:
                json.dump([], f)

        self.feedback_file = feedback_file
        self.current_weights = current_weights.copy() # Work on a copy
        self.learning_rate = learning_rate
        self.heuristic_keys = sorted(current_weights.keys())

    def load_feedback(self) -> List[Dict[str, Any]]:
        """Loads feedback data from the specified JSON file."""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            if not isinstance(feedback_data, list):
                # Handle case where file is valid JSON but not a list
                return []
            return feedback_data
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is empty, corrupt, or missing, return empty list.
            return []

    def train_on_feedback(self) -> Dict[str, float]:
        """
        Processes all available feedback and updates the weights.

        This uses a simple perceptron-like learning rule. For each piece of
        feedback, it nudges the weights in a direction that would have made
        the user's 'correct' choice more likely and the agent's 'incorrect'
        choice less likely.

        Returns:
            A dictionary containing the new, suggested weights.
        """
        feedback_data = self.load_feedback()
        if not feedback_data:
            print("No feedback data found to train on.")
            return self.current_weights

        updated_weights = self.current_weights.copy()

        for feedback_item in feedback_data:
            try:
                incorrect_action = feedback_item['original_decision']
                correct_action = feedback_item['corrected_action']
                context = feedback_item['decision_context']

                considered_actions = context.get('considered_actions', {})

                # Ensure both the incorrect and correct actions were actually considered
                if incorrect_action not in considered_actions or correct_action not in considered_actions:
                    continue

                heuristics_incorrect = considered_actions[incorrect_action]['heuristics']
                heuristics_correct = considered_actions[correct_action]['heuristics']

                # Align heuristic values into a consistent vector format
                h_vec_incorrect = [heuristics_incorrect.get(key, 0.0) for key in self.heuristic_keys]
                h_vec_correct = [heuristics_correct.get(key, 0.0) for key in self.heuristic_keys]

                # Apply the learning rule: W_new = W_old + lr * (H_correct - H_incorrect)
                for i, key in enumerate(self.heuristic_keys):
                    adjustment = self.learning_rate * (h_vec_correct[i] - h_vec_incorrect[i])
                    updated_weights[key] += adjustment

            except KeyError as e:
                print(f"Skipping malformed feedback item: missing key {e}")
                continue

        # Normalize weights to sum to 1 to prevent them from growing indefinitely
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            for key in updated_weights:
                updated_weights[key] /= total_weight

        # Clamp weights to be non-negative
        for key in updated_weights:
            updated_weights[key] = max(0.0, updated_weights[key])


        print(f"Training complete. Original weights: {self.current_weights}")
        print(f"New suggested weights: {updated_weights}")

        return updated_weights
