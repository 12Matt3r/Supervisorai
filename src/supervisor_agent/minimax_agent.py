import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

class Action(Enum):
    """Enumeration of possible supervisor interventions."""
    ALLOW = "ALLOW"  # Allow the agent's output as is
    WARN = "WARN"    # Log a warning but allow the output
    CORRECT = "CORRECT" # Attempt to correct the output
    ESCALATE = "ESCALATE" # Escalate the issue for human review

@dataclass
class AgentState:
    """Represents the state of the agent being supervised at a point in time."""
    quality_score: float  # A score from 0.0 to 1.0 representing output quality
    error_count: int      # Number of errors in the recent history
    resource_usage: float # A normalized score for resource consumption (e.g., tokens)
    task_progress: float  # Estimated progress towards task completion (0.0 to 1.0)

    def is_terminal(self) -> bool:
        """Determines if the state is a terminal state (e.g., task complete or failed)."""
        return self.task_progress >= 1.0 or self.quality_score <= 0.1

class MinimaxAgent:
    """A supervisor agent that uses the Minimax algorithm to choose interventions."""

    def __init__(self, depth: int = 2):
        """
        Initializes the Minimax agent.
        Args:
            depth: The maximum depth for the Minimax search tree.
        """
        self.depth = depth

    def _evaluate_state(self, state: AgentState) -> float:
        """
        The evaluation function for a given state.
        A higher score is better.
        """
        if state.is_terminal():
            if state.task_progress >= 1.0:
                return 1000.0
            else:
                return -1000.0

        # Final tuned weights
        score = (
            (state.quality_score * 70) +
            (state.task_progress * 30) -
            (state.error_count * 200) -    # Extremely high penalty for errors
            (state.resource_usage * 40)
        )
        return score

    def _get_possible_actions(self, state: AgentState) -> List[Action]:
        """Returns a list of sensible actions based on the current state."""

        # If there are too many errors, we must escalate.
        if state.error_count >= 3:
            return [Action.ESCALATE]

        actions = []
        if state.quality_score > 0.9:
            # If quality is very high, only allow or warn. No need to correct.
            actions.extend([Action.ALLOW, Action.WARN])
        elif state.quality_score < 0.4:
            # If quality is very low, allowing it is not an option.
            actions.extend([Action.CORRECT, Action.ESCALATE])
        else:
            # In the middle range, all actions are on the table.
            actions.extend([Action.ALLOW, Action.WARN, Action.CORRECT, Action.ESCALATE])

        return list(set(actions)) # Return unique actions

    def _apply_action(self, state: AgentState, action: Action) -> AgentState:
        """
        Simulates the effect of an action on a state, returning a new state.
        This is a simplified model of the world.
        """
        new_state = AgentState(
            quality_score=state.quality_score,
            error_count=state.error_count,
            resource_usage=state.resource_usage,
            task_progress=state.task_progress
        )

        # Simulate state changes based on action
        if action == Action.ALLOW:
            new_state.task_progress += 0.1
            new_state.quality_score *= 0.95 # Quality degrades without intervention
        elif action == Action.WARN:
            new_state.task_progress += 0.05
            new_state.resource_usage = min(1.0, new_state.resource_usage + 0.1) # Warnings have a cost
        elif action == Action.CORRECT:
            # Correction has a moderate cost
            new_state.resource_usage = min(1.0, new_state.resource_usage + 0.15)
            if state.quality_score < 0.85: # Only try to correct if quality is not high
                quality_improvement = 0.4 * (1 - state.quality_score)
                new_state.quality_score = min(1.0, state.quality_score + quality_improvement)
                new_state.error_count = max(0, state.error_count - 1)
            new_state.task_progress += 0.1
        elif action == Action.ESCALATE:
            # Escalation is a major setback
            new_state.task_progress *= 0.2 # Drastic progress reduction
            new_state.quality_score *= 0.1 # Catastrophic quality drop
            new_state.error_count += 1

        # All actions increase resource usage slightly
        new_state.resource_usage = min(1.0, new_state.resource_usage + 0.05)

        return new_state

    def minimax(self, state: AgentState, depth: int, maximizing_player: bool) -> float:
        """
        The core Minimax algorithm.
        """
        if depth == 0 or state.is_terminal():
            return self._evaluate_state(state)

        if maximizing_player:
            max_eval = -math.inf
            for action in self._get_possible_actions(state):
                new_state = self._apply_action(state, action)
                evaluation = self.minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, evaluation)
            return max_eval
        else: # Minimizing player (simulating the "environment" or "problem")
            min_eval = math.inf
            next_natural_state = state
            next_natural_state.quality_score *= 0.95 # Simulate slight decay or difficulty

            evaluation = self.minimax(next_natural_state, depth - 1, True)
            min_eval = min(min_eval, evaluation)
            return min_eval


    def get_best_action(self, state: AgentState) -> Dict[str, Any]:
        """
        Finds the best action to take from the current state.
        Returns a dictionary with the best action, its score, and all considered actions.
        """
        best_score = -math.inf
        best_action = None
        considered_actions = []

        for action in self._get_possible_actions(state):
            new_state = self._apply_action(state, action)
            score = self.minimax(new_state, self.depth, False)
            considered_actions.append({"action": action.value, "score": score})

            if score > best_score:
                best_score = score
                best_action = action

        best_action = best_action or Action.ALLOW

        return {
            "best_action": best_action,
            "best_score": best_score,
            "considered_actions": sorted(considered_actions, key=lambda x: x['score'], reverse=True),
            "state_evaluated": state
        }
