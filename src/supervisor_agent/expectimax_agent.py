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

class ExpectimaxAgent:
    """A supervisor agent that uses the Expectimax algorithm to choose interventions."""

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
        if state.quality_score >= 0.9:
            # If quality is very high, only allow or warn. No need to correct.
            actions.extend([Action.ALLOW, Action.WARN])
        elif state.quality_score < 0.4:
            # If quality is very low, allowing it is not an option.
            actions.extend([Action.CORRECT, Action.ESCALATE])
        else:
            # In the middle range, all actions are on the table.
            actions.extend([Action.ALLOW, Action.WARN, Action.CORRECT, Action.ESCALATE])

        return list(set(actions)) # Return unique actions

    def _get_action_outcomes(self, state: AgentState, action: Action) -> List[tuple[float, AgentState]]:
        """
        Returns a list of possible outcomes for a given action, each with a probability.
        Returns: A list of (probability, next_state) tuples.
        """
        outcomes = []
        base_state = AgentState(
            quality_score=state.quality_score,
            error_count=state.error_count,
            resource_usage=state.resource_usage,
            task_progress=state.task_progress
        )
        # All actions increase resource usage slightly
        base_state.resource_usage = min(1.0, base_state.resource_usage + 0.05)

        if action == Action.ALLOW:
            # 80% chance quality degrades slightly, 20% chance it stays the same
            state_degrade = AgentState(**base_state.__dict__)
            state_degrade.quality_score *= 0.95
            state_degrade.task_progress += 0.1
            outcomes.append((0.8, state_degrade))

            state_same = AgentState(**base_state.__dict__)
            state_same.task_progress += 0.1
            outcomes.append((0.2, state_same))

        elif action == Action.WARN:
            # Warning has a small cost and slightly less progress
            state_warn = AgentState(**base_state.__dict__)
            state_warn.task_progress += 0.05
            state_warn.resource_usage = min(1.0, state_warn.resource_usage + 0.1)
            outcomes.append((1.0, state_warn))

        elif action == Action.CORRECT:
            # 70% chance of successful correction, 30% chance of failure
            state_success = AgentState(**base_state.__dict__)
            state_success.resource_usage = min(1.0, state_success.resource_usage + 0.15)
            if state.quality_score < 0.85:
                quality_improvement = 0.4 * (1 - state.quality_score)
                state_success.quality_score = min(1.0, state.quality_score + quality_improvement)
                state_success.error_count = max(0, state.error_count - 1)
            state_success.task_progress += 0.1
            outcomes.append((0.7, state_success))

            state_fail = AgentState(**base_state.__dict__)
            state_fail.resource_usage = min(1.0, state_fail.resource_usage + 0.15)
            # No improvement on failure, just cost
            state_fail.task_progress += 0.02
            outcomes.append((0.3, state_fail))

        elif action == Action.ESCALATE:
            # Escalation is a deterministic major setback
            state_escalate = AgentState(**base_state.__dict__)
            state_escalate.task_progress *= 0.2
            state_escalate.quality_score *= 0.1
            state_escalate.error_count += 1
            outcomes.append((1.0, state_escalate))

        return outcomes

    def expectimax(self, state: AgentState, depth: int, action: Action) -> float:
        """
        The core Expectimax algorithm.
        Calculates the expected value of taking a given action from a given state.
        """
        if depth == 0 or state.is_terminal():
            return self._evaluate_state(state)

        # Get the possible outcomes for the given action
        outcomes = self._get_action_outcomes(state, action)

        # Calculate the weighted average of the scores of the outcomes
        expected_value = 0
        for probability, next_state in outcomes:
            # For each outcome, find the value of the best action the agent can take next
            max_eval = -math.inf
            for next_action in self._get_possible_actions(next_state):
                evaluation = self.expectimax(next_state, depth - 1, next_action)
                max_eval = max(max_eval, evaluation)

            expected_value += probability * max_eval

        return expected_value


    def get_best_action(self, state: AgentState) -> Dict[str, Any]:
        """
        Finds the best action to take from the current state.
        Returns a dictionary with the best action, its score, and all considered actions.
        """
        best_score = -math.inf
        best_action = None
        considered_actions = []

        for action in self._get_possible_actions(state):
            score = self.expectimax(state, self.depth, action)
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
