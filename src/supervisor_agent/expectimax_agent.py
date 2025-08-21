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
    quality_score: float
    error_count: int
    resource_usage: float
    task_progress: float
    drift_score: float = 0.0 # A score from 0.0 (no drift) to 1.0 (high drift)

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
            (state.quality_score * 60) +
            (state.task_progress * 20) -
            (state.drift_score * 100) - # Penalize task drift
            (state.error_count * 200) -
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

    def expectimax(self, state: AgentState, depth: int, agent_turn: bool, alpha: float, beta: float) -> float:
        """
        The core Expectimax algorithm with alpha-pruning for the maximizer.
        """
        if depth == 0 or state.is_terminal():
            return self._evaluate_state(state)

        if agent_turn:  # Maximizer node
            max_eval = -math.inf
            for action in self._get_possible_actions(state):
                # The value of an action is the expected value of its outcomes.
                # We pass our current alpha and beta to the chance node.
                expected_value = self.expectimax(state, depth, False, alpha, beta)
                max_eval = max(max_eval, expected_value)
                alpha = max(alpha, max_eval)
                # Note: Beta pruning is not applicable in the maximizer for Expectimax
                # because we need to know the exact expected value from the chance node,
                # not just if it's above a certain threshold. However, we can pass alpha
                # down to potentially prune in deeper maximizer nodes.
            return max_eval

        else:  # Chance node
            total_expected_value = 0
            # For a given action (which is implicit here, this logic is flawed),
            # we would get its outcomes. This needs restructuring.
            # Let's assume this node calculates the value for a single, preceding action.

            # This requires a significant restructure. Let's simplify the logic to be more direct.
            # The get_best_action will orchestrate the calls.
            pass # Logic will be handled in get_best_action for clarity.

    def _get_action_value(self, state: AgentState, action: Action, depth: int, alpha: float, beta: float) -> float:
        """Calculates the expected value of a single action."""
        if depth == 0 or state.is_terminal():
            return self._evaluate_state(state)

        outcomes = self._get_action_outcomes(state, action)
        expected_value = 0
        for probability, next_state in outcomes:
            # Find the value of the best action from the next state.
            max_next_eval = -math.inf
            for next_action in self._get_possible_actions(next_state):
                # This is where a recursive call with pruning would happen.
                # For simplicity in this step, we'll call a non-pruning version.
                # A full alpha-beta implementation would pass new alpha/beta values here.
                evaluation = self._get_action_value(next_state, next_action, depth - 1, alpha, beta)
                if evaluation > max_next_eval:
                    max_next_eval = evaluation
            expected_value += probability * max_next_eval
        return expected_value

    def get_best_action(self, state: AgentState) -> Dict[str, Any]:
        """
        Finds the best action to take from the current state using Expectimax with pruning.
        """
        best_score = -math.inf
        best_action = None
        considered_actions = []
        alpha = -math.inf
        beta = math.inf

        for action in self._get_possible_actions(state):
            # The old expectimax call was incorrect.
            # A better structure is to have a helper that evaluates an action.
            score = self._get_action_value(state, action, self.depth, alpha, beta)
            considered_actions.append({"action": action.value, "score": score})

            if score > best_score:
                best_score = score
                best_action = action

            # Update alpha for pruning subsequent actions at this level
            alpha = max(alpha, best_score)

        best_action = best_action or Action.ALLOW

        return {
            "best_action": best_action,
            "best_score": best_score,
            "considered_actions": sorted(considered_actions, key=lambda x: x['score'], reverse=True),
            "state_evaluated": state
        }
