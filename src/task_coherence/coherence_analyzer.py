import re
from typing import List, Dict

class CoherenceAnalyzer:
    """
    Analyzes the coherence of an agent's output against its original goals.
    """

    def __init__(self):
        # Patterns that indicate a potential deviation or topic shift.
        self.drift_patterns = [
            r"(?i)(instead|however|but let's focus on|alternatively|a different approach)",
            r"(?i)(let's try something else|let's switch gears|on another note)",
            r"(?i)(i can't do that, but i can|i'm unable to fulfill that request, so instead)",
        ]
        self.common_distractions = ["what's the weather", "tell me a joke", "who are you"]

    def analyze(self, output: str, original_goals: List[str]) -> Dict:
        """
        Analyzes the output for task coherence and drift.

        Returns:
            A dictionary containing a drift score (0.0 to 1.0) and a list of findings.
        """
        findings = []
        drift_score = 0.0

        # 1. Check for explicit drift patterns
        for pattern in self.drift_patterns:
            if re.search(pattern, output):
                drift_score += 0.6 # Increased weight for explicit drift
                findings.append(f"Detected potential topic shift with pattern: '{pattern}'")

        # 2. Check for common distractions
        for distraction in self.common_distractions:
            if distraction in output.lower():
                drift_score += 0.3
                findings.append(f"Detected common distraction: '{distraction}'")

        # 3. Check for keyword alignment with original goals
        if original_goals:
            goal_keywords = set()
            for goal in original_goals:
                goal_keywords.update(self._extract_keywords(goal))

            if goal_keywords:
                output_keywords = self._extract_keywords(output)

                if not output_keywords:
                    # If output has no keywords, it's likely not relevant
                    alignment_score = 0.0
                else:
                    matches = goal_keywords.intersection(output_keywords)
                    alignment_score = len(matches) / len(goal_keywords)

                if alignment_score < 0.2: # Very low alignment
                    drift_score += 0.5
                    findings.append(f"Output has very low keyword alignment with original goals. (Alignment: {alignment_score:.2f})")
                elif alignment_score < 0.5: # Moderate alignment
                    drift_score += 0.2
                    findings.append(f"Output has moderate keyword alignment with original goals. (Alignment: {alignment_score:.2f})")

        # Normalize score to be between 0.0 and 1.0
        final_drift_score = min(1.0, drift_score)

        return {
            "drift_score": final_drift_score,
            "is_coherent": final_drift_score < 0.5,
            "findings": findings
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """A simple keyword extractor."""
        stop_words = {'the', 'a', 'an', 'is', 'in', 'on', 'of', 'for', 'to', 'and', 'but'}
        words = re.findall(r'\b\w{3,}\b', text.lower())
        return [word for word in words if word not in stop_words]
