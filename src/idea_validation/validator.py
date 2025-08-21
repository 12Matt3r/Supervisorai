from .data_models import Idea, ValidationFinding, ValidationReport

class Validator:
    """
    Validates a project idea against a set of heuristics.
    """

    def __init__(self):
        # These would typically be loaded from a config file or a more sophisticated system.
        self.known_saturated_markets = ["social media", "photo sharing", "food delivery"]
        self.high_complexity_keywords = ["real-time multiplayer", "blockchain", "new programming language"]
        self.available_skills = ["python", "javascript", "web development", "api integration"]

    def validate(self, idea: Idea) -> ValidationReport:
        """
        Performs a full validation of a project idea.
        """
        findings = []

        findings.extend(self._check_market_viability(idea))
        findings.extend(self._check_technical_feasibility(idea))
        findings.extend(self._check_resource_mismatch(idea))

        overall_score = self._calculate_overall_score(findings)
        summary = self._generate_summary(overall_score)

        return ValidationReport(
            idea=idea,
            overall_score=overall_score,
            summary=summary,
            findings=findings
        )

    def _check_market_viability(self, idea: Idea) -> list[ValidationFinding]:
        """Checks for potential market-related issues."""
        findings = []
        for market in self.known_saturated_markets:
            if market in idea.market_niche.lower() or market in idea.description.lower():
                findings.append(ValidationFinding(
                    category="Market Viability",
                    risk_level="High",
                    message=f"The market for '{market}' is highly saturated.",
                    suggestion="Consider focusing on a very specific niche within this market to stand out."
                ))
        return findings

    def _check_technical_feasibility(self, idea: Idea) -> list[ValidationFinding]:
        """Checks for technical red flags."""
        findings = []
        for keyword in self.high_complexity_keywords:
            if keyword in idea.description.lower():
                findings.append(ValidationFinding(
                    category="Technical Feasibility",
                    risk_level="High",
                    message=f"The project involves '{keyword}', which is known to have high technical complexity.",
                    suggestion="Ensure you have a detailed technical plan and the necessary expertise before proceeding."
                ))
        return findings

    def _check_resource_mismatch(self, idea: Idea) -> list[ValidationFinding]:
        """Checks if required skills and time are realistic."""
        findings = []
        missing_skills = [skill for skill in idea.required_skills if skill.lower() not in self.available_skills]
        if missing_skills:
            findings.append(ValidationFinding(
                category="Resource Mismatch",
                risk_level="Medium",
                message=f"The project requires skills that are not listed as available: {', '.join(missing_skills)}.",
                suggestion="Plan for learning these skills or collaborating with someone who has them."
            ))

        if idea.estimated_time_hours > 100: # Arbitrary threshold for a hackathon-like project
            findings.append(ValidationFinding(
                category="Resource Mismatch",
                risk_level="Medium",
                message=f"The estimated time of {idea.estimated_time_hours} hours may be too high for a short-term project.",
                suggestion="Consider scoping the project down to a smaller MVP (Minimum Viable Product)."
            ))
        return findings

    def _calculate_overall_score(self, findings: list[ValidationFinding]) -> float:
        """Calculates a score from 0.0 to 1.0 based on the findings."""
        if not findings:
            return 1.0

        score = 1.0
        for finding in findings:
            if finding.risk_level == "High":
                score -= 0.3
            elif finding.risk_level == "Medium":
                score -= 0.15

        return max(0.0, score)

    def _generate_summary(self, score: float) -> str:
        """Generates a summary text based on the score."""
        if score >= 0.8:
            return "This idea seems highly viable with low risk."
        elif score >= 0.5:
            return "This idea is promising but has some moderate risks to consider."
        else:
            return "This idea has significant risks that should be addressed before proceeding."
