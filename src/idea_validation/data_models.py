from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Idea:
    """Represents a project idea to be validated."""
    description: str
    required_skills: List[str] = field(default_factory=list)
    required_apis: List[str] = field(default_factory=list)
    estimated_time_hours: int = 0
    market_niche: str = ""

@dataclass
class ValidationFinding:
    """Represents a single finding (positive or negative) from the validation process."""
    category: str  # e.g., "Technical Feasibility", "Market Viability"
    risk_level: str  # e.g., "Low", "Medium", "High", "Critical"
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationReport:
    """Represents the full validation report for an idea."""
    idea: Idea
    overall_score: float  # A score from 0.0 to 1.0
    summary: str
    findings: List[ValidationFinding] = field(default_factory=list)
