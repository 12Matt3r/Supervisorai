from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

class InterventionLevel(Enum):
    WARNING = "warning"
    CORRECTION = "correction"
    ESCALATION = "escalation"
    ASSISTANCE = "assistance"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class InterventionRequired:
    is_required: bool
    reason: str
    level: InterventionLevel

@dataclass
class ResourceUsage:
    """Resource tracking"""
    token_count: int = 0
    cpu_time: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class QualityMetrics:
    """Quality assessment"""
    confidence_score: float = 0.0
    coherence_score: float = 0.0
    structure_score: float = 0.0
    instruction_adherence: float = 0.0

@dataclass
class MonitoringRules:
    """Configuration rules"""
    quality_threshold: float = 0.7
    escalation_threshold: float = 0.4
    max_token_threshold: int = 10000
    enable_learning: bool = True

@dataclass
class AgentTask:
    """Represents a monitored agent task"""
    task_id: str
    agent_name: str
    framework: str
    original_input: str
    instructions: List[str]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    last_known_good_state: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0

@dataclass
class SupervisionReport:
    """Comprehensive reports"""
    report_id: str
    generated_at: datetime
    tasks_monitored: int
    total_interventions: int
    interventions_by_level: Dict[InterventionLevel, int]
    quality_trends: Dict[str, float]
    common_failures: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_distribution: Dict[ConfidenceLevel, int]

@dataclass
class EscalationConfig:
    """Escalation settings"""
    auto_pause_on_escalation: bool = True
    escalation_channel: str = "default"

@dataclass
class KnowledgeBaseEntry:
    """Knowledge base entries"""
    pattern_id: str
    pattern_description: str
    failure_type: str
    common_causes: List[str]
    suggested_fixes: List[str]
    confidence_score: float
    occurrences: int
    last_seen: datetime

__all__ = [
    "AgentTask",
    "TaskStatus",
    "InterventionLevel",
    "MonitoringRules",
    "QualityMetrics",
    "ResourceUsage",
    "SupervisionReport",
    "EscalationConfig",
    "KnowledgeBaseEntry",
    "ConfidenceLevel",
    "InterventionRequired",
]
