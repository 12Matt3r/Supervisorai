from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import uuid4
import json
import asyncio
from pathlib import Path

from . import (
    AgentTask,
    TaskStatus,
    InterventionLevel,
    MonitoringRules,
    QualityMetrics,
    ResourceUsage,
    SupervisionReport,
    EscalationConfig,
    KnowledgeBaseEntry,
    ConfidenceLevel,
    InterventionRequired
)
from monitoring.quality_analyzer import QualityAnalyzer
from monitoring.pattern_learner import PatternLearner
from reporting.audit_logger import AuditLogger
from reporting.audit_system import AuditEventType, AuditLevel
from .expectimax_agent import ExpectimaxAgent, AgentState, Action


class SupervisorCore:
    """Core supervisor engine for agent monitoring and intervention"""

    def __init__(self, data_dir: str = "./supervisor_data", audit_system: Optional[Any] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core components
        self.quality_analyzer = QualityAnalyzer()
        self.pattern_learner = PatternLearner(str(self.data_dir / "patterns.json"))
        self.audit_logger = AuditLogger(str(self.data_dir / "audit.jsonl")) # Legacy logger
        self.expectimax_agent = ExpectimaxAgent(depth=2)
        self.audit_system = audit_system # New, more comprehensive audit system
        
        # State management
        self.active_tasks: Dict[str, AgentTask] = {}
        self.monitoring_rules = MonitoringRules()
        self.escalation_config = EscalationConfig()
        self.knowledge_base: Dict[str, KnowledgeBaseEntry] = {}
        
        # Load persisted data
        asyncio.create_task(self._load_knowledge_base())

    async def monitor_agent(
        self,
        agent_name: str,
        framework: str,
        task_input: str,
        instructions: List[str],
        task_id: Optional[str] = None
    ) -> str:
        """Start monitoring an agent task"""
        if task_id is None:
            task_id = str(uuid4())
        
        task = AgentTask(
            task_id=task_id,
            agent_name=agent_name,
            framework=framework,
            original_input=task_input,
            instructions=instructions,
            status=TaskStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        
        # Log monitoring start
        await self.audit_logger.log_event(
            task_id=task_id,
            event_type="monitor_start",
            details={
                "agent_name": agent_name,
                "framework": framework,
                "input_length": len(task_input),
                "instruction_count": len(instructions)
            }
        )
        
        return task_id

    async def validate_output(
        self,
        task_id: str,
        output: str,
        output_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate agent output and apply interventions if needed"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        
        # Update resource usage
        task.resource_usage.token_count += len(output.split())
        task.updated_at = datetime.now()
        
        # Analyze output quality
        quality_metrics = await self.quality_analyzer.analyze(
            output=output,
            output_type=output_type,
            instructions=task.instructions,
            original_input=task.original_input
        )
        
        task.quality_metrics = quality_metrics
        
        # Check for intervention requirements
        intervention_result = await self._check_intervention_needed(
            task, output, quality_metrics
        )
        
        # Store output and intervention result
        output_record = {
            "timestamp": datetime.now().isoformat(),
            "output": output,
            "output_type": output_type,
            "quality_metrics": quality_metrics.__dict__,
            "intervention_result": intervention_result,
            "metadata": metadata or {}
        }
        
        task.outputs.append(output_record)
        
        # Log the enhanced decision data to the audit system
        if self.audit_system:
            self.audit_system.log(
                event_type=AuditEventType.DECISION_MADE,
                level=AuditLevel.INFO,
                source="ExpectimaxSupervisor",
                message=f"Intervention decision: {intervention_result.get('action', 'NONE')}",
                metadata={
                    "task_id": task_id,
                    "agent_name": task.agent_name,
                    "decision_details": intervention_result
                }
            )

        # Apply intervention if needed
        if intervention_result["intervention_required"]:
            await self._apply_intervention(task, intervention_result)
        
        # Log validation event
        await self.audit_logger.log_event(
            task_id=task_id,
            event_type="output_validation",
            details={
                "quality_score": quality_metrics.confidence_score,
                "intervention_level": intervention_result.get("level"),
                "output_length": len(output)
            }
        )
        
        return {
            "task_id": task_id,
            "quality_metrics": quality_metrics.__dict__,
            "intervention_result": intervention_result,
            "recommendations": await self._generate_recommendations(task)
        }

    async def _check_intervention_needed(
        self,
        task: AgentTask,
        output: str,
        quality_metrics: QualityMetrics
    ) -> Dict[str, Any]:
        """Determine if intervention is needed using the Minimax agent."""
        
        max_tokens = getattr(self.monitoring_rules, 'max_token_threshold', 10000)
        if max_tokens == 0: max_tokens = 10000

        current_state = AgentState(
            quality_score=quality_metrics.confidence_score,
            error_count=len([i for i in task.interventions if i["level"] == "ESCALATION"]),
            resource_usage=task.resource_usage.token_count / max_tokens,
            task_progress=len(task.outputs) / 10.0
        )

        decision_data = self.expectimax_agent.get_best_action(current_state)
        best_action = decision_data["best_action"]

        intervention_required = True
        reason = f"Minimax agent chose {best_action.value} with score {decision_data['best_score']:.2f}"

        level_map = {
            Action.ALLOW: None,
            Action.WARN: InterventionLevel.WARNING,
            Action.CORRECTION: InterventionLevel.CORRECTION,
            Action.ESCALATE: InterventionLevel.ESCALATION,
        }
        level = level_map[best_action]

        if level is None:
            intervention_required = False

        # Return the enhanced decision data
        return {
            "intervention_required": intervention_required,
            "level": level.value if level else None,
            "reason": reason,
            "confidence": decision_data['best_score'],
            "action": best_action.value,
            "minimax_details": {
                "considered_actions": decision_data['considered_actions'],
                "state_evaluated": decision_data['state_evaluated'].__dict__
            }
        }

    async def _apply_intervention(
        self,
        task: AgentTask,
        intervention_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply the appropriate intervention"""
        level = InterventionLevel(intervention_result["level"])
        
        intervention_record = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "reason": intervention_result["reason"],
            "confidence": intervention_result["confidence"],
            "action_taken": None
        }
        
        if level == InterventionLevel.WARNING:
            # Log warning, no direct action
            intervention_record["action_taken"] = "logged_warning"
        
        elif level == InterventionLevel.CORRECTION:
            # Attempt automatic correction
            if task.quality_metrics.structure_score < 0.6:
                # Try to fix format issues
                intervention_record["action_taken"] = "format_correction_attempted"
                
            # Save current state before correction
            task.last_known_good_state = {
                "timestamp": datetime.now().isoformat(),
                "outputs": task.outputs[-5:],  # Last 5 outputs
                "quality_metrics": task.quality_metrics.__dict__
            }
        
        elif level == InterventionLevel.ESCALATION:
            # Pause task and escalate
            task.status = TaskStatus.ESCALATED
            intervention_record["action_taken"] = "task_escalated"
            
            if self.escalation_config.auto_pause_on_escalation:
                task.status = TaskStatus.PAUSED
        
        task.interventions.append(intervention_record)
        
        # Learn from this intervention
        if self.monitoring_rules.enable_learning:
            await self.pattern_learner.add_pattern(
                task.outputs[-1]["output"],
                task.instructions,
                task.quality_metrics,
                intervention_result
            )
        
        return intervention_record

    async def _generate_recommendations(
        self,
        task: AgentTask
    ) -> List[str]:
        """Generate recommendations based on task analysis"""
        recommendations = []
        
        # Quality-based recommendations
        if task.quality_metrics.instruction_adherence < 0.7:
            recommendations.append(
                "Consider refining instructions for better adherence"
            )
        
        if task.quality_metrics.coherence_score < 0.6:
            recommendations.append(
                "Output coherence is low - check for logical consistency"
            )
        
        # Resource-based recommendations
        if task.resource_usage.token_count > self.monitoring_rules.max_token_threshold * 0.8:
            recommendations.append(
                "Approaching token limit - consider breaking down the task"
            )
        
        # Pattern-based recommendations
        similar_patterns = await self.pattern_learner.get_similar_patterns(
            task.outputs[-1]["output"] if task.outputs else "",
            task.instructions
        )
        
        if similar_patterns:
            for pattern in similar_patterns[:2]:  # Top 2 similar patterns
                recommendations.extend(pattern.suggested_fixes)
        
        return recommendations

    async def get_supervision_report(
        self,
        time_range_hours: int = 24
    ) -> SupervisionReport:
        """Generate comprehensive supervision report"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        # Filter tasks within time range
        recent_tasks = [
            task for task in self.active_tasks.values()
            if task.created_at >= cutoff_time
        ]
        
        # Calculate intervention statistics
        total_interventions = sum(
            len(task.interventions) for task in recent_tasks
        )
        
        interventions_by_level = {
            InterventionLevel.WARNING: 0,
            InterventionLevel.CORRECTION: 0,
            InterventionLevel.ESCALATION: 0
        }
        
        for task in recent_tasks:
            for intervention in task.interventions:
                level = InterventionLevel(intervention["level"])
                interventions_by_level[level] += 1
        
        # Calculate quality trends
        quality_trends = {
            "average_confidence": sum(
                task.quality_metrics.confidence_score for task in recent_tasks
            ) / max(len(recent_tasks), 1),
            "average_coherence": sum(
                task.quality_metrics.coherence_score for task in recent_tasks
            ) / max(len(recent_tasks), 1),
            "average_instruction_adherence": sum(
                task.quality_metrics.instruction_adherence for task in recent_tasks
            ) / max(len(recent_tasks), 1)
        }
        
        # Get common failure patterns
        common_failures = await self.pattern_learner.get_top_patterns(limit=5)
        
        # Generate recommendations
        recommendations = []
        if quality_trends["average_confidence"] < 0.7:
            recommendations.append(
                "Overall confidence is low - review agent configurations"
            )
        
        if total_interventions > len(recent_tasks) * 0.5:
            recommendations.append(
                "High intervention rate - consider adjusting monitoring thresholds"
            )
        
        # Calculate confidence distribution
        confidence_distribution = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.MEDIUM: 0,
            ConfidenceLevel.HIGH: 0
        }
        
        for task in recent_tasks:
            confidence = task.quality_metrics.confidence_score
            if confidence < 0.6:
                confidence_distribution[ConfidenceLevel.LOW] += 1
            elif confidence < 0.8:
                confidence_distribution[ConfidenceLevel.MEDIUM] += 1
            else:
                confidence_distribution[ConfidenceLevel.HIGH] += 1
        
        return SupervisionReport(
            report_id=str(uuid4()),
            generated_at=datetime.now(),
            tasks_monitored=len(recent_tasks),
            total_interventions=total_interventions,
            interventions_by_level=interventions_by_level,
            quality_trends=quality_trends,
            common_failures=common_failures,
            recommendations=recommendations,
            confidence_distribution=confidence_distribution
        )

    async def _load_knowledge_base(self):
        """Load knowledge base from persistent storage"""
        kb_file = self.data_dir / "knowledge_base.json"
        if kb_file.exists():
            try:
                with open(kb_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = KnowledgeBaseEntry(**entry_data)
                        self.knowledge_base[entry.pattern_id] = entry
            except Exception as e:
                print(f"Warning: Could not load knowledge base: {e}")

    async def save_knowledge_base(self):
        """Save knowledge base to persistent storage"""
        kb_file = self.data_dir / "knowledge_base.json"
        try:
            data = [
                {
                    "pattern_id": entry.pattern_id,
                    "pattern_description": entry.pattern_description,
                    "failure_type": entry.failure_type,
                    "common_causes": entry.common_causes,
                    "suggested_fixes": entry.suggested_fixes,
                    "confidence_score": entry.confidence_score,
                    "occurrences": entry.occurrences,
                    "last_seen": entry.last_seen.isoformat()
                }
                for entry in self.knowledge_base.values()
            ]
            
            with open(kb_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save knowledge base: {e}")