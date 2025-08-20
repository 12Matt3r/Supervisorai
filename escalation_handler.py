"""
Escalation Handler for Supervisor Agent - Human intervention and escalation management.

Adapted from escalation_system.py for integration with the supervisor error handling system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

from .error_types import SupervisorError, ErrorType, ErrorSeverity


class EscalationLevel(Enum):
    """Escalation levels for different types of interventions."""
    NONE = "none"
    AUTO_RECOVERY = "auto_recovery"
    SUPERVISOR_REVIEW = "supervisor_review"
    HUMAN_INTERVENTION = "human_intervention"
    CRITICAL_ALERT = "critical_alert"
    EMERGENCY = "emergency"


class EscalationStatus(Enum):
    """Status of an escalation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EscalationTicket:
    """Represents an escalation ticket."""
    ticket_id: str
    level: EscalationLevel
    status: EscalationStatus
    created_at: datetime
    error: SupervisorError
    context: Dict[str, Any]
    priority: int
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        data['level'] = self.level.value
        data['status'] = self.status.value
        data['error'] = self.error.to_dict() if self.error else None
        return data


class EscalationHandler:
    """System for managing error escalations and human interventions."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("supervisor_data/escalations")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Storage for escalation tickets
        self.tickets: Dict[str, EscalationTicket] = {}
        self.escalation_queue: List[str] = []
        
        # Configuration
        self.config = self._default_config()
        
        # Statistics
        self.stats = {
            'total_escalations': 0,
            'by_level': {level.value: 0 for level in EscalationLevel},
            'by_status': {status.value: 0 for status in EscalationStatus},
            'pending_tickets': 0
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for escalation handler."""
        return {
            'max_auto_recovery_attempts': 5,
            'escalation_timeout': 300,  # 5 minutes
            'critical_error_types': [
                ErrorType.INFINITE_LOOP.value,
                ErrorType.SYSTEM_ERROR.value,
                ErrorType.CORRUPTION.value,
                ErrorType.SECURITY_BREACH.value
            ],
            'auto_escalate_after_failures': 3,
            'escalation_rules': {
                ErrorSeverity.FATAL.value: EscalationLevel.EMERGENCY.value,
                ErrorSeverity.CRITICAL.value: EscalationLevel.CRITICAL_ALERT.value,
                ErrorSeverity.HIGH.value: EscalationLevel.HUMAN_INTERVENTION.value,
                ErrorSeverity.MEDIUM.value: EscalationLevel.SUPERVISOR_REVIEW.value,
                ErrorSeverity.LOW.value: EscalationLevel.AUTO_RECOVERY.value
            }
        }
    
    def determine_escalation_level(
        self,
        error: SupervisorError,
        context: Dict[str, Any]
    ) -> EscalationLevel:
        """Determine the appropriate escalation level for an error."""
        
        # Check for critical error types
        if error.error_type.value in self.config['critical_error_types']:
            return EscalationLevel.CRITICAL_ALERT
        
        # Check error severity
        if error.severity:
            level_name = self.config['escalation_rules'].get(
                error.severity.value,
                EscalationLevel.AUTO_RECOVERY.value
            )
            return EscalationLevel(level_name)
        
        # Check retry count
        if error.retry_count >= self.config['auto_escalate_after_failures']:
            return EscalationLevel.HUMAN_INTERVENTION
        
        # Default to auto recovery
        return EscalationLevel.AUTO_RECOVERY
    
    def escalate_error(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        level: Optional[EscalationLevel] = None
    ) -> str:
        """Create an escalation ticket for an error."""
        
        # Determine escalation level if not provided
        if level is None:
            level = self.determine_escalation_level(error, context)
        
        # Generate ticket ID
        ticket_id = self._generate_ticket_id(error)
        
        # Calculate priority
        priority = self._calculate_priority(error, context)
        
        # Create ticket
        ticket = EscalationTicket(
            ticket_id=ticket_id,
            level=level,
            status=EscalationStatus.PENDING,
            created_at=datetime.utcnow(),
            error=error,
            context=context,
            priority=priority,
            metadata={
                'created_by': 'error_handling_system',
                'auto_generated': True
            }
        )
        
        # Store ticket
        self.tickets[ticket_id] = ticket
        self.escalation_queue.append(ticket_id)
        
        # Update statistics
        self.stats['total_escalations'] += 1
        self.stats['by_level'][level.value] += 1
        self.stats['by_status'][EscalationStatus.PENDING.value] += 1
        self.stats['pending_tickets'] += 1
        
        # Store to disk
        self._store_ticket(ticket)
        
        self.logger.warning(
            f"Escalated error {error.error_id} to level {level.value} "
            f"with ticket {ticket_id}"
        )
        
        return ticket_id
    
    def resolve_ticket(
        self,
        ticket_id: str,
        resolution: str,
        resolved_by: Optional[str] = None
    ) -> bool:
        """Resolve an escalation ticket."""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            self.logger.error(f"Escalation ticket {ticket_id} not found")
            return False
        
        # Update ticket status
        ticket.status = EscalationStatus.RESOLVED
        ticket.resolved_at = datetime.utcnow()
        ticket.resolution = resolution
        ticket.assigned_to = resolved_by
        
        # Update statistics
        self.stats['by_status'][EscalationStatus.PENDING.value] -= 1
        self.stats['by_status'][EscalationStatus.RESOLVED.value] += 1
        self.stats['pending_tickets'] -= 1
        
        # Remove from queue
        if ticket_id in self.escalation_queue:
            self.escalation_queue.remove(ticket_id)
        
        # Update stored ticket
        self._store_ticket(ticket)
        
        self.logger.info(f"Resolved escalation ticket {ticket_id}: {resolution}")
        
        return True
    
    def get_pending_tickets(
        self,
        level: Optional[EscalationLevel] = None
    ) -> List[Dict[str, Any]]:
        """Get list of pending escalation tickets."""
        
        pending_tickets = []
        
        for ticket in self.tickets.values():
            if ticket.status != EscalationStatus.PENDING:
                continue
            
            if level and ticket.level != level:
                continue
            
            pending_tickets.append({
                'ticket_id': ticket.ticket_id,
                'level': ticket.level.value,
                'priority': ticket.priority,
                'created_at': ticket.created_at.isoformat(),
                'error_id': ticket.error.error_id,
                'error_type': ticket.error.error_type.value,
                'error_message': ticket.error.message
            })
        
        # Sort by priority (highest first)
        pending_tickets.sort(key=lambda x: x['priority'], reverse=True)
        
        return pending_tickets
    
    def generate_escalation_report(
        self,
        ticket_id: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive escalation report."""
        
        ticket = self.tickets.get(ticket_id)
        if not ticket:
            return None
        
        report = {
            'ticket_info': ticket.to_dict(),
            'error_analysis': {
                'error_type': ticket.error.error_type.value,
                'severity': ticket.error.severity.value if ticket.error.severity else 'unknown',
                'recoverable': ticket.error.recoverable,
                'retry_count': ticket.error.retry_count
            },
            'context_summary': self._summarize_context(ticket.context),
            'recommendations': self._generate_recommendations(ticket),
            'system_impact': self._assess_system_impact(ticket),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return report
    
    def _generate_ticket_id(self, error: SupervisorError) -> str:
        """Generate a unique ticket ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"ESC_{error.error_id[:8]}_{timestamp}"
    
    def _calculate_priority(self, error: SupervisorError, context: Dict[str, Any]) -> int:
        """Calculate priority score for escalation."""
        
        # Base priority from error severity
        severity_scores = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 3,
            ErrorSeverity.HIGH: 7,
            ErrorSeverity.CRITICAL: 10,
            ErrorSeverity.FATAL: 15
        }
        
        priority = severity_scores.get(error.severity, 5)
        
        # Add retry count penalty
        priority += error.retry_count * 2
        
        # Add error type weights
        type_weights = {
            ErrorType.INFINITE_LOOP: 10,
            ErrorType.SYSTEM_ERROR: 8,
            ErrorType.CORRUPTION: 15,
            ErrorType.SECURITY_BREACH: 20
        }
        
        priority += type_weights.get(error.error_type, 0)
        
        return min(priority, 100)  # Cap at 100
    
    def _store_ticket(self, ticket: EscalationTicket):
        """Store ticket to disk."""
        
        ticket_file = self.storage_path / f"{ticket.ticket_id}.json"
        
        try:
            with open(ticket_file, 'w') as f:
                json.dump(ticket.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to store ticket {ticket.ticket_id}: {str(e)}")
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize context for the report."""
        return {
            'agent_id': context.get('agent_id', 'unknown'),
            'task_id': context.get('task_id', 'unknown'),
            'context_keys': list(context.keys())
        }
    
    def _generate_recommendations(self, ticket: EscalationTicket) -> List[str]:
        """Generate recommendations for resolving the issue."""
        
        recommendations = []
        
        if ticket.error.error_type == ErrorType.INFINITE_LOOP:
            recommendations.extend([
                "Review agent logic for recursive patterns",
                "Implement circuit breakers",
                "Add execution limits"
            ])
        elif ticket.error.error_type == ErrorType.TIMEOUT:
            recommendations.extend([
                "Increase timeout limits",
                "Optimize task complexity",
                "Implement task splitting"
            ])
        elif ticket.error.error_type == ErrorType.SYSTEM_ERROR:
            recommendations.extend([
                "Check system resources",
                "Review error logs",
                "Verify system configuration"
            ])
        else:
            recommendations.append("Manual investigation required")
        
        return recommendations
    
    def _assess_system_impact(self, ticket: EscalationTicket) -> Dict[str, Any]:
        """Assess the impact of the error on the system."""
        
        return {
            'affected_agent': ticket.context.get('agent_id', 'unknown'),
            'affected_task': ticket.context.get('task_id', 'unknown'),
            'error_severity': ticket.error.severity.value if ticket.error.severity else 'unknown',
            'recoverable': ticket.error.recoverable
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the escalation handler."""
        return {
            'active_tickets': len(self.tickets),
            'queue_length': len(self.escalation_queue),
            'stats': self.stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the escalation handler."""
        self.logger.info("Shutting down escalation handler")
        
        # Log any unresolved critical tickets
        critical_tickets = [
            ticket for ticket in self.tickets.values()
            if ticket.level in [EscalationLevel.CRITICAL_ALERT, EscalationLevel.EMERGENCY]
            and ticket.status == EscalationStatus.PENDING
        ]
        
        for ticket in critical_tickets:
            self.logger.critical(
                f"Critical ticket {ticket.ticket_id} remains unresolved during shutdown"
            )
        
        self.tickets.clear()
        self.escalation_queue.clear()
        
        self.logger.info("Escalation handler shutdown complete")
