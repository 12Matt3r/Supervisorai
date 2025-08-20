"""
Complete Error Handling System for Supervisor Agent

Integrates all error handling components into a unified system with comprehensive
error management, recovery orchestration, and monitoring capabilities.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

from .error_types import SupervisorError, ErrorType, ErrorClassifier
from .retry_system import RetrySystem
from .rollback_manager import RollbackManager
from .escalation_handler import EscalationHandler
from .loop_detector import LoopDetector
from .history_manager import HistoryManager, HistoryEventType
from .recovery_orchestrator import RecoveryOrchestrator, RecoveryResult


class SupervisorErrorHandlingSystem:
    """
    Comprehensive error handling system for the Supervisor Agent.
    
    Integrates all error handling components:
    - Auto-retry with progressive strategies
    - State rollback and recovery
    - Human escalation management
    - Loop detection and circuit breakers
    - Versioned history tracking
    - Recovery orchestration
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_retries: int = 3,
        max_snapshots: int = 50,
        escalation_enabled: bool = True
    ):
        """
        Initialize the comprehensive error handling system.
        
        Args:
            storage_path: Base path for persistent storage
            max_retries: Maximum retry attempts
            max_snapshots: Maximum state snapshots to keep
            escalation_enabled: Whether to enable human escalation
        """
        
        self.storage_path = storage_path or Path("supervisor_data/error_handling")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Supervisor Error Handling System")
        
        # Initialize all subsystems
        self.retry_system = RetrySystem(max_retries=max_retries)
        self.rollback_manager = RollbackManager(
            storage_path=self.storage_path / "snapshots",
            max_snapshots=max_snapshots
        )
        self.escalation_handler = EscalationHandler(
            storage_path=self.storage_path / "escalations"
        ) if escalation_enabled else None
        
        self.loop_detector = LoopDetector()
        self.history_manager = HistoryManager(
            storage_path=self.storage_path / "history"
        )
        
        # Initialize recovery orchestrator with all components
        self.recovery_orchestrator = RecoveryOrchestrator(
            retry_system=self.retry_system,
            rollback_manager=self.rollback_manager,
            escalation_handler=self.escalation_handler,
            loop_detector=self.loop_detector,
            history_manager=self.history_manager
        )
        
        # System state
        self.is_initialized = True
        self.system_stats = {
            "total_errors_handled": 0,
            "successful_recoveries": 0,
            "escalated_errors": 0,
            "system_start_time": datetime.utcnow(),
            "loop_detections": 0,
            "agents_paused": 0
        }
        
        self.logger.info("Error handling system initialization complete")
    
    async def handle_error(
        self,
        error: Exception,
        agent_id: str,
        task_id: str,
        context: Optional[Dict[str, Any]] = None,
        state_data: Optional[Dict[str, Any]] = None,
        recovery_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive error handling entry point.
        
        Args:
            error: The exception/error that occurred
            agent_id: ID of the agent that encountered the error
            task_id: ID of the task being executed
            context: Additional context information
            state_data: Current system state for potential rollback
            recovery_callback: Optional callback for recovery operations
        
        Returns:
            Dict containing recovery result and metadata
        """
        
        self.logger.info(f"Handling error for agent {agent_id}, task {task_id}: {str(error)}")
        
        # Convert exception to SupervisorError if needed
        if isinstance(error, SupervisorError):
            supervisor_error = error
        else:
            error_type = ErrorClassifier.classify_exception(error)
            supervisor_error = SupervisorError(
                message=str(error),
                error_type=error_type,
                context={"agent_id": agent_id, "task_id": task_id, **(context or {})}
            )
        
        # Update statistics
        self.system_stats["total_errors_handled"] += 1
        
        # Create or get history
        history_id = self.history_manager.create_history(
            agent_id=agent_id,
            task_id=task_id,
            initial_data=state_data or {}
        )
        
        # Record error in history
        self.history_manager.add_entry(
            history_id=history_id,
            event_type=HistoryEventType.ERROR_OCCURRED,
            data=supervisor_error.to_dict(),
            metadata={"recovery_initiated": True},
            agent_id=agent_id,
            task_id=task_id
        )
        
        # Create state snapshot if state data provided
        snapshot_id = None
        if state_data:
            snapshot_id = self.rollback_manager.create_snapshot(
                state_data=state_data,
                tags=["error_handling", "pre_recovery"],
                metadata={
                    "error_type": supervisor_error.error_type.value,
                    "agent_id": agent_id,
                    "task_id": task_id
                },
                agent_id=agent_id,
                task_id=task_id
            )
        
        try:
            # Check for loops first
            loop_detection = None
            if state_data:
                loop_detection = self.loop_detector.record_execution_point(
                    agent_id=agent_id,
                    task_id=task_id,
                    state=state_data,
                    output=str(error),
                    context=context or {}
                )
                
                if loop_detection and loop_detection.severity in ["high", "critical"]:
                    self.system_stats["loop_detections"] += 1
                    return await self._handle_loop_error(loop_detection, supervisor_error, context or {})
            
            # Execute comprehensive recovery
            recovery_context = (context or {}).copy()
            recovery_context.update({
                "agent_id": agent_id,
                "task_id": task_id,
                "history_id": history_id,
                "snapshot_id": snapshot_id
            })
            
            recovery_result = await self.recovery_orchestrator.recover_from_error(
                error=supervisor_error,
                context=recovery_context,
                agent_id=agent_id,
                task_id=task_id,
                recovery_callback=recovery_callback
            )
            
            # Update statistics based on result
            if recovery_result == RecoveryResult.SUCCESS:
                self.system_stats["successful_recoveries"] += 1
            elif recovery_result == RecoveryResult.REQUIRES_ESCALATION:
                self.system_stats["escalated_errors"] += 1
            elif recovery_result == RecoveryResult.AGENT_PAUSED:
                self.system_stats["agents_paused"] += 1
            
            return {
                "success": recovery_result == RecoveryResult.SUCCESS,
                "recovery_result": recovery_result.value,
                "error_handled": True,
                "history_id": history_id,
                "snapshot_id": snapshot_id,
                "loop_detected": loop_detection is not None,
                "timestamp": datetime.utcnow().isoformat(),
                "error_id": supervisor_error.error_id
            }
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            
            # Emergency escalation if enabled
            ticket_id = None
            if self.escalation_handler:
                emergency_error = SupervisorError(
                    message=f"Recovery system failure: {str(recovery_error)}",
                    error_type=ErrorType.SYSTEM_ERROR,
                    context={
                        "original_error": supervisor_error.to_dict(),
                        "agent_id": agent_id,
                        "task_id": task_id
                    }
                )
                
                ticket_id = self.escalation_handler.escalate_error(
                    error=emergency_error,
                    context=recovery_context
                )
            
            return {
                "success": False,
                "recovery_result": "system_failure",
                "error_handled": False,
                "escalation_ticket": ticket_id,
                "history_id": history_id,
                "snapshot_id": snapshot_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error_id": supervisor_error.error_id
            }
    
    async def _handle_loop_error(
        self,
        loop_detection,
        error: SupervisorError,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle loop-specific error recovery."""
        
        self.logger.warning(f"Handling loop error: {loop_detection.loop_type.value}")
        
        # Pause the agent immediately
        self.loop_detector.pause_agent(
            loop_detection.agent_id,
            f"Loop detected: {loop_detection.loop_type.value}"
        )
        
        # Escalate if escalation is enabled
        ticket_id = None
        if self.escalation_handler:
            loop_error = SupervisorError(
                message=f"Infinite loop detected: {loop_detection.loop_type.value}",
                error_type=ErrorType.INFINITE_LOOP,
                context={
                    "loop_detection_id": loop_detection.detection_id,
                    "agent_id": loop_detection.agent_id,
                    "task_id": loop_detection.task_id,
                    "confidence_score": loop_detection.confidence_score
                }
            )
            
            ticket_id = self.escalation_handler.escalate_error(
                error=loop_error,
                context=context
            )
        
        return {
            "success": False,
            "recovery_result": "loop_detected_agent_paused",
            "error_handled": True,
            "loop_detection": loop_detection.to_dict(),
            "agent_paused": True,
            "escalation_ticket": ticket_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "system_stats": self.system_stats,
            "is_initialized": self.is_initialized,
            "components": {
                "retry_system": await self.retry_system.get_status(),
                "rollback_manager": await self.rollback_manager.get_status(),
                "escalation_handler": await self.escalation_handler.get_status() if self.escalation_handler else None,
                "loop_detector": await self.loop_detector.get_status(),
                "history_manager": await self.history_manager.get_status(),
                "recovery_orchestrator": await self.recovery_orchestrator.get_status()
            },
            "storage_path": str(self.storage_path),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def pause_agent(self, agent_id: str, reason: str = "Manual pause") -> bool:
        """Manually pause an agent."""
        self.loop_detector.pause_agent(agent_id, reason)
        self.system_stats["agents_paused"] += 1
        return True
    
    async def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        return self.loop_detector.resume_agent(agent_id)
    
    async def create_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a named checkpoint for easy rollback."""
        return self.rollback_manager.create_checkpoint(
            checkpoint_name=checkpoint_name,
            state_data=state_data,
            metadata=metadata
        )
    
    async def rollback_to_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Rollback to a named checkpoint."""
        return self.rollback_manager.rollback_to_checkpoint(checkpoint_name)
    
    async def get_pending_escalations(self) -> List[Dict[str, Any]]:
        """Get list of pending escalations."""
        if not self.escalation_handler:
            return []
        
        return self.escalation_handler.get_pending_tickets()
    
    async def resolve_escalation(
        self,
        ticket_id: str,
        resolution: str,
        resolved_by: Optional[str] = None
    ) -> bool:
        """Resolve an escalation ticket."""
        if not self.escalation_handler:
            return False
        
        return self.escalation_handler.resolve_ticket(
            ticket_id=ticket_id,
            resolution=resolution,
            resolved_by=resolved_by
        )
    
    async def shutdown(self):
        """Gracefully shutdown the error handling system."""
        self.logger.info("Shutting down error handling system")
        
        # Shutdown all subsystems
        await self.retry_system.shutdown()
        await self.rollback_manager.shutdown()
        if self.escalation_handler:
            await self.escalation_handler.shutdown()
        await self.loop_detector.shutdown()
        await self.history_manager.shutdown()
        await self.recovery_orchestrator.shutdown()
        
        self.is_initialized = False
        self.logger.info("Error handling system shutdown complete")
