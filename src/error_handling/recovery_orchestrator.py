"""
Recovery Orchestrator for Supervisor Agent.

Orchestrates comprehensive recovery strategies using all available error handling components.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from .error_types import SupervisorError, ErrorType, ErrorSeverity
from .retry_system import RetrySystem
from .rollback_manager import RollbackManager
from .escalation_handler import EscalationHandler, EscalationLevel
# from .loop_detector import LoopDetector # Commented out due to missing file
from .history_manager import HistoryManager, HistoryEventType


class RecoveryResult(Enum):
    """Results of recovery attempts."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    REQUIRES_ESCALATION = "requires_escalation"
    REQUIRES_ROLLBACK = "requires_rollback"
    AGENT_PAUSED = "agent_paused"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_WITH_ADJUSTMENT = "retry_with_adjustment"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    PAUSE_AGENT = "pause_agent"
    EMERGENCY_STOP = "emergency_stop"
    ADAPTIVE_RECOVERY = "adaptive_recovery"


@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan."""
    plan_id: str
    strategies: List[RecoveryStrategy]
    priority: int
    estimated_success_rate: float
    resource_requirements: Dict[str, Any]
    fallback_plan: Optional[str] = None


class RecoveryOrchestrator:
    """Orchestrates comprehensive recovery strategies."""
    
    def __init__(
        self,
        retry_system: RetrySystem,
        rollback_manager: RollbackManager,
        escalation_handler: Optional[EscalationHandler],
        loop_detector: Optional[Any], # LoopDetector, # Commented out
        history_manager: HistoryManager
    ):
        self.retry_system = retry_system
        self.rollback_manager = rollback_manager
        self.escalation_handler = escalation_handler
        self.loop_detector = loop_detector
        self.history_manager = history_manager
        
        self.logger = logging.getLogger(__name__)
        
        # Recovery statistics
        self.stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'escalated_recoveries': 0,
            'by_strategy': {strategy.value: 0 for strategy in RecoveryStrategy}
        }
        
        # Active recovery operations
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
    
    async def recover_from_error(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        agent_id: str,
        task_id: str,
        recovery_callback: Optional[Callable] = None
    ) -> RecoveryResult:
        """Main entry point for comprehensive error recovery."""
        
        recovery_id = self._generate_recovery_id(error)
        
        self.logger.info(
            f"Starting recovery {recovery_id} for error {error.error_id} "
            f"(type: {error.error_type.value}, severity: {error.severity.value if error.severity else 'unknown'})"
        )
        
        # Record recovery start in history
        history_id = context.get('history_id')
        if history_id:
            self.history_manager.add_entry(
                history_id=history_id,
                event_type=HistoryEventType.RECOVERY_FAILURE,  # Will update to success if needed
                data={
                    'recovery_id': recovery_id,
                    'error': error.to_dict(),
                    'recovery_started': True
                },
                metadata={'recovery_orchestrator': True},
                agent_id=agent_id,
                task_id=task_id
            )
        
        try:
            # Check if agent is paused
            # if self.loop_detector and self.loop_detector.is_agent_paused(agent_id): # Commented out
            #     self.logger.warning(f"Agent {agent_id} is paused, cannot recover")
            #     return RecoveryResult.AGENT_PAUSED
            
            # Generate recovery plan
            recovery_plan = self._generate_recovery_plan(error, context)
            
            # Record active recovery
            self.active_recoveries[recovery_id] = {
                'error': error,
                'context': context,
                'plan': recovery_plan,
                'started_at': datetime.utcnow(),
                'agent_id': agent_id,
                'task_id': task_id
            }
            
            # Execute recovery strategies
            result = await self._execute_recovery_plan(
                recovery_plan, error, context, agent_id, task_id, recovery_callback
            )
            
            # Update statistics
            self._update_recovery_stats(result, recovery_plan)
            
            # Record result in history
            if history_id:
                self.history_manager.add_entry(
                    history_id=history_id,
                    event_type=(
                        HistoryEventType.RECOVERY_SUCCESS if result == RecoveryResult.SUCCESS
                        else HistoryEventType.RECOVERY_FAILURE
                    ),
                    data={
                        'recovery_id': recovery_id,
                        'result': result.value,
                        'strategies_used': [s.value for s in recovery_plan.strategies]
                    },
                    metadata={'recovery_completed': True},
                    agent_id=agent_id,
                    task_id=task_id
                )
            
            self.logger.info(f"Recovery {recovery_id} completed with result: {result.value}")
            
            return result
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery {recovery_id} failed with exception: {str(recovery_error)}")
            
            # Emergency escalation if escalation handler available
            if self.escalation_handler:
                ticket_id = self.escalation_handler.escalate_error(
                    error=SupervisorError(
                        message=f"Recovery system failure: {str(recovery_error)}",
                        error_type=ErrorType.SYSTEM_ERROR,
                        context=context
                    ),
                    context=context,
                    level=EscalationLevel.EMERGENCY
                )
                self.logger.critical(f"Emergency escalation created: {ticket_id}")
            
            return RecoveryResult.FAILURE
        
        finally:
            # Clean up active recovery
            self.active_recoveries.pop(recovery_id, None)
            self.stats['total_recoveries'] += 1
    
    def _generate_recovery_plan(
        self,
        error: SupervisorError,
        context: Dict[str, Any]
    ) -> RecoveryPlan:
        """Generate a comprehensive recovery plan based on error analysis."""
        
        plan_id = f"plan_{error.error_id[:8]}"
        strategies = []
        priority = 50  # Default priority
        estimated_success_rate = 0.5  # Default estimate
        
        # Determine strategies based on error type and severity
        if error.error_type == ErrorType.TIMEOUT:
            strategies = [
                RecoveryStrategy.RETRY_WITH_ADJUSTMENT,
                RecoveryStrategy.ESCALATE_TO_HUMAN
            ]
            estimated_success_rate = 0.7
        
        elif error.error_type == ErrorType.INFINITE_LOOP:
            strategies = [
                RecoveryStrategy.PAUSE_AGENT,
                RecoveryStrategy.ROLLBACK_AND_RETRY,
                RecoveryStrategy.ESCALATE_TO_HUMAN
            ]
            priority = 90
            estimated_success_rate = 0.4
        
        elif error.error_type in [ErrorType.CORRUPTION, ErrorType.SECURITY_BREACH]:
            strategies = [
                RecoveryStrategy.EMERGENCY_STOP,
                RecoveryStrategy.ESCALATE_TO_HUMAN
            ]
            priority = 100
            estimated_success_rate = 0.2
        
        elif error.error_type in [ErrorType.NETWORK_ERROR, ErrorType.RATE_LIMIT]:
            strategies = [
                RecoveryStrategy.RETRY_WITH_ADJUSTMENT,
                RecoveryStrategy.ADAPTIVE_RECOVERY
            ]
            estimated_success_rate = 0.8
        
        elif error.error_type == ErrorType.AGENT_OVERLOAD:
            strategies = [
                RecoveryStrategy.PAUSE_AGENT,
                RecoveryStrategy.RETRY_WITH_ADJUSTMENT,
                RecoveryStrategy.ESCALATE_TO_HUMAN
            ]
            priority = 70
            estimated_success_rate = 0.6
        
        else:
            # Default recovery approach
            strategies = [
                RecoveryStrategy.RETRY_WITH_ADJUSTMENT,
                RecoveryStrategy.ESCALATE_TO_HUMAN
            ]
        
        # Adjust based on severity
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            priority += 30
            if RecoveryStrategy.ESCALATE_TO_HUMAN not in strategies:
                strategies.append(RecoveryStrategy.ESCALATE_TO_HUMAN)
        
        # Adjust based on retry count
        if error.retry_count >= 2:
            priority += 20
            if RecoveryStrategy.ROLLBACK_AND_RETRY not in strategies:
                strategies.insert(-1, RecoveryStrategy.ROLLBACK_AND_RETRY)
        
        return RecoveryPlan(
            plan_id=plan_id,
            strategies=strategies,
            priority=min(priority, 100),
            estimated_success_rate=min(estimated_success_rate, 1.0),
            resource_requirements={
                'snapshot_required': RecoveryStrategy.ROLLBACK_AND_RETRY in strategies,
                'human_intervention': RecoveryStrategy.ESCALATE_TO_HUMAN in strategies
            }
        )
    
    async def _execute_recovery_plan(
        self,
        plan: RecoveryPlan,
        error: SupervisorError,
        context: Dict[str, Any],
        agent_id: str,
        task_id: str,
        recovery_callback: Optional[Callable]
    ) -> RecoveryResult:
        """Execute the recovery plan strategies in sequence."""
        
        self.logger.info(f"Executing recovery plan {plan.plan_id} with {len(plan.strategies)} strategies")
        
        for i, strategy in enumerate(plan.strategies):
            self.logger.info(f"Executing strategy {i+1}/{len(plan.strategies)}: {strategy.value}")
            
            try:
                result = await self._execute_strategy(
                    strategy, error, context, agent_id, task_id, recovery_callback
                )
                
                # Update strategy usage stats
                self.stats['by_strategy'][strategy.value] += 1
                
                # Check if strategy succeeded
                if result in [RecoveryResult.SUCCESS, RecoveryResult.PARTIAL_SUCCESS]:
                    self.logger.info(f"Strategy {strategy.value} succeeded")
                    return result
                elif result == RecoveryResult.AGENT_PAUSED:
                    self.logger.warning(f"Agent paused by strategy {strategy.value}")
                    return result
                elif result == RecoveryResult.REQUIRES_ESCALATION:
                    # Skip to escalation strategy if available
                    if RecoveryStrategy.ESCALATE_TO_HUMAN in plan.strategies[i+1:]:
                        continue
                    else:
                        return result
                else:
                    self.logger.warning(f"Strategy {strategy.value} failed, trying next")
                    continue
                
            except Exception as e:
                self.logger.error(f"Strategy {strategy.value} raised exception: {str(e)}")
                continue
        
        # All strategies failed
        self.logger.error(f"All recovery strategies failed for plan {plan.plan_id}")
        return RecoveryResult.FAILURE
    
    async def _execute_strategy(
        self,
        strategy: RecoveryStrategy,
        error: SupervisorError,
        context: Dict[str, Any],
        agent_id: str,
        task_id: str,
        recovery_callback: Optional[Callable]
    ) -> RecoveryResult:
        """Execute a single recovery strategy."""
        
        if strategy == RecoveryStrategy.RETRY_WITH_ADJUSTMENT:
            return await self._retry_with_adjustment(error, context, recovery_callback)
        
        elif strategy == RecoveryStrategy.ROLLBACK_AND_RETRY:
            return await self._rollback_and_retry(error, context, recovery_callback)
        
        elif strategy == RecoveryStrategy.ESCALATE_TO_HUMAN:
            return await self._escalate_to_human(error, context)
        
        elif strategy == RecoveryStrategy.PAUSE_AGENT:
            return await self._pause_agent(error, context, agent_id)
        
        elif strategy == RecoveryStrategy.EMERGENCY_STOP:
            return await self._emergency_stop(error, context, agent_id)
        
        elif strategy == RecoveryStrategy.ADAPTIVE_RECOVERY:
            return await self._adaptive_recovery(error, context, recovery_callback)
        
        else:
            self.logger.error(f"Unknown recovery strategy: {strategy.value}")
            return RecoveryResult.FAILURE
    
    async def _retry_with_adjustment(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        recovery_callback: Optional[Callable]
    ) -> RecoveryResult:
        """Execute retry with intelligent adjustments."""
        
        if not await self.retry_system.should_retry(error):
            return RecoveryResult.REQUIRES_ESCALATION
        
        if not recovery_callback:
            self.logger.warning("No recovery callback provided for retry")
            return RecoveryResult.FAILURE
        
        retry_result = await self.retry_system.execute_retry(
            error=error,
            retry_callback=recovery_callback,
            original_prompt=context.get('original_prompt')
        )
        
        if retry_result.get('success', False):
            return RecoveryResult.SUCCESS
        else:
            # Update error retry count
            error.retry_count += 1
            return RecoveryResult.FAILURE
    
    async def _rollback_and_retry(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        recovery_callback: Optional[Callable]
    ) -> RecoveryResult:
        """Execute rollback to last known good state and retry."""
        
        snapshot_id = context.get('snapshot_id')
        if not snapshot_id:
            self.logger.warning("No snapshot available for rollback")
            return RecoveryResult.FAILURE
        
        # Execute rollback
        rollback_result = self.rollback_manager.rollback_to_snapshot(snapshot_id)
        
        if not rollback_result.get('success', False):
            self.logger.error(f"Rollback failed: {rollback_result.get('error')}")
            return RecoveryResult.FAILURE
        
        # Attempt retry after rollback
        if recovery_callback:
            try:
                result = await recovery_callback()
                return RecoveryResult.SUCCESS
            except Exception as e:
                self.logger.error(f"Retry after rollback failed: {str(e)}")
                return RecoveryResult.FAILURE
        
        return RecoveryResult.PARTIAL_SUCCESS
    
    async def _escalate_to_human(
        self,
        error: SupervisorError,
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """Escalate error to human intervention."""
        
        if not self.escalation_handler:
            self.logger.error("No escalation handler available")
            return RecoveryResult.FAILURE
        
        ticket_id = self.escalation_handler.escalate_error(
            error=error,
            context=context
        )
        
        self.logger.info(f"Escalated to human intervention: ticket {ticket_id}")
        return RecoveryResult.REQUIRES_ESCALATION
    
    async def _pause_agent(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        agent_id: str
    ) -> RecoveryResult:
        """Pause the agent to prevent further issues."""
        
        # if self.loop_detector: # Commented out
        #     self.loop_detector.pause_agent(
        #         agent_id=agent_id,
        #         reason=f"Recovery pause due to {error.error_type.value}: {error.message}"
        #     )
        
        return RecoveryResult.AGENT_PAUSED
    
    async def _emergency_stop(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        agent_id: str
    ) -> RecoveryResult:
        """Execute emergency stop procedures."""
        
        self.logger.critical(f"Emergency stop initiated for agent {agent_id} due to {error.error_type.value}")
        
        # Pause agent
        # if self.loop_detector: # Commented out
        #     self.loop_detector.pause_agent(
        #         agent_id=agent_id,
        #         reason=f"Emergency stop: {error.error_type.value}"
        #     )
        
        # Escalate to highest level
        if self.escalation_handler:
            self.escalation_handler.escalate_error(
                error=error,
                context=context,
                level=EscalationLevel.EMERGENCY
            )
        
        return RecoveryResult.AGENT_PAUSED
    
    async def _adaptive_recovery(
        self,
        error: SupervisorError,
        context: Dict[str, Any],
        recovery_callback: Optional[Callable]
    ) -> RecoveryResult:
        """Execute adaptive recovery based on historical success patterns."""
        
        # This would implement machine learning based recovery
        # For now, fall back to retry with adjustment
        return await self._retry_with_adjustment(error, context, recovery_callback)
    
    def _generate_recovery_id(self, error: SupervisorError) -> str:
        """Generate a unique recovery ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"recovery_{error.error_id[:8]}_{timestamp}"
    
    def _update_recovery_stats(self, result: RecoveryResult, plan: RecoveryPlan):
        """Update recovery statistics."""
        
        if result == RecoveryResult.SUCCESS:
            self.stats['successful_recoveries'] += 1
        elif result == RecoveryResult.REQUIRES_ESCALATION:
            self.stats['escalated_recoveries'] += 1
        else:
            self.stats['failed_recoveries'] += 1
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the recovery orchestrator."""
        return {
            'active_recoveries': len(self.active_recoveries),
            'stats': self.stats,
            'components_status': {
                'retry_system': await self.retry_system.get_status(),
                'rollback_manager': await self.rollback_manager.get_status(),
                'escalation_handler': await self.escalation_handler.get_status() if self.escalation_handler else None,
                # 'loop_detector': await self.loop_detector.get_status(), # Commented out
                'history_manager': await self.history_manager.get_status()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the recovery orchestrator."""
        self.logger.info("Shutting down recovery orchestrator")
        
        # Log any active recoveries
        for recovery_id, recovery_info in self.active_recoveries.items():
            self.logger.warning(f"Active recovery {recovery_id} interrupted during shutdown")
        
        self.active_recoveries.clear()
        self.logger.info("Recovery orchestrator shutdown complete")
