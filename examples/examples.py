"""
Comprehensive examples demonstrating the Error Handling System capabilities.

This module provides practical examples of how to use each component of the
error handling system in various scenarios.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all components
from .error_handling_system import SupervisorErrorHandlingSystem
from .error_types import SupervisorError, ErrorType, ErrorSeverity, ErrorClassifier
from .retry_system import RetrySystem
from .rollback_manager import RollbackManager
from .escalation_handler import EscalationHandler
# from .loop_detector import LoopDetector # Commented out
# from .history_manager import HistoryManager, HistoryEventType # Commented out


class ErrorHandlingExamples:
    """Complete examples of error handling system usage."""
    
    def __init__(self):
        self.error_system = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize_system(self):
        """Initialize the complete error handling system."""
        
        self.logger.info("Initializing Error Handling System")
        
        self.error_system = SupervisorErrorHandlingSystem(
            storage_path=Path("examples_data/error_handling"),
            max_retries=3,
            max_snapshots=20,
            escalation_enabled=True
        )
        
        self.logger.info("System initialization complete")
    
    async def example_1_basic_error_handling(self):
        """Example 1: Basic error handling with automatic recovery."""
        
        self.logger.info("=== Example 1: Basic Error Handling ===")
        
        # Simulate a task that might fail
        async def failing_task():
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise TimeoutError("Task execution timeout")
            return "Task completed successfully"
        
        # Recovery callback
        async def recovery_callback():
            self.logger.info("Executing recovery callback")
            # Simulate recovery logic
            await asyncio.sleep(0.1)
            return "Recovery completed"
        
        try:
            # Attempt the task
            result = await failing_task()
            self.logger.info(f"Task succeeded: {result}")
        except Exception as e:
            # Handle error through the system
            recovery_result = await self.error_system.handle_error(
                error=e,
                agent_id="example_agent_1",
                task_id="basic_task_001",
                context={"operation": "basic_example"},
                state_data={"progress": 50, "status": "processing"},
                recovery_callback=recovery_callback
            )
            
            self.logger.info(f"Error handling result: {recovery_result}")
    
    async def example_2_custom_error_types(self):
        """Example 2: Working with custom error types and classification."""
        
        self.logger.info("=== Example 2: Custom Error Types ===")
        
        # Create custom errors
        validation_error = SupervisorError(
            message="Data validation failed for user input",
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context={
                "validation_rule": "email_format",
                "invalid_value": "not-an-email",
                "field": "user_email"
            },
            recoverable=True
        )
        
        # Recovery with data cleaning
        async def data_cleaning_recovery():
            self.logger.info("Performing data cleaning recovery")
            # Simulate data cleaning
            cleaned_data = {
                "user_email": "corrected@example.com",
                "cleaned_at": datetime.utcnow().isoformat()
            }
            return cleaned_data
        
        # Handle the custom error
        result = await self.error_system.handle_error(
            error=validation_error,
            agent_id="data_processor",
            task_id="validation_task_001",
            context={"data_source": "user_input", "batch_size": 100},
            state_data={"processed_records": 45, "failed_records": 5},
            recovery_callback=data_cleaning_recovery
        )
        
        self.logger.info(f"Custom error handling result: {result}")
        
        # Demonstrate error classification
        standard_errors = [
            TimeoutError("Connection timeout"),
            ValueError("Invalid configuration value"),
            ConnectionError("Network unreachable"),
            MemoryError("Out of memory")
        ]
        
        for error in standard_errors:
            classified_type = ErrorClassifier.classify_exception(error)
            self.logger.info(f"{type(error).__name__} classified as: {classified_type.value}")
    
    async def example_3_retry_strategies(self):
        """Example 3: Different retry strategies and configurations."""
        
        self.logger.info("=== Example 3: Retry Strategies ===")
        
        retry_system = RetrySystem(max_retries=3)
        
        # Create different types of errors to test retry strategies
        errors = [
            SupervisorError(
                message="Network timeout occurred",
                error_type=ErrorType.TIMEOUT,
                retry_count=0
            ),
            SupervisorError(
                message="Rate limit exceeded",
                error_type=ErrorType.RATE_LIMIT,
                retry_count=1
            ),
            SupervisorError(
                message="Agent overloaded",
                error_type=ErrorType.AGENT_OVERLOAD,
                retry_count=2
            )
        ]
        
        for i, error in enumerate(errors):
            should_retry = await retry_system.should_retry(error)
            self.logger.info(
                f"Error {i+1} ({error.error_type.value}, retry_count={error.retry_count}): "
                f"Should retry = {should_retry}"
            )
            
            if should_retry:
                # Mock retry callback
                async def mock_retry(adjusted_prompt=None):
                    self.logger.info(f"Retry executed with prompt: {adjusted_prompt or 'original'}")
                    return f"Retry {i+1} successful"
                
                retry_result = await retry_system.execute_retry(
                    error=error,
                    retry_callback=mock_retry,
                    original_prompt=f"Original task prompt for error {i+1}"
                )
                
                self.logger.info(f"Retry result: {retry_result}")
    
    async def example_4_rollback_and_checkpoints(self):
        """Example 4: State management with rollback and checkpoints."""
        
        self.logger.info("=== Example 4: Rollback and Checkpoints ===")
        
        rollback_manager = RollbackManager(max_snapshots=10)
        
        # Simulate system states at different points
        states = [
            {
                "step": "initialization",
                "data": {"users": [], "settings": {}},
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "step": "data_loading",
                "data": {"users": ["user1", "user2"], "settings": {"loaded": True}},
                "timestamp": (datetime.utcnow() + timedelta(seconds=10)).isoformat()
            },
            {
                "step": "processing",
                "data": {"users": ["user1", "user2"], "processed": 1, "settings": {"loaded": True}},
                "timestamp": (datetime.utcnow() + timedelta(seconds=20)).isoformat()
            }
        ]
        
        # Create snapshots
        snapshot_ids = []
        for i, state in enumerate(states):
            snapshot_id = rollback_manager.create_snapshot(
                state_data=state,
                tags=["example", f"step_{i+1}"],
                metadata={"step_name": state["step"]},
                agent_id="state_manager",
                task_id="state_tracking_001"
            )
            snapshot_ids.append(snapshot_id)
            self.logger.info(f"Created snapshot {snapshot_id} for step: {state['step']}")
        
        # Create named checkpoints
        checkpoint_id = rollback_manager.create_checkpoint(
            checkpoint_name="stable_state",
            state_data=states[1],  # Use data_loading state as stable
            metadata={"reason": "Stable state after successful data loading"}
        )
        
        self.logger.info(f"Created checkpoint 'stable_state': {checkpoint_id}")
        
        # Simulate error and rollback
        self.logger.info("Simulating error during processing...")
        
        # Rollback to checkpoint
        rollback_result = rollback_manager.rollback_to_checkpoint("stable_state")
        
        if rollback_result["success"]:
            self.logger.info("Successfully rolled back to stable state")
            self.logger.info(f"Restored state: {rollback_result['state_data']['step']}")
        else:
            self.logger.error(f"Rollback failed: {rollback_result['error']}")
        
        # List available snapshots
        available_snapshots = rollback_manager.get_snapshots(limit=10)
        self.logger.info(f"Available snapshots: {len(available_snapshots)}")
        for snapshot in available_snapshots[:3]:  # Show first 3
            self.logger.info(
                f"  - {snapshot['snapshot_id']}: {snapshot['tags']} "
                f"at {snapshot['timestamp']}"
            )
    
    async def example_5_escalation_management(self):
        """Example 5: Escalation handling and human intervention."""
        
        self.logger.info("=== Example 5: Escalation Management ===")
        
        escalation_handler = EscalationHandler()
        
        # Create different severity errors for escalation
        errors = [
            SupervisorError(
                message="Minor validation issue",
                error_type=ErrorType.VALIDATION_ERROR,
                severity=ErrorSeverity.LOW,
                retry_count=1
            ),
            SupervisorError(
                message="Agent stuck in processing loop",
                error_type=ErrorType.INFINITE_LOOP,
                severity=ErrorSeverity.HIGH,
                retry_count=3
            ),
            SupervisorError(
                message="Critical system corruption detected",
                error_type=ErrorType.CORRUPTION,
                severity=ErrorSeverity.CRITICAL,
                retry_count=0
            )
        ]
        
        ticket_ids = []
        for i, error in enumerate(errors):
            # Determine escalation level
            level = escalation_handler.determine_escalation_level(
                error=error,
                context={"error_index": i}
            )
            
            # Create escalation
            ticket_id = escalation_handler.escalate_error(
                error=error,
                context={
                    "agent_id": f"agent_{i+1}",
                    "task_id": f"task_{i+1}",
                    "error_context": "example_scenario"
                }
            )
            
            ticket_ids.append(ticket_id)
            self.logger.info(
                f"Created escalation ticket {ticket_id} at level {level.value} "
                f"for {error.error_type.value}"
            )
        
        # Get pending tickets
        pending_tickets = escalation_handler.get_pending_tickets()
        self.logger.info(f"Pending escalation tickets: {len(pending_tickets)}")
        
        for ticket in pending_tickets:
            self.logger.info(
                f"  - Ticket {ticket['ticket_id']}: {ticket['level']} "
                f"(Priority: {ticket['priority']})"
            )
        
        # Resolve some tickets
        if ticket_ids:
            resolved = escalation_handler.resolve_ticket(
                ticket_id=ticket_ids[0],
                resolution="Issue resolved through automated data cleaning",
                resolved_by="system_auto_resolver"
            )
            
            if resolved:
                self.logger.info(f"Resolved ticket {ticket_ids[0]}")
        
        # Generate escalation report
        if len(ticket_ids) > 1:
            report = escalation_handler.generate_escalation_report(ticket_ids[1])
            if report:
                self.logger.info(f"Generated escalation report for {ticket_ids[1]}")
                self.logger.info(f"Report summary: {report['error_analysis']}")
    
    # async def example_6_loop_detection(self): # Commented out
    #     """Example 6: Loop detection and circuit breaker patterns."""
        
    #     self.logger.info("=== Example 6: Loop Detection ===")
        
    #     loop_detector = LoopDetector()
        
    #     # Simulate repetitive execution that might indicate a loop
    #     agent_id = "loop_prone_agent"
    #     task_id = "repetitive_task"
        
    #     # Simulate execution points that would trigger loop detection
    #     execution_scenarios = [
    #         # Normal execution
    #         {"state": {"step": 1}, "output": "Processing step 1"},
    #         {"state": {"step": 2}, "output": "Processing step 2"},
    #         {"state": {"step": 3}, "output": "Processing step 3"},
            
    #         # Start of potential loop
    #         {"state": {"step": 1}, "output": "Processing step 1"},  # Repeat
    #         {"state": {"step": 1}, "output": "Processing step 1"},  # Repeat
    #         {"state": {"step": 1}, "output": "Processing step 1"},  # Repeat - should trigger
    #     ]

    #     loop_detection = None
    #     for i, scenario in enumerate(execution_scenarios):
    #         self.logger.info(f"Recording execution point {i+1}: {scenario['state']}")
            
    #         detection = loop_detector.record_execution_point(
    #             agent_id=agent_id,
    #             task_id=task_id,
    #             state=scenario["state"],
    #             output=scenario["output"],
    #             context={"iteration": i+1}
    #         )
            
    #         if detection:
    #             loop_detection = detection
    #             self.logger.warning(
    #                 f"Loop detected! Type: {detection.loop_type.value}, "
    #                 f"Confidence: {detection.confidence_score:.2f}, "
    #                 f"Severity: {detection.severity}"
    #             )
    #             break

    #     # Demonstrate agent pausing
    #     if loop_detection:
    #         loop_detector.pause_agent(
    #             agent_id=agent_id,
    #             reason=f"Loop detected: {loop_detection.loop_type.value}"
    #         )
            
    #         is_paused = loop_detector.is_agent_paused(agent_id)
    #         self.logger.info(f"Agent {agent_id} is paused: {is_paused}")
            
    #         # Resume agent after some time
    #         await asyncio.sleep(0.1)  # Simulate investigation time
            
    #         resumed = loop_detector.resume_agent(agent_id)
    #         self.logger.info(f"Agent {agent_id} resumed: {resumed}")

    #     # Demonstrate circuit breaker
    #     circuit_breaker = loop_detector.get_circuit_breaker("example_service")

    #     # Simulate failures
    #     for i in range(7):  # Exceed failure threshold
    #         if circuit_breaker.is_call_allowed():
    #             # Simulate call that fails
    #             circuit_breaker.record_failure()
    #             self.logger.info(f"Call {i+1} failed, circuit breaker failure count: {circuit_breaker.failure_count}")
    #         else:
    #             self.logger.info(f"Call {i+1} blocked by circuit breaker (state: {circuit_breaker.state.value})")
    
    # async def example_7_history_and_versioning(self): # Commented out
    #     """Example 7: History management and versioning."""

    #     self.logger.info("=== Example 7: History and Versioning ===")

    #     history_manager = HistoryManager()

    #     # Create a history timeline
    #     history_id = history_manager.create_history(
    #         agent_id="versioned_agent",
    #         task_id="complex_task",
    #         initial_data={"status": "initialized", "version": "1.0"}
    #     )

    #     self.logger.info(f"Created history timeline: {history_id}")

    #     # Record various events
    #     events = [
    #         {
    #             "event_type": HistoryEventType.STATE_CHANGED,
    #             "data": {"status": "processing", "progress": 25},
    #             "metadata": {"trigger": "user_action"}
    #         },
    #         {
    #             "event_type": HistoryEventType.ERROR_OCCURRED,
    #             "data": {"error_type": "timeout", "message": "Processing timeout"},
    #             "metadata": {"auto_detected": True}
    #         },
    #         {
    #             "event_type": HistoryEventType.RETRY_ATTEMPTED,
    #             "data": {"attempt": 1, "strategy": "exponential_backoff"},
    #             "metadata": {"auto_retry": True}
    #         },
    #         {
    #             "event_type": HistoryEventType.RECOVERY_SUCCESS,
    #             "data": {"status": "completed", "progress": 100},
    #             "metadata": {"recovery_method": "retry_with_adjustment"}
    #         }
    #     ]

    #     entry_ids = []
    #     for event in events:
    #         entry_id = history_manager.add_entry(
    #             history_id=history_id,
    #             event_type=event["event_type"],
    #             data=event["data"],
    #             metadata=event["metadata"],
    #             agent_id="versioned_agent",
    #             task_id="complex_task"
    #         )
    #         entry_ids.append(entry_id)
    #         self.logger.info(f"Added {event['event_type'].value} entry: {entry_id}")

    #     # Create a version after significant progress
    #     version_id = history_manager.create_version(
    #         history_id=history_id,
    #         summary="Major milestone: Recovered from timeout error",
    #         tags=["milestone", "error_recovery", "timeout"]
    #     )

    #     self.logger.info(f"Created version snapshot: {version_id}")

    #     # Record intervention
    #     before_state = {"status": "error", "last_attempt": "failed"}
    #     after_state = {"status": "recovered", "last_attempt": "success"}

    #     intervention_id = history_manager.record_intervention(
    #         history_id=history_id,
    #         intervention_type="manual_recovery",
    #         intervention_data={"method": "state_reset", "operator": "system_admin"},
    #         before_state=before_state,
    #         after_state=after_state,
    #         metadata={"intervention_reason": "automated_recovery_failed"}
    #     )

    #     self.logger.info(f"Recorded intervention: {intervention_id}")

    #     # Query history
    #     recent_entries = history_manager.get_history(
    #         history_id=history_id,
    #         limit=5
    #     )

    #     self.logger.info(f"Recent history entries: {len(recent_entries)}")
    #     for entry in recent_entries[-3:]:  # Show last 3
    #         self.logger.info(
    #             f"  - {entry['event_type']}: {entry['timestamp']} "
    #             f"(Version: {entry['version']})"
    #         )

    #     # Search for specific events
    #     error_entries = history_manager.search_entries(
    #         history_id=history_id,
    #         event_type=HistoryEventType.ERROR_OCCURRED
    #     )

    #     self.logger.info(f"Found {len(error_entries)} error events in history")
    
    async def example_8_comprehensive_scenario(self):
        """Example 8: Comprehensive scenario using all components together."""
        
        self.logger.info("=== Example 8: Comprehensive Scenario ===")
        
        # Simulate a complex data processing task that encounters multiple issues
        
        async def complex_data_processing_task():
            """Simulates a complex task that might fail in various ways."""
            
            import random
            failure_type = random.choice([
                "timeout", "validation", "resource_exhaustion", "success"
            ])
            
            if failure_type == "timeout":
                raise TimeoutError("Data processing timeout after 30 seconds")
            elif failure_type == "validation":
                raise ValueError("Invalid data format in batch processing")
            elif failure_type == "resource_exhaustion":
                raise MemoryError("Insufficient memory for large dataset")
            else:
                return {
                    "status": "success",
                    "processed_records": 1000,
                    "processing_time": "45 seconds"
                }
        
        # Recovery callback with multiple strategies
        async def adaptive_recovery_callback(attempt=1):
            """Adaptive recovery that changes strategy based on attempt number."""
            
            strategies = {
                1: "reduce_batch_size",
                2: "increase_timeout",
                3: "fallback_processing"
            }
            
            strategy = strategies.get(attempt, "manual_intervention")
            self.logger.info(f"Executing recovery strategy: {strategy} (attempt {attempt})")
            
            # Simulate recovery work
            await asyncio.sleep(0.1)
            
            if attempt <= 2:
                # Simulate successful recovery for first 2 attempts
                return {
                    "recovery_strategy": strategy,
                    "success": True,
                    "processed_records": 800,  # Reduced due to recovery
                    "recovery_time": f"{attempt * 5} seconds"
                }
            else:
                # Third attempt might still fail
                import random
                if random.random() > 0.3:
                    return {
                        "recovery_strategy": strategy,
                        "success": True,
                        "processed_records": 600,
                        "recovery_time": "15 seconds"
                    }
                else:
                    raise Exception(f"Recovery strategy {strategy} also failed")
        
        # Create comprehensive state data
        initial_state = {
            "task_name": "comprehensive_data_processing",
            "start_time": datetime.utcnow().isoformat(),
            "dataset_size": 10000,
            "batch_size": 100,
            "timeout_seconds": 30,
            "processed_batches": 0,
            "failed_batches": 0,
            "current_memory_usage": "2.5GB",
            "configuration": {
                "retry_enabled": True,
                "auto_scaling": True,
                "error_reporting": True
            }
        }
        
        # Create a checkpoint before starting
        checkpoint_id = await self.error_system.create_checkpoint(
            checkpoint_name="pre_processing",
            state_data=initial_state,
            metadata={
                "checkpoint_reason": "Before starting complex processing",
                "created_by": "comprehensive_example"
            }
        )
        
        self.logger.info(f"Created pre-processing checkpoint: {checkpoint_id}")
        
        # Attempt the complex task
        try:
            result = await complex_data_processing_task()
            self.logger.info(f"Complex task succeeded: {result}")
            
        except Exception as e:
            self.logger.warning(f"Complex task failed: {str(e)}")
            
            # Handle error through comprehensive system
            recovery_result = await self.error_system.handle_error(
                error=e,
                agent_id="complex_processor",
                task_id="comprehensive_scenario_001",
                context={
                    "operation": "batch_data_processing",
                    "dataset_name": "customer_analytics",
                    "processing_stage": "data_transformation",
                    "batch_number": 42,
                    "retry_budget": 3,
                    "escalation_contact": "data_team@company.com"
                },
                state_data=initial_state,
                recovery_callback=adaptive_recovery_callback
            )
            
            self.logger.info(f"Comprehensive error handling completed:")
            self.logger.info(f"  - Success: {recovery_result['success']}")
            self.logger.info(f"  - Recovery Result: {recovery_result['recovery_result']}")
            self.logger.info(f"  - Error ID: {recovery_result.get('error_id', 'N/A')}")
            self.logger.info(f"  - Loop Detected: {recovery_result.get('loop_detected', False)}")
            
            if recovery_result.get('escalation_ticket'):
                self.logger.info(f"  - Escalation Ticket: {recovery_result['escalation_ticket']}")
        
        # Get comprehensive system status
        system_status = await self.error_system.get_system_status()
        
        self.logger.info("\n=== Final System Status ===")
        self.logger.info(f"Total Errors Handled: {system_status['system_stats']['total_errors_handled']}")
        self.logger.info(f"Successful Recoveries: {system_status['system_stats']['successful_recoveries']}")
        self.logger.info(f"Escalated Errors: {system_status['system_stats']['escalated_errors']}")
        self.logger.info(f"Loop Detections: {system_status['system_stats']['loop_detections']}")
        self.logger.info(f"Agents Paused: {system_status['system_stats']['agents_paused']}")
        
        # Show component status
        components = system_status['components']
        self.logger.info("\nComponent Status:")
        for component_name, status in components.items():
            if status:
                active_items = status.get('active_retries', status.get('cached_snapshots', 
                                         status.get('active_tickets', status.get('active_agents', 0))))
                self.logger.info(f"  - {component_name}: {active_items} active items")
    
    async def run_all_examples(self):
        """Run all examples in sequence."""
        
        await self.initialize_system()
        
        examples = [
            self.example_1_basic_error_handling,
            self.example_2_custom_error_types,
            self.example_3_retry_strategies,
            self.example_4_rollback_and_checkpoints,
            self.example_5_escalation_management,
            # self.example_6_loop_detection, # Commented out
            # self.example_7_history_and_versioning, # Commented out
            self.example_8_comprehensive_scenario
        ]
        
        for example in examples:
            try:
                await example()
                await asyncio.sleep(0.5)  # Brief pause between examples
            except Exception as e:
                self.logger.error(f"Example {example.__name__} failed: {str(e)}")
        
        # Cleanup
        await self.error_system.shutdown()
        self.logger.info("\n=== All Examples Completed ===")


async def main():
    """Main function to run examples."""
    
    examples = ErrorHandlingExamples()
    
    # Run all examples
    await examples.run_all_examples()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
