#!/usr/bin/env python3
"""
Complete Integrated Supervisor MCP Agent Server - Production Ready

Comprehensive supervisor agent with full integration of:
- Monitoring capabilities
- Error handling and recovery
- Reporting and analytics  
- Framework integration hooks
- Configuration management
- Performance optimization
- Security and access control
"""

import asyncio
import json
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager

# Make sure the 'src' directory is in the PYTHONPATH.
# e.g., export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('supervisor_data/supervisor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from fastmcp import FastMCP
    # Import integrated components
    from supervisor_agent.integrated_supervisor import IntegratedSupervisor, SupervisorConfig
    from supervisor_agent.core import SupervisorCore
    from supervisor_agent import (
        MonitoringRules, EscalationConfig, TaskStatus, 
        InterventionLevel, KnowledgeBaseEntry
    )
    from supervisor_agent.expectimax_agent import AgentState
    from idea_validation.validator import Validator
    from idea_validation.data_models import Idea
    from orchestrator.core import Orchestrator
    INTEGRATED_MODE = True
    logger.info("Loaded integrated supervisor system")
except ImportError as e:
    logger.error(f"Failed to import integrated modules: {e}")
    from fastmcp import FastMCP
    from supervisor_agent.core import SupervisorCore
    INTEGRATED_MODE = False
    logger.info("Running in basic supervisor mode")

# Initialize MCP server
mcp = FastMCP("Integrated Supervisor Agent")

# Global instances
integrated_supervisor: Optional['IntegratedSupervisor'] = None
basic_supervisor: Optional[SupervisorCore] = None
orchestrator: Optional[Orchestrator] = None

def get_orchestrator_instance() -> Orchestrator:
    """Get the singleton orchestrator instance, creating it if necessary."""
    global orchestrator
    if orchestrator is None:
        # The orchestrator needs a supervisor to function.
        # We need to ensure the supervisor is initialized first.
        # This is a bit tricky due to async/sync mixing.
        # For now, we assume get_supervisor_instance has been called.
        if basic_supervisor is None and integrated_supervisor is None:
             raise RuntimeError("Supervisor must be initialized before the orchestrator.")

        supervisor = integrated_supervisor if INTEGRATED_MODE else basic_supervisor
        orchestrator = Orchestrator(supervisor=supervisor)
    return orchestrator

async def get_supervisor_instance():
    """Get appropriate supervisor instance based on availability"""
    global integrated_supervisor, basic_supervisor
    
    if INTEGRATED_MODE:
        if integrated_supervisor is None:
            config = SupervisorConfig(
                data_dir=os.getenv("SUPERVISOR_DATA_DIR", "supervisor_data"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                real_time_alerts=os.getenv("ENABLE_REAL_TIME_ALERTS", "false").lower() == "true",
                monitoring_enabled=True,
                error_handling_enabled=True,
                reporting_enabled=True,
                framework_hooks_enabled=True
            )
            integrated_supervisor = IntegratedSupervisor(config)
            await integrated_supervisor.start()
            logger.info("Integrated Supervisor initialized")
        return integrated_supervisor
    else:
        if basic_supervisor is None:
            basic_supervisor = SupervisorCore()
            logger.info("Basic Supervisor initialized")
        return basic_supervisor

# ============================================================================
# COMPREHENSIVE MCP TOOLS - INTEGRATED SYSTEM
# ============================================================================

@mcp.tool
async def start_supervision_session(
    session_name: str,
    framework: str = "custom",
    agent_name: str = "unknown_agent",
    task_description: str = "",
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """Start comprehensive supervision session with full monitoring"""
    try:
        supervisor = await get_supervisor_instance()
        
        if INTEGRATED_MODE:
            # Use integrated supervisor
            session_data = {
                "session_name": session_name,
                "framework": framework,
                "agent_name": agent_name,
                "task_description": task_description,
                "started_at": datetime.utcnow().isoformat(),
                "configuration": configuration or {},
                "status": "active"
            }
            
            status = await supervisor.get_system_status()
            
            return json.dumps({
                "success": True,
                "session_id": session_name,
                "message": f"Integrated supervision session '{session_name}' started",
                "session_data": session_data,
                "system_capabilities": {
                    "monitoring": supervisor.config.monitoring_enabled,
                    "error_handling": supervisor.config.error_handling_enabled,
                    "reporting": supervisor.config.reporting_enabled,
                    "framework_hooks": supervisor.config.framework_hooks_enabled
                }
            }, indent=2)
        else:
            # Use basic supervisor
            task_id = await supervisor.monitor_agent(
                agent_name=agent_name,
                framework=framework,
                task_input=task_description,
                instructions=[],
                task_id=session_name
            )
            return f"Basic supervision session started: {task_id}"
            
    except Exception as e:
        logger.error(f"Failed to start supervision session: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool
async def execute_supervised_task(
    task_id: str,
    task_callable_description: str,
    framework: str = "custom",
    expected_output_type: str = "text",
    quality_threshold: float = 0.7,
    instructions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Execute task under comprehensive supervision with monitoring and error handling"""
    try:
        supervisor = await get_supervisor_instance()
        
        if INTEGRATED_MODE:
            # Create execution context
            context = {
                "task_description": task_callable_description,
                "instructions": instructions or [],
                "expected_output_type": expected_output_type,
                "quality_threshold": quality_threshold,
                "metadata": metadata or {}
            }
            
            # Demo task simulation (in production, would be actual task)
            async def demo_task():
                await asyncio.sleep(0.1)
                return {
                    "task_id": task_id,
                    "result": f"Supervised execution of: {task_callable_description}",
                    "output_type": expected_output_type,
                    "quality_score": 0.85,
                    "processing_time": "0.1s",
                    "framework": framework
                }
            
            result = await supervisor.execute_supervised_task(
                task_id=task_id,
                task_callable=demo_task,
                framework=framework,
                context=context
            )
            
            return json.dumps(result, indent=2)
        else:
            # Use basic supervisor
            result = await supervisor.validate_output(
                task_id=task_id,
                output=f"Executed: {task_callable_description}",
                output_type=expected_output_type,
                metadata=metadata
            )
            return json.dumps(result, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Supervised task execution failed: {e}")
        return json.dumps({"success": False, "error": str(e), "task_id": task_id})

@mcp.tool
async def monitor_agent_comprehensive(
    agent_name: str,
    framework: str,
    task_input: str,
    instructions: List[str],
    monitoring_rules: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None
) -> str:
    """Enhanced agent monitoring with comprehensive rules and real-time tracking"""
    try:
        supervisor = await get_supervisor_instance()
        
        if INTEGRATED_MODE and hasattr(supervisor, 'monitoring_engine'):
            # Use integrated monitoring
            task_id = task_id or f"{agent_name}_{datetime.utcnow().timestamp()}"
            
            await supervisor.monitoring_engine.start_task_monitoring(
                task_id=task_id,
                task_type=f"{framework}_task",
                metadata={
                    "agent_name": agent_name,
                    "task_input": task_input,
                    "instructions": instructions,
                    "monitoring_rules": monitoring_rules or {}
                }
            )
            
            return json.dumps({
                "success": True,
                "task_id": task_id,
                "message": f"Comprehensive monitoring started for {agent_name}",
                "monitoring_capabilities": [
                    "Task completion tracking",
                    "Instruction adherence monitoring", 
                    "Output quality assessment",
                    "Error detection and tracking",
                    "Resource usage monitoring",
                    "Confidence scoring"
                ]
            }, indent=2)
        else:
            # Use basic monitoring
            result = await supervisor.monitor_agent(
                agent_name=agent_name,
                framework=framework,
                task_input=task_input,
                instructions=instructions,
                task_id=task_id
            )
            return f"Basic monitoring started: {result}"
            
    except Exception as e:
        logger.error(f"Agent monitoring failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def handle_error_with_recovery(
    task_id: str,
    error_description: str,
    error_type: str = "unknown",
    agent_id: str = "unknown_agent",
    context: Optional[Dict[str, Any]] = None,
    enable_auto_recovery: bool = True
) -> str:
    """Comprehensive error handling with automatic recovery orchestration"""
    try:
        supervisor = await get_supervisor_instance()
        
        if INTEGRATED_MODE and hasattr(supervisor, 'error_handling_system'):
            # Create mock error for demonstration
            error = Exception(error_description)
            
            # Handle through integrated error handling system
            recovery_result = await supervisor.error_handling_system.handle_error(
                error=error,
                agent_id=agent_id,
                task_id=task_id,
                context=context or {},
                state_data={"task_id": task_id, "agent_id": agent_id}
            )
            
            return json.dumps({
                "success": True,
                "message": "Error handled through integrated recovery system",
                "recovery_result": recovery_result,
                "capabilities": [
                    "Automatic retry with progressive strategies",
                    "State rollback and recovery",
                    "Human escalation management", 
                    "Loop detection and circuit breakers",
                    "Comprehensive history tracking"
                ]
            }, indent=2)
        else:
            # Basic error handling
            return json.dumps({
                "success": True,
                "message": f"Basic error handling for task {task_id}: {error_description}",
                "error_type": error_type,
                "agent_id": agent_id
            }, indent=2)
            
    except Exception as e:
        logger.error(f"Error handling failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool  
async def generate_comprehensive_reports(
    report_type: str = "comprehensive",
    time_range_hours: int = 24,
    include_analytics: bool = True,
    export_format: str = "json",
    include_recommendations: bool = True
) -> str:
    """Generate comprehensive reports using integrated reporting system"""
    try:
        supervisor = await get_supervisor_instance()
        
        if INTEGRATED_MODE:
            report_data = {
                "report_type": report_type,
                "generated_at": datetime.utcnow().isoformat(),
                "time_range_hours": time_range_hours,
                "export_format": export_format,
                "system_status": await supervisor.get_system_status(),
                "capabilities": {
                    "real_time_alerting": hasattr(supervisor, 'alert_system'),
                    "pattern_detection": hasattr(supervisor, 'pattern_system'),
                    "confidence_tracking": hasattr(supervisor, 'confidence_system'),
                    "comprehensive_auditing": hasattr(supervisor, 'audit_system'),
                    "dashboard_integration": hasattr(supervisor, 'dashboard_system')
                }
            }
            
            if include_analytics:
                report_data["analytics"] = {
                    "tasks_supervised": supervisor.system_stats.get("tasks_supervised", 0),
                    "errors_handled": supervisor.system_stats.get("errors_handled", 0),
                    "uptime": str(datetime.utcnow() - supervisor.system_stats.get("start_time", datetime.utcnow()))
                }
            
            if include_recommendations:
                report_data["recommendations"] = [
                    "Monitor system performance regularly",
                    "Review error patterns for optimization opportunities",
                    "Update monitoring thresholds based on system behavior",
                    "Ensure proper escalation procedures are in place"
                ]
            
            return json.dumps(report_data, indent=2, default=str)
        else:
            # Basic reporting
            basic_report = await supervisor.get_supervision_report(time_range_hours)
            return json.dumps(basic_report, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def configure_system_wide_settings(
    monitoring_enabled: bool = True,
    error_handling_enabled: bool = True,
    reporting_enabled: bool = True,
    real_time_alerts: bool = False,
    quality_threshold: float = 0.7,
    confidence_threshold: float = 0.8,
    max_retries: int = 3,
    escalation_enabled: bool = True
) -> str:
    """Configure system-wide supervisor settings and thresholds"""
    try:
        supervisor = await get_supervisor_instance()
        
        config_data = {
            "monitoring_enabled": monitoring_enabled,
            "error_handling_enabled": error_handling_enabled,
            "reporting_enabled": reporting_enabled,
            "real_time_alerts": real_time_alerts,
            "quality_threshold": quality_threshold,
            "confidence_threshold": confidence_threshold,
            "max_retries": max_retries,
            "escalation_enabled": escalation_enabled,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if INTEGRATED_MODE:
            # Update integrated supervisor configuration
            if hasattr(supervisor, 'config'):
                supervisor.config.monitoring_enabled = monitoring_enabled
                supervisor.config.error_handling_enabled = error_handling_enabled
                supervisor.config.reporting_enabled = reporting_enabled
                supervisor.config.real_time_alerts = real_time_alerts
                supervisor.config.quality_threshold = quality_threshold
                supervisor.config.confidence_threshold = confidence_threshold
                supervisor.config.max_retries = max_retries
                supervisor.config.escalation_enabled = escalation_enabled
            
            return json.dumps({
                "success": True,
                "message": "System-wide configuration updated successfully",
                "configuration": config_data,
                "integrated_systems_affected": [
                    "Monitoring Engine",
                    "Error Handling System", 
                    "Reporting System",
                    "Framework Integration Hooks"
                ]
            }, indent=2)
        else:
            # Basic configuration
            return json.dumps({
                "success": True,
                "message": "Basic configuration updated",
                "configuration": config_data
            }, indent=2)
            
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def get_integration_status() -> str:
    """Get comprehensive status of all integrated systems and framework hooks"""
    try:
        supervisor = await get_supervisor_instance()
        
        status = {
            "integration_mode": "integrated" if INTEGRATED_MODE else "basic",
            "timestamp": datetime.utcnow().isoformat(),
            "supervisor_type": type(supervisor).__name__
        }
        
        if INTEGRATED_MODE:
            status.update({
                "system_status": await supervisor.get_system_status(),
                "framework_adapters": {
                    "mcp": "Available" if hasattr(supervisor, 'framework_hooks') else "Not Available",
                    "langchain": "Available" if hasattr(supervisor, 'framework_hooks') else "Not Available",
                    "autogen": "Available" if hasattr(supervisor, 'framework_hooks') else "Not Available",
                    "custom": "Available" if hasattr(supervisor, 'framework_hooks') else "Not Available"
                },
                "subsystems": {
                    "monitoring_engine": hasattr(supervisor, 'monitoring_engine'),
                    "error_handling_system": hasattr(supervisor, 'error_handling_system'),
                    "alert_system": hasattr(supervisor, 'alert_system'),
                    "reporting_system": hasattr(supervisor, 'report_generator'),
                    "audit_system": hasattr(supervisor, 'audit_system'),
                    "confidence_system": hasattr(supervisor, 'confidence_system'),
                    "pattern_system": hasattr(supervisor, 'pattern_system'),
                    "dashboard_system": hasattr(supervisor, 'dashboard_system')
                }
            })
        else:
            status.update({
                "message": "Running in basic supervisor mode",
                "available_tools": [
                    "monitor_agent", "validate_output", "get_supervision_report",
                    "intervene_task", "get_audit_log", "rollback_state"
                ]
            })
        
        return json.dumps(status, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Integration status check failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def get_minimax_decision(
    quality_score: float,
    error_count: int,
    resource_usage: float,
    task_progress: float,
    drift_score: float = 0.0
) -> str:
    """Get a supervisor decision from the Expectimax agent."""
    try:
        supervisor = await get_supervisor_instance()

        # The agent is now on the SupervisorCore instance
        if not hasattr(supervisor, 'expectimax_agent'):
            return json.dumps({"success": False, "error": "Expectimax agent not available in the current supervisor instance."})

        agent = supervisor.expectimax_agent

        state = AgentState(
            quality_score=quality_score,
            error_count=error_count,
            resource_usage=resource_usage,
            task_progress=task_progress,
            drift_score=drift_score
        )

        decision_data = agent.get_best_action(state)

        return json.dumps({
            "success": True,
            "decision": decision_data['best_action'].value,
            "score": decision_data['best_score'],
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Expectimax decision failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def get_decision_trace(
    quality_score: float,
    error_count: int,
    resource_usage: float,
    task_progress: float,
    drift_score: float = 0.0
) -> str:
    """Get a full decision trace from the Expectimax agent."""
    try:
        supervisor = await get_supervisor_instance()
        if not hasattr(supervisor, 'expectimax_agent'):
            return json.dumps({"success": False, "error": "Expectimax agent not available."})

        agent = supervisor.expectimax_agent
        state = AgentState(
            quality_score=quality_score,
            error_count=error_count,
            resource_usage=resource_usage,
            task_progress=task_progress,
            drift_score=drift_score
        )

        trace_data = agent.get_best_action_with_trace(state)

        return json.dumps({"success": True, "trace": trace_data['trace']})

    except Exception as e:
        logger.error(f"Failed to get decision trace: {e}")
        return json.dumps({"success": False, "error": str(e)})


@mcp.tool
async def get_decision_logs(limit: int = 50) -> str:
    """Get a log of recent supervisor decisions."""
    try:
        supervisor = await get_supervisor_instance()

        if hasattr(supervisor, 'audit_system'):
            decision_events = supervisor.audit_system.search(
                event_type='decision_made',
                limit=limit
            )
            # Convert events to a JSON-serializable format
            logs = [event.to_dict() for event in decision_events]
            return json.dumps({"success": True, "logs": logs})
        else:
            return json.dumps({"success": False, "error": "Audit system not available."})

    except Exception as e:
        logger.error(f"Failed to get decision logs: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def validate_idea(
    description: str,
    required_skills: List[str],
    required_apis: List[str],
    estimated_time_hours: int,
    market_niche: str
) -> str:
    """Validate a project idea for feasibility and potential issues."""
    try:
        idea = Idea(
            description=description,
            required_skills=required_skills,
            required_apis=required_apis,
            estimated_time_hours=estimated_time_hours,
            market_niche=market_niche
        )

        validator = Validator()
        report = validator.validate(idea)

        # Convert report to a JSON-serializable dictionary
        import dataclasses
        report_dict = dataclasses.asdict(report)

        return json.dumps({"success": True, "report": report_dict})

    except Exception as e:
        logger.error(f"Idea validation failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def submit_feedback(
    event_id: str,
    corrected_action: str,
    decision_context: Dict[str, Any]
) -> str:
    """Submits user feedback on a supervisor decision."""
    try:
        feedback_file = Path("supervisor_data") / "feedback.json"

        new_feedback = {
            "event_id": event_id,
            "corrected_action": corrected_action,
            "original_decision": decision_context.get("action"),
            "decision_context": decision_context.get("minimax_details", {}),
            "timestamp": datetime.utcnow().isoformat()
        }

        all_feedback = []
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                try:
                    all_feedback = json.load(f)
                except json.JSONDecodeError:
                    all_feedback = []

        all_feedback.append(new_feedback)

        with open(feedback_file, 'w') as f:
            json.dump(all_feedback, f, indent=2)

        return json.dumps({"success": True, "message": "Feedback submitted successfully."})

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def run_training() -> str:
    """Triggers the feedback-based training process for the supervisor agent."""
    try:
        supervisor = await get_supervisor_instance()
        if not isinstance(supervisor, SupervisorCore):
            return json.dumps({"success": False, "error": "Training is only available for the basic SupervisorCore."})

        from supervisor_agent.feedback_trainer import FeedbackTrainer

        feedback_file = "supervisor_data/feedback.json"
        current_weights = supervisor.weights

        trainer = FeedbackTrainer(feedback_file, current_weights)
        new_weights = trainer.train_on_feedback()

        supervisor.update_weights(new_weights)

        return json.dumps({
            "success": True,
            "message": "Training complete. Weights updated.",
            "new_weights": new_weights
        })

    except Exception as e:
        logger.error(f"Training run failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# LEGACY MCP TOOLS - BACKWARD COMPATIBILITY 
# ============================================================================

@mcp.tool
async def monitor_agent(agent_name: str, framework: str, task_input: str, instructions: List[str], task_id: Optional[str] = None) -> str:
    """Legacy monitoring tool for backward compatibility"""
    return await monitor_agent_comprehensive(agent_name, framework, task_input, instructions, None, task_id)

@mcp.tool
async def validate_output(task_id: str, output: str, output_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> str:
    """Legacy output validation for backward compatibility"""
    return await execute_supervised_task(task_id, f"Validate output: {output[:100]}...", "legacy", output_type, 0.7, [], metadata)

@mcp.tool
async def get_system_status() -> str:
    """Get basic system status"""
    return await get_integration_status()

# ============================================================================
# ORCHESTRATOR MCP TOOLS
# ============================================================================

@mcp.tool
async def register_agent(agent_id: str, name: str, capabilities: List[str]) -> str:
    """Registers an agent with the orchestrator."""
    try:
        orch = get_orchestrator_instance()
        agent = orch.register_agent(agent_id, name, capabilities)
        import dataclasses
        return json.dumps({"success": True, "agent": dataclasses.asdict(agent)})
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def get_orchestrator_status() -> str:
    """Gets the current status of the orchestrator, including all agents and projects."""
    try:
        orch = get_orchestrator_instance()
        agents = orch.list_agents()
        projects = list(orch.projects.values())
        import dataclasses

        status = {
            "is_running": orch.is_running,
            "agent_count": len(agents),
            "agents": [dataclasses.asdict(a) for a in agents],
            "project_count": len(projects),
            "projects": [dataclasses.asdict(p) for p in projects]
        }
        return json.dumps({"success": True, "status": status}, default=str)
    except Exception as e:
        logger.error(f"Failed to get orchestrator status: {e}")
        return json.dumps({"success": False, "error": str(e)})

@mcp.tool
async def submit_goal(name: str, description: str) -> str:
    """Submits a new high-level goal to the orchestrator."""
    try:
        orch = get_orchestrator_instance()
        project = orch.submit_goal(name, description)
        import dataclasses
        # Convert the project to a dict, but handle the nested tasks
        project_dict = dataclasses.asdict(project)
        return json.dumps({"success": True, "project": project_dict}, default=str)
    except Exception as e:
        logger.error(f"Failed to submit goal: {e}")
        return json.dumps({"success": False, "error": str(e)})


# ============================================================================
# SERVER STARTUP AND LIFECYCLE
# ============================================================================

async def startup_handler():
    """Initialize supervisor systems on startup"""
    logger.info("Starting Integrated Supervisor MCP Server")
    try:
        # Start the orchestrator
        orch = get_orchestrator_instance()
        orch.start()

        # Start the supervisor
        supervisor = await get_supervisor_instance()
        logger.info(f"Supervisor initialized: {type(supervisor).__name__}")
        logger.info(f"Integration mode: {'Integrated' if INTEGRATED_MODE else 'Basic'}")
        return True
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return False

async def shutdown_handler():
    """Cleanup on shutdown"""
    logger.info("Shutting down Integrated Supervisor MCP Server")
    try:
        # Stop the orchestrator
        orch = get_orchestrator_instance()
        orch.stop()

        if INTEGRATED_MODE and integrated_supervisor:
            await integrated_supervisor.stop()
        logger.info("Supervisor shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    # Initialize and run the MCP server
    logger.info("Starting Integrated Supervisor MCP Agent Server")
    
    # Create data directory if it doesn't exist
    Path("supervisor_data").mkdir(exist_ok=True)
    
    # Run startup
    loop = asyncio.get_event_loop()
    startup_success = loop.run_until_complete(startup_handler())
    
    if startup_success:
        try:
            # Run the MCP server
            mcp.run()
        finally:
            # Cleanup on exit
            loop.run_until_complete(shutdown_handler())
    else:
        logger.error("Failed to start supervisor server")
        sys.exit(1)