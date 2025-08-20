#!/usr/bin/env python3
"""
Integrated Supervisor System

Unifies all supervisor components into a single, production-ready system:
- Monitoring capabilities
- Error handling and recovery
- Reporting and analytics
- Framework integration hooks
- Configuration management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

# Import all subsystems
from .monitoring import (
    MonitoringEngine,
    TaskCompletionMonitor,
    InstructionAdherenceMonitor,
    OutputQualityMonitor,
    ErrorTracker,
    ResourceUsageMonitor,
    ConfidenceScorer
)

from .error_handling.error_handling_system import SupervisorErrorHandlingSystem
from .error_handling.error_types import SupervisorError, ErrorType

from .reporting.integrated_system import (
    IntegratedReportingConfig,
    SystemIntegrationEvent,
    EventRouter,
    BackgroundProcessor
)

from .reporting.alert_system import ComprehensiveAlertSystem
from .reporting.report_generator import PeriodicReportGenerator
from .reporting.audit_system import ComprehensiveAuditSystem, AuditEventType, AuditLevel
from .reporting.confidence_system import ConfidenceReportingSystem
from .reporting.pattern_system import ComprehensivePatternSystem
from .reporting.dashboard_system import ComprehensiveDashboardSystem

from .supervisor_agent.core import SupervisorCore

@dataclass
class SupervisorConfig:
    """Comprehensive configuration for integrated supervisor"""
    # Storage paths
    data_dir: str = "supervisor_data"
    
    # Monitoring settings
    monitoring_enabled: bool = True
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.8
    max_token_threshold: int = 10000
    max_loop_cycles: int = 50
    max_runtime_minutes: float = 30.0
    
    # Error handling settings
    error_handling_enabled: bool = True
    max_retries: int = 3
    rollback_enabled: bool = True
    escalation_enabled: bool = True
    
    # Reporting settings
    reporting_enabled: bool = True
    real_time_alerts: bool = True
    auto_reports: bool = True
    report_frequency_hours: int = 24
    
    # Framework integration
    framework_hooks_enabled: bool = True
    
    # Performance settings
    background_processing: bool = True
    max_workers: int = 4
    log_level: str = "INFO"

class FrameworkIntegrationHooks:
    """Hooks for integrating with different AI agent frameworks"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
        self.logger = logging.getLogger(__name__)
        
        # Framework-specific adapters
        self.mcp_adapter = MCPFrameworkAdapter(supervisor)
        self.langchain_adapter = LangChainFrameworkAdapter(supervisor)
        self.autogen_adapter = AutoGenFrameworkAdapter(supervisor)
        self.custom_adapter = CustomFrameworkAdapter(supervisor)
    
    def get_adapter(self, framework: str):
        """Get framework-specific adapter"""
        adapters = {
            'mcp': self.mcp_adapter,
            'langchain': self.langchain_adapter,
            'autogen': self.autogen_adapter,
            'custom': self.custom_adapter
        }
        return adapters.get(framework.lower(), self.custom_adapter)

class MCPFrameworkAdapter:
    """MCP-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_tool_call(self, tool_name: str, args: Dict[str, Any], task_id: str):
        """Wrap MCP tool calls with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="mcp_tool_call",
            metadata={"tool_name": tool_name, "args": args}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: self._execute_tool(tool_name, args),
            framework="mcp"
        )
    
    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]):
        """Execute MCP tool with error handling"""
        # Tool execution logic would be implemented here
        # This is a placeholder for demonstration
        return {"result": f"Executed {tool_name} with {args}"}

class LangChainFrameworkAdapter:
    """LangChain-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_chain_execution(self, chain, inputs: Dict[str, Any], task_id: str):
        """Wrap LangChain execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="langchain_execution",
            metadata={"chain_type": str(type(chain)), "inputs": inputs}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: chain.arun(inputs),
            framework="langchain"
        )

class AutoGenFrameworkAdapter:
    """AutoGen-specific integration adapter"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_agent_execution(self, agent, message: str, task_id: str):
        """Wrap AutoGen agent execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="autogen_execution",
            metadata={"agent_name": agent.name, "message": message}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=lambda: agent.generate_reply(message),
            framework="autogen"
        )

class CustomFrameworkAdapter:
    """Adapter for custom agent frameworks"""
    
    def __init__(self, supervisor: 'IntegratedSupervisor'):
        self.supervisor = supervisor
    
    async def wrap_execution(self, task_callable: Callable, task_id: str, metadata: Dict[str, Any] = None):
        """Wrap custom framework execution with supervision"""
        await self.supervisor.monitoring_engine.start_task_monitoring(
            task_id=task_id,
            task_type="custom_execution",
            metadata=metadata or {}
        )
        
        return await self.supervisor.execute_supervised_task(
            task_id=task_id,
            task_callable=task_callable,
            framework="custom"
        )

class IntegratedSupervisor:
    """Main integrated supervisor system that coordinates all components"""
    
    def __init__(self, config: Optional[SupervisorConfig] = None):
        self.config = config or SupervisorConfig()
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Integrated Supervisor System")
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Setup framework integration hooks
        self.framework_hooks = FrameworkIntegrationHooks(self) if self.config.framework_hooks_enabled else None
        
        # System state
        self.is_running = False
        self.tasks = {}
        self.system_stats = {
            "start_time": datetime.utcnow(),
            "tasks_supervised": 0,
            "errors_handled": 0,
            "reports_generated": 0
        }
        
        self.logger.info("Integrated Supervisor System initialized successfully")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'supervisor.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_subsystems(self):
        """Initialize all supervisor subsystems"""
        # Core supervisor
        self.supervisor_core = SupervisorCore()
        
        # Monitoring system
        if self.config.monitoring_enabled:
            self.monitoring_engine = MonitoringEngine()
            self.task_monitor = TaskCompletionMonitor()
            self.instruction_monitor = InstructionAdherenceMonitor()
            self.quality_monitor = OutputQualityMonitor()
            self.error_tracker = ErrorTracker()
            self.resource_monitor = ResourceUsageMonitor()
            self.confidence_scorer = ConfidenceScorer()
        
        # Error handling system
        if self.config.error_handling_enabled:
            self.error_handling_system = SupervisorErrorHandlingSystem(
                storage_path=self.data_dir / "error_handling",
                max_retries=self.config.max_retries,
                escalation_enabled=self.config.escalation_enabled
            )
        
        # Reporting system
        if self.config.reporting_enabled:
            self._initialize_reporting_system()
    
    def _initialize_reporting_system(self):
        """Initialize comprehensive reporting system"""
        output_dir = self.data_dir / "reporting"
        output_dir.mkdir(exist_ok=True)
        
        # Initialize individual reporting components
        self.alert_system = ComprehensiveAlertSystem()
        self.report_generator = PeriodicReportGenerator(str(output_dir))
        self.audit_system = ComprehensiveAuditSystem(
            log_file=str(output_dir / "audit.jsonl"),
            db_file=str(output_dir / "audit.db")
        )
        self.confidence_system = ConfidenceReportingSystem(
            data_file=str(output_dir / "confidence_data.json")
        )
        self.pattern_system = ComprehensivePatternSystem(
            patterns_file=str(output_dir / "patterns.json"),
            knowledge_base_file=str(output_dir / "knowledge_base.json")
        )
        
        if self.config.reporting_enabled:
            self.dashboard_system = ComprehensiveDashboardSystem(
                data_dir=str(output_dir),
                port=5000  # Could be configurable
            )
        
        # Setup event routing
        systems = {
            'alert_system': self.alert_system,
            'audit_system': self.audit_system,
            'confidence_system': self.confidence_system,
            'pattern_system': self.pattern_system,
            'report_generator': self.report_generator
        }
        
        self.event_router = EventRouter(systems)
        
        # Setup background processor
        if self.config.background_processing:
            reporting_config = IntegratedReportingConfig(
                base_output_dir=str(output_dir),
                auto_report_enabled=self.config.auto_reports,
                report_frequency_hours=self.config.report_frequency_hours,
                real_time_updates=self.config.real_time_alerts,
                background_processing=True,
                max_workers=self.config.max_workers
            )
            
            self.background_processor = BackgroundProcessor(systems, reporting_config)
    
    async def start(self):
        """Start the integrated supervisor system"""
        if self.is_running:
            return
        
        self.logger.info("Starting Integrated Supervisor System")
        
        # Start background processing
        if hasattr(self, 'background_processor'):
            self.background_processor.start()
        
        # Start dashboard if enabled
        if hasattr(self, 'dashboard_system') and self.config.reporting_enabled:
            await self.dashboard_system.start_server()
        
        self.is_running = True
        self.logger.info("Integrated Supervisor System started successfully")
    
    async def stop(self):
        """Stop the integrated supervisor system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Integrated Supervisor System")
        
        # Stop background processing
        if hasattr(self, 'background_processor'):
            self.background_processor.stop()
        
        # Stop dashboard if running
        if hasattr(self, 'dashboard_system') and self.config.reporting_enabled:
            await self.dashboard_system.stop_server()
        
        self.is_running = False
        self.logger.info("Integrated Supervisor System stopped")
    
    async def execute_supervised_task(
        self,
        task_id: str,
        task_callable: Callable,
        framework: str = "unknown",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task under full supervision"""
        
        self.system_stats["tasks_supervised"] += 1
        
        try:
            # Start monitoring
            if self.config.monitoring_enabled:
                await self.monitoring_engine.start_task_monitoring(
                    task_id=task_id,
                    task_type=f"{framework}_task",
                    metadata=context or {}
                )
            
            # Execute task with error handling
            if self.config.error_handling_enabled:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, task_callable
                    )
                except Exception as e:
                    # Handle error through error handling system
                    error_result = await self.error_handling_system.handle_error(
                        error=e,
                        agent_id=f"{framework}_agent",
                        task_id=task_id,
                        context=context
                    )
                    
                    self.system_stats["errors_handled"] += 1
                    
                    # Route error event
                    if hasattr(self, 'event_router'):
                        error_event = SystemIntegrationEvent(
                            event_id=f"error_{task_id}_{datetime.utcnow().timestamp()}",
                            timestamp=datetime.utcnow().isoformat(),
                            event_type="error_occurred",
                            source_system="integrated_supervisor",
                            data={
                                "task_id": task_id,
                                "error": str(e),
                                "framework": framework,
                                "error_handling_result": error_result
                            }
                        )
                        self.event_router.route_event(error_event)
                    
                    if not error_result.get("success", False):
                        raise e
                    
                    # If error handling succeeded, try to get result
                    result = error_result.get("recovered_result")
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, task_callable
                )
            
            # Validate output if monitoring enabled
            if self.config.monitoring_enabled and result:
                quality_score = await self.quality_monitor.analyze_output(
                    output=str(result),
                    expected_format="json" if isinstance(result, dict) else "text",
                    task_instructions=context.get("instructions", []) if context else []
                )
                
                confidence_score = self.confidence_scorer.calculate_confidence(
                    task_type=f"{framework}_task",
                    output_quality=quality_score,
                    error_count=0,  # Would be tracked properly in real implementation
                    completion_time=1.0  # Would be calculated properly
                )
            
            # Route success event
            if hasattr(self, 'event_router'):
                success_event = SystemIntegrationEvent(
                    event_id=f"success_{task_id}_{datetime.utcnow().timestamp()}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type="task_completed",
                    source_system="integrated_supervisor",
                    data={
                        "task_id": task_id,
                        "framework": framework,
                        "success": True,
                        "result_type": type(result).__name__
                    }
                )
                self.event_router.route_event(success_event)
            
            return {
                "success": True,
                "result": result,
                "task_id": task_id,
                "framework": framework,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_info": {
                "is_running": self.is_running,
                "uptime": str(datetime.utcnow() - self.system_stats["start_time"]),
                "config": asdict(self.config)
            },
            "statistics": self.system_stats,
            "subsystems": {
                "monitoring": self.config.monitoring_enabled,
                "error_handling": self.config.error_handling_enabled,
                "reporting": self.config.reporting_enabled,
                "framework_hooks": self.config.framework_hooks_enabled
            }
        }
        
        # Add detailed subsystem status
        if hasattr(self, 'error_handling_system'):
            status["error_handling_stats"] = self.error_handling_system.system_stats
        
        return status
    
    @asynccontextmanager
    async def supervision_context(self, task_id: str, framework: str = "unknown"):
        """Context manager for supervised task execution"""
        try:
            # Setup supervision context
            if self.config.monitoring_enabled:
                await self.monitoring_engine.start_task_monitoring(
                    task_id=task_id,
                    task_type=f"{framework}_context"
                )
            
            yield self
            
        finally:
            # Cleanup supervision context
            if self.config.monitoring_enabled:
                await self.monitoring_engine.stop_task_monitoring(task_id)
