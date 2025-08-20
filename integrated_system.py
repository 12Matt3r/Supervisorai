#!/usr/bin/env python3
"""
Comprehensive Reporting and Integration System for Supervisor Agent
Integrates all reporting components and provides unified interface.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all reporting components
from .alert_system import ComprehensiveAlertSystem, Alert
from .report_generator import PeriodicReportGenerator, TaskReport
from .audit_system import ComprehensiveAuditSystem, AuditEventType, AuditLevel
from .confidence_system import ConfidenceReportingSystem
from .pattern_system import ComprehensivePatternSystem
from .dashboard_system import ComprehensiveDashboardSystem

@dataclass
class IntegratedReportingConfig:
    """Configuration for integrated reporting system"""
    # Storage configurations
    base_output_dir: str = "reporting_output"
    audit_log_file: str = "audit.jsonl"
    audit_db_file: str = "audit.db"
    confidence_data_file: str = "confidence_data.json"
    patterns_file: str = "patterns.json"
    knowledge_base_file: str = "knowledge_base.json"
    
    # Alert configurations
    alert_config: Dict[str, Any] = None
    
    # Report generation settings
    auto_report_enabled: bool = True
    report_frequency_hours: int = 24
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_port: int = 5000
    
    # Integration settings
    real_time_updates: bool = True
    background_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.alert_config is None:
            self.alert_config = {
                'deduplication_window': 30,
                'email': {
                    'smtp_host': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'supervisor@example.com',
                    'password': 'app_password',
                    'recipients': ['admin@example.com']
                }
            }

@dataclass
class SystemIntegrationEvent:
    """Event for system integration"""
    event_id: str
    timestamp: str
    event_type: str
    source_system: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

class EventRouter:
    """Routes events between different reporting systems"""
    
    def __init__(self, systems: Dict[str, Any]):
        self.systems = systems
        self.logger = logging.getLogger(__name__)
        self.event_handlers = self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> Dict[str, List[callable]]:
        """Setup event handlers for different event types"""
        handlers = {
            'task_started': [self._handle_task_event, self._handle_audit_event],
            'task_completed': [self._handle_task_event, self._handle_confidence_update, self._handle_audit_event],
            'task_failed': [self._handle_task_event, self._handle_alert_generation, self._handle_audit_event],
            'error_occurred': [self._handle_error_event, self._handle_alert_generation, self._handle_audit_event],
            'confidence_recorded': [self._handle_confidence_event],
            'alert_generated': [self._handle_audit_event],
            'pattern_detected': [self._handle_pattern_event, self._handle_audit_event]
        }
        return handlers
    
    def route_event(self, event: SystemIntegrationEvent):
        """Route event to appropriate handlers"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error handling event {event.event_id} with {handler.__name__}: {e}")
    
    def _handle_task_event(self, event: SystemIntegrationEvent):
        """Handle task-related events"""
        # This would integrate with task tracking systems
        self.logger.debug(f"Handling task event: {event.event_type}")
    
    def _handle_confidence_update(self, event: SystemIntegrationEvent):
        """Handle confidence score updates"""
        if 'confidence_system' in self.systems and 'confidence_score' in event.data:
            confidence_system = self.systems['confidence_system']
            
            # Record outcome if task completed
            if event.event_type == 'task_completed':
                task_id = event.data.get('task_id')
                if task_id:
                    success = event.data.get('success', True)
                    confidence_system.record_task_outcome(task_id, success)
    
    def _handle_alert_generation(self, event: SystemIntegrationEvent):
        """Handle alert generation"""
        if 'alert_system' in self.systems:
            alert_system = self.systems['alert_system']
            
            severity = 'error' if event.event_type == 'error_occurred' else 'warning'
            title = f"{event.event_type.replace('_', ' ').title()}"
            message = event.data.get('message', f"Event: {event.event_type}")
            
            alert_system.send_alert(
                severity=severity,
                title=title,
                message=message,
                source=event.source_system,
                metadata=event.data
            )
    
    def _handle_audit_event(self, event: SystemIntegrationEvent):
        """Handle audit logging"""
        if 'audit_system' in self.systems:
            audit_system = self.systems['audit_system']
            
            # Map event types to audit event types
            audit_event_type = {
                'task_started': AuditEventType.TASK_STARTED,
                'task_completed': AuditEventType.TASK_COMPLETED,
                'task_failed': AuditEventType.TASK_FAILED,
                'error_occurred': AuditEventType.ERROR_OCCURRED,
                'alert_generated': AuditEventType.ALERT_GENERATED,
                'pattern_detected': AuditEventType.SYSTEM_STATE_CHANGE
            }.get(event.event_type, AuditEventType.SYSTEM_STATE_CHANGE)
            
            audit_level = AuditLevel.ERROR if 'error' in event.event_type or 'failed' in event.event_type else AuditLevel.INFO
            
            audit_system.log(
                event_type=audit_event_type,
                level=audit_level,
                source=event.source_system,
                message=event.data.get('message', f"Event: {event.event_type}"),
                metadata=event.data,
                correlation_id=event.correlation_id
            )
    
    def _handle_error_event(self, event: SystemIntegrationEvent):
        """Handle error events for pattern detection"""
        # This would feed into pattern detection
        self.logger.debug(f"Handling error event for pattern detection: {event.event_id}")
    
    def _handle_confidence_event(self, event: SystemIntegrationEvent):
        """Handle confidence recording events"""
        if 'confidence_system' in self.systems:
            confidence_system = self.systems['confidence_system']
            
            confidence_system.record_confidence(
                task_id=event.data.get('task_id', ''),
                agent_id=event.data.get('agent_id', ''),
                decision_type=event.data.get('decision_type', 'unknown'),
                confidence=event.data.get('confidence_score', 0.5),
                metadata=event.data
            )
    
    def _handle_pattern_event(self, event: SystemIntegrationEvent):
        """Handle pattern detection events"""
        # This could trigger knowledge base updates
        self.logger.debug(f"Handling pattern event: {event.event_id}")

class BackgroundProcessor:
    """Handles background processing tasks"""
    
    def __init__(self, systems: Dict[str, Any], config: IntegratedReportingConfig):
        self.systems = systems
        self.config = config
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start background processing"""
        self.running = True
        
        # Start periodic tasks
        if self.config.auto_report_enabled:
            threading.Thread(target=self._periodic_report_generation, daemon=True).start()
        
        if self.config.real_time_updates:
            threading.Thread(target=self._real_time_pattern_analysis, daemon=True).start()
        
        self.logger.info("Background processor started")
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Background processor stopped")
    
    def _periodic_report_generation(self):
        """Generate periodic reports"""
        while self.running:
            try:
                self.logger.info("Starting periodic report generation")
                
                # Generate comprehensive report
                if 'report_generator' in self.systems:
                    report_generator = self.systems['report_generator']
                    
                    # Create demo tasks for report (in real implementation, get from task system)
                    demo_tasks = self._create_demo_tasks()
                    
                    end_time = datetime.now().isoformat()
                    start_time = (datetime.now() - timedelta(hours=self.config.report_frequency_hours)).isoformat()
                    
                    report_generator.generate_and_save_report(
                        demo_tasks, start_time, end_time, "markdown"
                    )
                
                # Sleep until next report cycle
                time.sleep(self.config.report_frequency_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Error in periodic report generation: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def _real_time_pattern_analysis(self):
        """Perform real-time pattern analysis"""
        while self.running:
            try:
                if 'pattern_system' in self.systems and 'audit_system' in self.systems:
                    audit_system = self.systems['audit_system']
                    pattern_system = self.systems['pattern_system']
                    
                    # Get recent events for pattern analysis
                    recent_events = audit_system.search(
                        start_time=(datetime.now() - timedelta(hours=1)).isoformat(),
                        limit=1000
                    )
                    
                    # Convert to format expected by pattern system
                    events_for_analysis = [
                        {
                            'timestamp': event.timestamp,
                            'event_type': event.event_type,
                            'level': event.level,
                            'source': event.source,
                            'message': event.message,
                            'metadata': event.metadata
                        }
                        for event in recent_events
                    ]
                    
                    # Analyze patterns
                    if events_for_analysis:
                        analysis = pattern_system.analyze_events(events_for_analysis)
                        
                        if analysis.new_patterns:
                            self.logger.info(f"Detected {len(analysis.new_patterns)} new patterns")
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in real-time pattern analysis: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _create_demo_tasks(self) -> List[TaskReport]:
        """Create demo task reports (placeholder)"""
        # In real implementation, this would get tasks from the actual task system
        return []

class IntegratedReportingSystem:
    """Main integrated reporting and feedback system"""
    
    def __init__(self, config: Optional[IntegratedReportingConfig] = None):
        self.config = config or IntegratedReportingConfig()
        
        # Setup directories
        self.output_dir = Path(self.config.base_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all subsystems
        self.systems = self._initialize_systems()
        
        # Setup event routing
        self.event_router = EventRouter(self.systems)
        
        # Setup background processing
        self.background_processor = BackgroundProcessor(self.systems, self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Integrated reporting system initialized")
    
    def _initialize_systems(self) -> Dict[str, Any]:
        """Initialize all reporting subsystems"""
        systems = {}
        
        try:
            # Alert system
            systems['alert_system'] = ComprehensiveAlertSystem(
                self.config.alert_config
            )
            
            # Audit system
            systems['audit_system'] = ComprehensiveAuditSystem(
                str(self.output_dir / self.config.audit_log_file),
                str(self.output_dir / self.config.audit_db_file)
            )
            
            # Report generator
            systems['report_generator'] = PeriodicReportGenerator(
                str(self.output_dir / "reports")
            )
            
            # Confidence system
            systems['confidence_system'] = ConfidenceReportingSystem(
                str(self.output_dir / self.config.confidence_data_file)
            )
            
            # Pattern system
            systems['pattern_system'] = ComprehensivePatternSystem(
                str(self.output_dir / self.config.patterns_file),
                str(self.output_dir / self.config.knowledge_base_file)
            )
            
            # Dashboard system (if enabled)
            if self.config.dashboard_enabled:
                systems['dashboard_system'] = ComprehensiveDashboardSystem(
                    systems, 
                    {'dashboard_port': self.config.dashboard_port}
                )
            
            self.logger.info(f"Initialized {len(systems)} reporting systems")
            
        except Exception as e:
            self.logger.error(f"Error initializing systems: {e}")
            raise
        
        return systems
    
    def start(self):
        """Start the integrated reporting system"""
        self.logger.info("Starting integrated reporting system")
        
        # Start background processing
        if self.config.background_processing:
            self.background_processor.start()
        
        # Start dashboard if enabled
        if self.config.dashboard_enabled and 'dashboard_system' in self.systems:
            dashboard_thread = threading.Thread(
                target=self.systems['dashboard_system'].start_dashboard,
                kwargs={'debug': False},
                daemon=True
            )
            dashboard_thread.start()
            self.logger.info(f"Dashboard started on port {self.config.dashboard_port}")
    
    def stop(self):
        """Stop the integrated reporting system"""
        self.logger.info("Stopping integrated reporting system")
        
        # Stop background processing
        if self.config.background_processing:
            self.background_processor.stop()
        
        # Shutdown audit system
        if 'audit_system' in self.systems:
            self.systems['audit_system'].shutdown()
    
    def log_event(self, event_type: str, source: str, message: str, 
                  data: Optional[Dict[str, Any]] = None,
                  correlation_id: Optional[str] = None) -> str:
        """Log an event and route it through the system"""
        event_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source}"
        
        # Create integration event
        integration_event = SystemIntegrationEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            source_system=source,
            data=data or {},
            correlation_id=correlation_id
        )
        
        # Route through event system
        self.event_router.route_event(integration_event)
        
        return event_id
    
    def generate_comprehensive_report(self, output_format: str = "all") -> Dict[str, str]:
        """Generate comprehensive report from all systems"""
        report_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Generate reports from each system
            if output_format in ['markdown', 'all']:
                # Task performance report
                if 'report_generator' in self.systems:
                    demo_tasks = []  # In real implementation, get from task system
                    end_time = datetime.now().isoformat()
                    start_time = (datetime.now() - timedelta(days=1)).isoformat()
                    
                    report_file = self.systems['report_generator'].generate_and_save_report(
                        demo_tasks, start_time, end_time, "markdown"
                    )
                    report_files['task_report'] = report_file
            
            if output_format in ['json', 'all']:
                # Confidence analysis
                if 'confidence_system' in self.systems:
                    analysis = self.systems['confidence_system'].analyze_confidence()
                    confidence_file = str(self.output_dir / f"confidence_analysis_{timestamp}.json")
                    self.systems['confidence_system'].export_analysis(
                        analysis, confidence_file, "json"
                    )
                    report_files['confidence_analysis'] = confidence_file
                
                # Pattern analysis
                if 'pattern_system' in self.systems:
                    pattern_file = str(self.output_dir / f"patterns_{timestamp}.json")
                    self.systems['pattern_system'].export_patterns(pattern_file, "json")
                    report_files['patterns'] = pattern_file
                
                # Audit events
                if 'audit_system' in self.systems:
                    audit_file = str(self.output_dir / f"audit_events_{timestamp}.json")
                    self.systems['audit_system'].export_events(
                        audit_file, "json", limit=10000
                    )
                    report_files['audit_events'] = audit_file
            
            # Dashboard report
            if 'dashboard_system' in self.systems:
                dashboard_file = str(self.output_dir / f"dashboard_report_{timestamp}.html")
                self.systems['dashboard_system'].generate_static_report(dashboard_file)
                report_files['dashboard'] = dashboard_file
            
            self.logger.info(f"Generated comprehensive report with {len(report_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
        
        return report_files
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all systems"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'systems': {},
            'config': asdict(self.config),
            'background_processing': self.config.background_processing and self.background_processor.running
        }
        
        for system_name, system in self.systems.items():
            try:
                if system_name == 'audit_system':
                    stats = system.get_stats()
                    status['systems'][system_name] = {
                        'status': 'active',
                        'total_events': stats.get('total_events', 0),
                        'unique_sessions': stats.get('unique_sessions', 0)
                    }
                elif system_name == 'confidence_system':
                    stats = system.get_entry_statistics()
                    status['systems'][system_name] = {
                        'status': 'active',
                        'total_entries': stats.get('total_entries', 0),
                        'outcome_coverage': stats.get('outcome_coverage', 0)
                    }
                elif system_name == 'pattern_system':
                    insights = system.get_pattern_insights()
                    status['systems'][system_name] = {
                        'status': 'active',
                        'total_patterns': insights.get('total_patterns', 0)
                    }
                elif system_name == 'alert_system':
                    stats = system.get_alert_stats()
                    status['systems'][system_name] = {
                        'status': 'active',
                        'total_alerts': stats.get('total_alerts', 0),
                        'unresolved': stats.get('unresolved', 0)
                    }
                else:
                    status['systems'][system_name] = {'status': 'active'}
            except Exception as e:
                status['systems'][system_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def export_complete_system_state(self, output_file: str) -> str:
        """Export complete system state"""
        try:
            complete_state = {
                'export_timestamp': datetime.now().isoformat(),
                'system_status': self.get_system_status(),
                'reports': self.generate_comprehensive_report("json")
            }
            
            with open(output_file, 'w') as f:
                json.dump(complete_state, f, indent=2, default=str)
            
            self.logger.info(f"Complete system state exported to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting system state: {e}")
            return ""

# Demo and testing functions
def create_demo_integrated_system() -> IntegratedReportingSystem:
    """Create demo integrated reporting system"""
    config = IntegratedReportingConfig(
        base_output_dir="demo_reporting_output",
        dashboard_enabled=True,
        dashboard_port=5001,  # Different port for demo
        background_processing=False  # Disable for demo
    )
    
    return IntegratedReportingSystem(config)

def run_demo_scenario(system: IntegratedReportingSystem):
    """Run demo scenario with various events"""
    # Start the system
    system.start()
    
    # Log various events
    correlation_id = "demo_task_001"
    
    # Task started
    system.log_event(
        'task_started',
        'demo_agent',
        'Starting demo task processing',
        {
            'task_id': 'demo_task_001',
            'agent_id': 'demo_agent_1',
            'task_type': 'data_processing'
        },
        correlation_id
    )
    
    # Confidence recorded
    system.log_event(
        'confidence_recorded',
        'demo_agent',
        'Recording confidence for decision',
        {
            'task_id': 'demo_task_001',
            'agent_id': 'demo_agent_1',
            'decision_type': 'classification',
            'confidence_score': 0.87
        },
        correlation_id
    )
    
    # Task completed
    system.log_event(
        'task_completed',
        'demo_agent',
        'Demo task completed successfully',
        {
            'task_id': 'demo_task_001',
            'agent_id': 'demo_agent_1',
            'success': True,
            'duration_seconds': 45.2
        },
        correlation_id
    )
    
    # Generate comprehensive report
    reports = system.generate_comprehensive_report()
    print(f"\nGenerated reports: {reports}")
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem status: {json.dumps(status, indent=2, default=str)}")
    
    # Export system state
    state_file = system.export_complete_system_state("demo_system_state.json")
    print(f"\nSystem state exported to: {state_file}")
    
    return system

if __name__ == '__main__':
    # Demo usage
    print("Creating integrated reporting system...")
    integrated_system = create_demo_integrated_system()
    
    print("Running demo scenario...")
    integrated_system = run_demo_scenario(integrated_system)
    
    print("\n" + "="*50)
    print("INTEGRATED REPORTING SYSTEM DEMO COMPLETE")
    print("="*50)
    print(f"Dashboard available at: http://localhost:5001")
    print(f"Output directory: {integrated_system.output_dir}")
    print("\nPress Ctrl+C to stop the system")
    
    try:
        # Keep running for demo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping integrated reporting system...")
        integrated_system.stop()
        print("System stopped.")
