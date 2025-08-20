"""
Main Reporting System Integration for Supervisor Agent
Coordinates all reporting components and provides unified interface
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from alerts import AlertManager
from summaries import ReportGenerator
from audit_system import AuditTrailManager
from confidence import ConfidenceReporter
from patterns import PatternTracker
from dashboard import DashboardManager
from export_system import ExportManager


class MockDataSource:
    """Mock data source for demonstration purposes"""
    
    def get_agent_tasks(self, agent_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Mock method to get agent tasks"""
        return [
            {
                'task_id': f'task_{i}',
                'agent_id': agent_id,
                'start_time': (start_time + timedelta(hours=i)).isoformat(),
                'end_time': (start_time + timedelta(hours=i, minutes=30)).isoformat(),
                'status': 'completed' if i % 4 != 0 else 'failed',
                'confidence': 0.8 - (i * 0.05),
                'errors': [] if i % 4 != 0 else [{'type': 'timeout', 'message': 'Operation timed out'}]
            }
            for i in range(10)
        ]
    
    def get_tasks_in_period(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Mock method to get all tasks in period"""
        tasks = []
        agents = ['agent_001', 'agent_002', 'agent_003', 'agent_004', 'agent_005']
        
        for agent_id in agents:
            tasks.extend(self.get_agent_tasks(agent_id, start_time, end_time))
            
        return tasks


class SupervisorReportingSystem:
    """
    Main reporting system that coordinates all reporting components
    """
    
    def __init__(self, config_file: str = None):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data source
        self.data_source = MockDataSource()
        
        # Initialize components
        self.alert_manager = AlertManager(self.config.get('alerts', {}))
        self.report_generator = ReportGenerator(self.data_source, self.config.get('reports', {}))
        self.audit_manager = AuditTrailManager(self.config.get('audit', {}))
        self.confidence_reporter = ConfidenceReporter(self.config.get('confidence', {}))
        self.pattern_tracker = PatternTracker(self.config.get('patterns', {}))
        self.dashboard_manager = DashboardManager(self.config.get('dashboard', {}))
        self.export_manager = ExportManager(self.config.get('export', {}))
        
        # Setup component relationships
        self._setup_component_relationships()
        
        # System state
        self.running = False
        self.last_health_check = datetime.now()
        
        self.logger.info("Supervisor Reporting System initialized")
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'alerts': {
                'email': {'enabled': False},
                'slack': {'enabled': False},
                'webhook': {'enabled': False},
                'task_timeout_threshold': 300,
                'low_confidence_threshold': 0.3,
                'error_rate_threshold': 0.1
            },
            'reports': {
                'optimal_duration': 30,
                'long_task_threshold': 300
            },
            'audit': {
                'log_directory': 'audit_logs',
                'max_memory_events': 10000,
                'queue_size': 1000
            },
            'confidence': {
                'confidence_data_file': 'confidence_data.jsonl',
                'max_memory_entries': 10000,
                'calibration_bins': 10
            },
            'patterns': {
                'patterns_file': 'patterns.json',
                'knowledge_file': 'knowledge_base.json',
                'min_pattern_frequency': 3,
                'pattern_lookback_days': 30,
                'similarity_threshold': 0.8
            },
            'dashboard': {
                'update_interval': 30
            },
            'export': {
                'export_directory': 'exports',
                'max_concurrent_jobs': 3
            },
            'system': {
                'health_check_interval': 60,
                'log_level': 'INFO'
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations (user config overrides defaults)
                self._merge_configs(default_config, user_config)
            except Exception as e:
                print(f"Failed to load config file {config_file}: {e}")
                print("Using default configuration")
                
        return default_config
        
    def _merge_configs(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
                
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('system', {}).get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('supervisor_reporting.log')
            ]
        )
        
    def _setup_component_relationships(self):
        """Setup relationships between components"""
        # Set data sources for dashboard
        self.dashboard_manager.set_data_sources(
            alert_manager=self.alert_manager,
            report_generator=self.report_generator,
            confidence_reporter=self.confidence_reporter,
            pattern_tracker=self.pattern_tracker
        )
        
        # Set data sources for export manager
        self.export_manager.set_data_sources(
            audit_manager=self.audit_manager,
            report_generator=self.report_generator,
            confidence_reporter=self.confidence_reporter,
            pattern_tracker=self.pattern_tracker
        )
        
    async def start(self):
        """Start the reporting system"""
        self.running = True
        self.logger.info("Starting Supervisor Reporting System")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._periodic_report_generation()),
            asyncio.create_task(self._pattern_analysis_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the reporting system"""
        self.running = False
        self.logger.info("Stopping Supervisor Reporting System")
        
    async def _health_check_loop(self):
        """Periodic health check and system monitoring"""
        interval = self.config.get('system', {}).get('health_check_interval', 60)
        
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(interval)
                
    async def _perform_health_check(self):
        """Perform system health check"""
        now = datetime.now()
        
        # Check component health
        health_data = {
            'timestamp': now,
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'memory_usage': 0.5,  # Placeholder
            'cpu_usage': 0.3,     # Placeholder
            'error_rate': 0.02    # Placeholder
        }
        
        # Evaluate alert conditions
        self.alert_manager.evaluate_conditions(health_data, 'system', 'health_check')
        
        # Log audit event
        self.audit_manager.log_event(
            event_type=self.audit_manager.AuditEventType.PERFORMANCE_THRESHOLD_BREACH
            if health_data['error_rate'] > 0.1 else self.audit_manager.AuditEventType.TASK_COMPLETED,
            level=self.audit_manager.AuditLevel.INFO,
            agent_id='system',
            cause='Periodic health check',
            action='System monitoring',
            outcome='Health check completed',
            metadata=health_data
        )
        
        self.last_health_check = now
        
    async def _periodic_report_generation(self):
        """Generate periodic summary reports"""
        while self.running:
            try:
                # Generate hourly summary
                summary = self.report_generator.generate_period_summary(hours=1)
                
                # Save report
                report_path = self.report_generator.save_report(summary, format='markdown')
                
                self.logger.info(f"Generated periodic report: {report_path}")
                
                # Wait for next hour
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Failed to generate periodic report: {e}")
                await asyncio.sleep(3600)
                
    async def _pattern_analysis_loop(self):
        """Periodic pattern analysis"""
        while self.running:
            try:
                # Get recent audit events for pattern analysis
                recent_events = self.audit_manager.query_events(
                    start_time=datetime.now() - timedelta(hours=1),
                    limit=1000
                )
                
                # Convert audit events to pattern analysis format
                analysis_events = []
                for event in recent_events:
                    analysis_events.append({
                        'timestamp': event.timestamp.isoformat(),
                        'agent_id': event.agent_id,
                        'task_id': event.task_id,
                        'event_type': event.event_type.value,
                        'level': event.level.value,
                        'status': 'completed' if event.level.value == 'info' else 'failed',
                        'error_type': event.metadata.get('error_type', 'unknown') if event.level.value == 'error' else None
                    })
                
                # Analyze patterns
                detected_patterns = self.pattern_tracker.analyze_events(analysis_events)
                
                if detected_patterns:
                    self.logger.info(f"Detected {len(detected_patterns)} new patterns")
                    
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                self.logger.error(f"Pattern analysis failed: {e}")
                await asyncio.sleep(1800)
                
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Cleanup old alerts
                self.alert_manager.cleanup_old_alerts(days=30)
                
                # Cleanup old audit logs
                self.audit_manager.cleanup_old_logs(days=90)
                
                # Cleanup old export jobs
                self.export_manager.cleanup_old_jobs(days=7)
                
                self.logger.info("Completed periodic cleanup")
                
                # Wait 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")
                await asyncio.sleep(86400)
                
    # Public API methods
    
    def log_task_event(self, event_type: str, agent_id: str, task_id: str, 
                      data: Dict[str, Any]):
        """Log a task-related event"""
        # Determine audit event type and level
        audit_event_type = {
            'task_started': self.audit_manager.AuditEventType.TASK_STARTED,
            'task_completed': self.audit_manager.AuditEventType.TASK_COMPLETED,
            'task_failed': self.audit_manager.AuditEventType.TASK_FAILED,
            'decision_made': self.audit_manager.AuditEventType.DECISION_MADE,
            'error_occurred': self.audit_manager.AuditEventType.ERROR_OCCURRED
        }.get(event_type, self.audit_manager.AuditEventType.TASK_COMPLETED)
        
        audit_level = {
            'task_started': self.audit_manager.AuditLevel.INFO,
            'task_completed': self.audit_manager.AuditLevel.INFO,
            'task_failed': self.audit_manager.AuditLevel.ERROR,
            'decision_made': self.audit_manager.AuditLevel.INFO,
            'error_occurred': self.audit_manager.AuditLevel.ERROR
        }.get(event_type, self.audit_manager.AuditLevel.INFO)
        
        # Log audit event
        self.audit_manager.log_event(
            event_type=audit_event_type,
            level=audit_level,
            agent_id=agent_id,
            task_id=task_id,
            cause=data.get('cause', f'Task {event_type}'),
            action=data.get('action', 'Task execution'),
            outcome=data.get('outcome', event_type),
            confidence=data.get('confidence'),
            metadata=data
        )
        
        # Evaluate alert conditions
        self.alert_manager.evaluate_conditions(data, agent_id, task_id)
        
        # Record confidence decision if present
        if 'confidence' in data and event_type == 'decision_made':
            decision_id = data.get('decision_id', f"{task_id}_decision")
            self.confidence_reporter.record_decision(
                agent_id=agent_id,
                task_id=task_id,
                decision_id=decision_id,
                confidence=data['confidence'],
                decision_type=data.get('decision_type', 'task_decision'),
                context=data
            )
            
    def update_decision_outcome(self, decision_id: str, outcome: bool):
        """Update the outcome of a previously recorded decision"""
        self.confidence_reporter.update_outcome(decision_id, outcome)
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_manager.update_dashboard()
        
    def generate_report(self, report_type: str = 'summary', 
                       period_hours: int = 24, 
                       format: str = 'markdown') -> str:
        """Generate a report"""
        if report_type == 'summary':
            summary = self.report_generator.generate_period_summary(hours=period_hours)
            return self.report_generator.save_report(summary, format=format)
        elif report_type == 'confidence':
            metrics = self.confidence_reporter.generate_metrics(hours=period_hours)
            return self.confidence_reporter.generate_calibration_report(metrics)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
            
    def export_data(self, export_type: str, **kwargs) -> str:
        """Export data in specified format"""
        if export_type == 'audit_logs':
            return self.export_manager.export_audit_logs(**kwargs)
        elif export_type == 'performance_reports':
            return self.export_manager.export_performance_reports(**kwargs)
        elif export_type == 'confidence_analysis':
            return self.export_manager.export_confidence_analysis(**kwargs)
        elif export_type == 'patterns_knowledge':
            return self.export_manager.export_patterns_knowledge(**kwargs)
        elif export_type == 'complete_backup':
            return self.export_manager.export_complete_backup(**kwargs)
        else:
            raise ValueError(f"Unknown export type: {export_type}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'running': self.running,
            'last_health_check': self.last_health_check.isoformat(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'alert_statistics': self.alert_manager.get_alert_statistics(),
            'export_statistics': self.export_manager.get_export_statistics(),
            'components': {
                'alert_manager': 'operational',
                'report_generator': 'operational',
                'audit_manager': 'operational',
                'confidence_reporter': 'operational',
                'pattern_tracker': 'operational',
                'dashboard_manager': 'operational',
                'export_manager': 'operational'
            }
        }
        
    def search_audit_logs(self, query: str, limit: int = 100) -> List[Dict]:
        """Search audit logs"""
        events = self.audit_manager.search_events(query, limit)
        return [
            {
                'id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'level': event.level.value,
                'agent_id': event.agent_id,
                'task_id': event.task_id,
                'cause': event.cause,
                'action': event.action,
                'outcome': event.outcome
            }
            for event in events
        ]
        
    def get_agent_recommendations(self, agent_id: str) -> List[str]:
        """Get pattern-based recommendations for an agent"""
        return self.pattern_tracker.get_recommendations_for_agent(agent_id)


# CLI interface for standalone usage
async def main():
    """Main function for running the reporting system"""
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    system = SupervisorReportingSystem(config_file)
    
    # Demo mode: log some sample events
    demo_events = [
        ('task_started', 'agent_001', 'task_001', {'cause': 'User request', 'action': 'Starting task'}),
        ('decision_made', 'agent_001', 'task_001', {'confidence': 0.85, 'decision_type': 'classification'}),
        ('task_completed', 'agent_001', 'task_001', {'outcome': 'success', 'execution_time': 45}),
        ('task_failed', 'agent_002', 'task_002', {'outcome': 'timeout', 'execution_time': 305})
    ]
    
    for event_type, agent_id, task_id, data in demo_events:
        system.log_task_event(event_type, agent_id, task_id, data)
        
    # Update decision outcome
    system.update_decision_outcome('task_001_decision', True)
    
    print("Supervisor Reporting System Demo")
    print("================================")
    print()
    
    # Show system status
    status = system.get_system_status()
    print(f"System Status: {status['running']}")
    print(f"Active Alerts: {status['active_alerts']}")
    print()
    
    # Generate and show a sample report
    report_path = system.generate_report('summary', period_hours=1)
    print(f"Generated report: {report_path}")
    
    # Show dashboard data
    dashboard = system.get_dashboard_data()
    print(f"Dashboard metrics: {len(dashboard['metrics'])} metrics, {len(dashboard['charts'])} charts")
    
    # Start the system (this will run indefinitely)
    if '--run' in sys.argv:
        await system.start()


if __name__ == '__main__':
    asyncio.run(main())
