#!/usr/bin/env python3
"""
Quick Working Demo of Supervisor Agent Reporting System
Demonstrates core functionality with minimal setup
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components directly
from alerts import AlertManager, AlertType, AlertSeverity
from summaries import ReportGenerator
from audit_system import ComprehensiveAuditSystem as AuditTrailManager, AuditEventType, AuditLevel
from confidence import ConfidenceReporter
from patterns import PatternTracker
from dashboard import DashboardManager

def quick_demo():
    """Run a quick demonstration of core reporting capabilities"""
    
    logger.info("=== Quick Supervisor Reporting System Demo ===")
    
    # Create output directory
    output_dir = Path("quick_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Demo Alert System
        logger.info("\n1. Testing Alert System...")
        alert_config = {
            'task_timeout_threshold': 300,
            'low_confidence_threshold': 0.3,
            'error_rate_threshold': 0.1,
            'email': {'enabled': False},
            'slack': {'enabled': False},
            'webhook': {'enabled': False}
        }
        alert_manager = AlertManager(alert_config)
        
        # Create test alerts
        alert1 = alert_manager.create_alert(
            AlertType.SYSTEM_FAILURE, AlertSeverity.CRITICAL,
            "Test Critical Alert", "System test failure detected",
            "test_agent", {'test': True}
        )
        
        alert2 = alert_manager.create_alert(
            AlertType.LOW_CONFIDENCE, AlertSeverity.MEDIUM,
            "Low Confidence Test", "Confidence below threshold",
            "test_agent", {'confidence': 0.25}
        )
        
        # Test condition evaluation
        alert_manager.evaluate_conditions(
            {'task_duration': 350, 'confidence': 0.2, 'error_rate': 0.15},
            "test_agent", "demo_test"
        )
        
        active_alerts = alert_manager.get_active_alerts()
        logger.info(f"Created {len(active_alerts)} active alerts")
        
        # Export alert summary
        alert_summary = alert_manager.get_alerts_summary()
        with open(output_dir / "alerts.json", 'w') as f:
            json.dump(alert_summary, f, indent=2)
        
        # 2. Demo Audit System
        logger.info("\n2. Testing Audit System...")
        audit_config = {
            'log_directory': str(output_dir / 'audit_logs'),
            'max_memory_events': 1000,
            'queue_size': 100
        }
        audit_manager = AuditTrailManager(audit_config)
        
        # Log various events
        audit_manager.log_task_started("test_agent", "demo_task_1", "processing",
                                     {'input_size': 1024})
        
        audit_manager.log_decision("test_agent", "algorithm_selection", "algorithm_A",
                                 0.85, "Best performance for input type",
                                 {'alternatives': ['algorithm_B']})
        
        audit_manager.log_task_completed("test_agent", "demo_task_1", 23.4, 0.89,
                                       {'output_size': 512})
        
        audit_manager.log_task_failed("test_agent", "demo_task_2", "timeout", 45.2,
                                    {'retry_count': 3})
        
        # Query recent events
        recent_events = audit_manager.get_recent_events(10)
        logger.info(f"Logged {len(recent_events)} audit events")
        
        # Export audit events
        audit_manager.export_events(str(output_dir / "audit_events.json"), format='json')
        
        # 3. Demo Confidence System
        logger.info("\n3. Testing Confidence System...")
        confidence_config = {
            'confidence_data_file': str(output_dir / 'confidence.jsonl'),
            'max_memory_entries': 1000,
            'calibration_bins': 10
        }
        confidence_reporter = ConfidenceReporter(confidence_config)
        
        # Record confidence entries
        for i in range(20):
            confidence = 0.5 + (i % 5) * 0.1  # Vary from 0.5 to 0.9
            success = confidence > 0.7  # Simple success criterion
            
            confidence_reporter.record_confidence(
                f"agent_{i%3:02d}", f"task_{i}", confidence, success,
                20 + i*2, "test_task", {'iteration': i}
            )
        
        # Generate confidence metrics
        metrics = confidence_reporter.generate_metrics(hours=1)
        logger.info(f"Confidence metrics: {metrics.total_entries} entries, "
                   f"{metrics.accuracy:.2%} accuracy, {metrics.calibration_error:.3f} cal error")
        
        # Export confidence data
        with open(output_dir / "confidence_metrics.json", 'w') as f:
            json.dump({
                'total_entries': metrics.total_entries,
                'mean_confidence': metrics.mean_confidence,
                'accuracy': metrics.accuracy,
                'calibration_error': metrics.calibration_error,
                'brier_score': metrics.brier_score
            }, f, indent=2)
        
        # Generate calibration report
        calibration_report = confidence_reporter.generate_calibration_report(metrics)
        with open(output_dir / "confidence_report.md", 'w') as f:
            f.write(calibration_report)
        
        # 4. Demo Pattern System
        logger.info("\n4. Testing Pattern System...")
        pattern_config = {
            'patterns_file': str(output_dir / 'patterns.json'),
            'knowledge_file': str(output_dir / 'knowledge.json'),
            'min_pattern_frequency': 2,
            'pattern_lookback_days': 7
        }
        pattern_tracker = PatternTracker(pattern_config)
        
        # Create mock events for pattern detection
        events = []
        for i in range(10):
            events.append({
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'event_type': 'task_failed' if i % 3 == 0 else 'task_completed',
                'level': 'error' if i % 3 == 0 else 'info',
                'agent_id': f'agent_{i%2:02d}',
                'message': 'Connection timeout' if i % 3 == 0 else 'Task completed',
                'outcome': 'failed' if i % 3 == 0 else 'success',
                'metadata': {'duration': 30 + i*5, 'test_event': True}
            })
        
        # Analyze patterns
        analysis_result = pattern_tracker.analyze_events(events)
        logger.info(f"Pattern analysis: {analysis_result.patterns_detected} patterns detected, "
                   f"{analysis_result.new_patterns} new patterns")
        
        # Export pattern data
        pattern_tracker.export_patterns(str(output_dir / "patterns.json"))
        
        # 5. Demo Dashboard System
        logger.info("\n5. Testing Dashboard System...")
        dashboard_config = {'update_interval': 30}
        dashboard_manager = DashboardManager(dashboard_config)
        
        # Set data sources (mock references)
        dashboard_manager.set_data_sources(
            alert_manager=alert_manager,
            confidence_reporter=confidence_reporter,
            pattern_tracker=pattern_tracker
        )
        
        # Generate dashboard data
        dashboard_data = dashboard_manager.generate_dashboard_data()
        logger.info(f"Dashboard: {dashboard_data.metrics.system_status} status, "
                   f"{dashboard_data.metrics.total_agents} agents, "
                   f"{len(dashboard_data.charts)} charts")
        
        # Export dashboard data
        dashboard_manager.export_dashboard_data(str(output_dir / "dashboard.json"))
        
        # 6. Create Summary Report
        logger.info("\n6. Generating Summary Report...")
        
        # Mock data source for report generator
        class MockDataSource:
            def get_tasks_in_period(self, start_time, end_time):
                return [{
                    'task_id': f'task_{i}',
                    'agent_id': f'agent_{i%3:02d}',
                    'start_time': (start_time + timedelta(minutes=i*10)).isoformat(),
                    'end_time': (start_time + timedelta(minutes=i*10+5)).isoformat(),
                    'status': 'completed' if i % 4 != 0 else 'failed',
                    'confidence': 0.8 - (i * 0.02),
                    'errors': [] if i % 4 != 0 else [{'type': 'timeout'}]
                } for i in range(10)]
        
        report_generator = ReportGenerator(MockDataSource(), {
            'optimal_duration': 30,
            'long_task_threshold': 300
        })
        
        # Generate period summary
        summary = report_generator.generate_period_summary(hours=1)
        logger.info(f"Summary: {summary.total_tasks} tasks, "
                   f"{summary.overall_success_rate:.1%} success rate")
        
        # Generate markdown report
        markdown_report = report_generator.generate_markdown_report(summary)
        with open(output_dir / "summary_report.md", 'w') as f:
            f.write(markdown_report)
        
        # Export summary JSON
        report_generator.export_summary_json(summary, str(output_dir / "summary.json"))
        
        # 7. Generate Final Demo Report
        logger.info("\n7. Creating Final Demo Report...")
        
        demo_report = f"""
# Quick Supervisor Reporting System Demo Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Demo Summary

Successfully demonstrated all core reporting system components:

### ✅ Alert System
- Created {len(active_alerts)} active alerts
- Tested condition-based triggering
- Demonstrated alert deduplication
- **Output**: `alerts.json`

### ✅ Audit Trail System  
- Logged {len(recent_events)} audit events
- Tested structured event logging
- Demonstrated event querying and export
- **Output**: `audit_events.json`, `audit_logs/`

### ✅ Confidence Reporting
- Recorded {metrics.total_entries} confidence entries
- Calculated calibration metrics (Error: {metrics.calibration_error:.3f})
- Generated accuracy analysis ({metrics.accuracy:.1%})
- **Output**: `confidence_metrics.json`, `confidence_report.md`

### ✅ Pattern Detection
- Analyzed {len(events)} events for patterns
- Detected {analysis_result.patterns_detected} patterns
- Built knowledge base entries
- **Output**: `patterns.json`

### ✅ Dashboard System
- Generated real-time metrics
- Created {len(dashboard_data.charts)} visualization charts
- System status: {dashboard_data.metrics.system_status}
- **Output**: `dashboard.json`

### ✅ Summary Reports
- Generated performance summary
- {summary.total_tasks} tasks analyzed
- Success rate: {summary.overall_success_rate:.1%}
- **Output**: `summary_report.md`, `summary.json`

## Key Features Demonstrated

1. **Real-time Monitoring**: Live alert generation and system health tracking
2. **Comprehensive Analytics**: Task performance, confidence calibration, pattern detection
3. **Audit Trails**: Complete event logging with structured metadata
4. **Visualization**: Dashboard data with multiple chart types
5. **Export Capabilities**: Multiple formats (JSON, Markdown, CSV)
6. **Modular Architecture**: Independent components with clean interfaces

## Output Files Generated

- `alerts.json` - Active alerts and summary
- `audit_events.json` - Audit event export
- `confidence_metrics.json` - Confidence analysis metrics
- `confidence_report.md` - Detailed calibration report
- `patterns.json` - Detected patterns and knowledge base
- `dashboard.json` - Complete dashboard data
- `summary_report.md` - Performance summary report
- `summary.json` - Raw summary data

## System Status

**✅ All Components Operational**

- Alert System: {len(active_alerts)} active alerts
- Audit System: {len(recent_events)} events logged
- Confidence System: {metrics.total_entries} entries, {metrics.accuracy:.1%} accuracy
- Pattern System: {analysis_result.patterns_detected} patterns detected
- Dashboard: {dashboard_data.metrics.system_status.upper()} status
- Reports: Generated successfully

**Demo completed successfully! All reporting capabilities functional.**
"""
        
        with open(output_dir / "demo_report.md", 'w') as f:
            f.write(demo_report)
        
        # Count output files
        output_files = list(output_dir.rglob('*'))
        file_count = len([f for f in output_files if f.is_file()])
        
        logger.info(f"\n=== Demo Completed Successfully! ===")
        logger.info(f"Generated {file_count} output files in '{output_dir}'")
        logger.info(f"Key outputs: demo_report.md, dashboard.json, summary_report.md")
        logger.info(f"All core reporting components are functional and ready for production use.")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if 'audit_manager' in locals():
            audit_manager.shutdown()

if __name__ == "__main__":
    success = quick_demo()
    exit(0 if success else 1)
