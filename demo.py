#!/usr/bin/env python3
"""
Demo script for Supervisor Agent Reporting System
Shows all major features and capabilities
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from main import SupervisorReportingSystem


class ReportingDemo:
    def __init__(self):
        print("🤖 Supervisor Agent Reporting System Demo")
        print("==========================================")
        print()
        
        # Initialize the reporting system
        self.system = SupervisorReportingSystem()
        
    async def run_demo(self):
        """Run complete demonstration"""
        print("Starting comprehensive reporting system demo...\n")
        
        # 1. Basic event logging
        await self.demo_event_logging()
        
        # 2. Alert system
        await self.demo_alert_system()
        
        # 3. Confidence tracking
        await self.demo_confidence_tracking()
        
        # 4. Report generation
        await self.demo_report_generation()
        
        # 5. Dashboard
        await self.demo_dashboard()
        
        # 6. Pattern detection
        await self.demo_pattern_detection()
        
        # 7. Export system
        await self.demo_export_system()
        
        # 8. System status
        await self.demo_system_status()
        
        print("\n✅ Demo completed successfully!")
        print("\nThe reporting system is now ready for production use.")
        print("Check the generated files in the current directory.")
        
    async def demo_event_logging(self):
        """Demonstrate audit event logging"""
        print("📝 1. Audit Event Logging")
        print("-" * 30)
        
        # Log various types of events
        events = [
            ('task_started', 'agent_001', 'task_001', {
                'cause': 'User document upload',
                'action': 'Starting document analysis',
                'document_type': 'PDF',
                'file_size': 2048576
            }),
            ('decision_made', 'agent_001', 'task_001', {
                'confidence': 0.87,
                'decision_type': 'document_classification',
                'decision_id': 'doc_class_001',
                'categories': ['invoice', 'financial']
            }),
            ('task_completed', 'agent_001', 'task_001', {
                'outcome': 'success',
                'execution_time': 23.5,
                'pages_processed': 5,
                'confidence': 0.87
            }),
            ('task_started', 'agent_002', 'task_002', {
                'cause': 'Scheduled data processing',
                'action': 'Processing batch data',
                'batch_size': 1000
            }),
            ('task_failed', 'agent_002', 'task_002', {
                'outcome': 'timeout',
                'execution_time': 305,
                'error_type': 'timeout',
                'error_message': 'Task exceeded maximum execution time'
            }),
            ('error_occurred', 'agent_003', 'task_003', {
                'error_type': 'validation_error',
                'error_message': 'Invalid input format',
                'input_size': 0
            })
        ]
        
        for event_type, agent_id, task_id, data in events:
            self.system.log_task_event(event_type, agent_id, task_id, data)
            print(f"  ✓ Logged {event_type} for {agent_id}/{task_id}")
            
        # Update decision outcomes
        self.system.update_decision_outcome('doc_class_001', True)
        print(f"  ✓ Updated decision outcome for doc_class_001")
        
        print(f"  → Logged {len(events)} events to audit trail\n")
        
    async def demo_alert_system(self):
        """Demonstrate alert system"""
        print("🚨 2. Alert System")
        print("-" * 20)
        
        # Trigger some alerts with problematic data
        alert_scenarios = [
            ('agent_004', 'task_004', {
                'execution_time': 350,  # Over threshold
                'confidence': 0.2,      # Low confidence
                'memory_usage': 0.95,   # High memory
                'error_rate': 0.15      # High error rate
            }),
            ('agent_005', 'task_005', {
                'execution_time': 450,  # Very slow
                'confidence': 0.1,      # Very low confidence
                'cpu_usage': 0.98       # High CPU
            })
        ]
        
        for agent_id, task_id, data in alert_scenarios:
            self.system.alert_manager.evaluate_conditions(data, agent_id, task_id)
            
        # Show active alerts
        active_alerts = self.system.alert_manager.get_active_alerts()
        print(f"  ✓ Generated {len(active_alerts)} alerts")
        
        for alert in active_alerts[:3]:  # Show first 3
            print(f"    - {alert.priority.value.upper()}: {alert.title}")
            
        # Show alert statistics
        stats = self.system.alert_manager.get_alert_statistics(hours=1)
        print(f"  → Alert statistics: {stats['total_alerts']} total, {stats['active_alerts']} active\n")
        
    async def demo_confidence_tracking(self):
        """Demonstrate confidence tracking and calibration"""
        print("🎯 3. Confidence Tracking & Calibration")
        print("-" * 40)
        
        # Record multiple decisions with various confidence levels
        decisions = [
            ('agent_006', 'task_006', 'decision_006', 0.95, 'classification', True),
            ('agent_006', 'task_007', 'decision_007', 0.85, 'extraction', True),
            ('agent_007', 'task_008', 'decision_008', 0.75, 'classification', False),
            ('agent_007', 'task_009', 'decision_009', 0.65, 'validation', True),
            ('agent_008', 'task_010', 'decision_010', 0.45, 'classification', False),
            ('agent_008', 'task_011', 'decision_011', 0.35, 'extraction', False),
            ('agent_009', 'task_012', 'decision_012', 0.25, 'validation', False),
            ('agent_009', 'task_013', 'decision_013', 0.15, 'classification', False)
        ]
        
        for agent_id, task_id, decision_id, confidence, decision_type, outcome in decisions:
            # Record decision
            self.system.confidence_reporter.record_decision(
                agent_id=agent_id,
                task_id=task_id,
                decision_id=decision_id,
                confidence=confidence,
                decision_type=decision_type
            )
            
            # Update outcome
            self.system.confidence_reporter.update_outcome(decision_id, outcome)
            
        print(f"  ✓ Recorded {len(decisions)} decisions with outcomes")
        
        # Generate confidence metrics
        conf_metrics = self.system.confidence_reporter.generate_metrics(hours=1)
        print(f"  ✓ Average confidence: {conf_metrics.avg_confidence:.3f}")
        print(f"  ✓ Accuracy: {conf_metrics.accuracy:.3f}")
        print(f"  ✓ Calibration score: {conf_metrics.calibration_score:.3f}")
        print(f"  → {conf_metrics.total_decisions} total decisions analyzed\n")
        
    async def demo_report_generation(self):
        """Demonstrate report generation"""
        print("📊 4. Report Generation")
        print("-" * 25)
        
        # Generate summary report
        summary_path = self.system.generate_report('summary', period_hours=1, format='markdown')
        print(f"  ✓ Generated summary report: {Path(summary_path).name}")
        
        # Generate confidence report
        conf_metrics = self.system.confidence_reporter.generate_metrics(hours=1)
        conf_report = self.system.confidence_reporter.generate_calibration_report(conf_metrics)
        
        conf_report_path = 'confidence_report.md'
        with open(conf_report_path, 'w') as f:
            f.write(conf_report)
        print(f"  ✓ Generated confidence report: {conf_report_path}")
        
        print(f"  → Reports available for review\n")
        
    async def demo_dashboard(self):
        """Demonstrate dashboard functionality"""
        print("📈 5. Dashboard & Visualization")
        print("-" * 32)
        
        # Update dashboard
        dashboard_data = self.system.get_dashboard_data()
        
        print(f"  ✓ Dashboard updated with {len(dashboard_data['metrics'])} metrics")
        print(f"  ✓ Generated {len(dashboard_data['charts'])} charts")
        print(f"  ✓ Overall system status: {dashboard_data['status']}")
        
        # Show some key metrics
        metrics = dashboard_data['metrics']
        if 'success_rate' in metrics:
            print(f"    - Success Rate: {metrics['success_rate']['value']}{metrics['success_rate']['unit']}")
        if 'avg_confidence' in metrics:
            print(f"    - Avg Confidence: {metrics['avg_confidence']['value']}")
        if 'active_alerts' in metrics:
            print(f"    - Active Alerts: {metrics['active_alerts']['value']}")
            
        # Generate HTML dashboard
        dashboard_html = self.system.dashboard_manager.generate_dashboard_html()
        dashboard_path = 'supervisor_dashboard.html'
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        print(f"  ✓ Generated HTML dashboard: {dashboard_path}")
        
        print(f"  → Dashboard ready for viewing\n")
        
    async def demo_pattern_detection(self):
        """Demonstrate pattern detection"""
        print("🔍 6. Pattern Detection & Knowledge Base")
        print("-" * 42)
        
        # Create some events that will form patterns
        pattern_events = []
        
        # Create failure pattern
        for i in range(5):
            pattern_events.append({
                'timestamp': (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                'agent_id': 'agent_010',
                'task_id': f'task_{100+i}',
                'event_type': 'task_failed',
                'level': 'error',
                'status': 'failed',
                'error_type': 'timeout'
            })
            
        # Create performance pattern
        for i in range(4):
            pattern_events.append({
                'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                'agent_id': 'agent_011',
                'task_id': f'task_{200+i}',
                'event_type': 'task_completed',
                'level': 'info',
                'status': 'completed',
                'execution_time': 120 + i*30,  # Increasing execution time
                'confidence': 0.4 - i*0.05     # Decreasing confidence
            })
            
        # Analyze patterns
        detected_patterns = self.system.pattern_tracker.analyze_events(pattern_events)
        
        print(f"  ✓ Analyzed {len(pattern_events)} events")
        print(f"  ✓ Detected {len(detected_patterns)} patterns")
        
        for pattern in detected_patterns[:3]:  # Show first 3
            print(f"    - {pattern.pattern_type}: {pattern.name} (frequency: {pattern.frequency})")
            
        # Get recommendations
        recommendations = self.system.get_agent_recommendations('agent_010')
        if recommendations:
            print(f"  ✓ Generated {len(recommendations)} recommendations")
            for rec in recommendations[:2]:
                print(f"    - {rec}")
                
        print(f"  → Pattern analysis completed\n")
        
    async def demo_export_system(self):
        """Demonstrate export capabilities"""
        print("📤 7. Export System")
        print("-" * 20)
        
        # Export audit logs
        audit_job_id = self.system.export_data(
            'audit_logs',
            format='json',
            compress=True,
            start_time=datetime.now() - timedelta(hours=1)
        )
        print(f"  ✓ Started audit logs export: {audit_job_id}")
        
        # Export performance report
        perf_job_id = self.system.export_data(
            'performance_reports',
            format='json',
            period_hours=1
        )
        print(f"  ✓ Started performance report export: {perf_job_id}")
        
        # Export confidence analysis
        conf_job_id = self.system.export_data(
            'confidence_analysis',
            format='json',
            period_hours=1
        )
        print(f"  ✓ Started confidence analysis export: {conf_job_id}")
        
        # Wait a moment for exports to complete
        await asyncio.sleep(1)
        
        # Check export status
        export_stats = self.system.export_manager.get_export_statistics()
        print(f"  ✓ Export statistics: {export_stats['completed_jobs']} completed")
        
        # Create complete backup
        backup_job_id = self.system.export_data('complete_backup', compress=True)
        print(f"  ✓ Started complete backup: {backup_job_id}")
        
        print(f"  → Export jobs initiated\n")
        
    async def demo_system_status(self):
        """Show overall system status"""
        print("🔧 8. System Status & Health")
        print("-" * 30)
        
        # Get comprehensive system status
        status = self.system.get_system_status()
        
        print(f"  ✓ System running: {status['running']}")
        print(f"  ✓ Last health check: {status['last_health_check'][:19]}")
        print(f"  ✓ Active alerts: {status['active_alerts']}")
        
        # Component status
        print("  ✓ Components:")
        for component, state in status['components'].items():
            print(f"    - {component}: {state}")
            
        # Search audit logs
        search_results = self.system.search_audit_logs('task_completed', limit=5)
        print(f"  ✓ Audit search: found {len(search_results)} completed tasks")
        
        print(f"  → System health check completed\n")
        
    def show_summary(self):
        """Show demo summary"""
        print("📋 Demo Summary")
        print("=" * 15)
        print()
        print("Features Demonstrated:")
        print("✅ Real-time audit event logging")
        print("✅ Multi-channel alert system")
        print("✅ Confidence tracking and calibration")
        print("✅ Automated report generation")
        print("✅ Interactive dashboard")
        print("✅ Pattern detection and knowledge base")
        print("✅ Multi-format export system")
        print("✅ System health monitoring")
        print()
        print("Generated Files:")
        files = [
            'supervisor_dashboard.html',
            'confidence_report.md',
            'supervisor_reporting.log'
        ]
        
        for file in files:
            if Path(file).exists():
                print(f"📄 {file}")
                
        # Check export directory
        export_dir = Path('exports')
        if export_dir.exists():
            export_files = list(export_dir.glob('*'))
            for file in export_files:
                print(f"📦 {file}")
                
        print()
        print("Next Steps:")
        print("1. Open supervisor_dashboard.html in your browser")
        print("2. Review the generated reports")
        print("3. Configure alert channels (email/Slack/webhook)")
        print("4. Integrate with your existing monitoring systems")
        print("5. Customize patterns and thresholds for your use case")
        

async def main():
    """Run the complete demo"""
    demo = ReportingDemo()
    
    try:
        await demo.run_demo()
        demo.show_summary()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Thank you for trying the Supervisor Agent Reporting System!")


if __name__ == '__main__':
    asyncio.run(main())
