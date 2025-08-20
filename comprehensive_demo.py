#!/usr/bin/env python3
"""
Comprehensive demo script showcasing all reporting system capabilities
"""

import time
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Import all system components
from integrated_system import IntegratedReportingSystem, IntegratedReportingConfig
from alert_system import RealTimeAlertSystem
from confidence_system import ConfidenceReportingSystem
from pattern_system import ComprehensivePatternSystem
from audit_system import ComprehensiveAuditSystem

def create_comprehensive_demo():
    """Create comprehensive demo with realistic scenarios"""
    print("🚀 Starting Comprehensive Reporting System Demo")
    print("=" * 50)
    
    # Create demo output directory
    demo_dir = Path("comprehensive_demo_output")
    demo_dir.mkdir(exist_ok=True)
    
    # Configure system
    config = IntegratedReportingConfig(
        base_output_dir=str(demo_dir),
        dashboard_enabled=True,
        dashboard_port=5002,
        background_processing=False,  # Manual control for demo
        alert_config={
            'deduplication_window': 5,  # Short window for demo
            'email': {
                'smtp_host': 'demo.smtp.com',
                'smtp_port': 587,
                'username': 'demo@example.com',
                'password': 'demo_password',
                'recipients': ['admin@demo.com']
            }
        }
    )
    
    # Initialize system
    system = IntegratedReportingSystem(config)
    system.start()
    
    return system, demo_dir

def simulate_realistic_workload(system):
    """Simulate realistic AI agent workload"""
    print("\n📊 Simulating Realistic Agent Workload")
    print("-" * 40)
    
    scenarios = [
        # Normal operations
        {
            'task_type': 'data_processing',
            'success_rate': 0.92,
            'confidence_range': (0.75, 0.95),
            'duration_range': (10, 60),
            'count': 15
        },
        # Classification tasks
        {
            'task_type': 'classification',
            'success_rate': 0.88,
            'confidence_range': (0.65, 0.90),
            'duration_range': (5, 30),
            'count': 20
        },
        # Complex reasoning
        {
            'task_type': 'reasoning',
            'success_rate': 0.75,
            'confidence_range': (0.50, 0.85),
            'duration_range': (30, 120),
            'count': 10
        }
    ]
    
    agents = ['reasoning_agent', 'classifier_agent', 'processor_agent']
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\nRunning scenario {scenario_idx + 1}: {scenario['task_type']}")
        
        for task_idx in range(scenario['count']):
            agent_id = random.choice(agents)
            task_id = f"{scenario['task_type']}_{scenario_idx}_{task_idx:03d}"
            correlation_id = f"corr_{task_id}"
            
            # Task started
            system.log_event(
                'task_started',
                agent_id,
                f"Starting {scenario['task_type']} task",
                {
                    'task_id': task_id,
                    'agent_id': agent_id,
                    'task_type': scenario['task_type'],
                    'complexity': random.choice(['low', 'medium', 'high'])
                },
                correlation_id
            )
            
            # Confidence recording
            confidence = random.uniform(*scenario['confidence_range'])
            system.log_event(
                'confidence_recorded',
                agent_id,
                f"Recording confidence for {scenario['task_type']}",
                {
                    'task_id': task_id,
                    'agent_id': agent_id,
                    'decision_type': scenario['task_type'],
                    'confidence_score': confidence
                },
                correlation_id
            )
            
            # Simulate processing time
            duration = random.uniform(*scenario['duration_range'])
            
            # Determine outcome
            success = random.random() < scenario['success_rate']
            
            if success:
                # Task completed
                system.log_event(
                    'task_completed',
                    agent_id,
                    f"{scenario['task_type']} task completed successfully",
                    {
                        'task_id': task_id,
                        'agent_id': agent_id,
                        'success': True,
                        'duration_seconds': duration,
                        'confidence_score': confidence
                    },
                    correlation_id
                )
            else:
                # Task failed
                error_types = ['timeout', 'validation_error', 'resource_exhausted', 'network_error']
                error_type = random.choice(error_types)
                
                system.log_event(
                    'error_occurred',
                    agent_id,
                    f"{error_type} in {scenario['task_type']} task",
                    {
                        'task_id': task_id,
                        'agent_id': agent_id,
                        'error_type': error_type,
                        'duration_seconds': duration
                    },
                    correlation_id
                )
                
                system.log_event(
                    'task_failed',
                    agent_id,
                    f"{scenario['task_type']} task failed due to {error_type}",
                    {
                        'task_id': task_id,
                        'agent_id': agent_id,
                        'success': False,
                        'duration_seconds': duration,
                        'error_type': error_type
                    },
                    correlation_id
                )
            
            # Brief pause to simulate realistic timing
            time.sleep(0.1)
        
        print(f"  Completed {scenario['count']} {scenario['task_type']} tasks")
    
    print("\n✅ Workload simulation complete")

def demonstrate_system_capabilities(system, demo_dir):
    """Demonstrate all system capabilities"""
    print("\n🔧 Demonstrating System Capabilities")
    print("-" * 40)
    
    # 1. Generate comprehensive reports
    print("\n1. Generating Comprehensive Reports...")
    reports = system.generate_comprehensive_report()
    for report_type, file_path in reports.items():
        if file_path:
            print(f"   ✅ {report_type}: {file_path}")
        else:
            print(f"   ❌ {report_type}: Failed to generate")
    
    # 2. System status
    print("\n2. System Status Analysis...")
    status = system.get_system_status()
    print(f"   Active systems: {len([s for s in status['systems'].values() if s.get('status') == 'active'])}")
    print(f"   Background processing: {status['background_processing']}")
    
    for system_name, system_status in status['systems'].items():
        if system_status.get('status') == 'active':
            print(f"   ✅ {system_name}: Active")
            # Print specific metrics if available
            if 'total_events' in system_status:
                print(f"      - Events: {system_status['total_events']}")
            if 'total_entries' in system_status:
                print(f"      - Entries: {system_status['total_entries']}")
            if 'total_patterns' in system_status:
                print(f"      - Patterns: {system_status['total_patterns']}")
        else:
            print(f"   ❌ {system_name}: {system_status.get('status', 'unknown')}")
    
    # 3. Export system state
    print("\n3. Exporting Complete System State...")
    state_file = system.export_complete_system_state(str(demo_dir / "complete_system_state.json"))
    if state_file:
        print(f"   ✅ System state: {state_file}")
    else:
        print(f"   ❌ System state export failed")
    
    # 4. Individual system demonstrations
    print("\n4. Individual System Capabilities...")
    
    # Audit system
    if 'audit_system' in system.systems:
        audit_system = system.systems['audit_system']
        recent_events = audit_system.search(limit=10)
        print(f"   📝 Audit System: {len(recent_events)} recent events")
        
        # Event correlation example
        if recent_events and recent_events[0].correlation_id:
            correlation_id = recent_events[0].correlation_id
            trace = audit_system.trace_correlation(correlation_id)
            print(f"      - Correlation trace: {trace['summary']['event_count']} events")
    
    # Confidence system
    if 'confidence_system' in system.systems:
        confidence_system = system.systems['confidence_system']
        stats = confidence_system.get_entry_statistics()
        print(f"   🎯 Confidence System: {stats['total_entries']} entries")
        print(f"      - Outcome coverage: {stats['outcome_coverage']:.1%}")
        
        # Generate calibration plot
        analysis = confidence_system.analyze_confidence()
        plot_file = str(demo_dir / "calibration_plot.png")
        confidence_system.generate_calibration_plot(analysis, plot_file)
        print(f"      - Calibration plot: {plot_file}")
    
    # Pattern system
    if 'pattern_system' in system.systems:
        pattern_system = system.systems['pattern_system']
        insights = pattern_system.get_pattern_insights()
        print(f"   🔍 Pattern System: {insights['total_patterns']} patterns")
        if insights['most_frequent']:
            top_pattern = insights['most_frequent'][0]
            print(f"      - Top pattern: {top_pattern.title} (freq: {top_pattern.frequency})")
    
    # Alert system
    if 'alert_system' in system.systems:
        alert_system = system.systems['alert_system']
        stats = alert_system.get_alert_stats()
        print(f"   🚨 Alert System: {stats['total_alerts']} alerts")
        print(f"      - Unresolved: {stats['unresolved']}")
    
    # Dashboard system
    if 'dashboard_system' in system.systems:
        dashboard_system = system.systems['dashboard_system']
        dashboard_report = str(demo_dir / "dashboard_static_report.html")
        dashboard_system.generate_static_report(dashboard_report)
        print(f"   📊 Dashboard System: Static report generated")
        print(f"      - Report: {dashboard_report}")
        print(f"      - Live dashboard: http://localhost:{system.config.dashboard_port}")
    
    print("\n✅ System capabilities demonstration complete")

def generate_demo_insights(system, demo_dir):
    """Generate insights and recommendations from demo data"""
    print("\n💡 Generating Demo Insights and Recommendations")
    print("-" * 50)
    
    insights_file = demo_dir / "demo_insights.md"
    
    with open(insights_file, 'w') as f:
        f.write("# Comprehensive Reporting System Demo Insights\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # System overview
        status = system.get_system_status()
        f.write("## System Overview\n\n")
        f.write(f"- **Active Systems**: {len([s for s in status['systems'].values() if s.get('status') == 'active'])}\n")
        f.write(f"- **Configuration**: {len(status['config'])} settings\n")
        f.write(f"- **Background Processing**: {status['background_processing']}\n\n")
        
        # Individual system insights
        for system_name, system_status in status['systems'].items():
            f.write(f"### {system_name.title().replace('_', ' ')}\n\n")
            
            if system_status.get('status') == 'active':
                f.write(f"**Status**: ✅ Active\n\n")
                
                # Add specific insights based on system type
                if system_name == 'audit_system':
                    f.write(f"- Total Events: {system_status.get('total_events', 'N/A')}\n")
                    f.write(f"- Unique Sessions: {system_status.get('unique_sessions', 'N/A')}\n")
                    f.write("- **Insight**: Comprehensive event tracking is operational\n")
                elif system_name == 'confidence_system':
                    f.write(f"- Total Entries: {system_status.get('total_entries', 'N/A')}\n")
                    f.write(f"- Outcome Coverage: {system_status.get('outcome_coverage', 0)*100:.1f}%\n")
                    f.write("- **Insight**: Confidence calibration analysis is available\n")
                elif system_name == 'pattern_system':
                    f.write(f"- Total Patterns: {system_status.get('total_patterns', 'N/A')}\n")
                    f.write("- **Insight**: Pattern detection is identifying recurring behaviors\n")
                elif system_name == 'alert_system':
                    f.write(f"- Total Alerts: {system_status.get('total_alerts', 'N/A')}\n")
                    f.write(f"- Unresolved: {system_status.get('unresolved', 'N/A')}\n")
                    f.write("- **Insight**: Real-time alerting system is monitoring for issues\n")
                
                f.write("\n")
            else:
                f.write(f"**Status**: ❌ {system_status.get('status', 'Unknown')}\n\n")
        
        # Recommendations
        f.write("## Recommendations for Production Use\n\n")
        f.write("### 1. Configuration\n")
        f.write("- Configure SMTP settings for email alerts\n")
        f.write("- Set up Slack/webhook integrations for team notifications\n")
        f.write("- Adjust deduplication windows based on expected event volume\n\n")
        
        f.write("### 2. Monitoring\n")
        f.write("- Enable background processing for automatic report generation\n")
        f.write("- Set up regular health checks and system status monitoring\n")
        f.write("- Monitor disk usage for audit logs and pattern data\n\n")
        
        f.write("### 3. Integration\n")
        f.write("- Integrate with existing task management systems\n")
        f.write("- Set up automated confidence score recording\n")
        f.write("- Configure external system webhooks for real-time updates\n\n")
        
        f.write("### 4. Scaling\n")
        f.write("- Consider database migration for high-volume audit logs\n")
        f.write("- Implement log rotation and archival policies\n")
        f.write("- Set up distributed processing for pattern analysis\n\n")
    
    print(f"   📝 Insights report: {insights_file}")
    
    # Generate summary statistics
    summary_file = demo_dir / "demo_summary.json"
    summary = {
        'demo_timestamp': datetime.now().isoformat(),
        'systems_status': status,
        'files_generated': list(demo_dir.glob('*')),
        'recommendations': [
            'Configure production SMTP settings',
            'Enable background processing',
            'Set up external integrations',
            'Monitor system resources',
            'Implement log rotation policies'
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"   📊 Summary data: {summary_file}")
    print("\n✅ Demo insights generation complete")

def main():
    """Main demo execution"""
    try:
        # Create and configure system
        system, demo_dir = create_comprehensive_demo()
        
        # Simulate realistic workload
        simulate_realistic_workload(system)
        
        # Demonstrate capabilities
        demonstrate_system_capabilities(system, demo_dir)
        
        # Generate insights
        generate_demo_insights(system, demo_dir)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print(f"📁 Demo output directory: {demo_dir.absolute()}")
        print(f"🌐 Dashboard URL: http://localhost:{system.config.dashboard_port}")
        print("\n📋 Generated Files:")
        
        for file_path in sorted(demo_dir.glob('*')):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
        
        print("\n💡 Next Steps:")
        print("   1. Review generated reports and insights")
        print("   2. Explore the live dashboard")
        print("   3. Configure for your production environment")
        print("   4. Integrate with your existing systems")
        
        print("\n🚀 System is running - Press Ctrl+C to stop")
        
        # Keep system running for exploration
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
        if 'system' in locals():
            system.stop()
        print("✅ System shutdown complete")
    
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if 'system' in locals():
            system.stop()
        raise

if __name__ == '__main__':
    main()
