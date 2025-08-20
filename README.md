# Comprehensive Supervisor Agent Reporting System

A complete reporting and feedback infrastructure for the Supervisor Agent, providing real-time monitoring, analytics, alerts, and comprehensive observability capabilities.

## 🎯 System Overview

The Supervisor Agent Reporting System provides comprehensive monitoring and analytics capabilities including:

- **Real-Time Alerts** - Multi-channel notifications with deduplication
- **Periodic Summaries** - Automated performance analytics and trend analysis  
- **Audit Trails** - Machine-readable event logging with structured metadata
- **Confidence Reporting** - Decision calibration analysis and accuracy metrics
- **Pattern Tracking** - Automated failure pattern detection and knowledge building
- **Dashboard Visualization** - Real-time performance monitoring with charts
- **Export System** - Multi-format data export with job management

## 🏗️ Architecture

The system is built with a modular architecture:

```
src/reporting/
├── main.py                 # Main system coordinator
├── alerts.py              # Real-time alert system
├── summaries.py           # Periodic report generation
├── audit_system.py        # Audit trail management
├── confidence.py          # Confidence score analysis
├── patterns.py            # Pattern detection & knowledge base
├── dashboard.py           # Dashboard data & visualization
├── export_system.py       # Multi-format export system
├── config.json            # System configuration
└── comprehensive_demo.py  # Complete system demonstration
```

## 🚀 Quick Start

### Basic Usage

```python
from src.reporting.main import SupervisorReportingSystem

# Initialize with configuration
reporting_system = SupervisorReportingSystem('config.json')

# Start the system (async)
await reporting_system.start()
```

### Run Comprehensive Demo

```bash
# Run the complete system demonstration
cd src/reporting/
python comprehensive_demo.py
```

This will generate extensive output files demonstrating all system capabilities.

## 📊 Core Components

### 1. Real-Time Alert System

```python
from alerts import AlertManager, AlertType, AlertSeverity

# Create alerts with multi-channel notifications
alert_manager.create_alert(
    AlertType.SYSTEM_FAILURE,
    AlertSeverity.CRITICAL,
    "Database Connection Lost",
    "Primary database connection has been lost",
    "agent_001",
    {'database': 'primary', 'last_connection': '2025-08-18 16:45:00'}
)
```

**Features:**
- Multi-channel notifications (Email, Slack, Webhooks)
- Alert deduplication with time windows
- Priority-based alert routing
- Condition-based alert triggering
- Alert lifecycle management

### 2. Periodic Summary Reports

```python
from summaries import ReportGenerator

# Generate comprehensive period summary
summary = report_generator.generate_period_summary(hours=24)

# Export as Markdown report
markdown_report = report_generator.generate_markdown_report(summary)
```

**Features:**
- Task-level completion analytics
- Agent performance comparisons
- Success rate trends and metrics
- Actionable recommendations
- Multiple export formats (JSON, Markdown)

### 3. Machine-Readable Audit Trails

```python
from audit_system import AuditTrailManager, AuditEventType, AuditLevel

# Log structured audit events
audit_manager.log_task_completed(
    agent_id="agent_001",
    task_id="task_12345", 
    duration=23.4,
    confidence=0.89,
    metadata={'output_size': 1024}
)

# Query events with filters
events = audit_manager.query_events(
    start_time=datetime.now() - timedelta(hours=24),
    event_types=[AuditEventType.TASK_FAILED],
    agent_ids=["agent_001", "agent_002"]
)
```

**Features:**
- Structured event logging with metadata
- Searchable audit trails with filters
- Event correlation and session tracking
- Multiple storage formats (JSONL, JSON)
- Automatic cleanup and archiving

### 4. Confidence Score Reporting

```python
from confidence import ConfidenceReporter

# Record confidence predictions
confidence_reporter.record_confidence(
    agent_id="agent_001",
    task_id="task_12345",
    predicted_confidence=0.85,
    actual_success=True,
    task_duration=23.4
)

# Generate calibration analysis
metrics = confidence_reporter.generate_metrics(hours=24)
calibration_report = confidence_reporter.generate_calibration_report(metrics)
```

**Features:**
- Decision confidence tracking
- Calibration error analysis (ECE)
- Brier score calculation
- Over/under-confidence detection
- Agent-specific confidence metrics
- Trend analysis over time

### 5. Pattern Tracking & Knowledge Base

```python
from patterns import PatternTracker

# Analyze events for patterns
analysis_result = pattern_tracker.analyze_events(events)

# Get pattern-based recommendations
recommendations = pattern_tracker.get_pattern_recommendations(
    agent_id="agent_001",
    context={'task_type': 'data_processing'}
)
```

**Features:**
- Automated pattern detection (error, performance, temporal, agent-specific)
- Knowledge base building from patterns
- Pattern-based recommendations
- Failure sequence analysis
- Solution effectiveness tracking

### 6. Dashboard & Visualization

```python
from dashboard import DashboardManager

# Generate complete dashboard data
dashboard_data = dashboard_manager.generate_dashboard_data()

# Export for frontend consumption
dashboard_manager.export_dashboard_data('dashboard.json')
```

**Features:**
- Real-time system metrics
- Multiple chart types (line, bar, pie, histogram)
- Agent performance visualization
- System health indicators
- Responsive dashboard layouts

### 7. Export System

```python
from export_system import ExportManager

# Export audit logs with options
job_id = export_manager.export_audit_logs(
    format='jsonl',
    start_time=datetime.now() - timedelta(hours=24),
    agent_ids=['agent_001', 'agent_002'],
    compress=True
)

# Export performance reports
report_job_id = export_manager.export_performance_reports(
    format='pdf',
    period_hours=24,
    include_charts=True
)
```

**Features:**
- Multiple export formats (JSON, CSV, PDF, Markdown)
- Concurrent job management
- Progress tracking and status monitoring
- Compression and archiving
- Integration with external systems

## ⚙️ Configuration

The system uses a comprehensive JSON configuration:

```json
{
  "alerts": {
    "email": {
      "enabled": true,
      "smtp_host": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your-email@gmail.com",
      "to_addresses": ["admin@company.com"]
    },
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK"
    },
    "task_timeout_threshold": 300,
    "low_confidence_threshold": 0.3,
    "error_rate_threshold": 0.1
  },
  "audit": {
    "log_directory": "audit_logs",
    "max_memory_events": 10000,
    "queue_size": 1000
  },
  "confidence": {
    "calibration_bins": 10,
    "max_memory_entries": 10000
  },
  "patterns": {
    "min_pattern_frequency": 3,
    "pattern_lookback_days": 30,
    "similarity_threshold": 0.8
  }
}
```

## 🎮 Demo Outputs

The comprehensive demo generates extensive output files:

### Alert System
- `alert_summary.json` - Alert status and configurations

### Summary Reports  
- `summary_report_*.md` - Markdown performance reports
- `summary_*.json` - Raw summary data
- `combined_summary_report.md` - Multi-period analysis

### Audit System
- `audit_events.jsonl` - Machine-readable audit logs
- `audit_statistics.json` - System statistics

### Confidence Analysis
- `confidence_calibration_report_*.md` - Calibration analysis
- `confidence_metrics_*.json` - Detailed metrics
- `confidence_raw_data.json` - Raw confidence data

### Pattern Detection
- `detected_patterns.json` - All identified patterns
- `pattern_analysis_result.json` - Analysis summary
- `recommendations_*.json` - Agent-specific recommendations

### Dashboard
- `dashboard_data.json` - Complete dashboard data
- `dashboard.html` - Interactive HTML dashboard
- `charts/` - Individual chart data files
- `system_health_summary.json` - Health metrics

### Export System
- `export_job_summary.json` - Export job tracking
- Various format-specific export files

## 🔧 Integration

### With Existing Systems

```python
# Custom data source integration
class CustomDataSource:
    def get_agent_tasks(self, agent_id, start_time, end_time):
        # Your data retrieval logic
        return tasks

# Initialize with custom data source
report_generator = ReportGenerator(CustomDataSource(), config)
```

### API Endpoints

```python
# Flask/FastAPI integration example
@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    dashboard_data = dashboard_manager.generate_dashboard_data()
    return dashboard_data.metrics

@app.get("/alerts/active")
async def get_active_alerts():
    return alert_manager.get_active_alerts()

@app.post("/export/audit-logs")
async def export_audit_logs(request: ExportRequest):
    job_id = export_manager.export_audit_logs(**request.dict())
    return {"job_id": job_id}
```

## 📈 Performance

- **High Throughput**: Async processing with queuing
- **Memory Efficient**: Configurable memory limits with LRU eviction
- **Scalable**: Horizontal scaling support
- **Real-time**: Sub-second alert processing
- **Reliable**: Persistent storage with automatic failover

## 🛡️ Security

- **Data Privacy**: Configurable data retention policies
- **Access Control**: Role-based access to sensitive data
- **Encryption**: At-rest and in-transit data encryption
- **Audit Trail**: Complete audit trail of system access
- **Compliance**: GDPR/SOX compliance features

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run full system demonstration
python comprehensive_demo.py

# Run individual component tests  
python -m pytest tests/
```

## 📝 License

This project is part of the Supervisor Agent system and follows the same licensing terms.

## 🤝 Contributing

1. Follow the modular architecture patterns
2. Add comprehensive logging and error handling
3. Include configuration options for new features
4. Update the demo script with new capabilities
5. Add appropriate tests and documentation

---

**Status**: ✅ Production Ready  
**Last Updated**: 2025-08-18  
**Version**: 1.0.0