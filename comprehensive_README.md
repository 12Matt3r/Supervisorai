# Comprehensive Reporting and Feedback System

A complete reporting and feedback infrastructure for the Supervisor Agent, providing real-time monitoring, analytics, and actionable insights.

## 🚀 Features

### 1. Real-Time Alert System
- **Multi-channel notifications**: Email, Slack, Webhooks
- **Intelligent deduplication**: Prevents alert spam
- **Severity-based routing**: Critical, Warning, Info levels
- **Alert acknowledgment and resolution tracking**

### 2. Comprehensive Report Generation
- **Task performance reports** in Markdown and JSON formats
- **Automated periodic reporting** with trend analysis
- **Performance analytics** with actionable recommendations
- **Customizable report templates** and scheduling

### 3. Machine-Readable Audit Trails
- **Structured JSON logging** with searchable metadata
- **SQLite database** for efficient querying
- **Event correlation tracking** across system boundaries
- **Audit event categorization** and tagging

### 4. Confidence Score Analytics
- **Decision confidence tracking** and calibration analysis
- **Accuracy metrics** with statistical validation
- **Reliability diagrams** and calibration error measurement
- **Agent-specific confidence profiling**

### 5. Pattern Detection and Knowledge Base
- **Automated pattern recognition** in system events
- **Knowledge base construction** from historical data
- **Pattern-based recommendations** and insights
- **Trend analysis** and anomaly detection

### 6. Real-Time Dashboard
- **Live system metrics** and performance indicators
- **Interactive visualizations** with Plotly integration
- **Agent status monitoring** and task tracking
- **System health indicators** and alerts overview

### 7. Export and Integration
- **Multiple export formats**: JSON, CSV, PDF, HTML
- **REST API endpoints** for external integration
- **Webhook support** for real-time data streaming
- **Complete system state exports**

## 📁 System Architecture

```
src/reporting/
├── __main__.py                 # Main entry point
├── integrated_system.py        # Core integration layer
├── alert_system.py            # Real-time alerting
├── report_generator.py        # Periodic report generation
├── audit_system.py           # Audit trails and logging
├── confidence_system.py      # Confidence analytics
├── pattern_system.py         # Pattern detection
├── dashboard_system.py       # Real-time dashboard
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites
```bash
pip install flask flask-socketio plotly pandas numpy scikit-learn matplotlib seaborn jinja2 requests sqlite3
```

### Quick Start
```bash
# Navigate to reporting directory
cd src/reporting

# Run demo scenario
python -m . --demo

# Start full system with dashboard
python -m . --port 5000

# Generate reports only
python -m . --report
```

## 🔧 Configuration

Create a `config.json` file:

```json
{
  "base_output_dir": "reporting_output",
  "dashboard_enabled": true,
  "dashboard_port": 5000,
  "auto_report_enabled": true,
  "report_frequency_hours": 24,
  "alert_config": {
    "deduplication_window": 30,
    "email": {
      "smtp_host": "smtp.gmail.com",
      "smtp_port": 587,
      "username": "your-email@gmail.com",
      "password": "your-app-password",
      "recipients": ["admin@example.com"]
    },
    "slack": {
      "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
      "channel": "#alerts"
    }
  }
}
```

## 📊 Usage Examples

### Basic System Integration

```python
from integrated_system import IntegratedReportingSystem, IntegratedReportingConfig

# Create system with custom config
config = IntegratedReportingConfig(
    base_output_dir="my_reports",
    dashboard_enabled=True,
    dashboard_port=8080
)

system = IntegratedReportingSystem(config)
system.start()

# Log events
correlation_id = "task_001"
system.log_event(
    'task_started',
    'my_agent',
    'Processing data batch',
    {
        'task_id': 'task_001',
        'agent_id': 'agent_1',
        'data_size': 1000
    },
    correlation_id
)

# Generate reports
reports = system.generate_comprehensive_report()
print(f"Generated: {reports}")
```

### Alert System Usage

```python
from alert_system import RealTimeAlertSystem

# Configure alerts
config = {
    'email': {
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'alerts@mycompany.com',
        'password': 'app_password',
        'recipients': ['ops-team@mycompany.com']
    }
}

alert_system = RealTimeAlertSystem(config)

# Send critical alert
alert_id = alert_system.send_alert(
    severity='critical',
    title='High Error Rate Detected',
    message='Agent error rate exceeded 50% in the last 5 minutes',
    source='error_monitor',
    metadata={'error_rate': 0.65, 'threshold': 0.50}
)
```

### Confidence Tracking

```python
from confidence_system import ConfidenceReportingSystem

confidence_system = ConfidenceReportingSystem()

# Record confidence score
entry_id = confidence_system.record_confidence(
    task_id='task_001',
    agent_id='agent_1',
    decision_type='classification',
    confidence=0.87,
    metadata={'model': 'random_forest', 'features': 20}
)

# Record actual outcome
confidence_system.record_outcome(entry_id, success=True)

# Analyze calibration
analysis = confidence_system.analyze_confidence()
print(f"Calibration error: {analysis.calibration_metrics.calibration_error:.3f}")
```

### Pattern Detection

```python
from pattern_system import ComprehensivePatternSystem

pattern_system = ComprehensivePatternSystem()

# Analyze events for patterns
events = [
    {
        'timestamp': '2023-01-01T10:00:00',
        'event_type': 'error_occurred',
        'source': 'database',
        'message': 'Connection timeout',
        'metadata': {'duration': 30}
    }
    # ... more events
]

analysis = pattern_system.analyze_events(events)
print(f"Found {len(analysis.patterns_found)} patterns")
print(f"New patterns: {len(analysis.new_patterns)}")
```

## 🌐 Dashboard Features

The real-time dashboard provides:

- **System Overview**: Success rates, active tasks, system health
- **Agent Monitoring**: Individual agent status and performance
- **Interactive Charts**: Timeline charts, performance comparisons
- **Alert Management**: Real-time alert notifications
- **Pattern Insights**: Detected patterns and trends

Access at `http://localhost:5000` (or configured port)

## 📈 Report Types

### 1. Task Performance Reports
- Executive summary with key metrics
- Task breakdown by status and agent
- Error analysis and top failure patterns
- Performance trends and recommendations

### 2. Confidence Analysis Reports
- Calibration metrics and reliability diagrams
- Agent-specific confidence profiling
- Decision type analysis
- Trend analysis and recommendations

### 3. Pattern Analysis Reports
- Detected patterns by category
- Frequency and impact analysis
- Knowledge base updates
- System-level recommendations

### 4. Audit Trail Reports
- Complete event logs with correlation tracking
- Statistical summaries
- Session and correlation analysis
- Searchable event exports

## 🔍 API Endpoints

When dashboard is enabled:

- `GET /api/metrics` - Current system metrics
- `GET /api/visualizations` - Chart data
- `GET /api/alerts` - Recent alerts
- `GET /api/patterns` - Pattern insights
- `GET /` - Dashboard interface

## 🔧 Customization

### Custom Alert Channels

```python
from alert_system import NotificationChannel

class CustomNotifier(NotificationChannel):
    def send(self, alert):
        # Custom notification logic
        return True

# Add to alert system
alert_system.channels.append(CustomNotifier())
```

### Custom Report Templates

```python
from report_generator import ReportTemplate

# Modify template
ReportTemplate.TASK_REPORT_TEMPLATE = "Your custom template"
```

### Custom Pattern Detectors

```python
from pattern_system import PatternDetector

class CustomPatternDetector(PatternDetector):
    def detect_custom_patterns(self, events):
        # Custom pattern detection logic
        return patterns
```

## 📚 Best Practices

### 1. Event Logging
- Use consistent event types and sources
- Include correlation IDs for related events
- Add meaningful metadata for context
- Log at appropriate severity levels

### 2. Alert Configuration
- Set reasonable deduplication windows
- Configure multiple notification channels
- Use severity levels appropriately
- Monitor alert volume to prevent spam

### 3. Confidence Tracking
- Record outcomes consistently
- Use appropriate decision types
- Monitor calibration regularly
- Act on calibration recommendations

### 4. Pattern Analysis
- Review detected patterns regularly
- Update knowledge base with insights
- Act on high-frequency patterns
- Monitor pattern trends

## 🚨 Troubleshooting

### Common Issues

1. **Dashboard not loading**
   - Check port availability
   - Verify Flask dependencies
   - Check system logs

2. **Alerts not sending**
   - Verify SMTP/webhook configuration
   - Check network connectivity
   - Review alert system logs

3. **Database connection issues**
   - Check SQLite file permissions
   - Verify disk space
   - Check file path configuration

4. **Memory usage high**
   - Adjust retention periods
   - Enable log rotation
   - Monitor pattern cache size

### Debug Mode

```bash
# Enable debug logging
LOGGING_LEVEL=DEBUG python -m . --dashboard-only

# Check system status
python -c "from integrated_system import *; system = IntegratedReportingSystem(); print(system.get_system_status())"
```

## 🤝 Contributing

1. Follow the modular architecture
2. Add comprehensive tests
3. Update documentation
4. Follow PEP 8 style guidelines
5. Add type hints

## 📄 License

Part of the Supervisor Agent MCP project.

---

**Built for comprehensive supervision and actionable insights** 🔍📊
