# Production Supervisor MCP Agent

## Overview

A comprehensive, production-ready supervisor agent that provides inline firewall and auditing capabilities for AI agents, with integrated monitoring, error handling, reporting, and framework integration.

## 🚀 Key Features

### Core Capabilities
- **Inline Design**: Acts as firewall + auditor between user input and agent output
- **Multi-Agent Orchestration**: Tracks multiple workers in parallel
- **Framework Agnostic**: Compatible with MCP, LangChain, AutoGen, and custom frameworks
- **Tiered Response System**: Warning → Correction → Escalation

### Integrated Systems
- **Comprehensive Monitoring**: Task completion, instruction adherence, output quality, error tracking
- **Error Handling & Recovery**: Auto-retry, state rollback, loop detection, human escalation
- **Advanced Reporting**: Real-time alerts, audit trails, confidence scoring, pattern detection
- **Framework Integration**: Adapters for major AI agent frameworks

### Production Features
- **Security & Access Control**: Input validation, audit logging, secure state management
- **Performance Optimization**: Background processing, efficient snapshots, resource monitoring
- **Scalability**: Configurable worker pools, distributed processing support
- **Reliability**: Comprehensive error handling, automatic recovery, circuit breakers

## 📋 Installation

### Quick Start
```bash
cd supervisor-mcp-agent
uv sync
python test_comprehensive.py  # Validate installation
bash run.sh                    # Start the server
```

### Production Setup
```bash
# Set environment variables
export SUPERVISOR_DATA_DIR="/var/lib/supervisor-agent"
export LOG_LEVEL="INFO"
export ENABLE_REAL_TIME_ALERTS="true"
export MAX_WORKERS="8"

# Run the integrated server
python server_integrated.py
```

## 🛠 Architecture

### System Components

1. **Core Supervisor** (`src/supervisor_agent/`)
   - Basic monitoring and intervention
   - Task tracking and validation
   - Knowledge base management

2. **Monitoring Engine** (`src/monitoring/`)
   - Real-time task monitoring
   - Quality assessment and confidence scoring
   - Resource usage tracking
   - Error detection and tracking

3. **Error Handling System** (`src/error_handling/`)
   - Automatic retry with progressive strategies
   - State rollback and recovery
   - Loop detection and circuit breakers
   - Human escalation management

4. **Reporting System** (`src/reporting/`)
   - Comprehensive analytics and reports
   - Real-time alerting system
   - Pattern detection and learning
   - Dashboard and visualization

5. **Integration Layer** (`src/integrated_supervisor.py`)
   - Framework adapters (MCP, LangChain, AutoGen, Custom)
   - Unified orchestration
   - Configuration management
   - Performance optimization

## 🔧 Available MCP Tools

### Enhanced Integrated Tools
1. `start_supervision_session` - Start comprehensive supervision with framework integration
2. `execute_supervised_task` - Execute tasks under full supervision with monitoring
3. `monitor_agent_comprehensive` - Enhanced monitoring with real-time tracking
4. `handle_error_with_recovery` - Comprehensive error handling with auto-recovery
5. `generate_comprehensive_reports` - Advanced reporting with analytics
6. `configure_system_wide_settings` - System-wide configuration management
7. `get_integration_status` - Framework integration and system status

### Legacy Compatibility Tools
8. `monitor_agent` - Basic agent monitoring (backward compatibility)
9. `validate_output` - Output validation (backward compatibility)
10. `get_system_status` - Basic system status

## 📊 Framework Integration

### MCP Framework
```python
# Automatic supervision for MCP tools
result = await supervisor.mcp_adapter.wrap_tool_call(
    tool_name="your_tool",
    args={"param": "value"},
    task_id="task_1"
)
```

### LangChain Integration
```python
# Supervised chain execution
result = await supervisor.langchain_adapter.wrap_chain_execution(
    chain=your_chain,
    inputs={"input": "test"},
    task_id="task_2"
)
```

### AutoGen Integration
```python
# Supervised agent execution
result = await supervisor.autogen_adapter.wrap_agent_execution(
    agent=your_agent,
    message="Hello",
    task_id="task_3"
)
```

### Custom Framework
```python
# Context manager for custom frameworks
async with supervisor.supervision_context("task_4", "custom"):
    result = await your_custom_task()
```

## ⚙️ Configuration

### Environment Variables
- `SUPERVISOR_DATA_DIR`: Data storage directory (default: `supervisor_data`)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENABLE_REAL_TIME_ALERTS`: Enable real-time alerts (true/false)
- `MAX_WORKERS`: Maximum worker threads (default: 4)
- `MAX_AUDIT_LOG_SIZE_MB`: Max audit log size before rotation

### Configuration Object
```python
config = SupervisorConfig(
    monitoring_enabled=True,
    error_handling_enabled=True,
    reporting_enabled=True,
    quality_threshold=0.7,
    confidence_threshold=0.8,
    max_retries=3,
    real_time_alerts=True
)
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
python test_comprehensive.py
```

Tests include:
- Basic supervisor functionality
- Integrated system components
- MCP server tools
- Error handling and recovery
- Monitoring capabilities
- Reporting system
- Performance benchmarks

### Test Categories
- **Unit Tests**: Individual component validation
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Benchmarking and load testing
- **Error Scenario Tests**: Recovery and resilience testing

## 📈 Monitoring and Analytics

### Real-time Monitoring
- Task completion rates
- Quality score trends
- Error patterns and frequency
- Resource usage metrics
- Agent performance analytics

### Reporting Features
- Automated periodic reports
- Custom report generation
- Pattern detection and alerts
- Confidence tracking
- Audit trail management

### Dashboard Integration
- Web-based monitoring dashboard
- Real-time metrics visualization
- Historical trend analysis
- Alert management interface

## 🔒 Security Features

### Access Control
- Input validation and sanitization
- Secure state management
- Audit logging for all operations
- Configurable escalation thresholds

### Data Protection
- Encrypted state snapshots
- Secure audit trail storage
- Configurable data retention
- Privacy-aware logging

## 📚 Documentation

### Component Documentation
- [Monitoring System](src/monitoring/README.md)
- [Error Handling](src/error_handling/README.md)
- [Reporting System](src/reporting/README.md)
- [Deployment Guide](DEPLOYMENT.md)

### Examples
- [Usage Examples](examples/usage_examples.py)
- [Framework Integration Examples](examples/)
- [Configuration Examples](DEPLOYMENT.md)

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for comprehensive deployment instructions including:
- Development and production setups
- Container deployment
- Configuration management
- Monitoring and maintenance
- Troubleshooting guides

## 🔄 Version Information

- **Current Version**: 2.0.0 (Integrated)
- **Compatibility**: MCP Protocol v1.0+
- **Python Version**: 3.8+
- **Dependencies**: See `pyproject.toml`

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📞 Support

For support and questions:
- Run the comprehensive test suite for diagnostics
- Check component documentation
- Review troubleshooting guide in DEPLOYMENT.md