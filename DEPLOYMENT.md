# Comprehensive Deployment Guide

## Supervisor MCP Agent - Production Deployment

This guide covers deploying the integrated supervisor agent in production environments with full monitoring, error handling, and reporting capabilities.

## Quick Start

### 1. Basic Installation

```bash
cd supervisor-mcp-agent

# Install dependencies
uv sync

# Create data directory
mkdir -p supervisor_data

# Run tests
python test_comprehensive.py

# Start the server
bash run.sh
```

### 2. Environment Configuration

Set environment variables for production:

```bash
export SUPERVISOR_DATA_DIR="/var/lib/supervisor-agent"
export LOG_LEVEL="INFO"
export ENABLE_REAL_TIME_ALERTS="true"
export MAX_WORKERS="8"
export MAX_AUDIT_LOG_SIZE_MB="500"
```

## Architecture Overview

### Integrated Systems

1. **Core Supervisor** (`supervisor_agent/core.py`)
   - Basic monitoring and intervention
   - Task tracking and validation
   - Knowledge base management

2. **Monitoring Engine** (`monitoring/`)
   - Real-time task monitoring
   - Quality assessment
   - Resource usage tracking
   - Confidence scoring

3. **Error Handling System** (`error_handling/`)
   - Automatic retry mechanisms
   - State rollback and recovery
   - Loop detection
   - Human escalation

4. **Reporting System** (`reporting/`)
   - Comprehensive analytics
   - Real-time alerts
   - Pattern detection
   - Dashboard integration

5. **Framework Integration** (`integrated_supervisor.py`)
   - MCP adapter
   - LangChain integration
   - AutoGen orchestration
   - Custom framework hooks

## Production Features

### Security and Access Control

- Input validation and sanitization
- Audit logging for all operations
- Configurable escalation thresholds
- Secure state management

### Performance Optimization

- Background processing for reports
- Efficient state snapshots
- Resource usage monitoring
- Configurable worker pools

### Monitoring and Alerting

- Real-time quality assessment
- Automatic error detection
- Pattern-based learning
- Multi-channel notifications

### Error Recovery

- Progressive retry strategies
- Automatic rollback capabilities
- Circuit breaker patterns
- Human-in-the-loop escalation

## Framework Integration Examples

### MCP Integration

```python
from supervisor_mcp import SupervisorMCP

# Initialize with supervision
supervisor_mcp = SupervisorMCP()

# All MCP tools are automatically supervised
result = await supervisor_mcp.call_tool(
    "your_tool",
    {"param": "value"}
)
```

### LangChain Integration

```python
from langchain import LLMChain
from supervisor_integration import SupervisorLangChain

# Wrap your chain with supervision
supervised_chain = SupervisorLangChain(your_chain)

# Execute with monitoring
result = await supervised_chain.arun({"input": "test"})
```

### AutoGen Integration

```python
from autogen import ConversableAgent
from supervisor_integration import SupervisorAutoGen

# Create supervised agent
agent = SupervisorAutoGen.create_supervised_agent(
    agent_config,
    supervision_config
)
```

### Custom Framework Integration

```python
from integrated_supervisor import IntegratedSupervisor

supervisor = IntegratedSupervisor()

# Use context manager for supervision
async with supervisor.supervision_context("task_1", "custom"):
    result = await your_custom_task()
```

## Configuration Options

### Monitoring Configuration

```json
{
  "monitoring": {
    "quality_threshold": 0.7,
    "confidence_threshold": 0.8,
    "max_token_threshold": 10000,
    "max_loop_cycles": 50,
    "max_runtime_minutes": 30.0
  }
}
```

### Error Handling Configuration

```json
{
  "error_handling": {
    "max_retries": 3,
    "rollback_enabled": true,
    "escalation_enabled": true,
    "auto_recovery": true
  }
}
```

### Reporting Configuration

```json
{
  "reporting": {
    "real_time_alerts": true,
    "auto_reports": true,
    "report_frequency_hours": 24,
    "dashboard_enabled": true
  }
}
```

## Deployment Scenarios

### Development Environment

```bash
# Basic setup with file-based storage
export SUPERVISOR_DATA_DIR="./dev_data"
export LOG_LEVEL="DEBUG"
python server_integrated.py
```

### Production Environment

```bash
# Full production setup
export SUPERVISOR_DATA_DIR="/var/lib/supervisor"
export LOG_LEVEL="INFO"
export ENABLE_REAL_TIME_ALERTS="true"
export MAX_WORKERS="16"

# Run with process manager
supervisorctl start supervisor-mcp-agent
```

### Containerized Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY supervisor-mcp-agent/ .

RUN pip install uv && uv sync

EXPOSE 8000
CMD ["python", "server_integrated.py"]
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check system status
curl -X POST http://localhost:8000/tools/get_integration_status

# Validate configuration
python -c "from integrated_supervisor import SupervisorConfig; print('Config valid')"
```

### Log Monitoring

```bash
# Monitor supervisor logs
tail -f supervisor_data/supervisor.log

# Monitor audit logs
tail -f supervisor_data/reporting/audit.jsonl
```

### Performance Monitoring

```bash
# Run performance tests
python test_comprehensive.py

# Check resource usage
ps aux | grep supervisor
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `uv sync`
2. **Permission Issues**: Check write permissions to data directory
3. **Memory Usage**: Adjust `MAX_WORKERS` based on available resources
4. **Alert Failures**: Verify notification channel configurations

### Debug Mode

```bash
export LOG_LEVEL="DEBUG"
export SUPERVISOR_DEBUG="true"
python server_integrated.py
```

### Recovery Procedures

1. **State Corruption**: Use rollback functionality
2. **Memory Leaks**: Restart with clean state
3. **Configuration Issues**: Validate with test suite
4. **Performance Degradation**: Check monitoring thresholds

## Support and Resources

- **Documentation**: See README.md files in each component directory
- **Test Suite**: Run `python test_comprehensive.py` for validation
- **Configuration**: Check `integrated_supervisor.py` for all options
- **Examples**: See `examples/` directory for usage patterns

## Updates and Maintenance

### Regular Maintenance

- Review audit logs weekly
- Update monitoring thresholds based on usage patterns
- Clean up old snapshots and reports
- Update knowledge base with new patterns

### Version Updates

1. Test in development environment
2. Backup current configuration and data
3. Deploy with rollback plan
4. Verify all integrations work correctly