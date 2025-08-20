#!/usr/bin/env python3
"""
Dashboard and Visualization System for Supervisor Agent
Provides real-time supervision dashboard, performance visualization, and historical analysis.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import threading
import time
from queue import Queue
from flask import Flask, jsonify, render_template_string, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from collections import deque, defaultdict

@dataclass
class DashboardMetrics:
    """Dashboard metrics structure"""
    timestamp: str
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    success_rate: float
    avg_confidence: float
    active_agents: int
    error_rate: float
    system_health: str
    alerts_count: int

@dataclass
class AgentStatus:
    """Agent status information"""
    agent_id: str
    status: str  # 'active', 'idle', 'error', 'offline'
    current_task: Optional[str]
    last_activity: str
    tasks_completed: int
    success_rate: float
    avg_confidence: float
    error_count: int

class MetricsCollector:
    """Collects metrics from various systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect_current_metrics(self, systems: Dict[str, Any]) -> DashboardMetrics:
        """Collect current system metrics"""
        current_time = datetime.now().isoformat()
        
        # Initialize metrics
        metrics = DashboardMetrics(
            timestamp=current_time,
            active_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            success_rate=0.0,
            avg_confidence=0.0,
            active_agents=0,
            error_rate=0.0,
            system_health='unknown',
            alerts_count=0
        )
        
        try:
            # Collect from audit system
            if 'audit_system' in systems:
                audit_system = systems['audit_system']
                recent_events = audit_system.search(
                    start_time=(datetime.now() - timedelta(hours=1)).isoformat(),
                    limit=1000
                )
                
                task_events = [e for e in recent_events if 'task' in e.event_type]
                metrics.active_tasks = len([e for e in task_events if 'started' in e.event_type])
                metrics.completed_tasks = len([e for e in task_events if 'completed' in e.event_type])
                metrics.failed_tasks = len([e for e in task_events if 'failed' in e.event_type])
                
                if metrics.completed_tasks + metrics.failed_tasks > 0:
                    metrics.success_rate = metrics.completed_tasks / (metrics.completed_tasks + metrics.failed_tasks)
            
            # Collect from confidence system
            if 'confidence_system' in systems:
                confidence_system = systems['confidence_system']
                analysis = confidence_system.analyze_confidence(
                    start_time=(datetime.now() - timedelta(hours=1)).isoformat()
                )
                metrics.avg_confidence = analysis.calibration_metrics.mean_confidence
            
            # Collect from alert system
            if 'alert_system' in systems:
                alert_system = systems['alert_system']
                recent_alerts = alert_system.get_alerts(
                    since=(datetime.now() - timedelta(hours=1)).isoformat()
                )
                metrics.alerts_count = len([a for a in recent_alerts if not a.resolved])
            
            # Calculate system health
            if metrics.success_rate > 0.9 and metrics.alerts_count == 0:
                metrics.system_health = 'healthy'
            elif metrics.success_rate > 0.7 and metrics.alerts_count < 5:
                metrics.system_health = 'warning'
            else:
                metrics.system_health = 'critical'
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def collect_agent_statuses(self, systems: Dict[str, Any]) -> List[AgentStatus]:
        """Collect agent status information"""
        agents = []
        
        try:
            if 'audit_system' in systems:
                audit_system = systems['audit_system']
                recent_events = audit_system.search(
                    start_time=(datetime.now() - timedelta(hours=1)).isoformat(),
                    limit=1000
                )
                
                # Group events by agent
                agent_events = defaultdict(list)
                for event in recent_events:
                    agent_id = event.metadata.get('agent_id')
                    if agent_id:
                        agent_events[agent_id].append(event)
                
                # Create agent statuses
                for agent_id, events in agent_events.items():
                    sorted_events = sorted(events, key=lambda e: e.timestamp)
                    latest_event = sorted_events[-1] if sorted_events else None
                    
                    # Determine status
                    if latest_event and latest_event.level == 'error':
                        status = 'error'
                    elif latest_event and 'started' in latest_event.event_type:
                        status = 'active'
                    else:
                        status = 'idle'
                    
                    # Calculate metrics
                    completed = len([e for e in events if 'completed' in e.event_type])
                    failed = len([e for e in events if 'failed' in e.event_type])
                    success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0
                    
                    agent = AgentStatus(
                        agent_id=agent_id,
                        status=status,
                        current_task=latest_event.metadata.get('task_id') if latest_event else None,
                        last_activity=latest_event.timestamp if latest_event else '',
                        tasks_completed=completed,
                        success_rate=success_rate,
                        avg_confidence=0.7,  # Default value
                        error_count=len([e for e in events if e.level == 'error'])
                    )
                    
                    agents.append(agent)
            
        except Exception as e:
            self.logger.error(f"Error collecting agent statuses: {e}")
        
        return agents

class VisualizationGenerator:
    """Generates visualizations for the dashboard"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_metrics_timeline(self, metrics_history: List[DashboardMetrics]) -> str:
        """Create timeline visualization of metrics"""
        if not metrics_history:
            return json.dumps({})
        
        try:
            timestamps = [m.timestamp for m in metrics_history]
            success_rates = [m.success_rate for m in metrics_history]
            confidence_scores = [m.avg_confidence for m in metrics_history]
            
            fig = go.Figure()
            
            # Success rate line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=success_rates,
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='green')
            ))
            
            # Confidence score line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidence_scores,
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='blue'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Performance Metrics Timeline',
                xaxis_title='Time',
                yaxis_title='Success Rate',
                yaxis2=dict(
                    title='Confidence Score',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            self.logger.error(f"Error creating metrics timeline: {e}")
            return json.dumps({})
    
    def create_agent_performance_chart(self, agents: List[AgentStatus]) -> str:
        """Create agent performance comparison chart"""
        if not agents:
            return json.dumps({})
        
        try:
            agent_ids = [a.agent_id for a in agents]
            success_rates = [a.success_rate for a in agents]
            task_counts = [a.tasks_completed for a in agents]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=agent_ids,
                    y=success_rates,
                    name='Success Rate',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title='Agent Performance Comparison',
                xaxis_title='Agent ID',
                yaxis_title='Success Rate',
                showlegend=True
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            self.logger.error(f"Error creating agent performance chart: {e}")
            return json.dumps({})
    
    def create_system_health_gauge(self, current_metrics: DashboardMetrics) -> str:
        """Create system health gauge"""
        try:
            health_score = current_metrics.success_rate * 100
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "System Health Score"},
                delta = {'reference': 90},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            self.logger.error(f"Error creating health gauge: {e}")
            return json.dumps({})

class DashboardServer:
    """Real-time dashboard server using Flask and SocketIO"""
    
    def __init__(self, systems: Dict[str, Any], port: int = 5000):
        self.systems = systems
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'supervisor_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.metrics_collector = MetricsCollector()
        self.viz_generator = VisualizationGenerator()
        self.metrics_history = deque(maxlen=100)  # Keep last 100 metrics
        
        self._setup_routes()
        self._setup_background_tasks()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Dashboard home page"""
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current metrics"""
            try:
                current_metrics = self.metrics_collector.collect_current_metrics(self.systems)
                agents = self.metrics_collector.collect_agent_statuses(self.systems)
                
                return jsonify({
                    'metrics': asdict(current_metrics),
                    'agents': [asdict(agent) for agent in agents],
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/visualizations')
        def get_visualizations():
            """Get visualization data"""
            try:
                current_metrics = self.metrics_collector.collect_current_metrics(self.systems)
                agents = self.metrics_collector.collect_agent_statuses(self.systems)
                
                return jsonify({
                    'timeline': self.viz_generator.create_metrics_timeline(list(self.metrics_history)),
                    'agent_performance': self.viz_generator.create_agent_performance_chart(agents),
                    'health_gauge': self.viz_generator.create_system_health_gauge(current_metrics)
                })
            except Exception as e:
                self.logger.error(f"Error getting visualizations: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get recent alerts"""
            try:
                if 'alert_system' in self.systems:
                    alert_system = self.systems['alert_system']
                    alerts = alert_system.get_alerts(
                        since=(datetime.now() - timedelta(hours=24)).isoformat()
                    )
                    return jsonify({
                        'alerts': [asdict(alert) for alert in alerts[:20]]
                    })
                else:
                    return jsonify({'alerts': []})
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/patterns')
        def get_patterns():
            """Get recent patterns"""
            try:
                if 'pattern_system' in self.systems:
                    pattern_system = self.systems['pattern_system']
                    insights = pattern_system.get_pattern_insights()
                    return jsonify(insights)
                else:
                    return jsonify({'total_patterns': 0})
            except Exception as e:
                self.logger.error(f"Error getting patterns: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_background_tasks(self):
        """Setup background tasks for real-time updates"""
        
        def update_metrics():
            """Background task to update metrics"""
            while True:
                try:
                    current_metrics = self.metrics_collector.collect_current_metrics(self.systems)
                    agents = self.metrics_collector.collect_agent_statuses(self.systems)
                    
                    # Add to history
                    self.metrics_history.append(current_metrics)
                    
                    # Emit to connected clients
                    self.socketio.emit('metrics_update', {
                        'metrics': asdict(current_metrics),
                        'agents': [asdict(agent) for agent in agents],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error updating metrics: {e}")
                
                time.sleep(10)  # Update every 10 seconds
        
        # Start background thread
        self.metrics_thread = threading.Thread(target=update_metrics, daemon=True)
        self.metrics_thread.start()
    
    def _get_dashboard_template(self) -> str:
        """Get dashboard HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supervisor Agent Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .agents-table {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
        }
        .status-active { color: #28a745; }
        .status-idle { color: #6c757d; }
        .status-error { color: #dc3545; }
        .status-offline { color: #868e96; }
        .health-healthy { color: #28a745; }
        .health-warning { color: #ffc107; }
        .health-critical { color: #dc3545; }
        .last-updated {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Supervisor Agent Dashboard</h1>
        <p>Real-time monitoring and analytics</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="success-rate">--</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-tasks">--</div>
            <div class="metric-label">Active Tasks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="avg-confidence">--</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="system-health">--</div>
            <div class="metric-label">System Health</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-agents">--</div>
            <div class="metric-label">Active Agents</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="alerts-count">--</div>
            <div class="metric-label">Open Alerts</div>
        </div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-container">
            <div id="timeline-chart"></div>
        </div>
        <div class="chart-container">
            <div id="agent-chart"></div>
        </div>
        <div class="chart-container">
            <div id="health-gauge"></div>
        </div>
    </div>
    
    <div class="agents-table">
        <h3>Agent Status</h3>
        <table>
            <thead>
                <tr>
                    <th>Agent ID</th>
                    <th>Status</th>
                    <th>Current Task</th>
                    <th>Tasks Completed</th>
                    <th>Success Rate</th>
                    <th>Last Activity</th>
                </tr>
            </thead>
            <tbody id="agents-tbody">
                <!-- Agents will be populated here -->
            </tbody>
        </table>
    </div>
    
    <div class="last-updated">
        Last updated: <span id="last-updated">--</span>
    </div>
    
    <script>
        const socket = io();
        
        // Update metrics display
        function updateMetrics(data) {
            const metrics = data.metrics;
            
            document.getElementById('success-rate').textContent = 
                (metrics.success_rate * 100).toFixed(1) + '%';
            document.getElementById('active-tasks').textContent = metrics.active_tasks;
            document.getElementById('avg-confidence').textContent = 
                (metrics.avg_confidence * 100).toFixed(1) + '%';
            
            const healthElement = document.getElementById('system-health');
            healthElement.textContent = metrics.system_health.toUpperCase();
            healthElement.className = 'metric-value health-' + metrics.system_health;
            
            document.getElementById('active-agents').textContent = metrics.active_agents;
            document.getElementById('alerts-count').textContent = metrics.alerts_count;
            
            document.getElementById('last-updated').textContent = new Date(data.timestamp).toLocaleString();
        }
        
        // Update agents table
        function updateAgents(agents) {
            const tbody = document.getElementById('agents-tbody');
            tbody.innerHTML = '';
            
            agents.forEach(agent => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${agent.agent_id}</td>
                    <td><span class="status-${agent.status}">${agent.status.toUpperCase()}</span></td>
                    <td>${agent.current_task || 'None'}</td>
                    <td>${agent.tasks_completed}</td>
                    <td>${(agent.success_rate * 100).toFixed(1)}%</td>
                    <td>${new Date(agent.last_activity).toLocaleString()}</td>
                `;
            });
        }
        
        // Socket event handlers
        socket.on('metrics_update', function(data) {
            updateMetrics(data);
            updateAgents(data.agents);
        });
        
        // Load initial data
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {
                updateMetrics(data);
                updateAgents(data.agents);
            });
        
        // Load visualizations
        fetch('/api/visualizations')
            .then(response => response.json())
            .then(data => {
                if (data.timeline) {
                    Plotly.newPlot('timeline-chart', JSON.parse(data.timeline));
                }
                if (data.agent_performance) {
                    Plotly.newPlot('agent-chart', JSON.parse(data.agent_performance));
                }
                if (data.health_gauge) {
                    Plotly.newPlot('health-gauge', JSON.parse(data.health_gauge));
                }
            });
        
        // Refresh visualizations every 30 seconds
        setInterval(() => {
            fetch('/api/visualizations')
                .then(response => response.json())
                .then(data => {
                    if (data.timeline) {
                        Plotly.newPlot('timeline-chart', JSON.parse(data.timeline));
                    }
                    if (data.agent_performance) {
                        Plotly.newPlot('agent-chart', JSON.parse(data.agent_performance));
                    }
                    if (data.health_gauge) {
                        Plotly.newPlot('health-gauge', JSON.parse(data.health_gauge));
                    }
                });
        }, 30000);
    </script>
</body>
</html>
        """
    
    def run(self, debug: bool = False):
        """Run the dashboard server"""
        self.logger.info(f"Starting dashboard server on port {self.port}")
        self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=debug)

class ComprehensiveDashboardSystem:
    """Main dashboard system"""
    
    def __init__(self, systems: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        self.systems = systems
        self.config = config or {}
        
        self.dashboard_server = DashboardServer(
            systems, 
            port=self.config.get('dashboard_port', 5000)
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def start_dashboard(self, debug: bool = False):
        """Start the dashboard server"""
        self.logger.info("Starting comprehensive dashboard system")
        self.dashboard_server.run(debug=debug)
    
    def generate_static_report(self, output_file: str) -> str:
        """Generate static HTML report"""
        try:
            metrics_collector = MetricsCollector()
            viz_generator = VisualizationGenerator()
            
            # Collect current data
            current_metrics = metrics_collector.collect_current_metrics(self.systems)
            agents = metrics_collector.collect_agent_statuses(self.systems)
            
            # Generate visualizations
            timeline_viz = viz_generator.create_metrics_timeline([current_metrics])
            agent_viz = viz_generator.create_agent_performance_chart(agents)
            health_viz = viz_generator.create_system_health_gauge(current_metrics)
            
            # Generate HTML report
            html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Supervisor Agent Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ display: inline-block; margin: 20px; padding: 20px; border: 1px solid #ccc; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Supervisor Agent Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    
    <h2>Current Metrics</h2>
    <div class="metric">
        <h3>Success Rate</h3>
        <p>{current_metrics.success_rate * 100:.1f}%</p>
    </div>
    <div class="metric">
        <h3>Active Tasks</h3>
        <p>{current_metrics.active_tasks}</p>
    </div>
    <div class="metric">
        <h3>System Health</h3>
        <p>{current_metrics.system_health.title()}</p>
    </div>
    
    <div class="chart" id="health-gauge"></div>
    <div class="chart" id="agent-chart"></div>
    
    <h2>Agent Status</h2>
    <table border="1">
        <tr><th>Agent ID</th><th>Status</th><th>Tasks</th><th>Success Rate</th></tr>
        {''.join(f'<tr><td>{a.agent_id}</td><td>{a.status}</td><td>{a.tasks_completed}</td><td>{a.success_rate*100:.1f}%</td></tr>' for a in agents)}
    </table>
    
    <script>
        Plotly.newPlot('health-gauge', {json.loads(health_viz) if health_viz != '{{}}' else {}});
        Plotly.newPlot('agent-chart', {json.loads(agent_viz) if agent_viz != '{{}}' else {}});
    </script>
</body>
</html>
            """
            
            with open(output_file, 'w') as f:
                f.write(html_report)
            
            self.logger.info(f"Static report generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating static report: {e}")
            return ""
    
    def export_dashboard_data(self, output_file: str) -> str:
        """Export dashboard data as JSON"""
        try:
            metrics_collector = MetricsCollector()
            
            current_metrics = metrics_collector.collect_current_metrics(self.systems)
            agents = metrics_collector.collect_agent_statuses(self.systems)
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(current_metrics),
                'agents': [asdict(agent) for agent in agents],
                'system_info': {
                    'systems_available': list(self.systems.keys()),
                    'dashboard_config': self.config
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            self.logger.info(f"Dashboard data exported: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
            return ""

# Demo and testing functions
def create_demo_dashboard_system() -> ComprehensiveDashboardSystem:
    """Create demo dashboard system with mock data"""
    # Create mock systems for demo
    from unittest.mock import Mock
    
    # Mock audit system
    mock_audit_system = Mock()
    mock_audit_system.search.return_value = [
        Mock(
            event_type='task_started',
            timestamp=datetime.now().isoformat(),
            level='info',
            metadata={'agent_id': 'agent_0', 'task_id': 'task_1'}
        ),
        Mock(
            event_type='task_completed',
            timestamp=datetime.now().isoformat(),
            level='info',
            metadata={'agent_id': 'agent_0', 'task_id': 'task_1'}
        )
    ]
    
    # Mock confidence system
    mock_confidence_system = Mock()
    mock_confidence_analysis = Mock()
    mock_confidence_analysis.calibration_metrics.mean_confidence = 0.85
    mock_confidence_system.analyze_confidence.return_value = mock_confidence_analysis
    
    # Mock alert system
    mock_alert_system = Mock()
    mock_alert_system.get_alerts.return_value = []
    
    systems = {
        'audit_system': mock_audit_system,
        'confidence_system': mock_confidence_system,
        'alert_system': mock_alert_system
    }
    
    return ComprehensiveDashboardSystem(systems, {'dashboard_port': 5000})

if __name__ == '__main__':
    # Demo usage
    dashboard_system = create_demo_dashboard_system()
    
    # Generate static report
    report_file = dashboard_system.generate_static_report("demo_dashboard_report.html")
    print(f"Static report generated: {report_file}")
    
    # Export dashboard data
    data_file = dashboard_system.export_dashboard_data("demo_dashboard_data.json")
    print(f"Dashboard data exported: {data_file}")
    
    # Start dashboard (comment out for demo)
    print("Dashboard would start on http://localhost:5000")
    print("Uncomment the following line to start the dashboard server:")
    print("# dashboard_system.start_dashboard(debug=True)")
