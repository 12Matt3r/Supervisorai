"""
Dashboard and Visualization for Supervisor Agent
Real-time supervision dashboard with performance visualization
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class DashboardMetrics:
    timestamp: datetime
    system_status: str
    total_agents: int
    active_agents: int
    total_tasks_24h: int
    success_rate_24h: float
    avg_task_duration: float
    active_alerts: int
    critical_alerts: int
    confidence_score: float
    performance_trend: str
    error_rate_24h: float
    top_errors: List[Dict[str, Any]]
    agent_status: Dict[str, Dict[str, Any]]
    pattern_summary: Dict[str, Any]


@dataclass
class ChartData:
    chart_type: str
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Dict[str, Any]


@dataclass
class DashboardData:
    metrics: DashboardMetrics
    charts: List[ChartData]
    alerts: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    recommendations: List[str]
    last_updated: datetime


class DashboardManager:
    """Manages real-time dashboard data and visualizations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.update_interval = config.get('update_interval', 30)  # seconds
        
        # Data sources (will be set by main system)
        self.alert_manager = None
        self.report_generator = None
        self.confidence_reporter = None
        self.pattern_tracker = None
        
        self.logger.info("Dashboard Manager initialized")
        
    def set_data_sources(self, **sources):
        """Set references to data sources"""
        self.alert_manager = sources.get('alert_manager')
        self.report_generator = sources.get('report_generator')
        self.confidence_reporter = sources.get('confidence_reporter')
        self.pattern_tracker = sources.get('pattern_tracker')
        
    def generate_dashboard_data(self) -> DashboardData:
        """Generate complete dashboard data"""
        
        self.logger.debug("Generating dashboard data")
        
        # Generate metrics
        metrics = self._generate_metrics()
        
        # Generate charts
        charts = self._generate_charts()
        
        # Get alerts
        alerts = self._get_alert_data()
        
        # Get recent events
        recent_events = self._get_recent_events()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return DashboardData(
            metrics=metrics,
            charts=charts,
            alerts=alerts,
            recent_events=recent_events,
            recommendations=recommendations,
            last_updated=datetime.now()
        )
        
    def _generate_metrics(self) -> DashboardMetrics:
        """Generate key dashboard metrics"""
        
        # Mock data - in real implementation, this would query actual data sources
        now = datetime.now()
        
        # System status determination
        active_alerts = len(self.alert_manager.get_active_alerts()) if self.alert_manager else 0
        critical_alerts = len(self.alert_manager.get_alerts_by_severity(
            self.alert_manager.AlertSeverity.CRITICAL
        )) if self.alert_manager else 0
        
        if critical_alerts > 0:
            system_status = "critical"
        elif active_alerts > 5:
            system_status = "warning" 
        elif active_alerts > 0:
            system_status = "caution"
        else:
            system_status = "healthy"
            
        # Generate sample performance data
        performance_data = self._get_performance_data()
        
        return DashboardMetrics(
            timestamp=now,
            system_status=system_status,
            total_agents=performance_data['total_agents'],
            active_agents=performance_data['active_agents'],
            total_tasks_24h=performance_data['total_tasks_24h'],
            success_rate_24h=performance_data['success_rate_24h'],
            avg_task_duration=performance_data['avg_task_duration'],
            active_alerts=active_alerts,
            critical_alerts=critical_alerts,
            confidence_score=performance_data['confidence_score'],
            performance_trend=performance_data['performance_trend'],
            error_rate_24h=performance_data['error_rate_24h'],
            top_errors=performance_data['top_errors'],
            agent_status=performance_data['agent_status'],
            pattern_summary=performance_data['pattern_summary']
        )
        
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data from various sources"""
        
        # This is mock data - real implementation would query actual sources
        return {
            'total_agents': 5,
            'active_agents': 4,
            'total_tasks_24h': 847,
            'success_rate_24h': 0.912,
            'avg_task_duration': 23.4,
            'confidence_score': 0.78,
            'performance_trend': 'stable',
            'error_rate_24h': 0.088,
            'top_errors': [
                {'type': 'timeout', 'count': 23, 'percentage': 0.31},
                {'type': 'connection_error', 'count': 18, 'percentage': 0.24},
                {'type': 'validation_error', 'count': 12, 'percentage': 0.16}
            ],
            'agent_status': {
                'agent_001': {
                    'status': 'active',
                    'tasks_completed': 187,
                    'success_rate': 0.94,
                    'last_seen': datetime.now().isoformat()
                },
                'agent_002': {
                    'status': 'active', 
                    'tasks_completed': 203,
                    'success_rate': 0.89,
                    'last_seen': datetime.now().isoformat()
                },
                'agent_003': {
                    'status': 'warning',
                    'tasks_completed': 156,
                    'success_rate': 0.76,
                    'last_seen': (datetime.now() - timedelta(minutes=15)).isoformat()
                },
                'agent_004': {
                    'status': 'active',
                    'tasks_completed': 178,
                    'success_rate': 0.96,
                    'last_seen': datetime.now().isoformat()
                },
                'agent_005': {
                    'status': 'inactive',
                    'tasks_completed': 123,
                    'success_rate': 0.91,
                    'last_seen': (datetime.now() - timedelta(hours=2)).isoformat()
                }
            },
            'pattern_summary': {
                'total_patterns': 12,
                'critical_patterns': 2,
                'new_patterns_24h': 1
            }
        }
        
    def _generate_charts(self) -> List[ChartData]:
        """Generate chart data for visualization"""
        
        charts = []
        
        # Task completion rate over time (line chart)
        charts.append(self._create_task_completion_chart())
        
        # Agent performance comparison (bar chart)
        charts.append(self._create_agent_performance_chart())
        
        # Error distribution (pie chart)
        charts.append(self._create_error_distribution_chart())
        
        # Confidence score trend (line chart)
        charts.append(self._create_confidence_trend_chart())
        
        # Task duration distribution (histogram)
        charts.append(self._create_duration_distribution_chart())
        
        return charts
        
    def _create_task_completion_chart(self) -> ChartData:
        """Create task completion rate over time chart"""
        
        # Generate hourly data for last 24 hours
        now = datetime.now()
        hours = [(now - timedelta(hours=i)).strftime('%H:00') for i in range(23, -1, -1)]
        
        # Mock data - would come from actual metrics
        completion_rates = [0.89, 0.92, 0.95, 0.88, 0.91, 0.94, 0.87, 0.93,
                           0.96, 0.89, 0.92, 0.88, 0.94, 0.91, 0.95, 0.87,
                           0.93, 0.89, 0.92, 0.94, 0.88, 0.91, 0.93, 0.95]
        
        task_counts = [35, 42, 38, 41, 39, 45, 37, 44, 43, 36, 40, 38,
                      42, 41, 46, 34, 43, 37, 39, 45, 36, 40, 42, 44]
        
        return ChartData(
            chart_type="line",
            title="Task Completion Rate - Last 24 Hours",
            labels=hours,
            datasets=[
                {
                    "label": "Success Rate",
                    "data": completion_rates,
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "yAxisID": "y"
                },
                {
                    "label": "Task Count",
                    "data": task_counts,
                    "borderColor": "rgb(255, 159, 64)",
                    "backgroundColor": "rgba(255, 159, 64, 0.2)",
                    "yAxisID": "y1"
                }
            ],
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "min": 0,
                        "max": 1
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "grid": {"drawOnChartArea": False}
                    }
                }
            }
        )
        
    def _create_agent_performance_chart(self) -> ChartData:
        """Create agent performance comparison chart"""
        
        performance_data = self._get_performance_data()
        agent_data = performance_data['agent_status']
        
        agents = list(agent_data.keys())
        success_rates = [agent_data[agent]['success_rate'] for agent in agents]
        task_counts = [agent_data[agent]['tasks_completed'] for agent in agents]
        
        return ChartData(
            chart_type="bar",
            title="Agent Performance Comparison - Last 24 Hours",
            labels=agents,
            datasets=[
                {
                    "label": "Success Rate",
                    "data": success_rates,
                    "backgroundColor": "rgba(54, 162, 235, 0.8)",
                    "yAxisID": "y"
                },
                {
                    "label": "Tasks Completed",
                    "data": task_counts,
                    "backgroundColor": "rgba(255, 206, 86, 0.8)",
                    "yAxisID": "y1"
                }
            ],
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "min": 0,
                        "max": 1
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "grid": {"drawOnChartArea": False}
                    }
                }
            }
        )
        
    def _create_error_distribution_chart(self) -> ChartData:
        """Create error type distribution chart"""
        
        performance_data = self._get_performance_data()
        top_errors = performance_data['top_errors']
        
        labels = [error['type'] for error in top_errors]
        data = [error['count'] for error in top_errors]
        colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
        
        return ChartData(
            chart_type="pie",
            title="Error Distribution - Last 24 Hours",
            labels=labels,
            datasets=[{
                "data": data,
                "backgroundColor": colors[:len(data)],
                "hoverBackgroundColor": colors[:len(data)]
            }],
            options={
                "responsive": True,
                "plugins": {
                    "legend": {"position": "right"}
                }
            }
        )
        
    def _create_confidence_trend_chart(self) -> ChartData:
        """Create confidence score trend chart"""
        
        # Generate hourly confidence data
        now = datetime.now()
        hours = [(now - timedelta(hours=i)).strftime('%H:00') for i in range(23, -1, -1)]
        
        # Mock confidence data - would come from confidence reporter
        confidence_scores = [0.75, 0.78, 0.82, 0.76, 0.79, 0.83, 0.74, 0.81,
                           0.84, 0.77, 0.80, 0.75, 0.82, 0.79, 0.85, 0.73,
                           0.81, 0.76, 0.79, 0.83, 0.75, 0.78, 0.81, 0.84]
        
        calibration_errors = [0.12, 0.10, 0.08, 0.14, 0.11, 0.07, 0.15, 0.09,
                             0.06, 0.13, 0.10, 0.12, 0.08, 0.11, 0.05, 0.16,
                             0.09, 0.14, 0.11, 0.07, 0.13, 0.10, 0.09, 0.06]
        
        return ChartData(
            chart_type="line",
            title="Confidence Score Trend - Last 24 Hours",
            labels=hours,
            datasets=[
                {
                    "label": "Mean Confidence",
                    "data": confidence_scores,
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "yAxisID": "y"
                },
                {
                    "label": "Calibration Error",
                    "data": calibration_errors,
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "yAxisID": "y1"
                }
            ],
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "min": 0,
                        "max": 1
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "min": 0,
                        "max": 0.2,
                        "grid": {"drawOnChartArea": False}
                    }
                }
            }
        )
        
    def _create_duration_distribution_chart(self) -> ChartData:
        """Create task duration distribution chart"""
        
        # Mock duration distribution data
        duration_bins = ['0-10s', '10-30s', '30-60s', '1-2m', '2-5m', '5-10m', '10m+']
        task_counts = [156, 298, 187, 98, 67, 28, 13]
        
        return ChartData(
            chart_type="bar",
            title="Task Duration Distribution - Last 24 Hours",
            labels=duration_bins,
            datasets=[{
                "label": "Task Count",
                "data": task_counts,
                "backgroundColor": "rgba(153, 102, 255, 0.8)",
                "borderColor": "rgba(153, 102, 255, 1)",
                "borderWidth": 1
            }],
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        )
        
    def _get_alert_data(self) -> List[Dict[str, Any]]:
        """Get formatted alert data for dashboard"""
        
        if not self.alert_manager:
            return []
            
        alerts = self.alert_manager.get_active_alerts()
        
        formatted_alerts = []
        for alert in alerts[:10]:  # Limit to 10 most recent
            formatted_alerts.append({
                'id': alert.id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'type': alert.alert_type.value,
                'agent_id': alert.agent_id,
                'created_at': alert.created_at.isoformat(),
                'age_minutes': int((datetime.now() - alert.created_at).total_seconds() / 60)
            })
            
        return formatted_alerts
        
    def _get_recent_events(self) -> List[Dict[str, Any]]:
        """Get recent events for dashboard"""
        
        # Mock recent events - would come from audit system
        events = [
            {
                'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(),
                'type': 'task_completed',
                'agent_id': 'agent_001',
                'description': 'Data processing task completed successfully',
                'level': 'info'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'type': 'error_occurred',
                'agent_id': 'agent_003',
                'description': 'Connection timeout during API call',
                'level': 'warning'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=8)).isoformat(),
                'type': 'task_started',
                'agent_id': 'agent_002',
                'description': 'Started batch analysis task',
                'level': 'info'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=12)).isoformat(),
                'type': 'decision_made',
                'agent_id': 'agent_004',
                'description': 'Selected optimization strategy A (confidence: 0.89)',
                'level': 'info'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                'type': 'performance_breach',
                'agent_id': 'agent_003',
                'description': 'Task duration exceeded 5-minute threshold',
                'level': 'warning'
            }
        ]
        
        return events
        
    def _generate_recommendations(self, metrics: DashboardMetrics) -> List[str]:
        """Generate recommendations based on current metrics"""
        
        recommendations = []
        
        # System health recommendations
        if metrics.system_status == 'critical':
            recommendations.append("🚨 Critical alerts detected - immediate attention required")
        elif metrics.system_status == 'warning':
            recommendations.append("⚠️ Multiple alerts active - review system health")
            
        # Success rate recommendations
        if metrics.success_rate_24h < 0.9:
            recommendations.append(f"📊 Success rate ({metrics.success_rate_24h:.1%}) below target - investigate failure patterns")
            
        # Performance recommendations
        if metrics.avg_task_duration > 60:
            recommendations.append(f"⏱️ Average task duration ({metrics.avg_task_duration:.1f}s) is high - consider optimization")
            
        # Confidence recommendations
        if metrics.confidence_score < 0.7:
            recommendations.append(f"🎯 Confidence score ({metrics.confidence_score:.2f}) is low - review decision models")
            
        # Agent recommendations
        inactive_agents = [
            agent_id for agent_id, status in metrics.agent_status.items()
            if status['status'] == 'inactive'
        ]
        if inactive_agents:
            recommendations.append(f"🤖 {len(inactive_agents)} agent(s) inactive: {', '.join(inactive_agents[:3])}")
            
        # Pattern recommendations
        if metrics.pattern_summary.get('critical_patterns', 0) > 0:
            recommendations.append(f"🔍 {metrics.pattern_summary['critical_patterns']} critical patterns need attention")
            
        if not recommendations:
            recommendations.append("✅ System operating within normal parameters")
            
        return recommendations
        
    def export_dashboard_data(self, output_file: str) -> bool:
        """Export current dashboard data to file"""
        
        try:
            dashboard_data = self.generate_dashboard_data()
            
            # Convert to serializable format
            export_data = {
                'metrics': asdict(dashboard_data.metrics),
                'charts': [asdict(chart) for chart in dashboard_data.charts],
                'alerts': dashboard_data.alerts,
                'recent_events': dashboard_data.recent_events,
                'recommendations': dashboard_data.recommendations,
                'last_updated': dashboard_data.last_updated.isoformat()
            }
            
            # Convert datetime objects to strings
            export_data['metrics']['timestamp'] = dashboard_data.metrics.timestamp.isoformat()
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            self.logger.info(f"Dashboard data exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export dashboard data: {e}")
            return False
        
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get quick system health summary"""
        
        metrics = self._generate_metrics()
        
        return {
            'status': metrics.system_status,
            'active_agents': metrics.active_agents,
            'success_rate': metrics.success_rate_24h,
            'active_alerts': metrics.active_alerts,
            'critical_alerts': metrics.critical_alerts,
            'confidence_score': metrics.confidence_score,
            'timestamp': datetime.now().isoformat()
        }
