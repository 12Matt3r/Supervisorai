"""
Periodic Summary Reports for Supervisor Agent
Generates task-level completion reports and performance analytics
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class TaskSummary:
    task_id: str
    agent_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    status: str
    confidence: Optional[float]
    error_count: int
    error_types: List[str]
    metadata: Dict[str, Any]


@dataclass
class AgentSummary:
    agent_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    success_rate: float
    avg_duration: float
    avg_confidence: Optional[float]
    total_errors: int
    common_errors: List[str]
    performance_trend: str  # improving, stable, declining


@dataclass
class PeriodSummary:
    period_start: datetime
    period_end: datetime
    total_tasks: int
    total_agents: int
    overall_success_rate: float
    avg_task_duration: float
    avg_confidence: Optional[float]
    agent_summaries: List[AgentSummary]
    performance_metrics: Dict[str, Any]
    trends: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class ReportGenerator:
    """Generates periodic summary reports and analytics"""
    
    def __init__(self, data_source, config: Dict[str, Any]):
        self.data_source = data_source
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.optimal_duration = config.get('optimal_duration', 30)  # seconds
        self.long_task_threshold = config.get('long_task_threshold', 300)  # seconds
        
    def generate_period_summary(self, hours: int = 24) -> PeriodSummary:
        """Generate comprehensive summary for specified period"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        self.logger.info(f"Generating {hours}h summary from {start_time} to {end_time}")
        
        # Get tasks for the period
        tasks = self.data_source.get_tasks_in_period(start_time, end_time)
        
        if not tasks:
            return self._create_empty_summary(start_time, end_time)
            
        # Process tasks
        task_summaries = [self._create_task_summary(task) for task in tasks]
        agent_summaries = self._create_agent_summaries(task_summaries)
        
        # Calculate overall metrics
        total_tasks = len(task_summaries)
        completed_tasks = len([t for t in task_summaries if t.status == 'completed'])
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        durations = [t.duration for t in task_summaries]
        avg_duration = statistics.mean(durations) if durations else 0
        
        confidences = [t.confidence for t in task_summaries if t.confidence is not None]
        avg_confidence = statistics.mean(confidences) if confidences else None
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(task_summaries)
        
        # Trends analysis
        trends = self._analyze_trends(task_summaries, hours)
        
        # Recommendations
        recommendations = self._generate_recommendations(task_summaries, agent_summaries, performance_metrics)
        
        return PeriodSummary(
            period_start=start_time,
            period_end=end_time,
            total_tasks=total_tasks,
            total_agents=len(agent_summaries),
            overall_success_rate=success_rate,
            avg_task_duration=avg_duration,
            avg_confidence=avg_confidence,
            agent_summaries=agent_summaries,
            performance_metrics=performance_metrics,
            trends=trends,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
    def _create_task_summary(self, task: Dict[str, Any]) -> TaskSummary:
        """Create task summary from raw task data"""
        
        start_time = datetime.fromisoformat(task['start_time'])
        end_time = datetime.fromisoformat(task['end_time'])
        duration = (end_time - start_time).total_seconds()
        
        errors = task.get('errors', [])
        error_types = [error.get('type', 'unknown') for error in errors]
        
        return TaskSummary(
            task_id=task['task_id'],
            agent_id=task['agent_id'],
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            status=task['status'],
            confidence=task.get('confidence'),
            error_count=len(errors),
            error_types=error_types,
            metadata=task.get('metadata', {})
        )
        
    def _create_agent_summaries(self, task_summaries: List[TaskSummary]) -> List[AgentSummary]:
        """Create agent-level summaries"""
        
        agent_data = defaultdict(list)
        for task in task_summaries:
            agent_data[task.agent_id].append(task)
            
        summaries = []
        for agent_id, tasks in agent_data.items():
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.status == 'completed'])
            failed_tasks = total_tasks - completed_tasks
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            durations = [t.duration for t in tasks]
            avg_duration = statistics.mean(durations) if durations else 0
            
            confidences = [t.confidence for t in tasks if t.confidence is not None]
            avg_confidence = statistics.mean(confidences) if confidences else None
            
            all_errors = []
            for task in tasks:
                all_errors.extend(task.error_types)
            
            error_counts = defaultdict(int)
            for error in all_errors:
                error_counts[error] += 1
                
            common_errors = sorted(error_counts.keys(), key=lambda x: error_counts[x], reverse=True)[:3]
            
            # Determine performance trend (simplified)
            performance_trend = self._determine_performance_trend(tasks)
            
            summaries.append(AgentSummary(
                agent_id=agent_id,
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                success_rate=success_rate,
                avg_duration=avg_duration,
                avg_confidence=avg_confidence,
                total_errors=len(all_errors),
                common_errors=common_errors,
                performance_trend=performance_trend
            ))
            
        return summaries
        
    def _determine_performance_trend(self, tasks: List[TaskSummary]) -> str:
        """Determine performance trend for an agent"""
        if len(tasks) < 4:
            return "insufficient_data"
            
        # Sort by start time
        sorted_tasks = sorted(tasks, key=lambda x: x.start_time)
        
        # Split into first half and second half
        mid_point = len(sorted_tasks) // 2
        first_half = sorted_tasks[:mid_point]
        second_half = sorted_tasks[mid_point:]
        
        # Compare success rates
        first_success_rate = len([t for t in first_half if t.status == 'completed']) / len(first_half)
        second_success_rate = len([t for t in second_half if t.status == 'completed']) / len(second_half)
        
        if second_success_rate > first_success_rate + 0.1:
            return "improving"
        elif second_success_rate < first_success_rate - 0.1:
            return "declining"
        else:
            return "stable"
            
    def _calculate_performance_metrics(self, task_summaries: List[TaskSummary]) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        
        if not task_summaries:
            return {}
            
        durations = [t.duration for t in task_summaries]
        confidences = [t.confidence for t in task_summaries if t.confidence is not None]
        
        return {
            'duration_stats': {
                'min': min(durations) if durations else 0,
                'max': max(durations) if durations else 0,
                'median': statistics.median(durations) if durations else 0,
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
            },
            'confidence_stats': {
                'min': min(confidences) if confidences else None,
                'max': max(confidences) if confidences else None,
                'median': statistics.median(confidences) if confidences else None,
                'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else None
            },
            'efficiency_metrics': {
                'fast_tasks': len([t for t in task_summaries if t.duration <= self.optimal_duration]),
                'slow_tasks': len([t for t in task_summaries if t.duration >= self.long_task_threshold]),
                'optimal_ratio': len([t for t in task_summaries if t.duration <= self.optimal_duration]) / len(task_summaries)
            },
            'error_metrics': {
                'error_free_tasks': len([t for t in task_summaries if t.error_count == 0]),
                'high_error_tasks': len([t for t in task_summaries if t.error_count >= 3]),
                'avg_errors_per_task': statistics.mean([t.error_count for t in task_summaries])
            }
        }
        
    def _analyze_trends(self, task_summaries: List[TaskSummary], period_hours: int) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(task_summaries) < 6:
            return {'status': 'insufficient_data'}
            
        # Sort by time
        sorted_tasks = sorted(task_summaries, key=lambda x: x.start_time)
        
        # Split into time buckets
        bucket_size = period_hours // 4  # 4 time buckets
        buckets = [[] for _ in range(4)]
        
        start_time = sorted_tasks[0].start_time
        for task in sorted_tasks:
            hours_elapsed = (task.start_time - start_time).total_seconds() / 3600
            bucket_index = min(int(hours_elapsed / bucket_size), 3)
            buckets[bucket_index].append(task)
            
        # Calculate metrics per bucket
        bucket_metrics = []
        for bucket in buckets:
            if bucket:
                success_rate = len([t for t in bucket if t.status == 'completed']) / len(bucket)
                avg_duration = statistics.mean([t.duration for t in bucket])
                bucket_metrics.append({
                    'success_rate': success_rate,
                    'avg_duration': avg_duration,
                    'task_count': len(bucket)
                })
            else:
                bucket_metrics.append(None)
                
        return {
            'status': 'available',
            'bucket_metrics': bucket_metrics,
            'overall_trend': self._calculate_overall_trend(bucket_metrics)
        }
        
    def _calculate_overall_trend(self, bucket_metrics: List[Optional[Dict]]) -> str:
        """Calculate overall performance trend"""
        
        valid_buckets = [b for b in bucket_metrics if b is not None]
        if len(valid_buckets) < 2:
            return 'insufficient_data'
            
        first_success = valid_buckets[0]['success_rate']
        last_success = valid_buckets[-1]['success_rate']
        
        if last_success > first_success + 0.1:
            return 'improving'
        elif last_success < first_success - 0.1:
            return 'declining'
        else:
            return 'stable'
            
    def _generate_recommendations(self, task_summaries: List[TaskSummary], 
                                agent_summaries: List[AgentSummary],
                                performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if not task_summaries:
            return ['No data available for recommendations']
            
        # Overall success rate recommendations
        overall_success = len([t for t in task_summaries if t.status == 'completed']) / len(task_summaries)
        if overall_success < 0.9:
            recommendations.append(f"Overall success rate ({overall_success:.1%}) is below target. Review failed task patterns.")
            
        # Duration recommendations
        slow_tasks_ratio = performance_metrics.get('efficiency_metrics', {}).get('optimal_ratio', 1)
        if slow_tasks_ratio < 0.7:
            recommendations.append("High number of slow tasks detected. Consider optimizing task execution or resource allocation.")
            
        # Agent-specific recommendations
        declining_agents = [a for a in agent_summaries if a.performance_trend == 'declining']
        if declining_agents:
            agent_names = [a.agent_id for a in declining_agents[:3]]
            recommendations.append(f"Agents with declining performance: {', '.join(agent_names)}. Investigate recent changes.")
            
        # Error recommendations
        high_error_agents = [a for a in agent_summaries if a.total_errors > len(task_summaries) * 0.1]
        if high_error_agents:
            recommendations.append("Some agents showing high error rates. Review error patterns and consider retraining.")
            
        # Confidence recommendations
        low_confidence_tasks = [t for t in task_summaries if t.confidence and t.confidence < 0.5]
        if len(low_confidence_tasks) > len(task_summaries) * 0.2:
            recommendations.append("High number of low-confidence tasks. Review task complexity and agent capabilities.")
            
        return recommendations if recommendations else ['System performing within normal parameters']
        
    def _create_empty_summary(self, start_time: datetime, end_time: datetime) -> PeriodSummary:
        """Create empty summary when no data is available"""
        
        return PeriodSummary(
            period_start=start_time,
            period_end=end_time,
            total_tasks=0,
            total_agents=0,
            overall_success_rate=0,
            avg_task_duration=0,
            avg_confidence=None,
            agent_summaries=[],
            performance_metrics={},
            trends={'status': 'no_data'},
            recommendations=['No tasks found in specified period'],
            timestamp=datetime.now()
        )
        
    def generate_markdown_report(self, summary: PeriodSummary) -> str:
        """Generate formatted Markdown report"""
        
        report = f"""
# Supervisor Agent Performance Report

**Report Period:** {summary.period_start.strftime('%Y-%m-%d %H:%M')} to {summary.period_end.strftime('%Y-%m-%d %H:%M')}  
**Generated:** {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Tasks:** {summary.total_tasks:,}
- **Active Agents:** {summary.total_agents}
- **Overall Success Rate:** {summary.overall_success_rate:.1%}
- **Average Task Duration:** {summary.avg_task_duration:.1f}s
- **Average Confidence:** {summary.avg_confidence:.2f if summary.avg_confidence else 'N/A'}

## Agent Performance

| Agent ID | Tasks | Success Rate | Avg Duration | Confidence | Trend |
|----------|-------|--------------|--------------|------------|-------|
"""
        
        for agent in summary.agent_summaries:
            report += f"| {agent.agent_id} | {agent.total_tasks} | {agent.success_rate:.1%} | {agent.avg_duration:.1f}s | {agent.avg_confidence:.2f if agent.avg_confidence else 'N/A'} | {agent.performance_trend} |\n"
            
        report += f"""

## Performance Metrics

### Duration Analysis
- **Fastest Task:** {summary.performance_metrics.get('duration_stats', {}).get('min', 0):.1f}s
- **Slowest Task:** {summary.performance_metrics.get('duration_stats', {}).get('max', 0):.1f}s
- **Median Duration:** {summary.performance_metrics.get('duration_stats', {}).get('median', 0):.1f}s

### Efficiency Metrics
- **Fast Tasks (≤{self.optimal_duration}s):** {summary.performance_metrics.get('efficiency_metrics', {}).get('fast_tasks', 0)}
- **Slow Tasks (≥{self.long_task_threshold}s):** {summary.performance_metrics.get('efficiency_metrics', {}).get('slow_tasks', 0)}
- **Optimization Ratio:** {summary.performance_metrics.get('efficiency_metrics', {}).get('optimal_ratio', 0):.1%}

## Recommendations

"""
        
        for i, rec in enumerate(summary.recommendations, 1):
            report += f"{i}. {rec}\n"
            
        return report
        
    def export_summary_json(self, summary: PeriodSummary, output_file: str):
        """Export summary as JSON file"""
        
        data = asdict(summary)
        
        # Convert datetime objects to ISO strings
        data['period_start'] = summary.period_start.isoformat()
        data['period_end'] = summary.period_end.isoformat()
        data['timestamp'] = summary.timestamp.isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Exported summary to {output_file}")
