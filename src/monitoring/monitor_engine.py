"""Main Monitoring Engine - Coordinates all monitoring capabilities"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
import queue

from .task_monitor import TaskCompletionMonitor
from .instruction_monitor import InstructionAdherenceMonitor
from .quality_monitor import OutputQualityMonitor
from error_handling.error_tracker import ErrorTracker
from .resource_monitor import ResourceUsageMonitor
from .confidence_scorer import ConfidenceScorer

@dataclass
class MonitoringResult:
    """Comprehensive monitoring result"""
    timestamp: str
    task_completion: Dict[str, Any]
    instruction_adherence: Dict[str, Any]
    output_quality: Dict[str, Any]
    errors: List[Dict[str, Any]]
    resource_usage: Dict[str, Any]
    confidence_scores: Dict[str, float]
    overall_status: str
    recommendations: List[str]

class MonitoringEngine:
    """Main monitoring engine that coordinates all monitoring capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize monitoring components
        self.task_monitor = TaskCompletionMonitor(self.config.get('task_monitor', {}))
        self.instruction_monitor = InstructionAdherenceMonitor(self.config.get('instruction_monitor', {}))
        self.quality_monitor = OutputQualityMonitor(self.config.get('quality_monitor', {}))
        self.error_tracker = ErrorTracker(self.config.get('error_tracker', {}))
        self.resource_monitor = ResourceUsageMonitor(self.config.get('resource_monitor', {}))
        self.confidence_scorer = ConfidenceScorer(self.config.get('confidence_scorer', {}))
        
        # Real-time monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue()
        self.results_history = []
        
        # Performance tracking
        self.start_time = time.time()
        self.monitoring_stats = {
            'total_evaluations': 0,
            'error_count': 0,
            'average_confidence': 0.0,
            'last_evaluation_time': None
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            'real_time_enabled': True,
            'evaluation_interval': 5.0,  # seconds
            'history_limit': 1000,
            'confidence_threshold': 0.7,
            'alert_thresholds': {
                'task_completion': 0.8,
                'instruction_adherence': 0.9,
                'output_quality': 0.8,
                'resource_usage': 0.9
            }
        }
    
    def start_monitoring(self, session_data: Dict[str, Any]):
        """Start real-time monitoring for a session"""
        if self.monitoring_active:
            self.stop_monitoring()
        
        self.monitoring_active = True
        self.session_data = session_data
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Initialize resource monitoring
        self.resource_monitor.start_session()
        
        return {
            'status': 'monitoring_started',
            'timestamp': datetime.now().isoformat(),
            'session_id': session_data.get('session_id', 'unknown')
        }
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self.resource_monitor.end_session()
        
        return {
            'status': 'monitoring_stopped',
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': self.monitoring_stats['total_evaluations']
        }
    
    def evaluate_execution(self, execution_data: Dict[str, Any]) -> MonitoringResult:
        """Comprehensive evaluation of an execution step"""
        start_eval_time = time.time()
        
        try:
            # Task Completion Monitoring
            task_result = self.task_monitor.evaluate_task_completion(
                execution_data.get('task_data', {}),
                execution_data.get('original_goals', []),
                execution_data.get('current_progress', {})
            )
            
            # Instruction Adherence Monitoring
            instruction_result = self.instruction_monitor.evaluate_adherence(
                execution_data.get('instructions', []),
                execution_data.get('agent_steps', []),
                execution_data.get('constraints', {})
            )
            
            # Output Quality Monitoring
            quality_result = self.quality_monitor.evaluate_output_quality(
                execution_data.get('outputs', []),
                execution_data.get('expected_format', {})
            )
            
            # Error Tracking
            errors = self.error_tracker.detect_errors(
                execution_data.get('execution_logs', []),
                execution_data.get('api_responses', []),
                execution_data.get('outputs', [])
            )
            
            # Resource Usage Monitoring
            resource_result = self.resource_monitor.evaluate_usage(
                execution_data.get('resource_data', {})
            )
            
            # Calculate confidence scores
            confidence_scores = self.confidence_scorer.calculate_scores({
                'task_completion': task_result,
                'instruction_adherence': instruction_result,
                'output_quality': quality_result,
                'error_count': len(errors),
                'resource_usage': resource_result
            })
            
            # Determine overall status
            overall_status = self._determine_overall_status(
                task_result, instruction_result, quality_result, 
                errors, resource_result, confidence_scores
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                task_result, instruction_result, quality_result,
                errors, resource_result, confidence_scores
            )
            
            # Create monitoring result
            result = MonitoringResult(
                timestamp=datetime.now().isoformat(),
                task_completion=task_result,
                instruction_adherence=instruction_result,
                output_quality=quality_result,
                errors=errors,
                resource_usage=resource_result,
                confidence_scores=confidence_scores,
                overall_status=overall_status,
                recommendations=recommendations
            )
            
            # Update statistics
            self.monitoring_stats['total_evaluations'] += 1
            self.monitoring_stats['error_count'] += len(errors)
            self.monitoring_stats['average_confidence'] = (
                (self.monitoring_stats['average_confidence'] * 
                 (self.monitoring_stats['total_evaluations'] - 1) + 
                 confidence_scores.get('overall', 0)) / 
                self.monitoring_stats['total_evaluations']
            )
            self.monitoring_stats['last_evaluation_time'] = datetime.now().isoformat()
            
            # Store in history
            self._add_to_history(result)
            
            return result
            
        except Exception as e:
            self.error_tracker.log_error({
                'type': 'monitoring_error',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'evaluation_time': time.time() - start_eval_time
            })
            
            # Return minimal result on error
            return MonitoringResult(
                timestamp=datetime.now().isoformat(),
                task_completion={'status': 'error', 'score': 0.0},
                instruction_adherence={'status': 'error', 'score': 0.0},
                output_quality={'status': 'error', 'score': 0.0},
                errors=[{'type': 'monitoring_error', 'message': str(e)}],
                resource_usage={'status': 'error'},
                confidence_scores={'overall': 0.0},
                overall_status='error',
                recommendations=['Fix monitoring system error']
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time monitoring"""
        while self.monitoring_active:
            try:
                # Check for new data to monitor
                if not self.monitoring_queue.empty():
                    execution_data = self.monitoring_queue.get_nowait()
                    result = self.evaluate_execution(execution_data)
                    
                    # Check for alerts
                    self._check_alerts(result)
                
                time.sleep(self.config['evaluation_interval'])
                
            except Exception as e:
                self.error_tracker.log_error({
                    'type': 'monitoring_loop_error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                time.sleep(1.0)  # Brief pause on error
    
    def _determine_overall_status(self, task_result, instruction_result, 
                                 quality_result, errors, resource_result, 
                                 confidence_scores) -> str:
        """Determine overall monitoring status"""
        if errors and any(e.get('severity') == 'critical' for e in errors):
            return 'critical'
        
        scores = [
            task_result.get('score', 0),
            instruction_result.get('score', 0),
            quality_result.get('score', 0)
        ]
        avg_score = sum(scores) / len(scores) if scores else 0
        overall_confidence = confidence_scores.get('overall', 0)
        
        if avg_score >= 0.9 and overall_confidence >= 0.9:
            return 'excellent'
        elif avg_score >= 0.8 and overall_confidence >= 0.8:
            return 'good'
        elif avg_score >= 0.6 and overall_confidence >= 0.6:
            return 'acceptable'
        elif avg_score >= 0.4 and overall_confidence >= 0.4:
            return 'poor'
        else:
            return 'failing'
    
    def _generate_recommendations(self, task_result, instruction_result,
                                quality_result, errors, resource_result,
                                confidence_scores) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Task completion recommendations
        if task_result.get('score', 0) < 0.8:
            recommendations.append("Improve task alignment with original goals")
            if task_result.get('drift_detected'):
                recommendations.append("Address detected task drift")
        
        # Instruction adherence recommendations
        if instruction_result.get('score', 0) < 0.8:
            recommendations.append("Review and improve instruction following")
            if instruction_result.get('constraint_violations'):
                recommendations.append("Fix constraint violations")
        
        # Quality recommendations
        if quality_result.get('score', 0) < 0.8:
            recommendations.append("Improve output quality and structure")
            if quality_result.get('format_issues'):
                recommendations.append("Fix output format issues")
        
        # Error recommendations
        if errors:
            recommendations.append(f"Address {len(errors)} detected errors")
        
        # Resource recommendations
        if resource_result.get('token_usage', 0) > 0.9:
            recommendations.append("Optimize token usage")
        if resource_result.get('loop_detected'):
            recommendations.append("Break detected execution loops")
        
        # Confidence recommendations
        if confidence_scores.get('overall', 0) < 0.7:
            recommendations.append("Improve overall execution confidence")
        
        return recommendations or ["Continue current execution approach"]
    
    def _check_alerts(self, result: MonitoringResult):
        """Check for alert conditions"""
        thresholds = self.config['alert_thresholds']
        
        alerts = []
        
        if result.task_completion.get('score', 0) < thresholds['task_completion']:
            alerts.append('Low task completion score')
        
        if result.instruction_adherence.get('score', 0) < thresholds['instruction_adherence']:
            alerts.append('Low instruction adherence score')
        
        if result.output_quality.get('score', 0) < thresholds['output_quality']:
            alerts.append('Low output quality score')
        
        if result.resource_usage.get('usage_ratio', 0) > thresholds['resource_usage']:
            alerts.append('High resource usage detected')
        
        if alerts:
            self.error_tracker.log_error({
                'type': 'monitoring_alert',
                'alerts': alerts,
                'timestamp': result.timestamp,
                'overall_status': result.overall_status
            })
    
    def _add_to_history(self, result: MonitoringResult):
        """Add result to monitoring history"""
        self.results_history.append(result)
        
        # Limit history size
        if len(self.results_history) > self.config['history_limit']:
            self.results_history = self.results_history[-self.config['history_limit']:]
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            **self.monitoring_stats,
            'uptime_seconds': time.time() - self.start_time,
            'history_size': len(self.results_history),
            'is_monitoring_active': self.monitoring_active
        }
    
    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent monitoring results"""
        recent = self.results_history[-limit:] if self.results_history else []
        return [asdict(result) for result in recent]
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'monitoring_stats': self.get_monitoring_stats(),
            'recent_results': self.get_recent_results(100),
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return f"Monitoring data exported to {filepath}"
