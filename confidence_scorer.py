"""Confidence Scoring System - Calculates confidence scores for monitoring judgments"""

import math
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class ConfidenceMetrics:
    """Confidence calculation metrics"""
    base_score: float
    adjustments: Dict[str, float]
    final_score: float
    confidence_factors: Dict[str, Any]
    historical_accuracy: float

class ConfidenceScorer:
    """Calculates confidence scores for monitoring system judgments"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.historical_scores = deque(maxlen=self.config['history_size'])
        self.accuracy_tracking = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'accuracy_by_category': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
        self.score_calibration = self._initialize_calibration()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'history_size': 100,
            'min_confidence': 0.1,
            'max_confidence': 0.95,
            'uncertainty_penalty': 0.1,
            'consistency_weight': 0.3,
            'evidence_weight': 0.4,
            'historical_weight': 0.3,
            'calibration_enabled': True
        }
    
    def _initialize_calibration(self) -> Dict[str, Any]:
        """Initialize confidence calibration parameters"""
        return {
            'task_completion': {'bias': 0.0, 'variance': 1.0},
            'instruction_adherence': {'bias': 0.0, 'variance': 1.0},
            'output_quality': {'bias': 0.0, 'variance': 1.0},
            'error_detection': {'bias': 0.0, 'variance': 1.0},
            'resource_usage': {'bias': 0.0, 'variance': 1.0}
        }
    
    def calculate_scores(self, monitoring_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for all monitoring components"""
        try:
            scores = {}
            
            # Task completion confidence
            task_result = monitoring_results.get('task_completion', {})
            scores['task_completion'] = self._calculate_task_confidence(task_result)
            
            # Instruction adherence confidence
            instruction_result = monitoring_results.get('instruction_adherence', {})
            scores['instruction_adherence'] = self._calculate_instruction_confidence(instruction_result)
            
            # Output quality confidence
            quality_result = monitoring_results.get('output_quality', {})
            scores['output_quality'] = self._calculate_quality_confidence(quality_result)
            
            # Error detection confidence
            error_count = monitoring_results.get('error_count', 0)
            scores['error_detection'] = self._calculate_error_confidence(error_count, monitoring_results)
            
            # Resource usage confidence
            resource_result = monitoring_results.get('resource_usage', {})
            scores['resource_usage'] = self._calculate_resource_confidence(resource_result)
            
            # Calculate overall confidence
            scores['overall'] = self._calculate_overall_confidence(scores)
            
            # Apply calibration if enabled
            if self.config['calibration_enabled']:
                scores = self._apply_calibration(scores)
            
            # Store for historical tracking
            self._store_scores(scores, monitoring_results)
            
            return scores
            
        except Exception as e:
            return {
                'task_completion': 0.5,
                'instruction_adherence': 0.5,
                'output_quality': 0.5,
                'error_detection': 0.5,
                'resource_usage': 0.5,
                'overall': 0.5,
                'error': str(e)
            }
    
    def _calculate_task_confidence(self, task_result: Dict[str, Any]) -> float:
        """Calculate confidence in task completion assessment"""
        if not task_result or 'score' not in task_result:
            return 0.5
        
        base_score = task_result.get('score', 0.5)
        
        # Evidence factors
        evidence_factors = {
            'alignment_score': task_result.get('alignment_score', 0.5),
            'progress_score': task_result.get('progress_score', 0.5),
            'milestone_progress': task_result.get('milestone_progress', 0.5)
        }
        
        # Uncertainty factors
        uncertainty_factors = {
            'drift_detected': task_result.get('drift_detected', False),
            'incomplete_patterns': len(task_result.get('incomplete_patterns', [])),
            'drift_severity': task_result.get('drift_severity', 0.0)
        }
        
        # Calculate confidence adjustments
        confidence = base_score
        
        # Boost confidence if multiple evidence sources agree
        evidence_consistency = self._calculate_evidence_consistency(evidence_factors)
        confidence += evidence_consistency * 0.2
        
        # Reduce confidence for uncertainty indicators
        if uncertainty_factors['drift_detected']:
            confidence -= uncertainty_factors['drift_severity'] * 0.3
        
        if uncertainty_factors['incomplete_patterns'] > 0:
            confidence -= min(uncertainty_factors['incomplete_patterns'] * 0.1, 0.2)
        
        return self._clamp_confidence(confidence)
    
    def _calculate_instruction_confidence(self, instruction_result: Dict[str, Any]) -> float:
        """Calculate confidence in instruction adherence assessment"""
        if not instruction_result or 'score' not in instruction_result:
            return 0.5
        
        base_score = instruction_result.get('score', 0.5)
        
        # Evidence factors
        format_score = instruction_result.get('format_adherence', {}).get('score', 0.5)
        procedure_score = instruction_result.get('procedure_adherence', {}).get('score', 0.5)
        constraint_score = instruction_result.get('constraint_score', 0.5)
        
        # Violation factors
        violations = instruction_result.get('constraint_violations', [])
        high_severity_violations = sum(1 for v in violations if v.get('severity') == 'high')
        
        # Calculate confidence
        confidence = base_score
        
        # Evidence consistency boost
        scores = [format_score, procedure_score, constraint_score]
        consistency = 1.0 - (max(scores) - min(scores))
        confidence += consistency * 0.15
        
        # Violation penalty
        if high_severity_violations > 0:
            confidence -= min(high_severity_violations * 0.2, 0.4)
        
        return self._clamp_confidence(confidence)
    
    def _calculate_quality_confidence(self, quality_result: Dict[str, Any]) -> float:
        """Calculate confidence in output quality assessment"""
        if not quality_result or 'score' not in quality_result:
            return 0.5
        
        base_score = quality_result.get('score', 0.5)
        
        # Component scores
        structure_score = quality_result.get('structure_quality', {}).get('score', 0.5)
        coherence_score = quality_result.get('coherence_quality', {}).get('score', 0.5)
        relevance_score = quality_result.get('relevance_quality', {}).get('score', 0.5)
        
        # Quality issues
        quality_issues = quality_result.get('quality_issues', [])
        critical_issues = sum(1 for issue in quality_issues if issue.get('severity') == 'high')
        
        # Duplication analysis
        duplication = quality_result.get('duplication_analysis', {})
        duplication_detected = duplication.get('duplication_detected', False)
        
        # Calculate confidence
        confidence = base_score
        
        # Component consistency
        component_scores = [structure_score, coherence_score, relevance_score]
        consistency = 1.0 - (max(component_scores) - min(component_scores))
        confidence += consistency * 0.2
        
        # Issue penalties
        if critical_issues > 0:
            confidence -= min(critical_issues * 0.15, 0.3)
        
        if duplication_detected:
            confidence -= 0.1
        
        return self._clamp_confidence(confidence)
    
    def _calculate_error_confidence(self, error_count: int, monitoring_results: Dict[str, Any]) -> float:
        """Calculate confidence in error detection"""
        # Base confidence starts high for error detection
        confidence = 0.8
        
        # Adjust based on error count
        if error_count == 0:
            confidence = 0.9  # High confidence when no errors
        elif error_count <= 2:
            confidence = 0.8  # Good confidence for few errors
        elif error_count <= 5:
            confidence = 0.7  # Moderate confidence
        else:
            confidence = 0.6  # Lower confidence for many errors
        
        # Check for error detection patterns
        error_types = set()
        if 'errors' in monitoring_results:
            for error in monitoring_results['errors']:
                error_types.add(error.get('error_type', 'unknown'))
        
        # Boost confidence if diverse error types detected (indicates thorough checking)
        if len(error_types) > 3:
            confidence += 0.1
        
        return self._clamp_confidence(confidence)
    
    def _calculate_resource_confidence(self, resource_result: Dict[str, Any]) -> float:
        """Calculate confidence in resource usage assessment"""
        if not resource_result:
            return 0.5
        
        # System metrics confidence
        system_metrics = resource_result.get('system_metrics', {})
        has_system_data = bool(system_metrics and not system_metrics.get('error'))
        
        # Token usage confidence
        token_usage = resource_result.get('token_usage', {})
        has_token_data = bool(token_usage.get('total_tokens', 0) > 0)
        
        # Performance score
        performance_score = resource_result.get('performance_score', 0.5)
        
        # Loop detection
        loop_detection = resource_result.get('loop_detection', {})
        loop_confidence = loop_detection.get('confidence', 0.0)
        
        # Calculate base confidence
        confidence = 0.7  # Base confidence for resource monitoring
        
        # Adjust based on data availability
        if has_system_data:
            confidence += 0.1
        if has_token_data:
            confidence += 0.1
        
        # Adjust based on performance score
        if performance_score > 0.8:
            confidence += 0.1
        elif performance_score < 0.3:
            confidence -= 0.1
        
        # Loop detection confidence impact
        if loop_detection.get('loop_detected'):
            confidence = max(confidence, loop_confidence)
        
        return self._clamp_confidence(confidence)
    
    def _calculate_overall_confidence(self, individual_scores: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        # Remove non-component scores
        component_scores = {
            k: v for k, v in individual_scores.items() 
            if k in ['task_completion', 'instruction_adherence', 'output_quality', 
                    'error_detection', 'resource_usage']
        }
        
        if not component_scores:
            return 0.5
        
        # Weighted average
        weights = {
            'task_completion': 0.25,
            'instruction_adherence': 0.25,
            'output_quality': 0.25,
            'error_detection': 0.15,
            'resource_usage': 0.10
        }
        
        weighted_sum = sum(
            component_scores.get(component, 0.5) * weight 
            for component, weight in weights.items()
        )
        
        # Consistency bonus
        scores_list = list(component_scores.values())
        if len(scores_list) > 1:
            consistency = 1.0 - (max(scores_list) - min(scores_list))
            weighted_sum += consistency * 0.1
        
        # Historical accuracy adjustment
        historical_accuracy = self._get_historical_accuracy()
        if historical_accuracy > 0:
            weighted_sum = weighted_sum * 0.8 + historical_accuracy * 0.2
        
        return self._clamp_confidence(weighted_sum)
    
    def _calculate_evidence_consistency(self, evidence_factors: Dict[str, float]) -> float:
        """Calculate consistency of evidence factors"""
        if not evidence_factors:
            return 0.0
        
        values = list(evidence_factors.values())
        if len(values) < 2:
            return 1.0
        
        # Calculate standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Convert to consistency score (0-1, higher is better)
        consistency = max(0.0, 1.0 - std_dev)
        
        return consistency
    
    def _apply_calibration(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration to confidence scores"""
        calibrated_scores = {}
        
        for category, score in scores.items():
            if category in self.score_calibration:
                calibration = self.score_calibration[category]
                # Apply bias and variance adjustments
                calibrated_score = score + calibration['bias']
                calibrated_score *= calibration['variance']
                calibrated_scores[category] = self._clamp_confidence(calibrated_score)
            else:
                calibrated_scores[category] = score
        
        return calibrated_scores
    
    def _clamp_confidence(self, confidence: float) -> float:
        """Clamp confidence score to valid range"""
        return max(
            self.config['min_confidence'], 
            min(confidence, self.config['max_confidence'])
        )
    
    def _store_scores(self, scores: Dict[str, float], monitoring_results: Dict[str, Any]):
        """Store scores for historical tracking"""
        score_record = {
            'timestamp': datetime.now().isoformat(),
            'scores': scores.copy(),
            'context': {
                'task_score': monitoring_results.get('task_completion', {}).get('score', 0),
                'instruction_score': monitoring_results.get('instruction_adherence', {}).get('score', 0),
                'quality_score': monitoring_results.get('output_quality', {}).get('score', 0),
                'error_count': monitoring_results.get('error_count', 0)
            }
        }
        
        self.historical_scores.append(score_record)
    
    def _get_historical_accuracy(self) -> float:
        """Get historical accuracy of confidence predictions"""
        if self.accuracy_tracking['total_predictions'] == 0:
            return 0.0
        
        return self.accuracy_tracking['correct_predictions'] / self.accuracy_tracking['total_predictions']
    
    def update_accuracy(self, predicted_confidence: float, actual_outcome: bool, category: str = 'overall'):
        """Update accuracy tracking with actual outcomes"""
        # Convert confidence to binary prediction (>0.7 = positive prediction)
        predicted_positive = predicted_confidence > 0.7
        
        # Update overall accuracy
        self.accuracy_tracking['total_predictions'] += 1
        if predicted_positive == actual_outcome:
            self.accuracy_tracking['correct_predictions'] += 1
        
        # Update category-specific accuracy
        category_stats = self.accuracy_tracking['accuracy_by_category'][category]
        category_stats['total'] += 1
        if predicted_positive == actual_outcome:
            category_stats['correct'] += 1
        
        # Update calibration parameters
        self._update_calibration(category, predicted_confidence, actual_outcome)
    
    def _update_calibration(self, category: str, predicted: float, actual_outcome: bool):
        """Update calibration parameters based on actual outcomes"""
        if category not in self.score_calibration:
            return
        
        actual_score = 1.0 if actual_outcome else 0.0
        error = predicted - actual_score
        
        # Simple bias correction
        calibration = self.score_calibration[category]
        calibration['bias'] -= error * 0.01  # Small learning rate
        
        # Clamp bias to reasonable range
        calibration['bias'] = max(-0.2, min(calibration['bias'], 0.2))
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get summary of confidence scoring performance"""
        recent_scores = list(self.historical_scores)[-20:] if self.historical_scores else []
        
        if recent_scores:
            avg_confidence = {}
            for category in ['task_completion', 'instruction_adherence', 'output_quality', 
                           'error_detection', 'resource_usage', 'overall']:
                scores = [record['scores'].get(category, 0.5) for record in recent_scores]
                avg_confidence[category] = sum(scores) / len(scores)
        else:
            avg_confidence = {category: 0.5 for category in 
                            ['task_completion', 'instruction_adherence', 'output_quality', 
                             'error_detection', 'resource_usage', 'overall']}
        
        return {
            'historical_accuracy': self._get_historical_accuracy(),
            'total_predictions': self.accuracy_tracking['total_predictions'],
            'recent_average_confidence': avg_confidence,
            'calibration_status': self.score_calibration,
            'confidence_trend': self._calculate_confidence_trend(recent_scores),
            'summary_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_confidence_trend(self, recent_scores: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate trend in confidence scores"""
        if len(recent_scores) < 5:
            return {category: 'insufficient_data' for category in 
                   ['task_completion', 'instruction_adherence', 'output_quality', 
                    'error_detection', 'resource_usage', 'overall']}
        
        trends = {}
        for category in ['task_completion', 'instruction_adherence', 'output_quality', 
                        'error_detection', 'resource_usage', 'overall']:
            scores = [record['scores'].get(category, 0.5) for record in recent_scores]
            
            # Simple trend calculation
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            
            if second_half > first_half + 0.05:
                trends[category] = 'increasing'
            elif second_half < first_half - 0.05:
                trends[category] = 'decreasing'
            else:
                trends[category] = 'stable'
        
        return trends
