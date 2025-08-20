"""
Confidence Score Reporting and Analysis for Supervisor Agent
Tracks decision confidence, accuracy metrics, and calibration analysis
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import math


@dataclass
class ConfidenceEntry:
    timestamp: datetime
    agent_id: str
    task_id: str
    predicted_confidence: float
    actual_success: bool
    task_duration: float
    task_type: str
    context: Dict[str, Any]


@dataclass
class CalibrationBin:
    confidence_range: Tuple[float, float]
    predicted_prob: float
    actual_prob: float
    count: int
    entries: List[ConfidenceEntry]


@dataclass
class ConfidenceMetrics:
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    total_entries: int
    mean_confidence: float
    confidence_std: float
    accuracy: float
    calibration_error: float
    overconfidence_ratio: float
    underconfidence_ratio: float
    brier_score: float
    calibration_bins: List[CalibrationBin]
    agent_metrics: Dict[str, Dict[str, float]]
    trend_analysis: Dict[str, Any]


class ConfidenceReporter:
    """Manages confidence score tracking and calibration analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage configuration
        self.data_file = Path(config.get('confidence_data_file', 'confidence_data.jsonl'))
        self.max_memory_entries = config.get('max_memory_entries', 10000)
        self.calibration_bins = config.get('calibration_bins', 10)
        
        # Memory storage for recent entries
        self.memory_entries: deque = deque(maxlen=self.max_memory_entries)
        
        # Load existing data
        self._load_existing_data()
        
        self.logger.info("Confidence Reporter initialized")
        
    def record_confidence(self, agent_id: str, task_id: str, 
                         predicted_confidence: float, actual_success: bool,
                         task_duration: float, task_type: str = "unknown",
                         context: Optional[Dict[str, Any]] = None) -> str:
        """Record a confidence prediction and its outcome"""
        
        entry = ConfidenceEntry(
            timestamp=datetime.now(),
            agent_id=agent_id,
            task_id=task_id,
            predicted_confidence=max(0.0, min(1.0, predicted_confidence)),  # Clamp to [0,1]
            actual_success=actual_success,
            task_duration=task_duration,
            task_type=task_type,
            context=context or {}
        )
        
        # Add to memory
        self.memory_entries.append(entry)
        
        # Persist to file
        self._write_entry_to_file(entry)
        
        self.logger.debug(f"Recorded confidence entry for task {task_id}")
        return f"{entry.timestamp.isoformat()}_{task_id}"
        
    def generate_metrics(self, hours: int = 24) -> ConfidenceMetrics:
        """Generate comprehensive confidence metrics for specified period"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Filter entries for the period
        entries = [e for e in self.memory_entries 
                  if start_time <= e.timestamp <= end_time]
        
        if not entries:
            return self._create_empty_metrics(start_time, end_time)
            
        # Basic statistics
        confidences = [e.predicted_confidence for e in entries]
        mean_confidence = statistics.mean(confidences)
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # Accuracy
        accuracy = sum(1 for e in entries if e.actual_success) / len(entries)
        
        # Calibration analysis
        calibration_bins = self._calculate_calibration_bins(entries)
        calibration_error = self._calculate_calibration_error(calibration_bins)
        
        # Over/under confidence
        overconfidence_ratio = self._calculate_overconfidence_ratio(entries)
        underconfidence_ratio = self._calculate_underconfidence_ratio(entries)
        
        # Brier score (measure of prediction accuracy)
        brier_score = self._calculate_brier_score(entries)
        
        # Agent-specific metrics
        agent_metrics = self._calculate_agent_metrics(entries)
        
        # Trend analysis
        trend_analysis = self._analyze_confidence_trends(entries)
        
        return ConfidenceMetrics(
            timestamp=datetime.now(),
            period_start=start_time,
            period_end=end_time,
            total_entries=len(entries),
            mean_confidence=mean_confidence,
            confidence_std=confidence_std,
            accuracy=accuracy,
            calibration_error=calibration_error,
            overconfidence_ratio=overconfidence_ratio,
            underconfidence_ratio=underconfidence_ratio,
            brier_score=brier_score,
            calibration_bins=calibration_bins,
            agent_metrics=agent_metrics,
            trend_analysis=trend_analysis
        )
        
    def _calculate_calibration_bins(self, entries: List[ConfidenceEntry]) -> List[CalibrationBin]:
        """Calculate calibration bins for reliability analysis"""
        
        bins = []
        bin_size = 1.0 / self.calibration_bins
        
        for i in range(self.calibration_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size
            
            # Get entries in this bin
            bin_entries = [
                e for e in entries 
                if bin_start <= e.predicted_confidence < bin_end or 
                   (i == self.calibration_bins - 1 and e.predicted_confidence == 1.0)
            ]
            
            if bin_entries:
                predicted_prob = statistics.mean([e.predicted_confidence for e in bin_entries])
                actual_prob = sum(1 for e in bin_entries if e.actual_success) / len(bin_entries)
            else:
                predicted_prob = (bin_start + bin_end) / 2
                actual_prob = 0.0
                
            bins.append(CalibrationBin(
                confidence_range=(bin_start, bin_end),
                predicted_prob=predicted_prob,
                actual_prob=actual_prob,
                count=len(bin_entries),
                entries=bin_entries
            ))
            
        return bins
        
    def _calculate_calibration_error(self, bins: List[CalibrationBin]) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        
        total_entries = sum(bin.count for bin in bins)
        if total_entries == 0:
            return 0.0
            
        weighted_error = sum(
            (bin.count / total_entries) * abs(bin.predicted_prob - bin.actual_prob)
            for bin in bins if bin.count > 0
        )
        
        return weighted_error
        
    def _calculate_overconfidence_ratio(self, entries: List[ConfidenceEntry]) -> float:
        """Calculate ratio of overconfident predictions"""
        
        overconfident_count = sum(
            1 for e in entries 
            if e.predicted_confidence > 0.5 and not e.actual_success
        )
        
        high_confidence_count = sum(1 for e in entries if e.predicted_confidence > 0.5)
        
        return overconfident_count / high_confidence_count if high_confidence_count > 0 else 0.0
        
    def _calculate_underconfidence_ratio(self, entries: List[ConfidenceEntry]) -> float:
        """Calculate ratio of underconfident predictions"""
        
        underconfident_count = sum(
            1 for e in entries 
            if e.predicted_confidence < 0.5 and e.actual_success
        )
        
        low_confidence_count = sum(1 for e in entries if e.predicted_confidence < 0.5)
        
        return underconfident_count / low_confidence_count if low_confidence_count > 0 else 0.0
        
    def _calculate_brier_score(self, entries: List[ConfidenceEntry]) -> float:
        """Calculate Brier score (lower is better)"""
        
        if not entries:
            return 0.0
            
        score = sum(
            (e.predicted_confidence - (1 if e.actual_success else 0)) ** 2
            for e in entries
        ) / len(entries)
        
        return score
        
    def _calculate_agent_metrics(self, entries: List[ConfidenceEntry]) -> Dict[str, Dict[str, float]]:
        """Calculate per-agent confidence metrics"""
        
        agent_data = defaultdict(list)
        for entry in entries:
            agent_data[entry.agent_id].append(entry)
            
        agent_metrics = {}
        for agent_id, agent_entries in agent_data.items():
            if len(agent_entries) < 5:  # Skip agents with too few entries
                continue
                
            confidences = [e.predicted_confidence for e in agent_entries]
            accuracy = sum(1 for e in agent_entries if e.actual_success) / len(agent_entries)
            
            agent_metrics[agent_id] = {
                'count': len(agent_entries),
                'mean_confidence': statistics.mean(confidences),
                'accuracy': accuracy,
                'brier_score': self._calculate_brier_score(agent_entries),
                'overconfidence': self._calculate_overconfidence_ratio(agent_entries),
                'underconfidence': self._calculate_underconfidence_ratio(agent_entries)
            }
            
        return agent_metrics
        
    def _analyze_confidence_trends(self, entries: List[ConfidenceEntry]) -> Dict[str, Any]:
        """Analyze confidence trends over time"""
        
        if len(entries) < 10:
            return {'status': 'insufficient_data'}
            
        # Sort by time
        sorted_entries = sorted(entries, key=lambda x: x.timestamp)
        
        # Split into time buckets
        bucket_count = min(4, len(sorted_entries) // 5)  # At least 5 entries per bucket
        bucket_size = len(sorted_entries) // bucket_count
        
        bucket_metrics = []
        for i in range(bucket_count):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size if i < bucket_count - 1 else len(sorted_entries)
            bucket_entries = sorted_entries[start_idx:end_idx]
            
            confidences = [e.predicted_confidence for e in bucket_entries]
            accuracy = sum(1 for e in bucket_entries if e.actual_success) / len(bucket_entries)
            
            bucket_metrics.append({
                'mean_confidence': statistics.mean(confidences),
                'accuracy': accuracy,
                'count': len(bucket_entries)
            })
            
        # Calculate trends
        confidence_trend = self._calculate_trend(
            [m['mean_confidence'] for m in bucket_metrics]
        )
        accuracy_trend = self._calculate_trend(
            [m['accuracy'] for m in bucket_metrics]
        )
        
        return {
            'status': 'available',
            'bucket_count': bucket_count,
            'bucket_metrics': bucket_metrics,
            'confidence_trend': confidence_trend,
            'accuracy_trend': accuracy_trend
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        
        if len(values) < 2:
            return 'stable'
            
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        diff = second_avg - first_avg
        threshold = 0.05  # 5% change threshold
        
        if diff > threshold:
            return 'improving'
        elif diff < -threshold:
            return 'declining'
        else:
            return 'stable'
            
    def generate_calibration_report(self, metrics: ConfidenceMetrics) -> str:
        """Generate detailed calibration report in Markdown"""
        
        report = f"""
# Confidence Calibration Report

**Generated:** {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Period:** {metrics.period_start.strftime('%Y-%m-%d %H:%M')} to {metrics.period_end.strftime('%Y-%m-%d %H:%M')}

## Summary

- **Total Predictions:** {metrics.total_entries:,}
- **Mean Confidence:** {metrics.mean_confidence:.3f} ± {metrics.confidence_std:.3f}
- **Actual Accuracy:** {metrics.accuracy:.1%}
- **Calibration Error:** {metrics.calibration_error:.3f}
- **Brier Score:** {metrics.brier_score:.3f}

## Calibration Analysis

**Overconfidence Rate:** {metrics.overconfidence_ratio:.1%}  
**Underconfidence Rate:** {metrics.underconfidence_ratio:.1%}

### Calibration Bins

| Confidence Range | Predicted | Actual | Count | Difference |
|------------------|-----------|--------|-------|------------|
"""
        
        for bin in metrics.calibration_bins:
            if bin.count > 0:
                diff = bin.predicted_prob - bin.actual_prob
                report += f"| {bin.confidence_range[0]:.1f}-{bin.confidence_range[1]:.1f} | {bin.predicted_prob:.3f} | {bin.actual_prob:.3f} | {bin.count} | {diff:+.3f} |\n"
            
        report += f"""

## Agent Performance

| Agent ID | Predictions | Confidence | Accuracy | Brier Score | Calibration |
|----------|-------------|------------|----------|-------------|-------------|
"""
        
        for agent_id, metrics_data in metrics.agent_metrics.items():
            calibration_status = "Well-calibrated"
            if metrics_data['overconfidence'] > 0.2:
                calibration_status = "Overconfident"
            elif metrics_data['underconfidence'] > 0.2:
                calibration_status = "Underconfident"
                
            report += f"| {agent_id} | {metrics_data['count']} | {metrics_data['mean_confidence']:.3f} | {metrics_data['accuracy']:.1%} | {metrics_data['brier_score']:.3f} | {calibration_status} |\n"
            
        # Trend analysis
        if metrics.trend_analysis.get('status') == 'available':
            report += f"""

## Trend Analysis

- **Confidence Trend:** {metrics.trend_analysis['confidence_trend'].title()}
- **Accuracy Trend:** {metrics.trend_analysis['accuracy_trend'].title()}

"""
        
        # Recommendations
        recommendations = self._generate_calibration_recommendations(metrics)
        if recommendations:
            report += "## Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
                
        return report
        
    def _generate_calibration_recommendations(self, metrics: ConfidenceMetrics) -> List[str]:
        """Generate recommendations based on calibration analysis"""
        
        recommendations = []
        
        # Overall calibration
        if metrics.calibration_error > 0.1:
            recommendations.append(
                f"High calibration error ({metrics.calibration_error:.3f}). "
                "Consider retraining confidence estimation or adjusting thresholds."
            )
            
        # Overconfidence
        if metrics.overconfidence_ratio > 0.3:
            recommendations.append(
                f"High overconfidence rate ({metrics.overconfidence_ratio:.1%}). "
                "Review decision thresholds and consider confidence penalties."
            )
            
        # Underconfidence
        if metrics.underconfidence_ratio > 0.3:
            recommendations.append(
                f"High underconfidence rate ({metrics.underconfidence_ratio:.1%}). "
                "Consider boosting confidence in reliable predictions."
            )
            
        # Brier score
        if metrics.brier_score > 0.25:
            recommendations.append(
                f"High Brier score ({metrics.brier_score:.3f}) indicates poor probability estimates. "
                "Review confidence calculation methodology."
            )
            
        # Agent-specific recommendations
        poorly_calibrated_agents = [
            agent_id for agent_id, agent_metrics in metrics.agent_metrics.items()
            if agent_metrics['brier_score'] > 0.3 or agent_metrics['overconfidence'] > 0.4
        ]
        
        if poorly_calibrated_agents:
            recommendations.append(
                f"Agents with poor calibration: {', '.join(poorly_calibrated_agents[:3])}. "
                "Consider agent-specific confidence tuning."
            )
            
        return recommendations if recommendations else [
            "Confidence calibration is within acceptable ranges."
        ]
        
    def _write_entry_to_file(self, entry: ConfidenceEntry):
        """Write confidence entry to persistent storage"""
        
        try:
            entry_dict = {
                'timestamp': entry.timestamp.isoformat(),
                'agent_id': entry.agent_id,
                'task_id': entry.task_id,
                'predicted_confidence': entry.predicted_confidence,
                'actual_success': entry.actual_success,
                'task_duration': entry.task_duration,
                'task_type': entry.task_type,
                'context': entry.context
            }
            
            with open(self.data_file, 'a') as f:
                f.write(json.dumps(entry_dict) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write confidence entry to file: {e}")
            
    def _load_existing_data(self):
        """Load existing confidence data from file"""
        
        if not self.data_file.exists():
            return
            
        try:
            with open(self.data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry_dict = json.loads(line)
                        entry = ConfidenceEntry(
                            timestamp=datetime.fromisoformat(entry_dict['timestamp']),
                            agent_id=entry_dict['agent_id'],
                            task_id=entry_dict['task_id'],
                            predicted_confidence=entry_dict['predicted_confidence'],
                            actual_success=entry_dict['actual_success'],
                            task_duration=entry_dict['task_duration'],
                            task_type=entry_dict.get('task_type', 'unknown'),
                            context=entry_dict.get('context', {})
                        )
                        
                        # Only keep recent entries in memory
                        if datetime.now() - entry.timestamp < timedelta(days=7):
                            self.memory_entries.append(entry)
                            
            self.logger.info(f"Loaded {len(self.memory_entries)} recent confidence entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing confidence data: {e}")
            
    def _create_empty_metrics(self, start_time: datetime, end_time: datetime) -> ConfidenceMetrics:
        """Create empty metrics when no data is available"""
        
        return ConfidenceMetrics(
            timestamp=datetime.now(),
            period_start=start_time,
            period_end=end_time,
            total_entries=0,
            mean_confidence=0.0,
            confidence_std=0.0,
            accuracy=0.0,
            calibration_error=0.0,
            overconfidence_ratio=0.0,
            underconfidence_ratio=0.0,
            brier_score=0.0,
            calibration_bins=[],
            agent_metrics={},
            trend_analysis={'status': 'no_data'}
        )
        
    def get_agent_confidence_history(self, agent_id: str, hours: int = 24) -> List[ConfidenceEntry]:
        """Get confidence history for specific agent"""
        
        start_time = datetime.now() - timedelta(hours=hours)
        return [
            e for e in self.memory_entries
            if e.agent_id == agent_id and e.timestamp >= start_time
        ]
        
    def export_confidence_data(self, output_file: str, hours: int = 24) -> int:
        """Export confidence data to file"""
        
        start_time = datetime.now() - timedelta(hours=hours)
        entries = [
            e for e in self.memory_entries
            if e.timestamp >= start_time
        ]
        
        with open(output_file, 'w') as f:
            json.dump([
                {
                    'timestamp': e.timestamp.isoformat(),
                    'agent_id': e.agent_id,
                    'task_id': e.task_id,
                    'predicted_confidence': e.predicted_confidence,
                    'actual_success': e.actual_success,
                    'task_duration': e.task_duration,
                    'task_type': e.task_type,
                    'context': e.context
                } for e in entries
            ], f, indent=2)
            
        self.logger.info(f"Exported {len(entries)} confidence entries to {output_file}")
        return len(entries)
