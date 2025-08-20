#!/usr/bin/env python3
"""
Confidence Score Reporting System for Supervisor Agent
Tracks decision confidence, accuracy metrics, and calibration reports.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ConfidenceEntry:
    """Individual confidence score entry"""
    id: str
    timestamp: str
    task_id: str
    agent_id: str
    decision_type: str
    predicted_confidence: float
    actual_outcome: Optional[bool] = None  # True=success, False=failure, None=unknown
    outcome_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CalibrationMetrics:
    """Confidence calibration metrics"""
    mean_confidence: float
    mean_accuracy: float
    calibration_error: float
    brier_score: float
    log_loss_score: float
    reliability_diagram_data: Dict[str, List[float]]
    confidence_bins: List[Tuple[float, float, int, float]]  # (bin_start, bin_end, count, accuracy)

@dataclass
class ConfidenceAnalysis:
    """Comprehensive confidence analysis"""
    period_start: str
    period_end: str
    total_entries: int
    entries_with_outcomes: int
    calibration_metrics: CalibrationMetrics
    trend_data: Dict[str, List[float]]
    by_agent: Dict[str, CalibrationMetrics]
    by_decision_type: Dict[str, CalibrationMetrics]
    recommendations: List[str]

class ConfidenceCalibrator:
    """Calibrates and analyzes confidence scores"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_calibration_metrics(self, confidences: List[float], 
                                    outcomes: List[bool]) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""
        if len(confidences) == 0 or len(outcomes) == 0:
            return self._empty_calibration_metrics()
        
        confidences = np.array(confidences)
        outcomes = np.array(outcomes, dtype=int)
        
        # Basic metrics
        mean_confidence = float(np.mean(confidences))
        mean_accuracy = float(np.mean(outcomes))
        
        # Calibration error (Expected Calibration Error - ECE)
        calibration_error = self._calculate_ece(confidences, outcomes)
        
        # Brier score
        brier_score = brier_score_loss(outcomes, confidences)
        
        # Log loss (with small epsilon to avoid log(0))
        epsilon = 1e-15
        confidences_clipped = np.clip(confidences, epsilon, 1 - epsilon)
        log_loss_score = log_loss(outcomes, confidences_clipped)
        
        # Reliability diagram data
        reliability_data = self._calculate_reliability_diagram(confidences, outcomes)
        
        # Confidence bins
        confidence_bins = self._calculate_confidence_bins(confidences, outcomes)
        
        return CalibrationMetrics(
            mean_confidence=mean_confidence,
            mean_accuracy=mean_accuracy,
            calibration_error=calibration_error,
            brier_score=brier_score,
            log_loss_score=log_loss_score,
            reliability_diagram_data=reliability_data,
            confidence_bins=confidence_bins
        )
    
    def _empty_calibration_metrics(self) -> CalibrationMetrics:
        """Return empty calibration metrics"""
        return CalibrationMetrics(
            mean_confidence=0.0,
            mean_accuracy=0.0,
            calibration_error=0.0,
            brier_score=0.0,
            log_loss_score=0.0,
            reliability_diagram_data={'bin_centers': [], 'bin_accuracies': [], 'bin_confidences': []},
            confidence_bins=[]
        )
    
    def _calculate_ece(self, confidences: np.ndarray, outcomes: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        n_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = outcomes[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _calculate_reliability_diagram(self, confidences: np.ndarray, 
                                     outcomes: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
        """Calculate data for reliability diagram"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = outcomes[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                bin_count = in_bin.sum()
            else:
                bin_accuracy = 0.0
                bin_confidence = 0.0
                bin_count = 0
            
            bin_confidences.append(float(bin_confidence))
            bin_accuracies.append(float(bin_accuracy))
            bin_counts.append(int(bin_count))
        
        return {
            'bin_centers': [float(x) for x in bin_centers],
            'bin_confidences': bin_confidences,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts
        }
    
    def _calculate_confidence_bins(self, confidences: np.ndarray, 
                                 outcomes: np.ndarray, n_bins: int = 10) -> List[Tuple[float, float, int, float]]:
        """Calculate confidence bin statistics"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = int(in_bin.sum())
            
            if count > 0:
                accuracy = float(outcomes[in_bin].mean())
            else:
                accuracy = 0.0
            
            bins.append((float(bin_lower), float(bin_upper), count, accuracy))
        
        return bins

class TrendAnalyzer:
    """Analyzes confidence trends over time"""
    
    def analyze_trends(self, entries: List[ConfidenceEntry]) -> Dict[str, List[float]]:
        """Analyze confidence trends over time"""
        if not entries:
            return {}
        
        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        
        # Group by time periods (hourly)
        hourly_data = defaultdict(list)
        
        for entry in sorted_entries:
            hour_key = entry.timestamp[:13]  # YYYY-MM-DDTHH
            hourly_data[hour_key].append(entry)
        
        # Calculate trends
        hours = sorted(hourly_data.keys())
        confidence_trends = []
        accuracy_trends = []
        calibration_trends = []
        
        for hour in hours:
            hour_entries = hourly_data[hour]
            
            # Average confidence
            confidences = [e.predicted_confidence for e in hour_entries]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            confidence_trends.append(avg_confidence)
            
            # Accuracy (only for entries with outcomes)
            entries_with_outcomes = [e for e in hour_entries if e.actual_outcome is not None]
            if entries_with_outcomes:
                accuracy = sum(1 for e in entries_with_outcomes if e.actual_outcome) / len(entries_with_outcomes)
                
                # Simple calibration error for this hour
                hour_confidences = [e.predicted_confidence for e in entries_with_outcomes]
                hour_outcomes = [e.actual_outcome for e in entries_with_outcomes]
                
                if hour_confidences:
                    calibration_error = abs(sum(hour_confidences) / len(hour_confidences) - accuracy)
                else:
                    calibration_error = 0
            else:
                accuracy = 0
                calibration_error = 0
            
            accuracy_trends.append(accuracy)
            calibration_trends.append(calibration_error)
        
        return {
            'hours': hours,
            'confidence_trends': confidence_trends,
            'accuracy_trends': accuracy_trends,
            'calibration_trends': calibration_trends
        }

class RecommendationEngine:
    """Generates recommendations based on confidence analysis"""
    
    def generate_recommendations(self, analysis: ConfidenceAnalysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        metrics = analysis.calibration_metrics
        
        # Calibration recommendations
        if metrics.calibration_error > 0.1:
            recommendations.append(
                f"High calibration error ({metrics.calibration_error:.3f}). "
                "Consider recalibrating confidence scores or improving uncertainty estimation."
            )
        elif metrics.calibration_error > 0.05:
            recommendations.append(
                f"Moderate calibration error ({metrics.calibration_error:.3f}). "
                "Monitor confidence calibration and consider fine-tuning."
            )
        
        # Overconfidence/underconfidence
        confidence_bias = metrics.mean_confidence - metrics.mean_accuracy
        if confidence_bias > 0.1:
            recommendations.append(
                f"System appears overconfident (bias: +{confidence_bias:.3f}). "
                "Consider reducing confidence scores or improving accuracy."
            )
        elif confidence_bias < -0.1:
            recommendations.append(
                f"System appears underconfident (bias: {confidence_bias:.3f}). "
                "Consider increasing confidence scores for good decisions."
            )
        
        # Brier score recommendations
        if metrics.brier_score > 0.25:
            recommendations.append(
                f"High Brier score ({metrics.brier_score:.3f}). "
                "Focus on improving both confidence calibration and decision accuracy."
            )
        
        # Agent-specific recommendations
        if analysis.by_agent:
            worst_agent = min(
                analysis.by_agent.items(),
                key=lambda x: x[1].mean_accuracy - x[1].calibration_error
            )
            if worst_agent[1].calibration_error > 0.15:
                recommendations.append(
                    f"Agent '{worst_agent[0]}' has poor calibration "
                    f"(error: {worst_agent[1].calibration_error:.3f}). "
                    "Consider agent-specific confidence adjustments."
                )
        
        # Decision type recommendations
        if analysis.by_decision_type:
            for decision_type, metrics in analysis.by_decision_type.items():
                if metrics.calibration_error > 0.15:
                    recommendations.append(
                        f"Decision type '{decision_type}' has poor calibration "
                        f"(error: {metrics.calibration_error:.3f}). "
                        "Review confidence estimation for this decision category."
                    )
        
        # Data coverage recommendations
        if analysis.entries_with_outcomes < analysis.total_entries * 0.3:
            recommendations.append(
                f"Only {analysis.entries_with_outcomes}/{analysis.total_entries} "
                "confidence entries have outcome feedback. "
                "Increase outcome tracking for better calibration analysis."
            )
        
        return recommendations

class ConfidenceReportingSystem:
    """Main confidence score reporting system"""
    
    def __init__(self, storage_file: str = "confidence_data.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.entries = self._load_entries()
        self.calibrator = ConfidenceCalibrator()
        self.trend_analyzer = TrendAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_entries(self) -> List[ConfidenceEntry]:
        """Load confidence entries from storage"""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                return [ConfidenceEntry(**entry) for entry in data]
            except Exception as e:
                self.logger.error(f"Error loading confidence data: {e}")
        return []
    
    def _save_entries(self):
        """Save confidence entries to storage"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump([asdict(entry) for entry in self.entries], f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving confidence data: {e}")
    
    def record_confidence(self, task_id: str, agent_id: str, 
                         decision_type: str, confidence: float,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record a confidence score"""
        entry_id = f"{task_id}_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        entry = ConfidenceEntry(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            agent_id=agent_id,
            decision_type=decision_type,
            predicted_confidence=confidence,
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        self._save_entries()
        
        self.logger.info(f"Recorded confidence {confidence:.3f} for task {task_id}")
        return entry_id
    
    def record_outcome(self, entry_id: str, outcome: bool) -> bool:
        """Record the actual outcome for a confidence entry"""
        for entry in self.entries:
            if entry.id == entry_id:
                entry.actual_outcome = outcome
                entry.outcome_timestamp = datetime.now().isoformat()
                self._save_entries()
                self.logger.info(f"Recorded outcome {outcome} for entry {entry_id}")
                return True
        
        self.logger.warning(f"Entry {entry_id} not found")
        return False
    
    def record_task_outcome(self, task_id: str, outcome: bool) -> int:
        """Record outcome for all confidence entries of a task"""
        updated = 0
        for entry in self.entries:
            if entry.task_id == task_id and entry.actual_outcome is None:
                entry.actual_outcome = outcome
                entry.outcome_timestamp = datetime.now().isoformat()
                updated += 1
        
        if updated > 0:
            self._save_entries()
            self.logger.info(f"Updated {updated} entries for task {task_id} with outcome {outcome}")
        
        return updated
    
    def analyze_confidence(self, start_time: Optional[str] = None,
                         end_time: Optional[str] = None) -> ConfidenceAnalysis:
        """Perform comprehensive confidence analysis"""
        # Filter entries by time range
        filtered_entries = self.entries.copy()
        
        if start_time:
            filtered_entries = [
                e for e in filtered_entries 
                if e.timestamp >= start_time
            ]
        
        if end_time:
            filtered_entries = [
                e for e in filtered_entries 
                if e.timestamp <= end_time
            ]
        
        # Get entries with outcomes for calibration analysis
        entries_with_outcomes = [
            e for e in filtered_entries 
            if e.actual_outcome is not None
        ]
        
        # Overall calibration metrics
        if entries_with_outcomes:
            confidences = [e.predicted_confidence for e in entries_with_outcomes]
            outcomes = [e.actual_outcome for e in entries_with_outcomes]
            calibration_metrics = self.calibrator.calculate_calibration_metrics(confidences, outcomes)
        else:
            calibration_metrics = self.calibrator._empty_calibration_metrics()
        
        # Trend analysis
        trend_data = self.trend_analyzer.analyze_trends(filtered_entries)
        
        # Analysis by agent
        by_agent = {}
        agents = set(e.agent_id for e in entries_with_outcomes)
        for agent_id in agents:
            agent_entries = [e for e in entries_with_outcomes if e.agent_id == agent_id]
            if agent_entries:
                confidences = [e.predicted_confidence for e in agent_entries]
                outcomes = [e.actual_outcome for e in agent_entries]
                by_agent[agent_id] = self.calibrator.calculate_calibration_metrics(confidences, outcomes)
        
        # Analysis by decision type
        by_decision_type = {}
        decision_types = set(e.decision_type for e in entries_with_outcomes)
        for decision_type in decision_types:
            type_entries = [e for e in entries_with_outcomes if e.decision_type == decision_type]
            if type_entries:
                confidences = [e.predicted_confidence for e in type_entries]
                outcomes = [e.actual_outcome for e in type_entries]
                by_decision_type[decision_type] = self.calibrator.calculate_calibration_metrics(confidences, outcomes)
        
        # Create analysis object
        analysis = ConfidenceAnalysis(
            period_start=start_time or "beginning",
            period_end=end_time or datetime.now().isoformat(),
            total_entries=len(filtered_entries),
            entries_with_outcomes=len(entries_with_outcomes),
            calibration_metrics=calibration_metrics,
            trend_data=trend_data,
            by_agent=by_agent,
            by_decision_type=by_decision_type,
            recommendations=[]
        )
        
        # Generate recommendations
        analysis.recommendations = self.recommendation_engine.generate_recommendations(analysis)
        
        return analysis
    
    def generate_calibration_plot(self, analysis: ConfidenceAnalysis, 
                                output_file: str) -> str:
        """Generate calibration reliability diagram"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Get reliability data
            reliability_data = analysis.calibration_metrics.reliability_diagram_data
            
            if reliability_data['bin_centers']:
                # Plot reliability diagram
                plt.subplot(2, 2, 1)
                plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                plt.scatter(
                    reliability_data['bin_confidences'],
                    reliability_data['bin_accuracies'],
                    s=[c*10 + 10 for c in reliability_data['bin_counts']],
                    alpha=0.7,
                    label='Observed Calibration'
                )
                plt.xlabel('Mean Predicted Confidence')
                plt.ylabel('Mean Observed Accuracy')
                plt.title('Reliability Diagram')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot confidence histogram
                plt.subplot(2, 2, 2)
                plt.bar(
                    reliability_data['bin_centers'],
                    reliability_data['bin_counts'],
                    width=0.08,
                    alpha=0.7
                )
                plt.xlabel('Confidence Bin')
                plt.ylabel('Number of Predictions')
                plt.title('Confidence Distribution')
                plt.grid(True, alpha=0.3)
            
            # Plot trends if available
            if analysis.trend_data.get('hours'):
                plt.subplot(2, 2, 3)
                hours = range(len(analysis.trend_data['hours']))
                plt.plot(hours, analysis.trend_data['confidence_trends'], label='Confidence', marker='o')
                plt.plot(hours, analysis.trend_data['accuracy_trends'], label='Accuracy', marker='s')
                plt.xlabel('Time Period')
                plt.ylabel('Score')
                plt.title('Confidence vs Accuracy Trends')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(hours, analysis.trend_data['calibration_trends'], label='Calibration Error', marker='^', color='red')
                plt.xlabel('Time Period')
                plt.ylabel('Calibration Error')
                plt.title('Calibration Error Trend')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Calibration plot saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating calibration plot: {e}")
            return ""
    
    def export_analysis(self, analysis: ConfidenceAnalysis, 
                       output_file: str, format: str = "json") -> str:
        """Export confidence analysis"""
        try:
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(asdict(analysis), f, indent=2, default=str)
            elif format == "csv":
                # Export detailed entries
                entries_data = []
                for entry in self.entries:
                    entries_data.append(asdict(entry))
                
                df = pd.DataFrame(entries_data)
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Analysis exported to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
            return ""
    
    def get_entry_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about confidence entries"""
        entries_with_outcomes = [e for e in self.entries if e.actual_outcome is not None]
        
        stats = {
            'total_entries': len(self.entries),
            'entries_with_outcomes': len(entries_with_outcomes),
            'outcome_coverage': len(entries_with_outcomes) / len(self.entries) if self.entries else 0,
            'agents': list(set(e.agent_id for e in self.entries)),
            'decision_types': list(set(e.decision_type for e in self.entries)),
            'confidence_range': {
                'min': min((e.predicted_confidence for e in self.entries), default=0),
                'max': max((e.predicted_confidence for e in self.entries), default=0),
                'mean': sum(e.predicted_confidence for e in self.entries) / len(self.entries) if self.entries else 0
            }
        }
        
        if entries_with_outcomes:
            successful_outcomes = sum(1 for e in entries_with_outcomes if e.actual_outcome)
            stats['success_rate'] = successful_outcomes / len(entries_with_outcomes)
        else:
            stats['success_rate'] = 0
        
        return stats

# Demo and testing functions
def create_demo_confidence_system() -> ConfidenceReportingSystem:
    """Create demo confidence reporting system"""
    return ConfidenceReportingSystem("demo_confidence_data.json")

def generate_demo_confidence_data(system: ConfidenceReportingSystem):
    """Generate demo confidence data"""
    import random
    
    # Simulate confidence entries over the last week
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(200):
        # Simulate different agents and decision types
        agent_id = f"agent_{i % 3}"
        decision_types = ['classification', 'regression', 'planning', 'routing']
        decision_type = random.choice(decision_types)
        
        # Simulate confidence scores with some agent bias
        if agent_id == "agent_0":
            confidence = min(0.9, max(0.1, random.gauss(0.8, 0.15)))  # Overconfident
        elif agent_id == "agent_1":
            confidence = min(0.9, max(0.1, random.gauss(0.6, 0.15)))  # Underconfident  
        else:
            confidence = min(0.9, max(0.1, random.gauss(0.75, 0.1)))  # Well-calibrated
        
        # Record confidence
        entry_time = base_time + timedelta(minutes=i*20)
        
        # Temporarily set current time for realistic timestamps
        original_now = datetime.now
        datetime.now = lambda: entry_time
        
        entry_id = system.record_confidence(
            task_id=f"task_{i}",
            agent_id=agent_id,
            decision_type=decision_type,
            confidence=confidence,
            metadata={"complexity": random.choice(["low", "medium", "high"])}
        )
        
        # Restore original datetime.now
        datetime.now = original_now
        
        # Simulate outcomes (with some correlation to confidence)
        # Higher confidence should generally correlate with success, but not perfectly
        success_probability = 0.3 + 0.6 * confidence + random.gauss(0, 0.1)
        success_probability = max(0, min(1, success_probability))
        
        actual_outcome = random.random() < success_probability
        system.record_outcome(entry_id, actual_outcome)

if __name__ == '__main__':
    # Demo usage
    system = create_demo_confidence_system()
    
    # Generate demo data
    generate_demo_confidence_data(system)
    
    # Analyze confidence
    analysis = system.analyze_confidence()
    
    print("\n=== Confidence Analysis ===")
    print(f"Total entries: {analysis.total_entries}")
    print(f"Entries with outcomes: {analysis.entries_with_outcomes}")
    print(f"Mean confidence: {analysis.calibration_metrics.mean_confidence:.3f}")
    print(f"Mean accuracy: {analysis.calibration_metrics.mean_accuracy:.3f}")
    print(f"Calibration error: {analysis.calibration_metrics.calibration_error:.3f}")
    print(f"Brier score: {analysis.calibration_metrics.brier_score:.3f}")
    
    print("\n=== Agent Performance ===")
    for agent_id, metrics in analysis.by_agent.items():
        print(f"{agent_id}: accuracy={metrics.mean_accuracy:.3f}, calibration_error={metrics.calibration_error:.3f}")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"{i}. {rec}")
    
    # Export analysis
    system.export_analysis(analysis, "demo_confidence_analysis.json")
    
    # Generate plot
    plot_file = system.generate_calibration_plot(analysis, "demo_calibration_plot.png")
    print(f"\nCalibration plot saved to: {plot_file}")
    
    # Get statistics
    stats = system.get_entry_statistics()
    print(f"\n=== Statistics ===")
    print(json.dumps(stats, indent=2))
