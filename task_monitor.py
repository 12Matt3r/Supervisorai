"""Task Completion Monitoring - Tracks subtask alignment and progress"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TaskProgress:
    """Task progress tracking"""
    task_id: str
    original_goal: str
    current_status: str
    completion_percentage: float
    milestones_achieved: List[str]
    milestones_remaining: List[str]
    drift_detected: bool
    drift_severity: float

class TaskCompletionMonitor:
    """Monitors task completion alignment and progress"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.task_history = []
        self.drift_patterns = self._initialize_drift_patterns()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'drift_threshold': 0.3,
            'completion_threshold': 0.8,
            'milestone_weight': 0.4,
            'alignment_weight': 0.6,
            'progress_tracking_enabled': True
        }
    
    def _initialize_drift_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns that indicate task drift"""
        return [
            {
                'pattern': r'(?i)(instead|however|but|alternatively)',
                'weight': 0.3,
                'description': 'Direction change indicators'
            },
            {
                'pattern': r'(?i)(different|other|new|additional)\s+(task|goal|objective)',
                'weight': 0.5,
                'description': 'New task introduction'
            },
            {
                'pattern': r'(?i)(skip|ignore|bypass|omit)',
                'weight': 0.4,
                'description': 'Task avoidance indicators'
            },
            {
                'pattern': r'(?i)(unrelated|irrelevant|tangential)',
                'weight': 0.6,
                'description': 'Relevance loss indicators'
            }
        ]
    
    def evaluate_task_completion(self, task_data: Dict[str, Any], 
                               original_goals: List[str],
                               current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task completion alignment and progress"""
        try:
            # Extract task information
            current_outputs = task_data.get('outputs', [])
            execution_steps = task_data.get('steps', [])
            task_description = task_data.get('description', '')
            
            # Calculate alignment score
            alignment_score = self._calculate_alignment_score(
                current_outputs, execution_steps, original_goals
            )
            
            # Detect task drift
            drift_result = self._detect_task_drift(
                execution_steps, original_goals, task_description
            )
            
            # Calculate progress score
            progress_score = self._calculate_progress_score(
                current_progress, original_goals
            )
            
            # Check milestone completion
            milestone_result = self._evaluate_milestones(
                current_progress, original_goals
            )
            
            # Calculate overall completion score
            completion_score = (
                alignment_score * self.config['alignment_weight'] +
                progress_score * (1 - self.config['alignment_weight'])
            )
            
            # Detect incomplete execution patterns
            incomplete_patterns = self._detect_incomplete_execution(
                current_outputs, execution_steps
            )
            
            return {
                'score': completion_score,
                'alignment_score': alignment_score,
                'progress_score': progress_score,
                'drift_detected': drift_result['detected'],
                'drift_severity': drift_result['severity'],
                'drift_indicators': drift_result['indicators'],
                'milestones_achieved': milestone_result['achieved'],
                'milestones_remaining': milestone_result['remaining'],
                'milestone_progress': milestone_result['progress'],
                'incomplete_patterns': incomplete_patterns,
                'completion_status': self._determine_completion_status(
                    completion_score, drift_result, milestone_result
                ),
                'recommendations': self._generate_task_recommendations(
                    alignment_score, drift_result, milestone_result, incomplete_patterns
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'status': 'evaluation_failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_alignment_score(self, outputs: List[str], 
                                 steps: List[str], 
                                 goals: List[str]) -> float:
        """Calculate how well current outputs align with original goals"""
        if not goals or not (outputs or steps):
            return 0.0
        
        total_score = 0.0
        content_to_analyze = ' '.join(outputs + steps).lower()
        
        for goal in goals:
            goal_keywords = self._extract_keywords(goal.lower())
            
            if not goal_keywords:
                continue
                
            # Check keyword presence
            keyword_matches = sum(1 for keyword in goal_keywords 
                                if keyword in content_to_analyze)
            keyword_score = keyword_matches / len(goal_keywords)
            
            # Check semantic similarity (basic version)
            semantic_score = self._calculate_semantic_similarity(
                goal.lower(), content_to_analyze
            )
            
            # Combine scores
            goal_score = (keyword_score * 0.6 + semantic_score * 0.4)
            total_score += goal_score
        
        return min(total_score / len(goals), 1.0) if goals else 0.0
    
    def _detect_task_drift(self, steps: List[str], 
                          goals: List[str], 
                          description: str) -> Dict[str, Any]:
        """Detect if task execution has drifted from original goals"""
        content = ' '.join(steps + [description]).lower()
        
        drift_indicators = []
        total_drift_weight = 0.0
        
        for pattern_info in self.drift_patterns:
            matches = re.findall(pattern_info['pattern'], content)
            if matches:
                drift_indicators.append({
                    'pattern': pattern_info['description'],
                    'matches': matches,
                    'weight': pattern_info['weight']
                })
                total_drift_weight += pattern_info['weight'] * len(matches)
        
        # Normalize drift weight
        max_possible_weight = sum(p['weight'] for p in self.drift_patterns) * 3  # Assume max 3 matches per pattern
        drift_severity = min(total_drift_weight / max_possible_weight, 1.0) if max_possible_weight > 0 else 0.0
        
        # Check goal keyword presence in recent steps
        if steps:
            recent_content = ' '.join(steps[-3:]).lower()  # Last 3 steps
            goal_keywords = set()
            for goal in goals:
                goal_keywords.update(self._extract_keywords(goal.lower()))
            
            if goal_keywords:
                keyword_presence = sum(1 for keyword in goal_keywords 
                                     if keyword in recent_content) / len(goal_keywords)
                # Low keyword presence indicates possible drift
                if keyword_presence < 0.3:
                    drift_severity += 0.2
                    drift_indicators.append({
                        'pattern': 'Low goal keyword presence in recent steps',
                        'matches': [],
                        'weight': 0.2
                    })
        
        drift_detected = drift_severity > self.config['drift_threshold']
        
        return {
            'detected': drift_detected,
            'severity': min(drift_severity, 1.0),
            'indicators': drift_indicators
        }
    
    def _calculate_progress_score(self, progress: Dict[str, Any], 
                                goals: List[str]) -> float:
        """Calculate progress score based on milestones and completion"""
        if not progress:
            return 0.0
        
        # Extract progress indicators
        completed_items = progress.get('completed', [])
        total_items = progress.get('total', 0) or len(goals)
        percentage = progress.get('percentage', 0)
        
        if total_items > 0:
            completion_ratio = len(completed_items) / total_items
            return min((completion_ratio + percentage / 100.0) / 2.0, 1.0)
        
        return percentage / 100.0 if percentage else 0.0
    
    def _evaluate_milestones(self, progress: Dict[str, Any], 
                           goals: List[str]) -> Dict[str, Any]:
        """Evaluate milestone completion"""
        milestones = progress.get('milestones', goals)
        completed = progress.get('completed', [])
        
        achieved = [m for m in milestones if m in completed or 
                   any(keyword in str(completed).lower() 
                       for keyword in self._extract_keywords(str(m).lower()))]
        
        remaining = [m for m in milestones if m not in achieved]
        
        milestone_progress = len(achieved) / len(milestones) if milestones else 1.0
        
        return {
            'achieved': achieved,
            'remaining': remaining,
            'progress': milestone_progress,
            'total_milestones': len(milestones)
        }
    
    def _detect_incomplete_execution(self, outputs: List[str], 
                                   steps: List[str]) -> List[Dict[str, Any]]:
        """Detect patterns indicating incomplete execution"""
        incomplete_patterns = []
        content = ' '.join(outputs + steps).lower()
        
        # Pattern 1: Unfinished sentences or thoughts
        unfinished_patterns = [
            r'\b(will|would|should|need to|going to)\s+\w+(?:\s+\w+)*\s*$',
            r'\b(but|however|although)\s[^.!?]*$',
            r'\btodo\b|\bfixme\b|\btbd\b',
            r'\b(incomplete|unfinished|pending)\b'
        ]
        
        for pattern in unfinished_patterns:
            matches = re.findall(pattern, content)
            if matches:
                incomplete_patterns.append({
                    'type': 'unfinished_content',
                    'pattern': pattern,
                    'matches': matches
                })
        
        # Pattern 2: Error indicators
        error_patterns = [
            r'\berror\b|\bfailed\b|\bexception\b',
            r'\bnot found\b|\bmissing\b|\bunavailable\b',
            r'\btimeout\b|\bconnection\b.*\bfailed\b'
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, content)
            if matches:
                incomplete_patterns.append({
                    'type': 'error_indicator',
                    'pattern': pattern,
                    'matches': matches
                })
        
        # Pattern 3: Empty or minimal outputs
        if not outputs or (len(outputs) == 1 and len(outputs[0].strip()) < 10):
            incomplete_patterns.append({
                'type': 'minimal_output',
                'pattern': 'insufficient_content',
                'matches': ['Empty or minimal output detected']
            })
        
        return incomplete_patterns
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Basic semantic similarity calculation"""
        words1 = set(self._extract_keywords(text1))
        words2 = set(self._extract_keywords(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
                     'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'can',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                     'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Extract words (alphanumeric, at least 3 characters)
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
        
        # Filter out stop words
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def _determine_completion_status(self, score: float, 
                                   drift_result: Dict[str, Any],
                                   milestone_result: Dict[str, Any]) -> str:
        """Determine overall completion status"""
        if drift_result['detected'] and drift_result['severity'] > 0.7:
            return 'drifted'
        
        if score >= 0.9 and milestone_result['progress'] >= 0.9:
            return 'completed'
        elif score >= 0.8 and milestone_result['progress'] >= 0.7:
            return 'nearly_completed'
        elif score >= 0.6 and milestone_result['progress'] >= 0.5:
            return 'in_progress'
        elif score >= 0.3:
            return 'started'
        else:
            return 'not_started'
    
    def _generate_task_recommendations(self, alignment_score: float,
                                     drift_result: Dict[str, Any],
                                     milestone_result: Dict[str, Any],
                                     incomplete_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate task-specific recommendations"""
        recommendations = []
        
        if alignment_score < 0.7:
            recommendations.append("Realign execution with original goals")
        
        if drift_result['detected']:
            recommendations.append(f"Address task drift (severity: {drift_result['severity']:.2f})")
            recommendations.append("Review original goals and refocus execution")
        
        if milestone_result['progress'] < 0.5:
            recommendations.append("Accelerate milestone completion")
            recommendations.append(f"Focus on remaining milestones: {len(milestone_result['remaining'])}")
        
        if incomplete_patterns:
            recommendations.append(f"Address {len(incomplete_patterns)} incomplete execution patterns")
        
        return recommendations
