"""
Pattern Tracking and Knowledge Base for Supervisor Agent
Identifies recurring failure patterns and builds knowledge base
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import re
from difflib import SequenceMatcher


@dataclass
class Pattern:
    id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    frequency: int
    first_seen: datetime
    last_seen: datetime
    agents_affected: Set[str]
    severity: str
    resolution_suggestions: List[str]
    metadata: Dict[str, Any]


@dataclass
class KnowledgeEntry:
    id: str
    title: str
    category: str
    problem_description: str
    solution: str
    confidence: float
    effectiveness_score: float
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    related_patterns: List[str]
    usage_count: int
    success_rate: float


@dataclass
class PatternAnalysisResult:
    timestamp: datetime
    analysis_period: timedelta
    total_events: int
    patterns_detected: int
    new_patterns: int
    critical_patterns: int
    pattern_summary: List[Dict[str, Any]]
    recommendations: List[str]
    knowledge_base_updates: int


class PatternTracker:
    """Tracks patterns and builds knowledge base from historical data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage files
        self.patterns_file = Path(config.get('patterns_file', 'patterns.json'))
        self.knowledge_file = Path(config.get('knowledge_file', 'knowledge_base.json'))
        
        # Configuration
        self.min_frequency = config.get('min_pattern_frequency', 3)
        self.lookback_days = config.get('pattern_lookback_days', 30)
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        
        # Data storage
        self.patterns: Dict[str, Pattern] = {}
        self.knowledge_base: Dict[str, KnowledgeEntry] = {}
        self.event_history: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_patterns()
        self._load_knowledge_base()
        
        self.logger.info("Pattern Tracker initialized")
        
    def analyze_events(self, events: List[Dict[str, Any]]) -> PatternAnalysisResult:
        """Analyze events for patterns and update knowledge base"""
        
        analysis_start = datetime.now()
        self.logger.info(f"Starting pattern analysis on {len(events)} events")
        
        # Store events for future analysis
        self.event_history.extend(events)
        
        # Keep only recent events in memory
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        self.event_history = [
            event for event in self.event_history
            if datetime.fromisoformat(event.get('timestamp', '1970-01-01')) > cutoff_date
        ]
        
        # Detect patterns
        new_patterns = self._detect_patterns(events)
        self._update_existing_patterns(events)
        
        # Update knowledge base
        kb_updates = self._update_knowledge_base(new_patterns)
        
        # Generate analysis result
        critical_patterns = len([p for p in self.patterns.values() if p.severity == 'critical'])
        pattern_summary = self._generate_pattern_summary()
        recommendations = self._generate_pattern_recommendations()
        
        result = PatternAnalysisResult(
            timestamp=analysis_start,
            analysis_period=datetime.now() - analysis_start,
            total_events=len(events),
            patterns_detected=len(self.patterns),
            new_patterns=len(new_patterns),
            critical_patterns=critical_patterns,
            pattern_summary=pattern_summary,
            recommendations=recommendations,
            knowledge_base_updates=kb_updates
        )
        
        # Save updated patterns and knowledge
        self._save_patterns()
        self._save_knowledge_base()
        
        self.logger.info(f"Pattern analysis completed: {len(new_patterns)} new patterns detected")
        return result
        
    def _detect_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect new patterns in events"""
        
        new_patterns = []
        
        # Group events by various criteria
        pattern_candidates = self._identify_pattern_candidates(events)
        
        for candidate in pattern_candidates:
            if candidate['frequency'] >= self.min_frequency:
                pattern_id = self._generate_pattern_id(candidate)
                
                # Check if pattern already exists
                if pattern_id not in self.patterns:
                    pattern = Pattern(
                        id=pattern_id,
                        pattern_type=candidate['type'],
                        description=candidate['description'],
                        conditions=candidate['conditions'],
                        frequency=candidate['frequency'],
                        first_seen=candidate['first_seen'],
                        last_seen=candidate['last_seen'],
                        agents_affected=set(candidate['agents']),
                        severity=self._determine_pattern_severity(candidate),
                        resolution_suggestions=self._generate_resolution_suggestions(candidate),
                        metadata=candidate.get('metadata', {})
                    )
                    
                    self.patterns[pattern_id] = pattern
                    new_patterns.append(pattern)
                    
                    self.logger.info(f"New {pattern.severity} pattern detected: {pattern.description}")
                    
        return new_patterns
        
    def _identify_pattern_candidates(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential patterns from events"""
        
        candidates = []
        
        # Error pattern detection
        candidates.extend(self._detect_error_patterns(events))
        
        # Performance pattern detection
        candidates.extend(self._detect_performance_patterns(events))
        
        # Temporal pattern detection
        candidates.extend(self._detect_temporal_patterns(events))
        
        # Agent-specific pattern detection
        candidates.extend(self._detect_agent_patterns(events))
        
        # Task sequence pattern detection
        candidates.extend(self._detect_sequence_patterns(events))
        
        return candidates
        
    def _detect_error_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect recurring error patterns"""
        
        error_events = [
            event for event in events
            if event.get('level') in ['error', 'critical'] or 
               event.get('status') == 'failed'
        ]
        
        if not error_events:
            return []
            
        patterns = []
        
        # Group by error message similarity
        error_groups = self._group_by_similarity(
            error_events,
            key_func=lambda e: e.get('message', e.get('outcome', ''))
        )
        
        for group in error_groups:
            if len(group) >= self.min_frequency:
                agents = [e.get('agent_id', 'unknown') for e in group]
                timestamps = [datetime.fromisoformat(e['timestamp']) for e in group]
                
                # Extract common error characteristics
                common_error = self._extract_common_error_info(group)
                
                patterns.append({
                    'type': 'error_pattern',
                    'description': f"Recurring error: {common_error['message']}",
                    'conditions': {
                        'error_type': common_error['type'],
                        'error_message_pattern': common_error['pattern'],
                        'affected_components': common_error['components']
                    },
                    'frequency': len(group),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'agents': agents,
                    'metadata': {
                        'sample_events': group[:3],  # Keep sample events
                        'error_details': common_error
                    }
                })
                
        return patterns
        
    def _detect_performance_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance-related patterns"""
        
        perf_events = [
            event for event in events
            if 'duration' in event.get('metadata', {}) or 
               'performance' in event.get('event_type', '')
        ]
        
        patterns = []
        
        # Detect slow task patterns
        slow_tasks = [
            event for event in perf_events
            if event.get('metadata', {}).get('duration', 0) > 300  # 5 minutes
        ]
        
        if len(slow_tasks) >= self.min_frequency:
            agents = [e.get('agent_id', 'unknown') for e in slow_tasks]
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in slow_tasks]
            
            # Analyze task types and durations
            task_analysis = self._analyze_slow_tasks(slow_tasks)
            
            patterns.append({
                'type': 'performance_pattern',
                'description': f"Recurring slow tasks: {task_analysis['common_types']}",
                'conditions': {
                    'min_duration': task_analysis['min_duration'],
                    'task_types': task_analysis['task_types'],
                    'common_characteristics': task_analysis['characteristics']
                },
                'frequency': len(slow_tasks),
                'first_seen': min(timestamps),
                'last_seen': max(timestamps),
                'agents': agents,
                'metadata': task_analysis
            })
            
        return patterns
        
    def _detect_temporal_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect time-based patterns"""
        
        patterns = []
        
        # Group events by hour of day
        hourly_failures = defaultdict(list)
        for event in events:
            if event.get('level') in ['error', 'critical']:
                timestamp = datetime.fromisoformat(event['timestamp'])
                hourly_failures[timestamp.hour].append(event)
                
        # Find hours with significantly higher failure rates
        avg_failures = sum(len(events) for events in hourly_failures.values()) / 24
        
        for hour, hour_events in hourly_failures.items():
            if len(hour_events) > avg_failures * 2 and len(hour_events) >= self.min_frequency:
                agents = [e.get('agent_id', 'unknown') for e in hour_events]
                timestamps = [datetime.fromisoformat(e['timestamp']) for e in hour_events]
                
                patterns.append({
                    'type': 'temporal_pattern',
                    'description': f"High failure rate during hour {hour}:00-{hour+1}:00",
                    'conditions': {
                        'hour_of_day': hour,
                        'failure_rate_multiplier': len(hour_events) / avg_failures,
                        'common_error_types': self._extract_common_errors(hour_events)
                    },
                    'frequency': len(hour_events),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'agents': agents,
                    'metadata': {
                        'hourly_distribution': dict(hourly_failures),
                        'baseline_rate': avg_failures
                    }
                })
                
        return patterns
        
    def _detect_agent_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect agent-specific patterns"""
        
        patterns = []
        
        # Group events by agent
        agent_events = defaultdict(list)
        for event in events:
            agent_id = event.get('agent_id', 'unknown')
            agent_events[agent_id].append(event)
            
        # Analyze each agent's behavior
        for agent_id, agent_event_list in agent_events.items():
            if len(agent_event_list) < self.min_frequency:
                continue
                
            # Calculate failure rate
            failures = [e for e in agent_event_list if e.get('level') in ['error', 'critical']]
            failure_rate = len(failures) / len(agent_event_list)
            
            # Check for concerning patterns
            if failure_rate > 0.3:  # 30% failure rate
                timestamps = [datetime.fromisoformat(e['timestamp']) for e in agent_event_list]
                
                # Analyze failure characteristics
                failure_analysis = self._analyze_agent_failures(failures)
                
                patterns.append({
                    'type': 'agent_pattern',
                    'description': f"Agent {agent_id} showing high failure rate ({failure_rate:.1%})",
                    'conditions': {
                        'agent_id': agent_id,
                        'failure_rate': failure_rate,
                        'common_failure_types': failure_analysis['types'],
                        'failure_characteristics': failure_analysis['characteristics']
                    },
                    'frequency': len(failures),
                    'first_seen': min(timestamps),
                    'last_seen': max(timestamps),
                    'agents': [agent_id],
                    'metadata': failure_analysis
                })
                
        return patterns
        
    def _detect_sequence_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in task sequences"""
        
        patterns = []
        
        # Sort events by timestamp and agent
        agent_sequences = defaultdict(list)
        for event in sorted(events, key=lambda x: x['timestamp']):
            agent_id = event.get('agent_id', 'unknown')
            agent_sequences[agent_id].append(event)
            
        # Look for recurring sequences that lead to failures
        for agent_id, sequence in agent_sequences.items():
            if len(sequence) < 5:  # Need minimum sequence length
                continue
                
            failure_sequences = self._extract_failure_sequences(sequence)
            
            for seq_pattern in failure_sequences:
                if seq_pattern['frequency'] >= self.min_frequency:
                    patterns.append({
                        'type': 'sequence_pattern',
                        'description': f"Failure sequence pattern: {seq_pattern['description']}",
                        'conditions': {
                            'sequence_steps': seq_pattern['steps'],
                            'sequence_length': seq_pattern['length'],
                            'failure_probability': seq_pattern['failure_prob']
                        },
                        'frequency': seq_pattern['frequency'],
                        'first_seen': seq_pattern['first_seen'],
                        'last_seen': seq_pattern['last_seen'],
                        'agents': [agent_id],
                        'metadata': seq_pattern
                    })
                    
        return patterns
        
    def _update_existing_patterns(self, events: List[Dict[str, Any]]):
        """Update frequency and metadata of existing patterns"""
        
        for pattern in self.patterns.values():
            # Find events matching this pattern
            matching_events = self._find_matching_events(events, pattern)
            
            if matching_events:
                # Update frequency and last seen
                pattern.frequency += len(matching_events)
                pattern.last_seen = max(
                    datetime.fromisoformat(e['timestamp']) 
                    for e in matching_events
                )
                
                # Update agents affected
                new_agents = {e.get('agent_id', 'unknown') for e in matching_events}
                pattern.agents_affected.update(new_agents)
                
                # Update severity if needed
                new_severity = self._determine_pattern_severity({
                    'frequency': pattern.frequency,
                    'type': pattern.pattern_type,
                    'agents': list(pattern.agents_affected)
                })
                pattern.severity = new_severity
                
    def _update_knowledge_base(self, new_patterns: List[Pattern]) -> int:
        """Update knowledge base with new patterns and solutions"""
        
        updates = 0
        
        for pattern in new_patterns:
            # Generate knowledge entry from pattern
            kb_entry = self._pattern_to_knowledge_entry(pattern)
            
            if kb_entry:
                self.knowledge_base[kb_entry.id] = kb_entry
                updates += 1
                
        # Update existing entries based on pattern frequency changes
        for kb_id, entry in self.knowledge_base.items():
            related_patterns = [
                p for p in self.patterns.values()
                if p.id in entry.related_patterns
            ]
            
            if related_patterns:
                # Update effectiveness based on pattern frequency trends
                self._update_knowledge_effectiveness(entry, related_patterns)
                entry.updated_at = datetime.now()
                updates += 1
                
        return updates
        
    def _pattern_to_knowledge_entry(self, pattern: Pattern) -> Optional[KnowledgeEntry]:
        """Convert a pattern to a knowledge base entry"""
        
        if pattern.pattern_type == 'error_pattern':
            return KnowledgeEntry(
                id=f"kb_{pattern.id}",
                title=f"Resolve: {pattern.description}",
                category="Error Resolution",
                problem_description=pattern.description,
                solution=self._generate_error_solution(pattern),
                confidence=0.7,
                effectiveness_score=0.0,  # Will be updated based on usage
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=self._generate_tags(pattern),
                related_patterns=[pattern.id],
                usage_count=0,
                success_rate=0.0
            )
        elif pattern.pattern_type == 'performance_pattern':
            return KnowledgeEntry(
                id=f"kb_{pattern.id}",
                title=f"Optimize: {pattern.description}",
                category="Performance Optimization",
                problem_description=pattern.description,
                solution=self._generate_performance_solution(pattern),
                confidence=0.6,
                effectiveness_score=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                tags=self._generate_tags(pattern),
                related_patterns=[pattern.id],
                usage_count=0,
                success_rate=0.0
            )
            
        return None
        
    def get_pattern_recommendations(self, agent_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on patterns for specific agent/context"""
        
        recommendations = []
        
        # Find relevant patterns for this agent
        relevant_patterns = [
            pattern for pattern in self.patterns.values()
            if agent_id in pattern.agents_affected or 
               self._pattern_matches_context(pattern, context)
        ]
        
        # Sort by severity and frequency
        relevant_patterns.sort(
            key=lambda p: (p.severity == 'critical', p.frequency), 
            reverse=True
        )
        
        for pattern in relevant_patterns[:5]:  # Top 5 recommendations
            # Find related knowledge base entries
            kb_entries = [
                entry for entry in self.knowledge_base.values()
                if pattern.id in entry.related_patterns
            ]
            
            recommendation = {
                'pattern_id': pattern.id,
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
                'severity': pattern.severity,
                'frequency': pattern.frequency,
                'resolution_suggestions': pattern.resolution_suggestions,
                'knowledge_entries': [
                    {
                        'title': entry.title,
                        'solution': entry.solution,
                        'confidence': entry.confidence,
                        'success_rate': entry.success_rate
                    }
                    for entry in sorted(kb_entries, key=lambda x: x.effectiveness_score, reverse=True)[:3]
                ]
            }
            
            recommendations.append(recommendation)
            
        return recommendations
        
    def export_patterns(self, output_file: str) -> int:
        """Export patterns to JSON file"""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_patterns': len(self.patterns),
            'patterns': {
                pattern_id: {
                    'id': pattern.id,
                    'type': pattern.pattern_type,
                    'description': pattern.description,
                    'conditions': pattern.conditions,
                    'frequency': pattern.frequency,
                    'first_seen': pattern.first_seen.isoformat(),
                    'last_seen': pattern.last_seen.isoformat(),
                    'agents_affected': list(pattern.agents_affected),
                    'severity': pattern.severity,
                    'resolution_suggestions': pattern.resolution_suggestions,
                    'metadata': pattern.metadata
                }
                for pattern_id, pattern in self.patterns.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Exported {len(self.patterns)} patterns to {output_file}")
        return len(self.patterns)
        
    def _load_patterns(self):
        """Load patterns from file"""
        if not self.patterns_file.exists():
            return
            
        try:
            with open(self.patterns_file, 'r') as f:
                data = json.load(f)
                
            for pattern_data in data.get('patterns', {}).values():
                pattern = Pattern(
                    id=pattern_data['id'],
                    pattern_type=pattern_data['type'],
                    description=pattern_data['description'],
                    conditions=pattern_data['conditions'],
                    frequency=pattern_data['frequency'],
                    first_seen=datetime.fromisoformat(pattern_data['first_seen']),
                    last_seen=datetime.fromisoformat(pattern_data['last_seen']),
                    agents_affected=set(pattern_data['agents_affected']),
                    severity=pattern_data['severity'],
                    resolution_suggestions=pattern_data['resolution_suggestions'],
                    metadata=pattern_data['metadata']
                )
                self.patterns[pattern.id] = pattern
                
            self.logger.info(f"Loaded {len(self.patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")
            
    def _save_patterns(self):
        """Save patterns to file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_patterns': len(self.patterns),
                'patterns': {
                    pattern_id: {
                        'id': pattern.id,
                        'type': pattern.pattern_type,
                        'description': pattern.description,
                        'conditions': pattern.conditions,
                        'frequency': pattern.frequency,
                        'first_seen': pattern.first_seen.isoformat(),
                        'last_seen': pattern.last_seen.isoformat(),
                        'agents_affected': list(pattern.agents_affected),
                        'severity': pattern.severity,
                        'resolution_suggestions': pattern.resolution_suggestions,
                        'metadata': pattern.metadata
                    }
                    for pattern_id, pattern in self.patterns.items()
                }
            }
            
            with open(self.patterns_file, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")
            
    # Additional helper methods would be implemented here...
    # (Truncated for brevity - the complete implementation would include all helper methods)
    
    def _generate_pattern_id(self, candidate: Dict[str, Any]) -> str:
        """Generate unique pattern ID"""
        content = f"{candidate['type']}:{candidate['description']}:{candidate['conditions']}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
        
    def _determine_pattern_severity(self, candidate: Dict[str, Any]) -> str:
        """Determine pattern severity based on frequency and impact"""
        frequency = candidate['frequency']
        pattern_type = candidate['type']
        
        if frequency > 20 or pattern_type == 'error_pattern':
            return 'critical'
        elif frequency > 10 or pattern_type == 'performance_pattern':
            return 'high'
        elif frequency > 5:
            return 'medium'
        else:
            return 'low'
            
    def _generate_resolution_suggestions(self, candidate: Dict[str, Any]) -> List[str]:
        """Generate resolution suggestions for pattern"""
        suggestions = []
        
        if candidate['type'] == 'error_pattern':
            suggestions = [
                "Review error logs for common root causes",
                "Implement additional error handling",
                "Consider retrying failed operations with backoff"
            ]
        elif candidate['type'] == 'performance_pattern':
            suggestions = [
                "Analyze task resource requirements",
                "Optimize task execution algorithms",
                "Consider task parallelization or caching"
            ]
            
        return suggestions
        
    def _generate_pattern_summary(self) -> List[Dict[str, Any]]:
        """Generate summary of all patterns"""
        summary = []
        
        for pattern in sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True):
            summary.append({
                'id': pattern.id,
                'type': pattern.pattern_type,
                'description': pattern.description,
                'frequency': pattern.frequency,
                'severity': pattern.severity,
                'agents_count': len(pattern.agents_affected),
                'last_seen': pattern.last_seen.isoformat()
            })
            
        return summary[:20]  # Top 20 patterns
        
    def _generate_pattern_recommendations(self) -> List[str]:
        """Generate overall recommendations based on patterns"""
        recommendations = []
        
        critical_patterns = [p for p in self.patterns.values() if p.severity == 'critical']
        if critical_patterns:
            recommendations.append(f"{len(critical_patterns)} critical patterns require immediate attention")
            
        error_patterns = [p for p in self.patterns.values() if p.pattern_type == 'error_pattern']
        if len(error_patterns) > 5:
            recommendations.append("High number of error patterns detected - review error handling strategies")
            
        return recommendations
        
    # Placeholder methods for brevity - complete implementation would include:
    # _load_knowledge_base, _save_knowledge_base, _group_by_similarity, 
    # _extract_common_error_info, _analyze_slow_tasks, _extract_common_errors,
    # _analyze_agent_failures, _extract_failure_sequences, _find_matching_events,
    # _update_knowledge_effectiveness, _generate_error_solution, 
    # _generate_performance_solution, _generate_tags, _pattern_matches_context
    
    def _load_knowledge_base(self):
        """Load knowledge base - placeholder"""
        pass
        
    def _save_knowledge_base(self):
        """Save knowledge base - placeholder"""
        pass
