#!/usr/bin/env python3
"""
Pattern Tracking and Knowledge Base System for Supervisor Agent
Identifies recurring patterns, builds knowledge base, and provides recommendations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class Pattern:
    """Represents an identified pattern"""
    id: str
    pattern_type: str  # 'error', 'performance', 'usage', 'success'
    title: str
    description: str
    frequency: int
    first_seen: str
    last_seen: str
    confidence: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    impact: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class KnowledgeEntry:
    """Knowledge base entry"""
    id: str
    category: str
    title: str
    content: str
    related_patterns: List[str] = field(default_factory=list)
    source_events: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    usage_count: int = 0

@dataclass
class PatternAnalysisResult:
    """Results of pattern analysis"""
    analysis_period: Tuple[str, str]
    total_events: int
    patterns_found: List[Pattern]
    pattern_categories: Dict[str, int]
    trending_patterns: List[Pattern]
    new_patterns: List[Pattern]
    knowledge_updates: List[str]
    recommendations: List[str]

class EventProcessor:
    """Processes events for pattern detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from an event for pattern analysis"""
        features = {
            'event_type': event.get('event_type', ''),
            'source': event.get('source', ''),
            'level': event.get('level', ''),
            'message_tokens': self._tokenize_message(event.get('message', '')),
            'timestamp_hour': self._extract_hour(event.get('timestamp', '')),
            'timestamp_dow': self._extract_day_of_week(event.get('timestamp', '')),
            'metadata_keys': list(event.get('metadata', {}).keys()),
            'has_error': 'error' in event.get('message', '').lower(),
            'has_timeout': 'timeout' in event.get('message', '').lower(),
            'has_retry': 'retry' in event.get('message', '').lower(),
        }
        
        # Extract error types
        if features['has_error']:
            features['error_type'] = self._extract_error_type(event.get('message', ''))
        
        # Extract numerical values
        features['numerical_values'] = self._extract_numbers(event.get('message', ''))
        
        # Extract agent/task identifiers
        features['agent_id'] = event.get('metadata', {}).get('agent_id', '')
        features['task_id'] = event.get('metadata', {}).get('task_id', '')
        
        return features
    
    def _tokenize_message(self, message: str) -> List[str]:
        """Tokenize message into meaningful terms"""
        # Simple tokenization - could be improved with NLP libraries
        tokens = re.findall(r'\b\w+\b', message.lower())
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def _extract_hour(self, timestamp: str) -> int:
        """Extract hour from timestamp"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.hour
        except:
            return 0
    
    def _extract_day_of_week(self, timestamp: str) -> int:
        """Extract day of week from timestamp"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.weekday()
        except:
            return 0
    
    def _extract_error_type(self, message: str) -> str:
        """Extract error type from message"""
        error_patterns = {
            'timeout': r'timeout|timed?\s*out',
            'connection': r'connection|connect|network',
            'permission': r'permission|access|denied|forbidden',
            'not_found': r'not\s*found|404|missing',
            'validation': r'validation|invalid|malformed',
            'resource': r'resource|memory|disk|cpu',
            'authentication': r'auth|login|credential|token'
        }
        
        message_lower = message.lower()
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, message_lower):
                return error_type
        
        return 'unknown'
    
    def _extract_numbers(self, message: str) -> List[float]:
        """Extract numerical values from message"""
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        return [float(n) for n in numbers]

class PatternDetector:
    """Detects patterns in processed events"""
    
    def __init__(self, min_frequency: int = 3, similarity_threshold: float = 0.7):
        self.min_frequency = min_frequency
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self, events: List[Dict[str, Any]], 
                       existing_patterns: List[Pattern] = None) -> List[Pattern]:
        """Detect patterns in events"""
        if not events:
            return []
        
        processor = EventProcessor()
        processed_events = []
        
        # Process events
        for event in events:
            features = processor.extract_features(event)
            processed_events.append({
                'original': event,
                'features': features,
                'timestamp': event.get('timestamp', '')
            })
        
        patterns = []
        existing_patterns = existing_patterns or []
        
        # Detect different types of patterns
        patterns.extend(self._detect_error_patterns(processed_events))
        patterns.extend(self._detect_temporal_patterns(processed_events))
        patterns.extend(self._detect_sequence_patterns(processed_events))
        patterns.extend(self._detect_performance_patterns(processed_events))
        patterns.extend(self._detect_usage_patterns(processed_events))
        
        # Merge with existing patterns
        patterns = self._merge_patterns(patterns, existing_patterns)
        
        # Filter by minimum frequency
        patterns = [p for p in patterns if p.frequency >= self.min_frequency]
        
        return patterns
    
    def _detect_error_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect error patterns"""
        error_events = [
            e for e in events 
            if e['features']['has_error'] or e['features']['level'] in ['error', 'critical']
        ]
        
        if not error_events:
            return []
        
        patterns = []
        
        # Group by error type and source
        error_groups = defaultdict(list)
        for event in error_events:
            key = (
                event['features']['error_type'],
                event['features']['source'],
                event['original'].get('message', '')[:100]  # First 100 chars
            )
            error_groups[key].append(event)
        
        # Create patterns for frequent error combinations
        for (error_type, source, message_prefix), group_events in error_groups.items():
            if len(group_events) >= self.min_frequency:
                timestamps = [e['timestamp'] for e in group_events]
                timestamps.sort()
                
                pattern = Pattern(
                    id=hashlib.md5(f"error_{error_type}_{source}_{message_prefix}".encode()).hexdigest()[:12],
                    pattern_type='error',
                    title=f"Recurring {error_type} error in {source}",
                    description=f"Error pattern: {message_prefix}...",
                    frequency=len(group_events),
                    first_seen=timestamps[0],
                    last_seen=timestamps[-1],
                    confidence=min(1.0, len(group_events) / 10),
                    examples=[e['original'] for e in group_events[:5]],
                    conditions={
                        'error_type': error_type,
                        'source': source,
                        'level': ['error', 'critical']
                    },
                    impact={
                        'frequency': len(group_events),
                        'time_span': self._calculate_time_span(timestamps[0], timestamps[-1])
                    },
                    tags=['error', error_type, source]
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect time-based patterns"""
        patterns = []
        
        # Group events by hour of day
        hourly_counts = defaultdict(list)
        for event in events:
            hour = event['features']['timestamp_hour']
            hourly_counts[hour].append(event)
        
        # Find hours with significantly high activity
        avg_hourly = sum(len(events) for events in hourly_counts.values()) / 24
        
        for hour, hour_events in hourly_counts.items():
            if len(hour_events) > avg_hourly * 2 and len(hour_events) >= self.min_frequency:
                timestamps = [e['timestamp'] for e in hour_events]
                timestamps.sort()
                
                pattern = Pattern(
                    id=hashlib.md5(f"temporal_hour_{hour}".encode()).hexdigest()[:12],
                    pattern_type='usage',
                    title=f"High activity during hour {hour}:00",
                    description=f"Consistently high event volume at {hour}:00 ({len(hour_events)} events)",
                    frequency=len(hour_events),
                    first_seen=timestamps[0],
                    last_seen=timestamps[-1],
                    confidence=min(1.0, len(hour_events) / (avg_hourly * 3)),
                    examples=hour_events[:3],
                    conditions={'hour': hour},
                    impact={'hourly_volume': len(hour_events), 'avg_hourly': avg_hourly},
                    tags=['temporal', 'usage', f'hour_{hour}']
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_sequence_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect event sequence patterns"""
        patterns = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e['timestamp'])
        
        # Look for common sequences (sliding window)
        sequence_counts = defaultdict(list)
        window_size = 3
        
        for i in range(len(sorted_events) - window_size + 1):
            window = sorted_events[i:i + window_size]
            sequence_key = tuple(
                (e['features']['event_type'], e['features']['source']) 
                for e in window
            )
            sequence_counts[sequence_key].append({
                'start_time': window[0]['timestamp'],
                'events': window
            })
        
        # Create patterns for frequent sequences
        for sequence_key, occurrences in sequence_counts.items():
            if len(occurrences) >= self.min_frequency:
                timestamps = [occ['start_time'] for occ in occurrences]
                timestamps.sort()
                
                sequence_desc = " -> ".join(f"{et}({src})" for et, src in sequence_key)
                
                pattern = Pattern(
                    id=hashlib.md5(f"sequence_{sequence_desc}".encode()).hexdigest()[:12],
                    pattern_type='sequence',
                    title=f"Recurring event sequence",
                    description=f"Sequence pattern: {sequence_desc}",
                    frequency=len(occurrences),
                    first_seen=timestamps[0],
                    last_seen=timestamps[-1],
                    confidence=min(1.0, len(occurrences) / 20),
                    examples=[occ['events'][0]['original'] for occ in occurrences[:3]],
                    conditions={'sequence': list(sequence_key)},
                    impact={'occurrences': len(occurrences)},
                    tags=['sequence', 'workflow']
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_performance_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect performance-related patterns"""
        patterns = []
        
        # Look for events with performance implications
        perf_keywords = ['slow', 'timeout', 'duration', 'time', 'performance', 'latency']
        perf_events = [
            e for e in events
            if any(keyword in ' '.join(e['features']['message_tokens']) for keyword in perf_keywords)
        ]
        
        if not perf_events:
            return patterns
        
        # Group by source for performance analysis
        source_groups = defaultdict(list)
        for event in perf_events:
            source_groups[event['features']['source']].append(event)
        
        for source, source_events in source_groups.items():
            if len(source_events) >= self.min_frequency:
                timestamps = [e['timestamp'] for e in source_events]
                timestamps.sort()
                
                pattern = Pattern(
                    id=hashlib.md5(f"performance_{source}".encode()).hexdigest()[:12],
                    pattern_type='performance',
                    title=f"Performance issues in {source}",
                    description=f"Recurring performance-related events from {source}",
                    frequency=len(source_events),
                    first_seen=timestamps[0],
                    last_seen=timestamps[-1],
                    confidence=min(1.0, len(source_events) / 15),
                    examples=[e['original'] for e in source_events[:3]],
                    conditions={'source': source, 'performance_related': True},
                    impact={'affected_source': source, 'frequency': len(source_events)},
                    tags=['performance', source]
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _detect_usage_patterns(self, events: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect usage patterns"""
        patterns = []
        
        # Agent usage patterns
        agent_counts = Counter(e['features']['agent_id'] for e in events if e['features']['agent_id'])
        
        if agent_counts:
            avg_usage = sum(agent_counts.values()) / len(agent_counts)
            
            for agent_id, count in agent_counts.items():
                if count > avg_usage * 1.5 and count >= self.min_frequency:
                    agent_events = [e for e in events if e['features']['agent_id'] == agent_id]
                    timestamps = [e['timestamp'] for e in agent_events]
                    timestamps.sort()
                    
                    pattern = Pattern(
                        id=hashlib.md5(f"usage_agent_{agent_id}".encode()).hexdigest()[:12],
                        pattern_type='usage',
                        title=f"High usage by agent {agent_id}",
                        description=f"Agent {agent_id} has unusually high activity ({count} events)",
                        frequency=count,
                        first_seen=timestamps[0],
                        last_seen=timestamps[-1],
                        confidence=min(1.0, count / (avg_usage * 3)),
                        examples=agent_events[:3],
                        conditions={'agent_id': agent_id},
                        impact={'usage_count': count, 'avg_usage': avg_usage},
                        tags=['usage', 'agent', agent_id]
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _merge_patterns(self, new_patterns: List[Pattern], 
                       existing_patterns: List[Pattern]) -> List[Pattern]:
        """Merge new patterns with existing ones"""
        merged = existing_patterns.copy()
        existing_ids = {p.id for p in existing_patterns}
        
        for new_pattern in new_patterns:
            if new_pattern.id in existing_ids:
                # Update existing pattern
                for i, existing in enumerate(merged):
                    if existing.id == new_pattern.id:
                        merged[i].frequency += new_pattern.frequency
                        merged[i].last_seen = new_pattern.last_seen
                        merged[i].examples.extend(new_pattern.examples)
                        # Keep only recent examples
                        merged[i].examples = merged[i].examples[-10:]
                        break
            else:
                # Add new pattern
                merged.append(new_pattern)
        
        return merged
    
    def _calculate_time_span(self, start_time: str, end_time: str) -> str:
        """Calculate human-readable time span"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            delta = end - start
            
            if delta.days > 0:
                return f"{delta.days} days"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600} hours"
            elif delta.seconds > 60:
                return f"{delta.seconds // 60} minutes"
            else:
                return f"{delta.seconds} seconds"
        except:
            return "unknown"

class KnowledgeBase:
    """Manages the knowledge base built from patterns"""
    
    def __init__(self, storage_file: str = "knowledge_base.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = self._load_entries()
        self.logger = logging.getLogger(__name__)
    
    def _load_entries(self) -> Dict[str, KnowledgeEntry]:
        """Load knowledge base entries"""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                return {
                    entry_id: KnowledgeEntry(**entry_data)
                    for entry_id, entry_data in data.items()
                }
            except Exception as e:
                self.logger.error(f"Error loading knowledge base: {e}")
        return {}
    
    def _save_entries(self):
        """Save knowledge base entries"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(
                    {entry_id: asdict(entry) for entry_id, entry in self.entries.items()},
                    f, indent=2
                )
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")
    
    def add_from_pattern(self, pattern: Pattern) -> str:
        """Add knowledge entry from a pattern"""
        entry_id = f"kb_{pattern.id}"
        
        # Generate content based on pattern
        content = self._generate_content_from_pattern(pattern)
        
        # Generate recommendations
        recommendations = self._generate_pattern_recommendations(pattern)
        
        entry = KnowledgeEntry(
            id=entry_id,
            category=pattern.pattern_type,
            title=pattern.title,
            content=content,
            related_patterns=[pattern.id],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=pattern.tags + ['auto_generated'],
            confidence=pattern.confidence
        )
        
        self.entries[entry_id] = entry
        self._save_entries()
        
        self.logger.info(f"Added knowledge entry {entry_id} from pattern {pattern.id}")
        return entry_id
    
    def _generate_content_from_pattern(self, pattern: Pattern) -> str:
        """Generate knowledge content from pattern"""
        content_parts = [
            f"## Pattern: {pattern.title}",
            f"**Type:** {pattern.pattern_type}",
            f"**Frequency:** {pattern.frequency} occurrences",
            f"**Confidence:** {pattern.confidence:.2f}",
            f"**First Seen:** {pattern.first_seen}",
            f"**Last Seen:** {pattern.last_seen}",
            "",
            "### Description",
            pattern.description,
            ""
        ]
        
        if pattern.conditions:
            content_parts.extend([
                "### Conditions",
                json.dumps(pattern.conditions, indent=2),
                ""
            ])
        
        if pattern.impact:
            content_parts.extend([
                "### Impact",
                json.dumps(pattern.impact, indent=2),
                ""
            ])
        
        if pattern.recommendations:
            content_parts.extend([
                "### Recommendations",
                "\n".join(f"- {rec}" for rec in pattern.recommendations)
            ])
        
        return "\n".join(content_parts)
    
    def _generate_pattern_recommendations(self, pattern: Pattern) -> List[str]:
        """Generate recommendations based on pattern type"""
        recommendations = []
        
        if pattern.pattern_type == 'error':
            recommendations.extend([
                f"Monitor error frequency for pattern '{pattern.title}'",
                "Consider implementing specific error handling for this pattern",
                "Investigate root cause if frequency increases"
            ])
        
        elif pattern.pattern_type == 'performance':
            recommendations.extend([
                f"Profile performance for pattern '{pattern.title}'",
                "Consider optimization strategies",
                "Set up performance monitoring alerts"
            ])
        
        elif pattern.pattern_type == 'usage':
            recommendations.extend([
                f"Analyze usage pattern '{pattern.title}' for optimization opportunities",
                "Consider load balancing if pattern indicates bottlenecks",
                "Monitor for capacity planning"
            ])
        
        elif pattern.pattern_type == 'sequence':
            recommendations.extend([
                f"Document workflow sequence '{pattern.title}'",
                "Consider automation opportunities",
                "Monitor for sequence deviations"
            ])
        
        return recommendations
    
    def search_entries(self, query: str, category: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Search knowledge base entries"""
        results = []
        query_lower = query.lower()
        
        for entry in self.entries.values():
            # Category filter
            if category and entry.category != category:
                continue
            
            # Tags filter
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            
            # Text search
            if (query_lower in entry.title.lower() or 
                query_lower in entry.content.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)):
                results.append(entry)
        
        # Sort by confidence and usage
        results.sort(key=lambda e: (e.confidence, e.usage_count), reverse=True)
        return results
    
    def get_recommendations_for_event(self, event: Dict[str, Any]) -> List[str]:
        """Get recommendations based on an event"""
        processor = EventProcessor()
        features = processor.extract_features(event)
        
        recommendations = []
        
        # Search for relevant knowledge entries
        search_terms = [
            features['source'],
            features['event_type'],
            features['error_type'] if 'error_type' in features else ''
        ]
        
        for term in search_terms:
            if term:
                entries = self.search_entries(term)
                for entry in entries[:3]:  # Top 3 matches
                    entry.usage_count += 1
                    # Extract recommendations from content
                    content_lines = entry.content.split('\n')
                    in_recommendations = False
                    for line in content_lines:
                        if '### Recommendations' in line:
                            in_recommendations = True
                            continue
                        if in_recommendations and line.startswith('- '):
                            recommendations.append(line[2:])  # Remove '- '
        
        self._save_entries()
        return list(set(recommendations))  # Remove duplicates

class ComprehensivePatternSystem:
    """Main pattern tracking and knowledge base system"""
    
    def __init__(self, patterns_file: str = "patterns.json",
                 knowledge_file: str = "knowledge_base.json"):
        self.patterns_file = Path(patterns_file)
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.detector = PatternDetector()
        self.knowledge_base = KnowledgeBase(knowledge_file)
        self.patterns = self._load_patterns()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_patterns(self) -> List[Pattern]:
        """Load existing patterns"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                return [Pattern(**pattern_data) for pattern_data in data]
            except Exception as e:
                self.logger.error(f"Error loading patterns: {e}")
        return []
    
    def _save_patterns(self):
        """Save patterns to file"""
        try:
            with open(self.patterns_file, 'w') as f:
                json.dump([asdict(pattern) for pattern in self.patterns], f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")
    
    def analyze_events(self, events: List[Dict[str, Any]]) -> PatternAnalysisResult:
        """Analyze events for patterns"""
        if not events:
            return PatternAnalysisResult(
                analysis_period=("", ""),
                total_events=0,
                patterns_found=[],
                pattern_categories={},
                trending_patterns=[],
                new_patterns=[],
                knowledge_updates=[],
                recommendations=[]
            )
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', ''))
        period_start = sorted_events[0].get('timestamp', '')
        period_end = sorted_events[-1].get('timestamp', '')
        
        # Detect patterns
        new_patterns = self.detector.detect_patterns(events, self.patterns)
        
        # Identify truly new patterns
        existing_ids = {p.id for p in self.patterns}
        truly_new = [p for p in new_patterns if p.id not in existing_ids]
        
        # Update patterns
        self.patterns = new_patterns
        self._save_patterns()
        
        # Update knowledge base
        knowledge_updates = []
        for pattern in truly_new:
            if pattern.frequency >= 5:  # Only add significant patterns
                kb_id = self.knowledge_base.add_from_pattern(pattern)
                knowledge_updates.append(f"Added knowledge entry {kb_id}")
        
        # Identify trending patterns (increased frequency)
        trending_patterns = []
        for pattern in new_patterns:
            if pattern.frequency > 10:  # Threshold for trending
                trending_patterns.append(pattern)
        
        # Generate pattern categories
        pattern_categories = Counter(p.pattern_type for p in new_patterns)
        
        # Generate recommendations
        recommendations = self._generate_system_recommendations(new_patterns)
        
        return PatternAnalysisResult(
            analysis_period=(period_start, period_end),
            total_events=len(events),
            patterns_found=new_patterns,
            pattern_categories=dict(pattern_categories),
            trending_patterns=trending_patterns,
            new_patterns=truly_new,
            knowledge_updates=knowledge_updates,
            recommendations=recommendations
        )
    
    def _generate_system_recommendations(self, patterns: List[Pattern]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []
        
        # Error pattern recommendations
        error_patterns = [p for p in patterns if p.pattern_type == 'error']
        if len(error_patterns) > 5:
            recommendations.append(
                f"High number of error patterns detected ({len(error_patterns)}). "
                "Consider systematic error handling improvements."
            )
        
        # Performance pattern recommendations
        perf_patterns = [p for p in patterns if p.pattern_type == 'performance']
        if len(perf_patterns) > 2:
            recommendations.append(
                f"Multiple performance patterns detected ({len(perf_patterns)}). "
                "Recommend comprehensive performance audit."
            )
        
        # High-frequency patterns
        high_freq_patterns = [p for p in patterns if p.frequency > 20]
        if high_freq_patterns:
            recommendations.append(
                f"High-frequency patterns detected: {[p.title for p in high_freq_patterns[:3]]}. "
                "These may indicate systemic issues requiring attention."
            )
        
        return recommendations
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about current patterns"""
        if not self.patterns:
            return {'total_patterns': 0}
        
        insights = {
            'total_patterns': len(self.patterns),
            'by_type': Counter(p.pattern_type for p in self.patterns),
            'most_frequent': sorted(self.patterns, key=lambda p: p.frequency, reverse=True)[:5],
            'most_recent': sorted(self.patterns, key=lambda p: p.last_seen, reverse=True)[:5],
            'high_confidence': [p for p in self.patterns if p.confidence > 0.8],
            'avg_frequency': sum(p.frequency for p in self.patterns) / len(self.patterns)
        }
        
        return insights
    
    def export_patterns(self, output_file: str, format: str = "json") -> str:
        """Export patterns to file"""
        try:
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump([asdict(pattern) for pattern in self.patterns], f, indent=2)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame([asdict(pattern) for pattern in self.patterns])
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Patterns exported to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error exporting patterns: {e}")
            return ""

# Demo and testing functions
def create_demo_pattern_system() -> ComprehensivePatternSystem:
    """Create demo pattern system"""
    return ComprehensivePatternSystem("demo_patterns.json", "demo_knowledge_base.json")

def generate_demo_events() -> List[Dict[str, Any]]:
    """Generate demo events for pattern detection"""
    import random
    
    events = []
    base_time = datetime.now() - timedelta(hours=24)
    
    # Generate various types of events
    error_types = ['timeout', 'connection', 'validation', 'not_found']
    sources = ['api_gateway', 'database', 'auth_service', 'file_processor']
    
    for i in range(200):
        event_time = base_time + timedelta(minutes=i*5)
        
        # Create different patterns
        if i % 10 == 0:  # Error pattern
            error_type = random.choice(error_types)
            source = random.choice(sources)
            event = {
                'timestamp': event_time.isoformat(),
                'event_type': 'error_occurred',
                'level': 'error',
                'source': source,
                'message': f'{error_type} error in {source}: Operation failed',
                'metadata': {
                    'error_type': error_type,
                    'agent_id': f'agent_{i % 3}',
                    'task_id': f'task_{i}'
                }
            }
        elif i % 15 == 0:  # Performance pattern
            source = random.choice(sources)
            duration = random.randint(5, 30)
            event = {
                'timestamp': event_time.isoformat(),
                'event_type': 'performance_metric',
                'level': 'warning',
                'source': source,
                'message': f'Slow operation in {source}: took {duration} seconds',
                'metadata': {
                    'duration': duration,
                    'agent_id': f'agent_{i % 3}',
                    'task_id': f'task_{i}'
                }
            }
        else:  # Normal events
            event = {
                'timestamp': event_time.isoformat(),
                'event_type': 'task_completed',
                'level': 'info',
                'source': random.choice(sources),
                'message': f'Task {i} completed successfully',
                'metadata': {
                    'agent_id': f'agent_{i % 3}',
                    'task_id': f'task_{i}'
                }
            }
        
        events.append(event)
    
    return events

if __name__ == '__main__':
    # Demo usage
    pattern_system = create_demo_pattern_system()
    demo_events = generate_demo_events()
    
    # Analyze patterns
    analysis = pattern_system.analyze_events(demo_events)
    
    print("\n=== Pattern Analysis Results ===")
    print(f"Total events analyzed: {analysis.total_events}")
    print(f"Patterns found: {len(analysis.patterns_found)}")
    print(f"New patterns: {len(analysis.new_patterns)}")
    print(f"Trending patterns: {len(analysis.trending_patterns)}")
    
    print("\n=== Pattern Categories ===")
    for category, count in analysis.pattern_categories.items():
        print(f"{category}: {count}")
    
    print("\n=== Top Patterns ===")
    top_patterns = sorted(analysis.patterns_found, key=lambda p: p.frequency, reverse=True)[:5]
    for pattern in top_patterns:
        print(f"- {pattern.title} (frequency: {pattern.frequency}, confidence: {pattern.confidence:.2f})")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n=== Knowledge Updates ===")
    for update in analysis.knowledge_updates:
        print(f"- {update}")
    
    # Export patterns
    pattern_system.export_patterns("demo_patterns_export.json")
    print("\nPatterns exported to demo_patterns_export.json")
    
    # Get insights
    insights = pattern_system.get_pattern_insights()
    print("\n=== Pattern Insights ===")
    print(json.dumps(insights, indent=2, default=str))
