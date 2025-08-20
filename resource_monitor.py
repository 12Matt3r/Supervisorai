"""Resource Usage Monitoring - Tracks token usage, loops, and performance"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

@dataclass
class ResourceSnapshot:
    """Resource usage snapshot"""
    timestamp: str
    cpu_percent: float
    memory_mb: float
    token_count: int
    api_calls: int
    execution_time: float
    loop_detected: bool
    performance_score: float

class ResourceUsageMonitor:
    """Monitors resource usage including tokens, performance, and loop detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.session_start_time = None
        self.resource_history = deque(maxlen=self.config['history_size'])
        self.token_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'api_calls': 0
        }
        self.performance_metrics = {
            'execution_times': deque(maxlen=100),
            'cpu_samples': deque(maxlen=100),
            'memory_samples': deque(maxlen=100)
        }
        self.loop_detection = {
            'execution_patterns': deque(maxlen=50),
            'output_hashes': deque(maxlen=30),
            'state_history': deque(maxlen=20)
        }
        self.monitoring_thread = None
        self.monitoring_active = False
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'token_limit': 100000,  # Token usage limit
            'token_warning_threshold': 0.8,  # 80% of limit
            'loop_detection_enabled': True,
            'loop_similarity_threshold': 0.8,
            'performance_monitoring': True,
            'history_size': 1000,
            'cpu_threshold': 90.0,  # CPU usage threshold
            'memory_threshold': 4096,  # Memory threshold in MB
            'monitoring_interval': 5.0,  # seconds
            'max_execution_time': 300  # 5 minutes
        }
    
    def start_session(self):
        """Start monitoring session"""
        self.session_start_time = time.time()
        
        # Reset counters for new session
        self.token_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'api_calls': 0
        }
        
        # Start background monitoring
        if self.config['performance_monitoring']:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._performance_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
        
        return {
            'status': 'session_started',
            'start_time': datetime.now().isoformat(),
            'monitoring_enabled': self.monitoring_active
        }
    
    def end_session(self):
        """End monitoring session"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        return {
            'status': 'session_ended',
            'end_time': datetime.now().isoformat(),
            'session_duration': session_duration,
            'total_tokens_used': self.token_usage['total_tokens'],
            'api_calls_made': self.token_usage['api_calls']
        }
    
    def evaluate_usage(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current resource usage"""
        try:
            # Update token usage
            self._update_token_usage(resource_data.get('token_data', {}))
            
            # Get current system metrics
            system_metrics = self._get_system_metrics()
            
            # Detect execution loops
            loop_result = self._detect_execution_loops(
                resource_data.get('execution_data', {})
            )
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                system_metrics, self.token_usage
            )
            
            # Check resource limits
            limit_status = self._check_resource_limits(
                system_metrics, self.token_usage
            )
            
            # Create resource snapshot
            snapshot = ResourceSnapshot(
                timestamp=datetime.now().isoformat(),
                cpu_percent=system_metrics['cpu_percent'],
                memory_mb=system_metrics['memory_mb'],
                token_count=self.token_usage['total_tokens'],
                api_calls=self.token_usage['api_calls'],
                execution_time=resource_data.get('execution_time', 0),
                loop_detected=loop_result['loop_detected'],
                performance_score=performance_score
            )
            
            # Store in history
            self.resource_history.append(snapshot)
            
            # Calculate usage trends
            usage_trends = self._calculate_usage_trends()
            
            return {
                'usage_ratio': self._calculate_overall_usage_ratio(system_metrics, self.token_usage),
                'token_usage': self.token_usage.copy(),
                'token_limit_status': {
                    'limit': self.config['token_limit'],
                    'used': self.token_usage['total_tokens'],
                    'percentage': (self.token_usage['total_tokens'] / self.config['token_limit']) * 100,
                    'warning_triggered': limit_status['token_warning']
                },
                'system_metrics': system_metrics,
                'performance_score': performance_score,
                'loop_detection': loop_result,
                'resource_limits': limit_status,
                'usage_trends': usage_trends,
                'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
                'recommendations': self._generate_resource_recommendations(
                    system_metrics, self.token_usage, loop_result, limit_status
                ),
                'status': 'evaluated',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'usage_ratio': 1.0,  # Assume high usage on error
                'error': str(e),
                'status': 'evaluation_failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_token_usage(self, token_data: Dict[str, Any]):
        """Update token usage statistics"""
        if not token_data:
            return
        
        # Add tokens from current request
        input_tokens = token_data.get('input_tokens', 0)
        output_tokens = token_data.get('output_tokens', 0)
        total_tokens = token_data.get('total_tokens', input_tokens + output_tokens)
        
        self.token_usage['input_tokens'] += input_tokens
        self.token_usage['output_tokens'] += output_tokens
        self.token_usage['total_tokens'] += total_tokens
        
        if token_data.get('api_call', False):
            self.token_usage['api_calls'] += 1
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Get process-specific metrics if available
            current_process = psutil.Process()
            process_memory_mb = current_process.memory_info().rss / (1024 * 1024)
            process_cpu_percent = current_process.cpu_percent()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'process_memory_mb': process_memory_mb,
                'process_cpu_percent': process_cpu_percent,
                'available_memory_mb': memory.available / (1024 * 1024)
            }
            
            # Store samples for trend analysis
            self.performance_metrics['cpu_samples'].append(cpu_percent)
            self.performance_metrics['memory_samples'].append(memory_mb)
            
            return metrics
            
        except Exception as e:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_percent': 0.0,
                'process_memory_mb': 0.0,
                'process_cpu_percent': 0.0,
                'available_memory_mb': 0.0,
                'error': str(e)
            }
    
    def _detect_execution_loops(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect execution loops and cycles"""
        if not self.config['loop_detection_enabled']:
            return {
                'loop_detected': False,
                'loop_type': None,
                'confidence': 0.0,
                'pattern_info': {},
                'detection_disabled': True
            }
        
        loop_detected = False
        loop_type = None
        confidence = 0.0
        pattern_info = {}
        
        # Check for pattern loops in execution steps
        current_pattern = execution_data.get('execution_pattern', '')
        if current_pattern:
            self.loop_detection['execution_patterns'].append({
                'pattern': current_pattern,
                'timestamp': time.time()
            })
            
            # Analyze recent patterns for similarity
            recent_patterns = list(self.loop_detection['execution_patterns'])[-10:]
            if len(recent_patterns) >= 3:
                similarity_score = self._calculate_pattern_similarity(recent_patterns)
                if similarity_score > self.config['loop_similarity_threshold']:
                    loop_detected = True
                    loop_type = 'execution_pattern'
                    confidence = similarity_score
                    pattern_info['similar_patterns'] = len(recent_patterns)
        
        # Check for output loops (similar outputs being generated)
        current_output_hash = execution_data.get('output_hash', '')
        if current_output_hash:
            self.loop_detection['output_hashes'].append({
                'hash': current_output_hash,
                'timestamp': time.time()
            })
            
            # Check for repeated output hashes
            recent_hashes = [item['hash'] for item in list(self.loop_detection['output_hashes'])[-10:]]
            if len(recent_hashes) >= 3:
                hash_repetition = self._calculate_hash_repetition(recent_hashes)
                if hash_repetition > 0.6:  # 60% repetition
                    if not loop_detected or confidence < hash_repetition:
                        loop_detected = True
                        loop_type = 'output_repetition'
                        confidence = hash_repetition
                        pattern_info['repeated_hashes'] = hash_repetition
        
        # Check for state loops (same state being reached repeatedly)
        current_state = execution_data.get('state_summary', '')
        if current_state:
            self.loop_detection['state_history'].append({
                'state': current_state,
                'timestamp': time.time()
            })
            
            # Analyze state repetition
            recent_states = [item['state'] for item in list(self.loop_detection['state_history'])[-8:]]
            if len(recent_states) >= 4:
                state_repetition = self._calculate_state_repetition(recent_states)
                if state_repetition > 0.5:  # 50% state repetition
                    if not loop_detected or confidence < state_repetition:
                        loop_detected = True
                        loop_type = 'state_loop'
                        confidence = state_repetition
                        pattern_info['state_repetition'] = state_repetition
        
        return {
            'loop_detected': loop_detected,
            'loop_type': loop_type,
            'confidence': confidence,
            'pattern_info': pattern_info,
            'detection_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_performance_score(self, system_metrics: Dict[str, Any], 
                                   token_usage: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 1.0
        
        # CPU usage penalty
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > self.config['cpu_threshold']:
            score -= (cpu_percent - self.config['cpu_threshold']) / 100.0
        
        # Memory usage penalty
        memory_mb = system_metrics.get('memory_mb', 0)
        if memory_mb > self.config['memory_threshold']:
            score -= (memory_mb - self.config['memory_threshold']) / self.config['memory_threshold']
        
        # Token usage penalty
        token_ratio = token_usage['total_tokens'] / self.config['token_limit']
        if token_ratio > self.config['token_warning_threshold']:
            score -= (token_ratio - self.config['token_warning_threshold']) * 0.5
        
        # Session duration penalty (for very long sessions)
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            if session_duration > self.config['max_execution_time']:
                score -= (session_duration - self.config['max_execution_time']) / self.config['max_execution_time']
        
        return max(score, 0.0)
    
    def _check_resource_limits(self, system_metrics: Dict[str, Any], 
                             token_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Check if resource limits are being approached or exceeded"""
        limits = {
            'token_warning': False,
            'token_exceeded': False,
            'cpu_warning': False,
            'memory_warning': False,
            'execution_time_warning': False,
            'warnings': []
        }
        
        # Token usage checks
        token_ratio = token_usage['total_tokens'] / self.config['token_limit']
        if token_ratio > self.config['token_warning_threshold']:
            limits['token_warning'] = True
            limits['warnings'].append(f"Token usage at {token_ratio*100:.1f}% of limit")
        
        if token_usage['total_tokens'] > self.config['token_limit']:
            limits['token_exceeded'] = True
            limits['warnings'].append("Token limit exceeded")
        
        # CPU usage checks
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > self.config['cpu_threshold']:
            limits['cpu_warning'] = True
            limits['warnings'].append(f"High CPU usage: {cpu_percent:.1f}%")
        
        # Memory usage checks
        memory_mb = system_metrics.get('memory_mb', 0)
        if memory_mb > self.config['memory_threshold']:
            limits['memory_warning'] = True
            limits['warnings'].append(f"High memory usage: {memory_mb:.1f} MB")
        
        # Execution time checks
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            if session_duration > self.config['max_execution_time']:
                limits['execution_time_warning'] = True
                limits['warnings'].append(f"Long execution time: {session_duration/60:.1f} minutes")
        
        return limits
    
    def _calculate_overall_usage_ratio(self, system_metrics: Dict[str, Any], 
                                     token_usage: Dict[str, Any]) -> float:
        """Calculate overall resource usage ratio"""
        # Combine different usage metrics
        token_ratio = min(token_usage['total_tokens'] / self.config['token_limit'], 1.0)
        cpu_ratio = min(system_metrics.get('cpu_percent', 0) / 100.0, 1.0)
        memory_ratio = min(system_metrics.get('memory_mb', 0) / self.config['memory_threshold'], 1.0)
        
        # Weight the ratios
        overall_ratio = (
            token_ratio * 0.4 +  # Token usage is most important
            cpu_ratio * 0.3 +    # CPU usage
            memory_ratio * 0.3   # Memory usage
        )
        
        return min(overall_ratio, 1.0)
    
    def _calculate_usage_trends(self) -> Dict[str, Any]:
        """Calculate usage trends over time"""
        if len(self.resource_history) < 5:
            return {
                'trend_available': False,
                'insufficient_data': True
            }
        
        recent_snapshots = list(self.resource_history)[-10:]
        
        # Calculate trends
        token_trend = self._calculate_trend([s.token_count for s in recent_snapshots])
        cpu_trend = self._calculate_trend([s.cpu_percent for s in recent_snapshots])
        memory_trend = self._calculate_trend([s.memory_mb for s in recent_snapshots])
        performance_trend = self._calculate_trend([s.performance_score for s in recent_snapshots])
        
        return {
            'trend_available': True,
            'token_trend': token_trend,
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'performance_trend': performance_trend,
            'samples_analyzed': len(recent_snapshots)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and slope for a list of values"""
        if len(values) < 3:
            return {'direction': 'unknown', 'slope': 0.0, 'confidence': 0.0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Very small slope
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate confidence (simplified)
        confidence = min(abs(slope) * 10, 1.0)
        
        return {
            'direction': direction,
            'slope': slope,
            'confidence': confidence
        }
    
    def _performance_monitoring_loop(self):
        """Background performance monitoring loop"""
        while self.monitoring_active:
            try:
                # Record execution time
                start_time = time.time()
                
                # Simulate some monitoring work
                system_metrics = self._get_system_metrics()
                
                execution_time = time.time() - start_time
                self.performance_metrics['execution_times'].append(execution_time)
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception:
                time.sleep(1.0)  # Brief pause on error
    
    def _calculate_pattern_similarity(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate similarity between execution patterns"""
        if len(patterns) < 3:
            return 0.0
        
        pattern_strings = [p['pattern'] for p in patterns]
        
        # Simple string similarity calculation
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(pattern_strings)):
            for j in range(i + 1, len(pattern_strings)):
                similarity = self._calculate_string_similarity(pattern_strings[i], pattern_strings[j])
                total_similarity += similarity
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_hash_repetition(self, hashes: List[str]) -> float:
        """Calculate repetition ratio in hash list"""
        if len(hashes) < 2:
            return 0.0
        
        hash_counts = defaultdict(int)
        for hash_val in hashes:
            hash_counts[hash_val] += 1
        
        # Find the most repeated hash
        max_repetition = max(hash_counts.values())
        
        return max_repetition / len(hashes)
    
    def _calculate_state_repetition(self, states: List[str]) -> float:
        """Calculate state repetition ratio"""
        if len(states) < 2:
            return 0.0
        
        state_counts = defaultdict(int)
        for state in states:
            state_counts[state] += 1
        
        # Calculate repetition ratio
        max_repetition = max(state_counts.values())
        
        return max_repetition / len(states)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Simple character-based similarity
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_resource_recommendations(self, system_metrics: Dict[str, Any],
                                         token_usage: Dict[str, Any],
                                         loop_result: Dict[str, Any],
                                         limit_status: Dict[str, Any]) -> List[str]:
        """Generate resource optimization recommendations"""
        recommendations = []
        
        # Token usage recommendations
        token_ratio = token_usage['total_tokens'] / self.config['token_limit']
        if token_ratio > 0.8:
            recommendations.append("Optimize token usage - approaching limit")
            recommendations.append("Consider shorter prompts or response chunking")
        
        # Performance recommendations
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider optimization")
        
        memory_mb = system_metrics.get('memory_mb', 0)
        if memory_mb > self.config['memory_threshold'] * 0.8:
            recommendations.append("High memory usage - consider memory optimization")
        
        # Loop detection recommendations
        if loop_result['loop_detected']:
            recommendations.append(f"Execution loop detected ({loop_result['loop_type']})")
            recommendations.append("Review execution logic to prevent cycles")
        
        # Limit status recommendations
        if limit_status['warnings']:
            recommendations.extend([f"Address: {warning}" for warning in limit_status['warnings'][:2]])
        
        return recommendations or ["Resource usage is within normal limits"]
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage"""
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        
        return {
            'session_duration': session_duration,
            'token_usage': self.token_usage.copy(),
            'snapshots_recorded': len(self.resource_history),
            'monitoring_active': self.monitoring_active,
            'average_performance': {
                'cpu': sum(self.performance_metrics['cpu_samples']) / len(self.performance_metrics['cpu_samples']) if self.performance_metrics['cpu_samples'] else 0,
                'memory': sum(self.performance_metrics['memory_samples']) / len(self.performance_metrics['memory_samples']) if self.performance_metrics['memory_samples'] else 0
            },
            'summary_timestamp': datetime.now().isoformat()
        }
