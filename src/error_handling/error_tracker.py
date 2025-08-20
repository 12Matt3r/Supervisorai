"""Error Tracking - Detects API errors, hallucinations, and incomplete responses"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class ErrorEvent:
    """Represents an error event"""
    error_id: str
    error_type: str
    severity: str
    message: str
    timestamp: str
    location: str
    context: Dict[str, Any]
    suggested_resolution: str
    recurrence_count: int = 1

class ErrorTracker:
    """Tracks and analyzes various types of errors during execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.error_history = []
        self.error_patterns = self._initialize_error_patterns()
        self.hallucination_indicators = self._initialize_hallucination_indicators()
        self.api_error_patterns = self._initialize_api_error_patterns()
        self.error_stats = defaultdict(int)
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_history_size': 1000,
            'error_correlation_window': 300,  # seconds
            'hallucination_confidence_threshold': 0.7,
            'api_timeout_threshold': 30,  # seconds
            'critical_error_threshold': 3,  # errors per minute
            'track_error_patterns': True
        }
    
    def _initialize_error_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for error detection"""
        return [
            {
                'type': 'syntax_error',
                'patterns': [
                    r'SyntaxError|ParseError|IndentationError',
                    r'unexpected token|invalid syntax',
                    r'missing \w+|expected \w+'
                ],
                'severity': 'high',
                'resolution': 'Fix syntax issues in the code or data'
            },
            {
                'type': 'runtime_error',
                'patterns': [
                    r'RuntimeError|Exception|Error:',
                    r'failed to execute|execution failed',
                    r'\w+Error: [^\n]+'
                ],
                'severity': 'high',
                'resolution': 'Debug and fix runtime execution issues'
            },
            {
                'type': 'data_error',
                'patterns': [
                    r'KeyError|IndexError|TypeError',
                    r'not found|missing data|invalid format',
                    r'null|undefined|empty result'
                ],
                'severity': 'medium',
                'resolution': 'Validate data sources and formats'
            },
            {
                'type': 'logic_error',
                'patterns': [
                    r'assertion failed|logic error',
                    r'unexpected result|incorrect output',
                    r'validation failed'
                ],
                'severity': 'medium',
                'resolution': 'Review and correct logical flow'
            },
            {
                'type': 'incomplete_response',
                'patterns': [
                    r'\.\.\.$|\s+$',  # Trailing dots or spaces
                    r'\b(incomplete|partial|truncated)\b',
                    r'\b(continued|more|etc)\s*$'
                ],
                'severity': 'medium',
                'resolution': 'Complete the interrupted response'
            }
        ]
    
    def _initialize_hallucination_indicators(self) -> List[Dict[str, Any]]:
        """Initialize patterns that might indicate hallucinations"""
        return [
            {
                'type': 'factual_inconsistency',
                'patterns': [
                    r'\b(according to|based on|as stated)\b.*\b(however|but|contradicts)\b',
                    r'\b(fact|truth|reality)\b.*\b(false|incorrect|wrong)\b'
                ],
                'confidence_impact': 0.3
            },
            {
                'type': 'impossible_claims',
                'patterns': [
                    r'\b(100% accurate|completely certain|absolutely true)\b',
                    r'\b(never fails|always works|guaranteed)\b',
                    r'\b(impossible|cannot happen)\b.*\b(but|however)\b.*\b(possible|can happen)\b'
                ],
                'confidence_impact': 0.4
            },
            {
                'type': 'contradictory_statements',
                'patterns': [
                    r'\b(yes|correct)\b.*\b(no|incorrect)\b',
                    r'\b(true|valid)\b.*\b(false|invalid)\b',
                    r'\b(exists|present)\b.*\b(does not exist|absent)\b'
                ],
                'confidence_impact': 0.5
            },
            {
                'type': 'fabricated_details',
                'patterns': [
                    r'\b(specific|exact|precise)\b.*\b(unknown|unclear|uncertain)\b',
                    r'\d{4}-\d{2}-\d{2}.*\b(approximate|roughly|around)\b',  # Precise dates with uncertainty
                    r'\b(detailed|comprehensive)\b.*\b(limited|insufficient) (information|data)\b'
                ],
                'confidence_impact': 0.25
            }
        ]
    
    def _initialize_api_error_patterns(self) -> List[Dict[str, Any]]:
        """Initialize API error patterns"""
        return [
            {
                'type': 'timeout_error',
                'patterns': [
                    r'timeout|timed out|request timeout',
                    r'connection timeout|read timeout',
                    r'operation timed out'
                ],
                'severity': 'high'
            },
            {
                'type': 'authentication_error',
                'patterns': [
                    r'401|unauthorized|authentication failed',
                    r'invalid credentials|access denied',
                    r'token expired|invalid token'
                ],
                'severity': 'high'
            },
            {
                'type': 'rate_limit_error',
                'patterns': [
                    r'429|rate limit|too many requests',
                    r'quota exceeded|limit exceeded',
                    r'throttling|rate limiting'
                ],
                'severity': 'medium'
            },
            {
                'type': 'server_error',
                'patterns': [
                    r'5\d{2}|server error|internal error',
                    r'service unavailable|server down',
                    r'maintenance mode|temporarily unavailable'
                ],
                'severity': 'high'
            },
            {
                'type': 'client_error',
                'patterns': [
                    r'4\d{2}|bad request|invalid request',
                    r'malformed|invalid parameters',
                    r'missing required'
                ],
                'severity': 'medium'
            }
        ]
    
    def detect_errors(self, execution_logs: List[str], 
                     api_responses: List[str],
                     outputs: List[str]) -> List[Dict[str, Any]]:
        """Detect errors across different sources"""
        try:
            detected_errors = []
            
            # Detect execution errors
            execution_errors = self._detect_execution_errors(execution_logs)
            detected_errors.extend(execution_errors)
            
            # Detect API errors
            api_errors = self._detect_api_errors(api_responses)
            detected_errors.extend(api_errors)
            
            # Detect output errors and hallucinations
            output_errors = self._detect_output_errors(outputs)
            detected_errors.extend(output_errors)
            
            # Detect hallucinations
            hallucinations = self._detect_hallucinations(outputs)
            detected_errors.extend(hallucinations)
            
            # Check for missing or incomplete data
            completeness_errors = self._detect_incomplete_responses(outputs)
            detected_errors.extend(completeness_errors)
            
            # Analyze error patterns and correlations
            pattern_analysis = self._analyze_error_patterns(detected_errors)
            
            # Update error statistics
            self._update_error_stats(detected_errors)
            
            # Store in history
            for error in detected_errors:
                self._add_to_history(error)
            
            # Convert to serializable format and add metadata
            result_errors = []
            for error in detected_errors:
                if isinstance(error, ErrorEvent):
                    result_errors.append(asdict(error))
                else:
                    result_errors.append(error)
            
            # Add pattern analysis
            if pattern_analysis:
                result_errors.append({
                    'error_type': 'pattern_analysis',
                    'severity': 'info',
                    'message': 'Error pattern analysis',
                    'timestamp': datetime.now().isoformat(),
                    'context': pattern_analysis
                })
            
            return result_errors
            
        except Exception as e:
            return [{
                'error_type': 'error_detection_failed',
                'severity': 'critical',
                'message': f"Error detection failed: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'location': 'error_tracker',
                'context': {}
            }]
    
    def _detect_execution_errors(self, logs: List[str]) -> List[ErrorEvent]:
        """Detect errors in execution logs"""
        errors = []
        
        for i, log_entry in enumerate(logs):
            for pattern_group in self.error_patterns:
                for pattern in pattern_group['patterns']:
                    matches = re.finditer(pattern, log_entry, re.IGNORECASE)
                    for match in matches:
                        error_id = self._generate_error_id('execution', i, match.start())
                        
                        errors.append(ErrorEvent(
                            error_id=error_id,
                            error_type=pattern_group['type'],
                            severity=pattern_group['severity'],
                            message=match.group(0),
                            timestamp=datetime.now().isoformat(),
                            location=f"log_entry_{i}",
                            context={
                                'log_entry': log_entry[:200],
                                'pattern_matched': pattern,
                                'match_position': match.span()
                            },
                            suggested_resolution=pattern_group['resolution']
                        ))
        
        return errors
    
    def _detect_api_errors(self, responses: List[str]) -> List[ErrorEvent]:
        """Detect API-related errors"""
        errors = []
        
        for i, response in enumerate(responses):
            # Try to parse as JSON first
            try:
                response_data = json.loads(response)
                
                # Check for error fields in JSON
                if isinstance(response_data, dict):
                    if 'error' in response_data:
                        error_id = self._generate_error_id('api_json', i, 0)
                        errors.append(ErrorEvent(
                            error_id=error_id,
                            error_type='api_error',
                            severity='high',
                            message=str(response_data['error']),
                            timestamp=datetime.now().isoformat(),
                            location=f"api_response_{i}",
                            context={
                                'response_data': response_data,
                                'error_details': response_data.get('error')
                            },
                            suggested_resolution='Review API call parameters and authentication'
                        ))
                        continue
            except json.JSONDecodeError:
                pass  # Not JSON, continue with pattern matching
            
            # Pattern-based API error detection
            for pattern_group in self.api_error_patterns:
                for pattern in pattern_group['patterns']:
                    matches = re.finditer(pattern, response, re.IGNORECASE)
                    for match in matches:
                        error_id = self._generate_error_id('api', i, match.start())
                        
                        errors.append(ErrorEvent(
                            error_id=error_id,
                            error_type=pattern_group['type'],
                            severity=pattern_group['severity'],
                            message=match.group(0),
                            timestamp=datetime.now().isoformat(),
                            location=f"api_response_{i}",
                            context={
                                'response': response[:300],
                                'pattern_matched': pattern
                            },
                            suggested_resolution=self._get_api_error_resolution(pattern_group['type'])
                        ))
        
        return errors
    
    def _detect_output_errors(self, outputs: List[str]) -> List[ErrorEvent]:
        """Detect errors in outputs"""
        errors = []
        
        for i, output in enumerate(outputs):
            # Check for incomplete patterns
            for pattern_group in self.error_patterns:
                if pattern_group['type'] == 'incomplete_response':
                    for pattern in pattern_group['patterns']:
                        if re.search(pattern, output):
                            error_id = self._generate_error_id('output', i, 0)
                            
                            errors.append(ErrorEvent(
                                error_id=error_id,
                                error_type='incomplete_output',
                                severity='medium',
                                message='Output appears incomplete or truncated',
                                timestamp=datetime.now().isoformat(),
                                location=f"output_{i}",
                                context={
                                    'output_length': len(output),
                                    'output_preview': output[-100:] if len(output) > 100 else output,
                                    'pattern_matched': pattern
                                },
                                suggested_resolution='Complete the interrupted response'
                            ))
            
            # Check for empty or minimal outputs
            if len(output.strip()) < 10:
                error_id = self._generate_error_id('output', i, 0)
                
                errors.append(ErrorEvent(
                    error_id=error_id,
                    error_type='minimal_output',
                    severity='medium',
                    message='Output is too short or empty',
                    timestamp=datetime.now().isoformat(),
                    location=f"output_{i}",
                    context={
                        'output_length': len(output.strip()),
                        'output': output
                    },
                    suggested_resolution='Provide more substantial output content'
                ))
        
        return errors
    
    def _detect_hallucinations(self, outputs: List[str]) -> List[ErrorEvent]:
        """Detect potential hallucinations in outputs"""
        errors = []
        combined_output = ' '.join(outputs)
        
        confidence_penalty = 0.0
        detected_indicators = []
        
        for indicator_group in self.hallucination_indicators:
            for pattern in indicator_group['patterns']:
                matches = list(re.finditer(pattern, combined_output, re.IGNORECASE))
                
                if matches:
                    detected_indicators.append({
                        'type': indicator_group['type'],
                        'matches': [m.group(0) for m in matches],
                        'confidence_impact': indicator_group['confidence_impact']
                    })
                    confidence_penalty += indicator_group['confidence_impact']
        
        # If significant hallucination indicators detected
        if confidence_penalty > self.config['hallucination_confidence_threshold']:
            error_id = self._generate_error_id('hallucination', 0, 0)
            
            errors.append(ErrorEvent(
                error_id=error_id,
                error_type='potential_hallucination',
                severity='high',
                message=f'Potential hallucination detected (confidence penalty: {confidence_penalty:.2f})',
                timestamp=datetime.now().isoformat(),
                location='combined_output',
                context={
                    'detected_indicators': detected_indicators,
                    'confidence_penalty': confidence_penalty,
                    'output_sample': combined_output[:200]
                },
                suggested_resolution='Verify factual accuracy and logical consistency'
            ))
        
        return errors
    
    def _detect_incomplete_responses(self, outputs: List[str]) -> List[ErrorEvent]:
        """Detect incomplete or missing responses"""
        errors = []
        
        if not outputs:
            error_id = self._generate_error_id('missing', 0, 0)
            
            errors.append(ErrorEvent(
                error_id=error_id,
                error_type='missing_output',
                severity='critical',
                message='No outputs generated',
                timestamp=datetime.now().isoformat(),
                location='output_collection',
                context={'output_count': 0},
                suggested_resolution='Ensure output generation is working properly'
            ))
            return errors
        
        # Check for truncated JSON/structured data
        for i, output in enumerate(outputs):
            if output.strip().startswith('{') and not output.strip().endswith('}'):
                error_id = self._generate_error_id('truncated', i, 0)
                
                errors.append(ErrorEvent(
                    error_id=error_id,
                    error_type='truncated_json',
                    severity='medium',
                    message='JSON output appears truncated',
                    timestamp=datetime.now().isoformat(),
                    location=f"output_{i}",
                    context={
                        'output_length': len(output),
                        'starts_with': output[:50],
                        'ends_with': output[-50:]
                    },
                    suggested_resolution='Complete the JSON structure'
                ))
        
        return errors
    
    def _analyze_error_patterns(self, errors: List[ErrorEvent]) -> Dict[str, Any]:
        """Analyze patterns in detected errors"""
        if not errors:
            return {}
        
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        location_counts = defaultdict(int)
        
        for error in errors:
            error_types[error.error_type] += 1
            severity_counts[error.severity] += 1
            location_counts[error.location] += 1
        
        # Detect error clustering
        recent_errors = self._get_recent_errors(minutes=5)
        error_rate = len(recent_errors) / 5.0  # errors per minute
        
        return {
            'error_types': dict(error_types),
            'severity_distribution': dict(severity_counts),
            'location_distribution': dict(location_counts),
            'recent_error_rate': error_rate,
            'critical_threshold_exceeded': error_rate > self.config['critical_error_threshold'],
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_recent_errors(self, minutes: int = 5) -> List[ErrorEvent]:
        """Get errors from recent time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_errors = []
        
        for error in self.error_history:
            error_time = datetime.fromisoformat(error.timestamp)
            if error_time >= cutoff_time:
                recent_errors.append(error)
        
        return recent_errors
    
    def _update_error_stats(self, errors: List[ErrorEvent]):
        """Update error statistics"""
        for error in errors:
            self.error_stats[error.error_type] += 1
            self.error_stats[f"severity_{error.severity}"] += 1
            self.error_stats['total_errors'] += 1
    
    def _add_to_history(self, error: ErrorEvent):
        """Add error to history"""
        # Check for duplicate errors
        for existing_error in self.error_history:
            if (existing_error.error_type == error.error_type and 
                existing_error.message == error.message and
                existing_error.location == error.location):
                existing_error.recurrence_count += 1
                return
        
        self.error_history.append(error)
        
        # Limit history size
        if len(self.error_history) > self.config['max_history_size']:
            self.error_history = self.error_history[-self.config['max_history_size']:]
    
    def _generate_error_id(self, error_type: str, index: int, position: int) -> str:
        """Generate unique error ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{error_type}_{index}_{position}_{timestamp}"
    
    def _get_api_error_resolution(self, error_type: str) -> str:
        """Get resolution suggestion for API errors"""
        resolutions = {
            'timeout_error': 'Increase timeout or retry with exponential backoff',
            'authentication_error': 'Check API credentials and authentication method',
            'rate_limit_error': 'Implement rate limiting and retry logic',
            'server_error': 'Retry request or contact API provider',
            'client_error': 'Validate request parameters and format'
        }
        return resolutions.get(error_type, 'Review API documentation and request format')
    
    def log_error(self, error_data: Dict[str, Any]):
        """Log a custom error"""
        error = ErrorEvent(
            error_id=self._generate_error_id('custom', 0, 0),
            error_type=error_data.get('type', 'unknown'),
            severity=error_data.get('severity', 'medium'),
            message=error_data.get('message', 'Custom error'),
            timestamp=error_data.get('timestamp', datetime.now().isoformat()),
            location=error_data.get('location', 'unknown'),
            context=error_data.get('context', {}),
            suggested_resolution=error_data.get('resolution', 'Review error details')
        )
        
        self._add_to_history(error)
        self._update_error_stats([error])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics"""
        recent_errors = self._get_recent_errors(minutes=60)  # Last hour
        
        return {
            'total_errors': self.error_stats['total_errors'],
            'recent_errors': len(recent_errors),
            'error_types': dict(self.error_stats),
            'most_recent_error': self.error_history[-1].timestamp if self.error_history else None,
            'error_rate_last_hour': len(recent_errors) / 60.0,
            'critical_errors': sum(1 for e in recent_errors if e.severity == 'critical'),
            'summary_timestamp': datetime.now().isoformat()
        }
