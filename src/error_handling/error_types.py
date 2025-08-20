"""
Error Types and Classification System for Supervisor Agent

Defines error types, custom exception classes, and error classification logic.
"""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


class ErrorType(Enum):
    """Categories of errors that can occur in the system."""
    
    # Recoverable errors - can be retried with adjustments
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    TEMPORARY_FAILURE = "temporary_failure"
    
    # Agent-specific errors
    AGENT_OVERLOAD = "agent_overload"
    TASK_COMPLEXITY = "task_complexity"
    CONTEXT_OVERFLOW = "context_overflow"
    
    # Loop and control flow errors
    INFINITE_LOOP = "infinite_loop"
    STUCK_STATE = "stuck_state"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    
    # Critical system errors
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    AUTHENTICATION_ERROR = "authentication_error"
    
    # Fatal errors - require immediate intervention
    CORRUPTION = "corruption"
    SECURITY_BREACH = "security_breach"
    HARDWARE_FAILURE = "hardware_failure"
    
    # Other
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class SupervisorError(Exception):
    """Custom exception class for Supervisor Agent errors."""
    
    message: str
    error_type: ErrorType
    severity: Optional[ErrorSeverity] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    stack_trace: Optional[str] = None
    error_id: Optional[str] = None
    recoverable: bool = True
    retry_count: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if self.error_id is None:
            import uuid
            self.error_id = str(uuid.uuid4())
        
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()
        
        if self.severity is None:
            self.severity = ErrorClassifier.determine_severity(self.error_type)
        
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        data = asdict(self)
        data['error_type'] = self.error_type.value
        data['severity'] = self.severity.value if self.severity else None
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SupervisorError':
        """Create SupervisorError from dictionary."""
        data['error_type'] = ErrorType(data['error_type'])
        if data.get('severity'):
            data['severity'] = ErrorSeverity(data['severity'])
        if data.get('timestamp'):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def __str__(self) -> str:
        return f"SupervisorError({self.error_type.value}): {self.message}"


class ErrorClassifier:
    """Utility class for error classification and analysis."""
    
    # Error type mapping based on exception types
    EXCEPTION_TYPE_MAPPING = {
        TimeoutError: ErrorType.TIMEOUT,
        ConnectionError: ErrorType.NETWORK_ERROR,
        ConnectionResetError: ErrorType.NETWORK_ERROR,
        ConnectionRefusedError: ErrorType.NETWORK_ERROR,
        ValueError: ErrorType.CONFIGURATION_ERROR,
        KeyError: ErrorType.CONFIGURATION_ERROR,
        FileNotFoundError: ErrorType.CONFIGURATION_ERROR,
        PermissionError: ErrorType.AUTHENTICATION_ERROR,
        MemoryError: ErrorType.SYSTEM_ERROR,
        OSError: ErrorType.SYSTEM_ERROR,
    }
    
    # Error severity mapping
    SEVERITY_MAPPING = {
        ErrorType.TIMEOUT: ErrorSeverity.MEDIUM,
        ErrorType.RATE_LIMIT: ErrorSeverity.LOW,
        ErrorType.NETWORK_ERROR: ErrorSeverity.MEDIUM,
        ErrorType.TEMPORARY_FAILURE: ErrorSeverity.LOW,
        ErrorType.AGENT_OVERLOAD: ErrorSeverity.HIGH,
        ErrorType.TASK_COMPLEXITY: ErrorSeverity.MEDIUM,
        ErrorType.CONTEXT_OVERFLOW: ErrorSeverity.HIGH,
        ErrorType.INFINITE_LOOP: ErrorSeverity.CRITICAL,
        ErrorType.STUCK_STATE: ErrorSeverity.HIGH,
        ErrorType.CIRCULAR_DEPENDENCY: ErrorSeverity.HIGH,
        ErrorType.SYSTEM_ERROR: ErrorSeverity.CRITICAL,
        ErrorType.CONFIGURATION_ERROR: ErrorSeverity.MEDIUM,
        ErrorType.AUTHENTICATION_ERROR: ErrorSeverity.HIGH,
        ErrorType.CORRUPTION: ErrorSeverity.FATAL,
        ErrorType.SECURITY_BREACH: ErrorSeverity.FATAL,
        ErrorType.HARDWARE_FAILURE: ErrorSeverity.FATAL,
        ErrorType.UNKNOWN: ErrorSeverity.MEDIUM,
    }
    
    # Recoverable error types
    RECOVERABLE_ERRORS = {
        ErrorType.TIMEOUT,
        ErrorType.RATE_LIMIT,
        ErrorType.NETWORK_ERROR,
        ErrorType.TEMPORARY_FAILURE,
        ErrorType.AGENT_OVERLOAD,
        ErrorType.TASK_COMPLEXITY,
        ErrorType.CONTEXT_OVERFLOW,
        ErrorType.STUCK_STATE,
    }
    
    @classmethod
    def classify_exception(cls, exception: Exception) -> ErrorType:
        """Classify an exception into an ErrorType."""
        
        # Check direct mapping
        for exc_type, error_type in cls.EXCEPTION_TYPE_MAPPING.items():
            if isinstance(exception, exc_type):
                return error_type
        
        # Check error message patterns
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['timeout', 'timed out']):
            return ErrorType.TIMEOUT
        
        if any(keyword in error_message for keyword in ['rate limit', 'too many requests']):
            return ErrorType.RATE_LIMIT
        
        if any(keyword in error_message for keyword in ['network', 'connection', 'dns']):
            return ErrorType.NETWORK_ERROR
        
        if any(keyword in error_message for keyword in ['loop', 'circular', 'recursion']):
            return ErrorType.INFINITE_LOOP
        
        if any(keyword in error_message for keyword in ['memory', 'resource']):
            return ErrorType.SYSTEM_ERROR
        
        if any(keyword in error_message for keyword in ['auth', 'permission', 'credential']):
            return ErrorType.AUTHENTICATION_ERROR
        
        # Default classification
        return ErrorType.UNKNOWN
    
    @classmethod
    def determine_severity(cls, error_type: ErrorType) -> ErrorSeverity:
        """Determine severity level for an error type."""
        return cls.SEVERITY_MAPPING.get(error_type, ErrorSeverity.MEDIUM)
    
    @classmethod
    def is_recoverable(cls, error_type: ErrorType) -> bool:
        """Check if an error type is recoverable."""
        return error_type in cls.RECOVERABLE_ERRORS
    
    @classmethod
    def analyze_error_pattern(
        cls, 
        errors: List[SupervisorError]
    ) -> Dict[str, Any]:
        """Analyze patterns in a list of errors."""
        
        if not errors:
            return {'pattern_found': False}
        
        # Count error types
        error_type_counts = {}
        severity_counts = {}
        
        for error in errors:
            error_type = error.error_type.value
            severity = error.severity.value if error.severity else 'unknown'
            
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Find most common patterns
        most_common_type = max(error_type_counts, key=error_type_counts.get)
        most_common_severity = max(severity_counts, key=severity_counts.get)
        
        # Analyze temporal patterns
        if len(errors) > 1:
            time_diffs = []
            sorted_errors = sorted(errors, key=lambda x: x.timestamp or datetime.min)
            
            for i in range(1, len(sorted_errors)):
                if sorted_errors[i].timestamp and sorted_errors[i-1].timestamp:
                    diff = (sorted_errors[i].timestamp - sorted_errors[i-1].timestamp).total_seconds()
                    time_diffs.append(diff)
            
            avg_time_between = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        else:
            avg_time_between = 0
        
        # Detect potential infinite loop
        loop_indicators = [
            error_type_counts.get(ErrorType.INFINITE_LOOP.value, 0) > 0,
            len(errors) > 10 and avg_time_between < 5,  # Many errors in short time
            error_type_counts.get(ErrorType.STUCK_STATE.value, 0) > 2
        ]
        
        return {
            'pattern_found': True,
            'total_errors': len(errors),
            'error_type_distribution': error_type_counts,
            'severity_distribution': severity_counts,
            'most_common_type': most_common_type,
            'most_common_severity': most_common_severity,
            'average_time_between_errors': avg_time_between,
            'potential_infinite_loop': any(loop_indicators),
            'recoverable_ratio': sum(
                count for error_type, count in error_type_counts.items()
                if cls.is_recoverable(ErrorType(error_type))
            ) / len(errors)
        }
