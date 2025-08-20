"""
Loop Detector and Circuit Breaker for Supervisor Agent.

Detects infinite loops, circular dependencies, and implements circuit breaker patterns
to prevent system overload and stuck states.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import deque, Counter
import hashlib
import json


class LoopType(Enum):
    """Types of loops that can be detected."""
    EXECUTION_LOOP = "execution_loop"  # Same operation repeated
    STATE_LOOP = "state_loop"  # Same state reached repeatedly
    CIRCULAR_DEPENDENCY = "circular_dependency"  # A->B->C->A pattern
    STUCK_AGENT = "stuck_agent"  # Agent not making progress


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ExecutionPoint:
    """Represents a point in execution for loop detection."""
    timestamp: datetime
    agent_id: str
    task_id: str
    operation: str
    state_hash: str
    output_hash: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LoopDetection:
    """Results of loop detection analysis."""
    detection_id: str
    loop_type: LoopType
    agent_id: str
    task_id: str
    confidence_score: float
    severity: str  # "low", "medium", "high", "critical"
    evidence: List[Dict[str, Any]]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['loop_type'] = self.loop_type.value
        return data


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    name: str
    state: CircuitBreakerState
    failure_count: int
    last_failure_time: Optional[datetime]
    failure_threshold: int
    recovery_timeout: int  # seconds
    half_open_max_calls: int
    half_open_calls: int
    
    def is_call_allowed(self) -> bool:
        """Check if a call is allowed through the circuit breaker."""
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record a successful call."""
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record a failed call."""
        
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class LoopDetector:
    """System for detecting infinite loops and implementing circuit breakers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Execution history for loop detection
        self.execution_history: Dict[str, deque] = {}  # agent_id -> deque of ExecutionPoint
        self.max_history_size = 100
        
        # Circuit breakers per agent
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Paused agents
        self.paused_agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> pause_info
        
        # Configuration
        self.config = {
            'loop_detection_window': 300,  # seconds
            'min_repetitions_for_loop': 3,
            'similarity_threshold': 0.9,
            'max_execution_time': 600,  # 10 minutes
            'circuit_breaker_failure_threshold': 5,
            'circuit_breaker_recovery_timeout': 60,  # 1 minute
            'stuck_agent_threshold': 300  # 5 minutes without progress
        }
        
        # Statistics
        self.stats = {
            'loops_detected': 0,
            'agents_paused': 0,
            'circuit_breakers_opened': 0,
            'by_loop_type': {loop_type.value: 0 for loop_type in LoopType}
        }
    
    def record_execution_point(
        self,
        agent_id: str,
        task_id: str,
        state: Dict[str, Any],
        output: str,
        context: Dict[str, Any],
        operation: str = "execute"
    ) -> Optional[LoopDetection]:
        """Record an execution point and check for loops."""
        
        # Create execution point
        execution_point = ExecutionPoint(
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            task_id=task_id,
            operation=operation,
            state_hash=self._hash_data(state),
            output_hash=self._hash_data(output),
            context=context
        )
        
        # Initialize history for agent if needed
        if agent_id not in self.execution_history:
            self.execution_history[agent_id] = deque(maxlen=self.max_history_size)
        
        # Add to history
        self.execution_history[agent_id].append(execution_point)
        
        # Check for loops
        return self._analyze_for_loops(agent_id, task_id)
    
    def _analyze_for_loops(
        self,
        agent_id: str,
        task_id: str
    ) -> Optional[LoopDetection]:
        """Analyze execution history for loop patterns."""
        
        history = self.execution_history.get(agent_id, deque())
        if len(history) < self.config['min_repetitions_for_loop']:
            return None
        
        # Convert to list for easier analysis
        recent_history = list(history)[-50:]  # Analyze last 50 points
        
        # Check for different types of loops
        loop_detections = [
            self._detect_execution_loop(recent_history, agent_id, task_id),
            self._detect_state_loop(recent_history, agent_id, task_id),
            self._detect_stuck_agent(recent_history, agent_id, task_id)
        ]
        
        # Return the most severe detection
        valid_detections = [d for d in loop_detections if d is not None]
        if not valid_detections:
            return None
        
        # Sort by severity and confidence
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        best_detection = max(
            valid_detections,
            key=lambda d: (severity_order.get(d.severity, 0), d.confidence_score)
        )
        
        # Update statistics
        self.stats['loops_detected'] += 1
        self.stats['by_loop_type'][best_detection.loop_type.value] += 1
        
        self.logger.warning(
            f"Loop detected for agent {agent_id}: {best_detection.loop_type.value} "
            f"(confidence: {best_detection.confidence_score:.2f})"
        )
        
        return best_detection
    
    def _detect_execution_loop(
        self,
        history: List[ExecutionPoint],
        agent_id: str,
        task_id: str
    ) -> Optional[LoopDetection]:
        """Detect loops in execution patterns."""
        
        if len(history) < 3:
            return None
        
        # Look for repeated operation + output hash combinations
        recent_operations = [(point.operation, point.output_hash) for point in history[-10:]]
        
        # Count repetitions
        counter = Counter(recent_operations)
        most_common = counter.most_common(1)[0]
        
        if most_common[1] >= self.config['min_repetitions_for_loop']:
            confidence = min(most_common[1] / len(recent_operations), 1.0)
            
            return LoopDetection(
                detection_id=self._generate_detection_id(),
                loop_type=LoopType.EXECUTION_LOOP,
                agent_id=agent_id,
                task_id=task_id,
                confidence_score=confidence,
                severity="high" if confidence > 0.7 else "medium",
                evidence=[
                    {
                        'pattern': f"{most_common[0][0]}:{most_common[0][1][:8]}",
                        'repetitions': most_common[1],
                        'window_size': len(recent_operations)
                    }
                ],
                recommendation="Break execution loop by modifying operation or adding randomization"
            )
        
        return None
    
    def _detect_state_loop(
        self,
        history: List[ExecutionPoint],
        agent_id: str,
        task_id: str
    ) -> Optional[LoopDetection]:
        """Detect loops in system state."""
        
        if len(history) < 3:
            return None
        
        # Look for repeated state hashes
        state_hashes = [point.state_hash for point in history[-15:]]
        
        # Find cycles in state transitions
        for cycle_length in range(2, min(len(state_hashes) // 2, 6)):
            if self._has_cycle(state_hashes, cycle_length):
                confidence = 0.8  # High confidence for state cycles
                
                return LoopDetection(
                    detection_id=self._generate_detection_id(),
                    loop_type=LoopType.STATE_LOOP,
                    agent_id=agent_id,
                    task_id=task_id,
                    confidence_score=confidence,
                    severity="critical",
                    evidence=[
                        {
                            'cycle_length': cycle_length,
                            'state_pattern': state_hashes[-cycle_length:]
                        }
                    ],
                    recommendation="Agent is cycling through states without progress"
                )
        
        return None
    
    def _detect_stuck_agent(
        self,
        history: List[ExecutionPoint],
        agent_id: str,
        task_id: str
    ) -> Optional[LoopDetection]:
        """Detect agents that are stuck without making progress."""
        
        if len(history) < 2:
            return None
        
        # Check time since last different output
        now = datetime.utcnow()
        last_different_output_time = None
        last_output_hash = None
        
        for point in reversed(history):
            if last_output_hash is None:
                last_output_hash = point.output_hash
                continue
            
            if point.output_hash != last_output_hash:
                last_different_output_time = point.timestamp
                break
        
        if last_different_output_time:
            time_stuck = (now - last_different_output_time).total_seconds()
            if time_stuck > self.config['stuck_agent_threshold']:
                confidence = min(time_stuck / self.config['stuck_agent_threshold'], 1.0)
                
                return LoopDetection(
                    detection_id=self._generate_detection_id(),
                    loop_type=LoopType.STUCK_AGENT,
                    agent_id=agent_id,
                    task_id=task_id,
                    confidence_score=confidence,
                    severity="high",
                    evidence=[
                        {
                            'time_stuck_seconds': time_stuck,
                            'last_progress_time': last_different_output_time.isoformat(),
                            'stuck_threshold': self.config['stuck_agent_threshold']
                        }
                    ],
                    recommendation="Agent appears stuck - consider intervention or restart"
                )
        
        return None
    
    def pause_agent(self, agent_id: str, reason: str):
        """Pause an agent to prevent further execution."""
        
        self.paused_agents[agent_id] = {
            'paused_at': datetime.utcnow().isoformat(),
            'reason': reason,
            'pause_count': self.paused_agents.get(agent_id, {}).get('pause_count', 0) + 1
        }
        
        self.stats['agents_paused'] += 1
        self.logger.warning(f"Paused agent {agent_id}: {reason}")
    
    def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        
        if agent_id in self.paused_agents:
            pause_info = self.paused_agents.pop(agent_id)
            self.logger.info(
                f"Resumed agent {agent_id} (was paused: {pause_info['reason']})"
            )
            return True
        
        return False
    
    def is_agent_paused(self, agent_id: str) -> bool:
        """Check if an agent is currently paused."""
        return agent_id in self.paused_agents
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                state=CircuitBreakerState.CLOSED,
                failure_count=0,
                last_failure_time=None,
                failure_threshold=self.config['circuit_breaker_failure_threshold'],
                recovery_timeout=self.config['circuit_breaker_recovery_timeout'],
                half_open_max_calls=3,
                half_open_calls=0
            )
        
        return self.circuit_breakers[name]
    
    def _hash_data(self, data: Any) -> str:
        """Create a hash of data for comparison."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _has_cycle(self, sequence: List[str], cycle_length: int) -> bool:
        """Check if a sequence has a repeating cycle."""
        
        if len(sequence) < cycle_length * 2:
            return False
        
        pattern = sequence[-cycle_length:]
        previous_pattern = sequence[-(cycle_length * 2):-cycle_length]
        
        return pattern == previous_pattern
    
    def _generate_detection_id(self) -> str:
        """Generate a unique detection ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the loop detector."""
        return {
            'active_agents': len(self.execution_history),
            'paused_agents': len(self.paused_agents),
            'circuit_breakers': len(self.circuit_breakers),
            'circuit_breaker_states': {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            },
            'stats': self.stats,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the loop detector."""
        self.logger.info("Shutting down loop detector")
        
        # Resume all paused agents
        for agent_id in list(self.paused_agents.keys()):
            self.resume_agent(agent_id)
        
        self.execution_history.clear()
        self.circuit_breakers.clear()
        
        self.logger.info("Loop detector shutdown complete")
