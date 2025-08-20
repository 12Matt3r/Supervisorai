"""
Retry System for Supervisor Agent - Progressive retry strategies with intelligent adjustments.

Adapted from auto_retry_system.py for integration with the supervisor error handling system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import json

from .error_types import ErrorType, SupervisorError, ErrorSeverity


class RetryStrategy(Enum):
    """Retry strategies for different error types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    ADAPTIVE = "adaptive"


class PromptAdjustmentType(Enum):
    """Types of prompt adjustments for retries."""
    ADD_CONTEXT = "add_context"
    SIMPLIFY = "simplify"
    REPHRASE = "rephrase"
    ADD_EXAMPLES = "add_examples"
    INCREASE_SPECIFICITY = "increase_specificity"
    CHANGE_APPROACH = "change_approach"


@dataclass
class RetryContext:
    """Context for a retry operation."""
    attempt: int
    max_attempts: int
    last_error: str
    strategy: RetryStrategy
    delay: float
    prompt_adjustments: List[PromptAdjustmentType]
    adjusted_prompt: Optional[str] = None
    metadata: Dict[str, Any] = None


class RetrySystem:
    """Retry system with progressive strategies and intelligent adjustments."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self._default_config()
        
        # Active retry operations
        self.active_retries: Dict[str, RetryContext] = {}
        
        # Retry statistics
        self.stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'strategy_usage': {strategy.value: 0 for strategy in RetryStrategy}
        }
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for retry system."""
        return {
            'max_retries': self.max_retries,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'backoff_multiplier': 2.0,
            'strategy_mapping': {
                ErrorType.TIMEOUT.value: RetryStrategy.EXPONENTIAL_BACKOFF.value,
                ErrorType.NETWORK_ERROR.value: RetryStrategy.EXPONENTIAL_BACKOFF.value,
                ErrorType.RATE_LIMIT.value: RetryStrategy.LINEAR_BACKOFF.value,
                ErrorType.TEMPORARY_FAILURE.value: RetryStrategy.ADAPTIVE.value,
                ErrorType.AGENT_OVERLOAD.value: RetryStrategy.ADAPTIVE.value,
                ErrorType.UNKNOWN.value: RetryStrategy.LINEAR_BACKOFF.value
            },
            'prompt_adjustments': {
                'attempt_1': [PromptAdjustmentType.ADD_CONTEXT.value],
                'attempt_2': [PromptAdjustmentType.REPHRASE.value, PromptAdjustmentType.SIMPLIFY.value],
                'attempt_3': [PromptAdjustmentType.CHANGE_APPROACH.value, PromptAdjustmentType.ADD_EXAMPLES.value]
            },
            'adaptive_learning': True,
            'success_rate_threshold': 0.3
        }
    
    async def should_retry(
        self,
        error: SupervisorError,
        current_attempt: int = 0
    ) -> bool:
        """Determine if an error should be retried."""
        
        # Check maximum retry attempts
        if current_attempt >= self.max_retries:
            self.logger.info(f"Max retries ({self.max_retries}) reached for error {error.error_id}")
            return False
        
        # Check if error type is retryable
        non_retryable_errors = {
            ErrorType.INFINITE_LOOP,
            ErrorType.CONFIGURATION_ERROR,
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.CORRUPTION,
            ErrorType.SECURITY_BREACH,
            ErrorType.HARDWARE_FAILURE
        }
        
        if error.error_type in non_retryable_errors:
            self.logger.info(f"Error type {error.error_type.value} is not retryable")
            return False
        
        # Check if explicitly marked as non-recoverable
        if not error.recoverable:
            self.logger.info(f"Error {error.error_id} marked as non-recoverable")
            return False
        
        # Check severity - don't retry fatal errors
        if error.severity == ErrorSeverity.FATAL:
            self.logger.info(f"Fatal error {error.error_id} cannot be retried")
            return False
        
        return True
    
    async def execute_retry(
        self,
        error: SupervisorError,
        retry_callback: Callable,
        original_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a retry with progressive strategy and prompt adjustment."""
        
        retry_id = f"{error.error_id}_retry_{error.retry_count + 1}"
        
        # Determine retry strategy
        strategy = self._get_retry_strategy(error)
        
        # Calculate delay
        delay = self._calculate_delay(strategy, error.retry_count)
        
        # Get prompt adjustments
        adjustments = self._get_prompt_adjustments(error.retry_count + 1)
        
        # Create retry context
        retry_context = RetryContext(
            attempt=error.retry_count + 1,
            max_attempts=self.max_retries,
            last_error=error.message,
            strategy=strategy,
            delay=delay,
            prompt_adjustments=adjustments,
            metadata={
                'error_type': error.error_type.value,
                'severity': error.severity.value if error.severity else 'unknown',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Adjust prompt if provided
        if original_prompt:
            retry_context.adjusted_prompt = self._adjust_prompt(
                original_prompt, adjustments, error
            )
        
        self.active_retries[retry_id] = retry_context
        
        self.logger.info(
            f"Starting retry {retry_context.attempt}/{retry_context.max_attempts} "
            f"for {error.error_id} with strategy {strategy.value} and delay {delay}s"
        )
        
        # Wait for delay
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            # Execute retry
            if retry_context.adjusted_prompt:
                result = await retry_callback(retry_context.adjusted_prompt)
            else:
                result = await retry_callback()
            
            # Success
            self.stats['successful_retries'] += 1
            self.stats['strategy_usage'][strategy.value] += 1
            self.active_retries.pop(retry_id, None)
            
            self.logger.info(f"Retry {retry_id} successful")
            
            return {
                'success': True,
                'result': result,
                'retry_context': retry_context,
                'attempts_used': retry_context.attempt
            }
            
        except Exception as e:
            # Retry failed
            self.stats['failed_retries'] += 1
            self.active_retries.pop(retry_id, None)
            
            self.logger.error(f"Retry {retry_id} failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'retry_context': retry_context,
                'attempts_used': retry_context.attempt
            }
        
        finally:
            self.stats['total_retries'] += 1
    
    def _get_retry_strategy(self, error: SupervisorError) -> RetryStrategy:
        """Determine the retry strategy based on error type."""
        strategy_name = self.config['strategy_mapping'].get(
            error.error_type.value,
            RetryStrategy.LINEAR_BACKOFF.value
        )
        return RetryStrategy(strategy_name)
    
    def _calculate_delay(self, strategy: RetryStrategy, attempt: int) -> float:
        """Calculate retry delay based on strategy and attempt number."""
        base_delay = self.config['base_delay']
        max_delay = self.config['max_delay']
        multiplier = self.config['backoff_multiplier']
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif strategy == RetryStrategy.FIXED_DELAY:
            return min(base_delay, max_delay)
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(base_delay * (attempt + 1), max_delay)
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(base_delay * (multiplier ** attempt), max_delay)
        elif strategy == RetryStrategy.ADAPTIVE:
            # Adaptive strategy considers recent success rates
            success_rate = self._calculate_recent_success_rate()
            adaptive_multiplier = 1.0 + (1.0 - success_rate)
            return min(base_delay * adaptive_multiplier * (attempt + 1), max_delay)
        
        return base_delay
    
    def _get_prompt_adjustments(self, attempt: int) -> List[PromptAdjustmentType]:
        """Get prompt adjustments for the given attempt number."""
        attempt_key = f'attempt_{attempt}'
        adjustment_names = self.config['prompt_adjustments'].get(
            attempt_key,
            [PromptAdjustmentType.ADD_CONTEXT.value]
        )
        
        return [PromptAdjustmentType(name) for name in adjustment_names]
    
    def _adjust_prompt(
        self,
        original_prompt: str,
        adjustments: List[PromptAdjustmentType],
        error: SupervisorError
    ) -> str:
        """Apply prompt adjustments based on retry strategy."""
        
        adjusted_prompt = original_prompt
        
        for adjustment in adjustments:
            if adjustment == PromptAdjustmentType.ADD_CONTEXT:
                adjusted_prompt = self._add_context_to_prompt(adjusted_prompt, error)
            elif adjustment == PromptAdjustmentType.SIMPLIFY:
                adjusted_prompt = self._simplify_prompt(adjusted_prompt)
            elif adjustment == PromptAdjustmentType.REPHRASE:
                adjusted_prompt = self._rephrase_prompt(adjusted_prompt)
            elif adjustment == PromptAdjustmentType.ADD_EXAMPLES:
                adjusted_prompt = self._add_examples_to_prompt(adjusted_prompt)
            elif adjustment == PromptAdjustmentType.INCREASE_SPECIFICITY:
                adjusted_prompt = self._increase_prompt_specificity(adjusted_prompt)
            elif adjustment == PromptAdjustmentType.CHANGE_APPROACH:
                adjusted_prompt = self._change_prompt_approach(adjusted_prompt, error)
        
        return adjusted_prompt
    
    def _add_context_to_prompt(self, prompt: str, error: SupervisorError) -> str:
        """Add error context to the prompt."""
        context_addition = f"\n\nNote: The previous attempt failed with error: {error.message}. Please consider this when formulating your response."
        return prompt + context_addition
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify the prompt by breaking it down."""
        simplification = "\n\nPlease focus on the core requirement and provide a simpler, more direct approach."
        return prompt + simplification
    
    def _rephrase_prompt(self, prompt: str) -> str:
        """Rephrase the prompt for clarity."""
        rephrase_prefix = "Let me rephrase this request more clearly: "
        return rephrase_prefix + prompt
    
    def _add_examples_to_prompt(self, prompt: str) -> str:
        """Add examples to the prompt."""
        examples_addition = "\n\nPlease provide concrete examples in your response to illustrate the solution."
        return prompt + examples_addition
    
    def _increase_prompt_specificity(self, prompt: str) -> str:
        """Make the prompt more specific."""
        specificity_addition = "\n\nPlease be very specific in your response, including detailed steps and exact parameters."
        return prompt + specificity_addition
    
    def _change_prompt_approach(self, prompt: str, error: SupervisorError) -> str:
        """Change the approach based on error type."""
        approach_change = "\n\nGiven the previous failure, please try a different approach or methodology to solve this problem."
        return prompt + approach_change
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent overall success rate."""
        total = self.stats['successful_retries'] + self.stats['failed_retries']
        if total == 0:
            return 1.0
        return self.stats['successful_retries'] / total
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the retry system."""
        return {
            'active_retries': len(self.active_retries),
            'stats': self.stats,
            'config': self.config,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the retry system."""
        self.logger.info("Shutting down retry system")
        self.active_retries.clear()
        self.logger.info("Retry system shutdown complete")
