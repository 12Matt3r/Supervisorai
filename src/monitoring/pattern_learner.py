import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import uuid4
from pathlib import Path

from . import QualityMetrics, KnowledgeBaseEntry


class PatternLearner:
    """Learns from agent failures and successes to build a knowledge base"""

    def __init__(self, patterns_file: str):
        self.patterns_file = Path(patterns_file)
        self.patterns: Dict[str, KnowledgeBaseEntry] = {}
        self._load_patterns()

    def _load_patterns(self):
        """Load existing patterns from file"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    for pattern_data in data:
                        pattern = KnowledgeBaseEntry(
                            pattern_id=pattern_data['pattern_id'],
                            pattern_description=pattern_data['pattern_description'],
                            failure_type=pattern_data['failure_type'],
                            common_causes=pattern_data['common_causes'],
                            suggested_fixes=pattern_data['suggested_fixes'],
                            confidence_score=pattern_data['confidence_score'],
                            occurrences=pattern_data['occurrences'],
                            last_seen=datetime.fromisoformat(pattern_data['last_seen'])
                        )
                        self.patterns[pattern.pattern_id] = pattern
            except Exception as e:
                print(f"Warning: Could not load patterns: {e}")

    async def _save_patterns(self):
        """Save patterns to file"""
        try:
            data = []
            for pattern in self.patterns.values():
                data.append({
                    'pattern_id': pattern.pattern_id,
                    'pattern_description': pattern.pattern_description,
                    'failure_type': pattern.failure_type,
                    'common_causes': pattern.common_causes,
                    'suggested_fixes': pattern.suggested_fixes,
                    'confidence_score': pattern.confidence_score,
                    'occurrences': pattern.occurrences,
                    'last_seen': pattern.last_seen.isoformat()
                })
            
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save patterns: {e}")

    async def add_pattern(
        self,
        output: str,
        instructions: List[str],
        quality_metrics: QualityMetrics,
        intervention_result: Dict[str, Any]
    ) -> str:
        """Add a new failure pattern to the knowledge base"""
        
        # Analyze the failure to create pattern
        failure_type = self._classify_failure(quality_metrics, intervention_result)
        pattern_description = self._generate_pattern_description(
            output, instructions, quality_metrics, failure_type
        )
        
        # Check if similar pattern already exists
        existing_pattern = await self._find_similar_pattern(
            pattern_description, failure_type
        )
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.occurrences += 1
            existing_pattern.last_seen = datetime.now()
            existing_pattern.confidence_score = min(
                1.0,
                existing_pattern.confidence_score + 0.1
            )
            pattern_id = existing_pattern.pattern_id
        else:
            # Create new pattern
            pattern_id = str(uuid4())[:8]
            
            common_causes = self._identify_common_causes(
                quality_metrics, intervention_result
            )
            
            suggested_fixes = self._generate_suggested_fixes(
                failure_type, quality_metrics
            )
            
            new_pattern = KnowledgeBaseEntry(
                pattern_id=pattern_id,
                pattern_description=pattern_description,
                failure_type=failure_type,
                common_causes=common_causes,
                suggested_fixes=suggested_fixes,
                confidence_score=0.5,  # Initial confidence
                occurrences=1,
                last_seen=datetime.now()
            )
            
            self.patterns[pattern_id] = new_pattern
        
        await self._save_patterns()
        return pattern_id

    def _classify_failure(
        self,
        quality_metrics: QualityMetrics,
        intervention_result: Dict[str, Any]
    ) -> str:
        """Classify the type of failure"""
        
        if quality_metrics.structure_score < 0.5:
            return "structure_failure"
        elif quality_metrics.coherence_score < 0.5:
            return "coherence_failure"
        elif quality_metrics.instruction_adherence < 0.5:
            return "instruction_failure"
        elif quality_metrics.completeness_score < 0.5:
            return "completeness_failure"
        elif quality_metrics.confidence_score < 0.5:
            return "quality_failure"
        else:
            return "unknown_failure"

    def _generate_pattern_description(
        self,
        output: str,
        instructions: List[str],
        quality_metrics: QualityMetrics,
        failure_type: str
    ) -> str:
        """Generate a description of the failure pattern"""
        
        output_length = len(output.split())
        instruction_count = len(instructions)
        
        if failure_type == "structure_failure":
            return f"Output with {output_length} words fails structure validation"
        elif failure_type == "coherence_failure":
            return f"Output lacks coherence (score: {quality_metrics.coherence_score:.2f})"
        elif failure_type == "instruction_failure":
            return f"Output fails to follow {instruction_count} instructions"
        elif failure_type == "completeness_failure":
            return f"Output appears incomplete (score: {quality_metrics.completeness_score:.2f})"
        else:
            return f"Low quality output (confidence: {quality_metrics.confidence_score:.2f})"

    async def _find_similar_pattern(
        self,
        pattern_description: str,
        failure_type: str
    ) -> Optional[KnowledgeBaseEntry]:
        """Find similar existing pattern"""
        
        for pattern in self.patterns.values():
            if pattern.failure_type == failure_type:
                # Simple similarity check based on description keywords
                desc_words = set(pattern_description.lower().split())
                pattern_words = set(pattern.pattern_description.lower().split())
                
                # Calculate Jaccard similarity
                intersection = len(desc_words & pattern_words)
                union = len(desc_words | pattern_words)
                
                if union > 0 and intersection / union > 0.6:
                    return pattern
        
        return None

    def _identify_common_causes(
        self,
        quality_metrics: QualityMetrics,
        intervention_result: Dict[str, Any]
    ) -> List[str]:
        """Identify common causes of the failure"""
        
        causes = []
        
        if quality_metrics.structure_score < 0.5:
            causes.append("Invalid output format")
            causes.append("Parsing errors")
        
        if quality_metrics.coherence_score < 0.5:
            causes.append("Logical inconsistencies")
            causes.append("Contradictory statements")
        
        if quality_metrics.instruction_adherence < 0.5:
            causes.append("Unclear instructions")
            causes.append("Agent configuration issues")
        
        if quality_metrics.completeness_score < 0.5:
            causes.append("Premature termination")
            causes.append("Insufficient context")
        
        if intervention_result.get("reason"):
            causes.append(f"Trigger: {intervention_result['reason']}")
        
        return causes

    def _generate_suggested_fixes(
        self,
        failure_type: str,
        quality_metrics: QualityMetrics
    ) -> List[str]:
        """Generate suggested fixes for the failure type"""
        
        fixes = []
        
        if failure_type == "structure_failure":
            fixes.extend([
                "Validate output format before submission",
                "Use structured output templates",
                "Add format verification step"
            ])
        
        elif failure_type == "coherence_failure":
            fixes.extend([
                "Review output for logical consistency",
                "Add coherence checking step",
                "Break down complex tasks"
            ])
        
        elif failure_type == "instruction_failure":
            fixes.extend([
                "Clarify instructions",
                "Add instruction compliance check",
                "Use more specific prompts"
            ])
        
        elif failure_type == "completeness_failure":
            fixes.extend([
                "Extend maximum response length",
                "Add completion verification",
                "Check for premature termination"
            ])
        
        else:
            fixes.extend([
                "Review agent configuration",
                "Adjust quality thresholds",
                "Monitor for repeated patterns"
            ])
        
        return fixes

    async def check_pattern(
        self,
        output: str,
        instructions: List[str],
        quality_metrics: QualityMetrics
    ) -> Optional[Dict[str, Any]]:
        """Check if current output matches known failure patterns"""
        
        failure_type = self._classify_failure(quality_metrics, {})
        
        # Look for matching patterns
        best_match = None
        best_confidence = 0.0
        
        for pattern in self.patterns.values():
            if pattern.failure_type == failure_type:
                # Calculate confidence based on pattern history and similarity
                confidence = pattern.confidence_score * min(1.0, pattern.occurrences / 5)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern
        
        if best_match and best_confidence > 0.6:
            return {
                "pattern_id": best_match.pattern_id,
                "confidence": best_confidence,
                "description": best_match.pattern_description,
                "suggested_fixes": best_match.suggested_fixes
            }
        
        return None

    async def get_similar_patterns(
        self,
        output: str,
        instructions: List[str],
        limit: int = 5
    ) -> List[KnowledgeBaseEntry]:
        """Get similar patterns for recommendations"""
        
        # Simple similarity based on output characteristics
        output_words = len(output.split())
        instruction_count = len(instructions)
        
        similar_patterns = []
        
        for pattern in self.patterns.values():
            # Basic similarity heuristic
            similarity_score = 0.0
            
            # Consider patterns with similar output lengths
            if "words" in pattern.pattern_description:
                try:
                    pattern_words = int(
                        pattern.pattern_description.split("words")[0].split()[-1]
                    )
                    length_similarity = 1.0 - abs(output_words - pattern_words) / max(output_words, pattern_words, 1)
                    similarity_score += length_similarity * 0.3
                except:
                    pass
            
            # Consider pattern confidence and occurrence frequency
            similarity_score += pattern.confidence_score * 0.4
            similarity_score += min(1.0, pattern.occurrences / 10) * 0.3
            
            if similarity_score > 0.3:
                similar_patterns.append((pattern, similarity_score))
        
        # Sort by similarity and return top patterns
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, _ in similar_patterns[:limit]]

    async def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top failure patterns for reporting"""
        
        # Sort patterns by occurrence frequency and confidence
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: (p.occurrences, p.confidence_score),
            reverse=True
        )
        
        return [{
            "pattern_id": pattern.pattern_id,
            "description": pattern.pattern_description,
            "failure_type": pattern.failure_type,
            "occurrences": pattern.occurrences,
            "confidence": pattern.confidence_score,
            "suggested_fixes": pattern.suggested_fixes[:3]  # Top 3 fixes
        } for pattern in sorted_patterns[:limit]]