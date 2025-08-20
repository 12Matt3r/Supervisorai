import json
import re
from typing import List, Dict, Any
from datetime import datetime

from supervisor_agent import QualityMetrics


class QualityAnalyzer:
    """Analyzes output quality across multiple dimensions"""

    def __init__(self):
        self.json_pattern = re.compile(r'^\s*[\{\[].*[\}\]]\s*$', re.DOTALL)
        self.markdown_patterns = {
            'headers': re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            'lists': re.compile(r'^\s*[-*+]\s+.+$', re.MULTILINE),
            'code_blocks': re.compile(r'```[\s\S]*?```'),
            'links': re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        }

    async def analyze(
        self,
        output: str,
        output_type: str,
        instructions: List[str],
        original_input: str
    ) -> QualityMetrics:
        """Comprehensive quality analysis of agent output"""
        
        structure_score = await self._analyze_structure(output, output_type)
        coherence_score = await self._analyze_coherence(output)
        instruction_adherence = await self._analyze_instruction_adherence(
            output, instructions, original_input
        )
        completeness_score = await self._analyze_completeness(
            output, instructions, original_input
        )
        
        # Calculate overall confidence score
        confidence_score = (
            structure_score * 0.2 +
            coherence_score * 0.3 +
            instruction_adherence * 0.3 +
            completeness_score * 0.2
        )
        
        return QualityMetrics(
            structure_score=structure_score,
            coherence_score=coherence_score,
            instruction_adherence=instruction_adherence,
            completeness_score=completeness_score,
            confidence_score=confidence_score
        )

    async def _analyze_structure(self, output: str, output_type: str) -> float:
        """Analyze structural validity of output"""
        if not output or not output.strip():
            return 0.0
        
        score = 0.5  # Base score for non-empty output
        
        if output_type.lower() == "json":
            try:
                json.loads(output)
                score = 1.0  # Valid JSON
            except json.JSONDecodeError:
                # Check if it looks like JSON but has issues
                if self.json_pattern.match(output.strip()):
                    score = 0.3  # Looks like JSON but invalid
                else:
                    score = 0.1  # Doesn't look like JSON at all
        
        elif output_type.lower() == "markdown":
            # Check for markdown patterns
            markdown_features = 0
            for pattern_name, pattern in self.markdown_patterns.items():
                if pattern.search(output):
                    markdown_features += 1
            
            # Score based on markdown features present
            score = min(1.0, 0.4 + (markdown_features * 0.15))
        
        elif output_type.lower() == "code":
            # Basic code structure checks
            code_indicators = [
                r'\bdef\s+\w+\s*\(',  # Python functions
                r'\bclass\s+\w+',     # Python classes
                r'\bfunction\s+\w+\s*\(',  # JavaScript functions
                r'\{[\s\S]*\}',       # Code blocks
                r'\bif\s*\(',         # Conditional statements
                r'\bfor\s*\(',        # Loops
                r'\bimport\s+\w+',    # Imports
            ]
            
            code_features = sum(
                1 for pattern in code_indicators
                if re.search(pattern, output)
            )
            
            score = min(1.0, 0.3 + (code_features * 0.1))
        
        else:  # text
            # Basic text quality checks
            sentences = re.split(r'[.!?]+', output)
            valid_sentences = [
                s.strip() for s in sentences 
                if s.strip() and len(s.strip()) > 10
            ]
            
            if len(valid_sentences) > 0:
                score = min(1.0, 0.6 + (len(valid_sentences) * 0.05))
        
        return score

    async def _analyze_coherence(self, output: str) -> float:
        """Analyze logical coherence and consistency"""
        if not output or len(output.strip()) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for basic coherence indicators
        sentences = re.split(r'[.!?]+', output)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(valid_sentences) == 0:
            return 0.0
        
        # Coherence factors
        factors = {
            'length_consistency': self._check_sentence_length_consistency(valid_sentences),
            'repetition_penalty': self._check_repetition(output),
            'contradiction_penalty': self._check_contradictions(output),
            'flow_score': self._check_logical_flow(valid_sentences)
        }
        
        # Calculate weighted coherence score
        score = (
            factors['length_consistency'] * 0.2 +
            factors['repetition_penalty'] * 0.3 +
            factors['contradiction_penalty'] * 0.3 +
            factors['flow_score'] * 0.2
        )
        
        return max(0.0, min(1.0, score))

    def _check_sentence_length_consistency(self, sentences: List[str]) -> float:
        """Check if sentence lengths are reasonably consistent"""
        if len(sentences) < 2:
            return 0.7
        
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        if avg_length == 0:
            return 0.0
        
        # Calculate coefficient of variation
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / avg_length
        
        # Lower CV means more consistent (better)
        return max(0.3, 1.0 - min(cv, 1.0))

    def _check_repetition(self, output: str) -> float:
        """Check for excessive repetition (penalty factor)"""
        words = output.lower().split()
        if len(words) < 10:
            return 0.8
        
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        # Penalize high repetition
        return max(0.3, repetition_ratio)

    def _check_contradictions(self, output: str) -> float:
        """Check for obvious contradictions"""
        # Simple contradiction patterns
        contradiction_patterns = [
            (r'\bnot\s+\w+', r'\byes\b'),
            (r'\bno\b', r'\byes\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
            (r'\bnever\b', r'\balways\b'),
            (r'\bfalse\b', r'\btrue\b')
        ]
        
        contradictions = 0
        for neg_pattern, pos_pattern in contradiction_patterns:
            if (re.search(neg_pattern, output, re.IGNORECASE) and 
                re.search(pos_pattern, output, re.IGNORECASE)):
                contradictions += 1
        
        # Penalize contradictions
        penalty = min(0.5, contradictions * 0.1)
        return max(0.3, 1.0 - penalty)

    def _check_logical_flow(self, sentences: List[str]) -> float:
        """Check for logical flow indicators"""
        if len(sentences) < 2:
            return 0.7
        
        # Look for transition words and logical connectors
        flow_indicators = [
            r'\btherefore\b', r'\bhowever\b', r'\bmoreover\b',
            r'\bfurthermore\b', r'\bconsequently\b', r'\bnevertheless\b',
            r'\badditionally\b', r'\bfinally\b', r'\bfirst\b', r'\bsecond\b',
            r'\bnext\b', r'\bthen\b', r'\bafter\b', r'\bbefore\b'
        ]
        
        flow_count = sum(
            1 for pattern in flow_indicators
            for sentence in sentences
            if re.search(pattern, sentence, re.IGNORECASE)
        )
        
        # Normalize by sentence count
        flow_score = min(1.0, flow_count / len(sentences) + 0.5)
        return flow_score

    async def _analyze_instruction_adherence(
        self,
        output: str,
        instructions: List[str],
        original_input: str
    ) -> float:
        """Analyze how well output follows given instructions"""
        if not instructions:
            return 0.8  # No instructions to violate
        
        adherence_scores = []
        
        for instruction in instructions:
            instruction_lower = instruction.lower()
            output_lower = output.lower()
            
            # Check for specific instruction patterns
            if 'format' in instruction_lower or 'json' in instruction_lower:
                if 'json' in instruction_lower:
                    try:
                        json.loads(output)
                        adherence_scores.append(1.0)
                    except json.JSONDecodeError:
                        adherence_scores.append(0.2)
                else:
                    adherence_scores.append(0.7)  # Generic format instruction
            
            elif 'list' in instruction_lower or 'bullet' in instruction_lower:
                # Check for list-like structure
                list_patterns = [r'^\s*[-*+•]', r'^\s*\d+\.']
                has_list = any(
                    re.search(pattern, output, re.MULTILINE)
                    for pattern in list_patterns
                )
                adherence_scores.append(1.0 if has_list else 0.3)
            
            elif 'summary' in instruction_lower or 'summarize' in instruction_lower:
                # Check if output is significantly shorter than input
                input_length = len(original_input.split())
                output_length = len(output.split())
                
                if input_length > 0:
                    ratio = output_length / input_length
                    # Good summary should be 10-50% of original length
                    if 0.1 <= ratio <= 0.5:
                        adherence_scores.append(1.0)
                    elif ratio <= 0.8:
                        adherence_scores.append(0.7)
                    else:
                        adherence_scores.append(0.3)
                else:
                    adherence_scores.append(0.7)
            
            elif 'explain' in instruction_lower or 'describe' in instruction_lower:
                # Check for explanatory content
                explanation_indicators = [
                    r'\bbecause\b', r'\bdue to\b', r'\bresult\b',
                    r'\btherefore\b', r'\bmeans\b', r'\bindicates\b'
                ]
                
                has_explanation = any(
                    re.search(indicator, output_lower)
                    for indicator in explanation_indicators
                )
                
                adherence_scores.append(0.9 if has_explanation else 0.5)
            
            else:
                # Generic instruction - check for keyword presence
                instruction_keywords = [
                    word for word in instruction_lower.split()
                    if len(word) > 3 and word not in ['the', 'and', 'for', 'with']
                ]
                
                keyword_matches = sum(
                    1 for keyword in instruction_keywords
                    if keyword in output_lower
                )
                
                if instruction_keywords:
                    match_ratio = keyword_matches / len(instruction_keywords)
                    adherence_scores.append(max(0.3, match_ratio))
                else:
                    adherence_scores.append(0.7)
        
        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.7

    async def _analyze_completeness(
        self,
        output: str,
        instructions: List[str],
        original_input: str
    ) -> float:
        """Analyze if the output appears complete"""
        if not output or not output.strip():
            return 0.0
        
        # Basic completeness indicators
        completeness_factors = []
        
        # 1. Output length relative to input complexity
        input_words = len(original_input.split())
        output_words = len(output.split())
        
        if input_words > 0:
            # For complex inputs, expect substantial outputs
            if input_words > 100:
                expected_min_words = 20
            elif input_words > 50:
                expected_min_words = 10
            else:
                expected_min_words = 5
            
            length_factor = min(1.0, output_words / expected_min_words)
            completeness_factors.append(length_factor)
        
        # 2. Check for completion indicators
        completion_indicators = [
            r'\bconclude\b', r'\bin summary\b', r'\bfinally\b',
            r'\bto summarize\b', r'\boverall\b', r'\bin conclusion\b'
        ]
        
        has_completion = any(
            re.search(indicator, output, re.IGNORECASE)
            for indicator in completion_indicators
        )
        
        completeness_factors.append(0.8 if has_completion else 0.5)
        
        # 3. Check for abrupt endings
        last_sentence = output.strip().split('.')[-1]
        if len(last_sentence.strip()) < 5:  # Very short last sentence
            completeness_factors.append(0.9)
        else:
            completeness_factors.append(1.0)
        
        # 4. Instruction coverage
        if instructions:
            # Simple heuristic: longer outputs more likely to be complete
            instruction_complexity = sum(len(inst.split()) for inst in instructions)
            coverage_factor = min(1.0, output_words / max(instruction_complexity, 10))
            completeness_factors.append(coverage_factor)
        
        return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.5