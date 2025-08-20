"""Output Quality Monitoring - Validates structure, coherence, and content relevance"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import hashlib

@dataclass
class QualityIssue:
    """Represents an output quality issue"""
    issue_type: str
    description: str
    severity: str
    location: str
    score_impact: float
    suggested_fix: str

class OutputQualityMonitor:
    """Monitors output quality including structure, coherence, and relevance"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.quality_patterns = self._initialize_quality_patterns()
        self.coherence_indicators = self._initialize_coherence_indicators()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'structure_weight': 0.3,
            'coherence_weight': 0.4,
            'relevance_weight': 0.3,
            'minimum_quality_threshold': 0.6,
            'coherence_window_size': 3,  # sentences to analyze together
            'duplicate_threshold': 0.8
        }
    
    def _initialize_quality_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize patterns for quality assessment"""
        return {
            'structure_issues': [
                {
                    'pattern': r'^[a-z]',  # Sentences not starting with capital
                    'description': 'Improper capitalization',
                    'severity': 'low',
                    'score_impact': 0.05
                },
                {
                    'pattern': r'[.!?]\s*[a-z]',  # Lowercase after sentence end
                    'description': 'Missing capitalization after punctuation',
                    'severity': 'medium',
                    'score_impact': 0.1
                },
                {
                    'pattern': r'\b\w{50,}\b',  # Extremely long words (likely errors)
                    'description': 'Extremely long words detected',
                    'severity': 'medium',
                    'score_impact': 0.1
                },
                {
                    'pattern': r'\s{3,}',  # Multiple consecutive spaces
                    'description': 'Excessive whitespace',
                    'severity': 'low',
                    'score_impact': 0.02
                }
            ],
            'coherence_issues': [
                {
                    'pattern': r'\b(however|but|although)\s+[^.!?]*\b(however|but|although)\b',
                    'description': 'Contradictory conjunctions in same sentence',
                    'severity': 'high',
                    'score_impact': 0.2
                },
                {
                    'pattern': r'\b(the same|identical|similar)\b.*\b(different|distinct|unique)\b',
                    'description': 'Logical contradiction detected',
                    'severity': 'high',
                    'score_impact': 0.25
                }
            ],
            'content_issues': [
                {
                    'pattern': r'\b(TODO|FIXME|PLACEHOLDER|TBD)\b',
                    'description': 'Incomplete content markers',
                    'severity': 'high',
                    'score_impact': 0.3
                },
                {
                    'pattern': r'\b(error|failed|exception|null|undefined)\b',
                    'description': 'Error indicators in output',
                    'severity': 'medium',
                    'score_impact': 0.15
                }
            ]
        }
    
    def _initialize_coherence_indicators(self) -> Dict[str, List[str]]:
        """Initialize coherence analysis indicators"""
        return {
            'transition_words': [
                'however', 'therefore', 'furthermore', 'moreover', 'consequently',
                'nevertheless', 'additionally', 'similarly', 'in contrast', 'meanwhile'
            ],
            'pronoun_references': [
                'it', 'this', 'that', 'these', 'those', 'they', 'them', 'their'
            ],
            'logical_connectors': [
                'because', 'since', 'due to', 'as a result', 'leads to',
                'causes', 'results in', 'leads to', 'implies', 'suggests'
            ]
        }
    
    def evaluate_output_quality(self, outputs: List[str], 
                              expected_format: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate comprehensive output quality"""
        try:
            if not outputs:
                return {
                    'score': 0.0,
                    'status': 'no_outputs',
                    'timestamp': datetime.now().isoformat()
                }
            
            combined_output = ' '.join(outputs)
            
            # Structure validation
            structure_result = self._evaluate_structure_quality(
                outputs, expected_format
            )
            
            # Coherence analysis
            coherence_result = self._evaluate_coherence(
                outputs, combined_output
            )
            
            # Content relevance analysis
            relevance_result = self._evaluate_content_relevance(
                outputs, expected_format
            )
            
            # Detect quality issues
            quality_issues = self._detect_quality_issues(
                combined_output, outputs
            )
            
            # Check for duplicates and repetition
            duplication_result = self._analyze_duplication(
                outputs
            )
            
            # Calculate overall quality score
            structure_score = structure_result['score']
            coherence_score = coherence_result['score']
            relevance_score = relevance_result['score']
            
            overall_score = (
                structure_score * self.config['structure_weight'] +
                coherence_score * self.config['coherence_weight'] +
                relevance_score * self.config['relevance_weight']
            )
            
            # Apply penalty for quality issues
            issue_penalty = sum(issue.score_impact for issue in quality_issues)
            overall_score = max(overall_score - issue_penalty, 0.0)
            
            # Apply duplication penalty
            if duplication_result['duplication_detected']:
                overall_score *= (1.0 - duplication_result['duplication_ratio'] * 0.3)
            
            return {
                'score': min(overall_score, 1.0),
                'structure_quality': structure_result,
                'coherence_quality': coherence_result,
                'relevance_quality': relevance_result,
                'quality_issues': [{
                    'type': issue.issue_type,
                    'description': issue.description,
                    'severity': issue.severity,
                    'location': issue.location,
                    'impact': issue.score_impact,
                    'fix': issue.suggested_fix
                } for issue in quality_issues],
                'duplication_analysis': duplication_result,
                'quality_status': self._determine_quality_status(
                    overall_score, quality_issues
                ),
                'recommendations': self._generate_quality_recommendations(
                    structure_result, coherence_result, relevance_result, quality_issues
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
    
    def _evaluate_structure_quality(self, outputs: List[str], 
                                  expected_format: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate structural quality of outputs"""
        issues = []
        score_penalties = 0.0
        
        combined_text = ' '.join(outputs)
        
        # Check format compliance
        format_score = 1.0
        format_type = expected_format.get('type', '').lower()
        
        if format_type == 'json':
            format_score = self._validate_json_structure(combined_text)
        elif format_type == 'markdown':
            format_score = self._validate_markdown_structure(combined_text)
        elif format_type == 'structured':
            format_score = self._validate_structured_format(combined_text, expected_format)
        
        # Check for structural issues using patterns
        for category, patterns in self.quality_patterns.items():
            if category == 'structure_issues':
                for pattern_info in patterns:
                    matches = re.findall(pattern_info['pattern'], combined_text)
                    if matches:
                        issues.append(QualityIssue(
                            issue_type='structure',
                            description=pattern_info['description'],
                            severity=pattern_info['severity'],
                            location=f"Found {len(matches)} instances",
                            score_impact=pattern_info['score_impact'] * len(matches),
                            suggested_fix=f"Fix {pattern_info['description'].lower()}"
                        ))
                        score_penalties += pattern_info['score_impact'] * len(matches)
        
        final_score = max((format_score - score_penalties), 0.0)
        
        return {
            'score': min(final_score, 1.0),
            'format_compliance': format_score,
            'structure_issues': issues,
            'penalty_applied': score_penalties,
            'status': 'evaluated'
        }
    
    def _evaluate_coherence(self, outputs: List[str], combined_text: str) -> Dict[str, Any]:
        """Evaluate logical coherence and consistency"""
        coherence_issues = []
        
        # Analyze sentence-level coherence
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return {
                'score': 0.8,  # Neutral score for very short content
                'coherence_issues': [],
                'transition_score': 0.0,
                'consistency_score': 1.0,
                'status': 'insufficient_content'
            }
        
        # Check for transition word usage
        transition_score = self._analyze_transitions(sentences)
        
        # Check for pronoun reference coherence
        reference_score = self._analyze_pronoun_references(sentences)
        
        # Check for logical contradictions
        contradiction_score = self._detect_logical_contradictions(combined_text)
        
        # Check topic consistency
        consistency_score = self._analyze_topic_consistency(sentences)
        
        # Detect coherence issues using patterns
        for category, patterns in self.quality_patterns.items():
            if category == 'coherence_issues':
                for pattern_info in patterns:
                    matches = re.findall(pattern_info['pattern'], combined_text)
                    if matches:
                        coherence_issues.append(QualityIssue(
                            issue_type='coherence',
                            description=pattern_info['description'],
                            severity=pattern_info['severity'],
                            location=f"Found in: {matches[0][:50]}...",
                            score_impact=pattern_info['score_impact'],
                            suggested_fix="Review and resolve logical inconsistency"
                        ))
        
        # Calculate overall coherence score
        coherence_score = (
            transition_score * 0.25 +
            reference_score * 0.25 +
            contradiction_score * 0.3 +
            consistency_score * 0.2
        )
        
        # Apply penalties for detected issues
        issue_penalty = sum(issue.score_impact for issue in coherence_issues)
        coherence_score = max(coherence_score - issue_penalty, 0.0)
        
        return {
            'score': min(coherence_score, 1.0),
            'transition_score': transition_score,
            'reference_score': reference_score,
            'contradiction_score': contradiction_score,
            'consistency_score': consistency_score,
            'coherence_issues': coherence_issues,
            'status': 'evaluated'
        }
    
    def _evaluate_content_relevance(self, outputs: List[str], 
                                  expected_format: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate content relevance and accuracy"""
        combined_text = ' '.join(outputs)
        
        # Check content completeness
        completeness_score = self._analyze_content_completeness(
            combined_text, expected_format
        )
        
        # Check for error indicators
        error_score = self._detect_error_indicators(combined_text)
        
        # Check content density (information per word)
        density_score = self._analyze_content_density(combined_text)
        
        # Check for vague or filler content
        specificity_score = self._analyze_content_specificity(combined_text)
        
        relevance_score = (
            completeness_score * 0.3 +
            error_score * 0.3 +
            density_score * 0.2 +
            specificity_score * 0.2
        )
        
        return {
            'score': min(relevance_score, 1.0),
            'completeness_score': completeness_score,
            'error_score': error_score,
            'density_score': density_score,
            'specificity_score': specificity_score,
            'status': 'evaluated'
        }
    
    def _detect_quality_issues(self, combined_text: str, outputs: List[str]) -> List[QualityIssue]:
        """Detect various quality issues"""
        issues = []
        
        # Check for content issues using patterns
        for category, patterns in self.quality_patterns.items():
            if category == 'content_issues':
                for pattern_info in patterns:
                    matches = re.findall(pattern_info['pattern'], combined_text, re.IGNORECASE)
                    if matches:
                        issues.append(QualityIssue(
                            issue_type='content',
                            description=pattern_info['description'],
                            severity=pattern_info['severity'],
                            location=f"Found: {', '.join(matches[:3])}",
                            score_impact=pattern_info['score_impact'],
                            suggested_fix="Remove or replace problematic content"
                        ))
        
        # Check for empty or minimal content
        for i, output in enumerate(outputs):
            if len(output.strip()) < 10:
                issues.append(QualityIssue(
                    issue_type='content',
                    description='Minimal or empty content',
                    severity='medium',
                    location=f"Output {i+1}",
                    score_impact=0.2,
                    suggested_fix="Provide more substantial content"
                ))
        
        return issues
    
    def _analyze_duplication(self, outputs: List[str]) -> Dict[str, Any]:
        """Analyze content duplication and repetition"""
        if len(outputs) < 2:
            return {
                'duplication_detected': False,
                'duplication_ratio': 0.0,
                'duplicate_pairs': [],
                'repetitive_phrases': []
            }
        
        # Check for duplicate outputs
        duplicate_pairs = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = self._calculate_text_similarity(outputs[i], outputs[j])
                if similarity > self.config['duplicate_threshold']:
                    duplicate_pairs.append({
                        'output_indices': [i, j],
                        'similarity': similarity
                    })
        
        # Check for repetitive phrases
        combined_text = ' '.join(outputs)
        repetitive_phrases = self._find_repetitive_phrases(combined_text)
        
        duplication_ratio = len(duplicate_pairs) / max(len(outputs) - 1, 1)
        
        return {
            'duplication_detected': len(duplicate_pairs) > 0 or len(repetitive_phrases) > 0,
            'duplication_ratio': duplication_ratio,
            'duplicate_pairs': duplicate_pairs,
            'repetitive_phrases': repetitive_phrases
        }
    
    def _validate_json_structure(self, text: str) -> float:
        """Validate JSON structure"""
        try:
            json.loads(text)
            return 1.0
        except json.JSONDecodeError:
            # Check if it looks like attempted JSON
            if '{' in text and '}' in text:
                return 0.3
            return 0.0
    
    def _validate_markdown_structure(self, text: str) -> float:
        """Validate Markdown structure"""
        markdown_elements = [
            r'^#{1,6}\s',  # Headers
            r'\*\*[^*]+\*\*',  # Bold
            r'\*[^*]+\*',  # Italic
            r'```[\s\S]*?```',  # Code blocks
            r'\[[^\]]+\]\([^)]+\)',  # Links
            r'^\s*[-*+]\s',  # Lists
        ]
        
        element_count = 0
        for pattern in markdown_elements:
            if re.search(pattern, text, re.MULTILINE):
                element_count += 1
        
        return min(element_count / 3.0, 1.0)  # Normalize to max 3 elements
    
    def _validate_structured_format(self, text: str, expected_format: Dict[str, Any]) -> float:
        """Validate structured format based on expectations"""
        required_sections = expected_format.get('sections', [])
        if not required_sections:
            return 1.0
        
        found_sections = 0
        for section in required_sections:
            if section.lower() in text.lower():
                found_sections += 1
        
        return found_sections / len(required_sections)
    
    def _analyze_transitions(self, sentences: List[str]) -> float:
        """Analyze transition word usage for coherence"""
        if len(sentences) < 2:
            return 0.5
        
        transition_words = self.coherence_indicators['transition_words']
        sentences_with_transitions = 0
        
        for sentence in sentences[1:]:  # Skip first sentence
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words):
                sentences_with_transitions += 1
        
        # Good transition usage: 30-70% of sentences (excluding first)
        transition_ratio = sentences_with_transitions / (len(sentences) - 1)
        
        if 0.3 <= transition_ratio <= 0.7:
            return 1.0
        elif transition_ratio < 0.3:
            return transition_ratio / 0.3
        else:
            return max(1.0 - (transition_ratio - 0.7) * 2, 0.0)
    
    def _analyze_pronoun_references(self, sentences: List[str]) -> float:
        """Analyze pronoun reference coherence"""
        if len(sentences) < 2:
            return 1.0
        
        pronouns = self.coherence_indicators['pronoun_references']
        problematic_references = 0
        total_pronouns = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = sentence.lower().split()
            sentence_pronouns = [word for word in sentence_words if word in pronouns]
            total_pronouns += len(sentence_pronouns)
            
            # Simple check: if sentence starts with pronoun but is first sentence
            if i == 0 and sentence_pronouns and sentence_words[0] in pronouns:
                problematic_references += 1
        
        if total_pronouns == 0:
            return 1.0
        
        return max(1.0 - (problematic_references / total_pronouns), 0.0)
    
    def _detect_logical_contradictions(self, text: str) -> float:
        """Detect logical contradictions in text"""
        contradiction_patterns = [
            (r'\b(yes|true|correct)\b.*\b(no|false|incorrect)\b', 0.3),
            (r'\b(always|never)\b.*\b(sometimes|occasionally)\b', 0.2),
            (r'\b(all|every)\b.*\b(some|few|none)\b', 0.2),
            (r'\b(impossible|cannot)\b.*\b(possible|can)\b', 0.25)
        ]
        
        contradiction_score = 1.0
        
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, text.lower()):
                contradiction_score -= penalty
        
        return max(contradiction_score, 0.0)
    
    def _analyze_topic_consistency(self, sentences: List[str]) -> float:
        """Analyze topic consistency across sentences"""
        if len(sentences) < 3:
            return 1.0
        
        # Simple keyword-based consistency check
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
            all_words.extend(words)
        
        if not all_words:
            return 0.5
        
        # Calculate word frequency
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find common words (appearing in multiple sentences)
        common_words = [word for word, freq in word_freq.items() if freq >= 2]
        
        consistency_ratio = len(common_words) / len(set(all_words))
        return min(consistency_ratio * 2, 1.0)  # Scale up for better scoring
    
    def _analyze_content_completeness(self, text: str, expected_format: Dict[str, Any]) -> float:
        """Analyze content completeness"""
        expected_elements = expected_format.get('required_elements', [])
        if not expected_elements:
            return 1.0  # No specific requirements
        
        found_elements = 0
        for element in expected_elements:
            if element.lower() in text.lower():
                found_elements += 1
        
        return found_elements / len(expected_elements)
    
    def _detect_error_indicators(self, text: str) -> float:
        """Detect error indicators in content"""
        error_patterns = self.quality_patterns['content_issues']
        error_score = 1.0
        
        for pattern_info in error_patterns:
            matches = re.findall(pattern_info['pattern'], text, re.IGNORECASE)
            if matches:
                error_score -= pattern_info['score_impact']
        
        return max(error_score, 0.0)
    
    def _analyze_content_density(self, text: str) -> float:
        """Analyze information density"""
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        # Count information-bearing words (nouns, verbs, adjectives)
        info_words = len(re.findall(r'\b[a-zA-Z]{4,}\b', text))
        
        # Count filler words
        filler_words = ['very', 'really', 'quite', 'somewhat', 'rather', 
                       'pretty', 'fairly', 'just', 'simply', 'basically']
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        
        density_ratio = (info_words - filler_count) / len(words)
        return min(max(density_ratio, 0.0), 1.0)
    
    def _analyze_content_specificity(self, text: str) -> float:
        """Analyze content specificity vs vagueness"""
        vague_indicators = [
            r'\b(thing|stuff|something|anything|everything)\b',
            r'\b(some|many|several|various|different)\s+\w+\b',
            r'\b(might|could|may|perhaps|possibly)\b',
            r'\b(generally|usually|often|sometimes)\b'
        ]
        
        vague_count = 0
        for pattern in vague_indicators:
            matches = re.findall(pattern, text.lower())
            vague_count += len(matches)
        
        words = len(text.split())
        if words == 0:
            return 0.0
        
        vague_ratio = vague_count / words
        return max(1.0 - vague_ratio * 2, 0.0)  # Penalize vagueness
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _find_repetitive_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Find repetitive phrases in text"""
        # Find phrases that appear multiple times
        phrases = []
        words = text.split()
        
        # Check for repeated 3-word phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3]).lower()
            
            # Count occurrences
            count = text.lower().count(phrase)
            if count > 1:
                phrases.append({
                    'phrase': phrase,
                    'count': count,
                    'length': 3
                })
        
        # Remove duplicates and return top repetitive phrases
        unique_phrases = {}
        for phrase_info in phrases:
            phrase = phrase_info['phrase']
            if phrase not in unique_phrases or phrase_info['count'] > unique_phrases[phrase]['count']:
                unique_phrases[phrase] = phrase_info
        
        return sorted(unique_phrases.values(), key=lambda x: x['count'], reverse=True)[:5]
    
    def _determine_quality_status(self, score: float, issues: List[QualityIssue]) -> str:
        """Determine overall quality status"""
        critical_issues = sum(1 for issue in issues if issue.severity == 'high')
        
        if critical_issues > 2:
            return 'poor_quality'
        elif score >= 0.9:
            return 'excellent_quality'
        elif score >= 0.8:
            return 'good_quality'
        elif score >= 0.6:
            return 'acceptable_quality'
        elif score >= 0.4:
            return 'below_average_quality'
        else:
            return 'poor_quality'
    
    def _generate_quality_recommendations(self, structure_result: Dict[str, Any],
                                       coherence_result: Dict[str, Any],
                                       relevance_result: Dict[str, Any],
                                       quality_issues: List[QualityIssue]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if structure_result['score'] < 0.7:
            recommendations.append("Improve output structure and formatting")
            if structure_result.get('format_compliance', 1.0) < 0.8:
                recommendations.append("Fix format compliance issues")
        
        if coherence_result['score'] < 0.7:
            recommendations.append("Improve logical coherence and flow")
            if coherence_result.get('transition_score', 1.0) < 0.5:
                recommendations.append("Add more transition words between sentences")
        
        if relevance_result['score'] < 0.7:
            recommendations.append("Improve content relevance and specificity")
            if relevance_result.get('specificity_score', 1.0) < 0.5:
                recommendations.append("Reduce vague language and add specific details")
        
        # Add specific recommendations for high-severity issues
        high_severity_issues = [issue for issue in quality_issues if issue.severity == 'high']
        for issue in high_severity_issues[:3]:  # Top 3
            recommendations.append(issue.suggested_fix)
        
        return recommendations or ["Maintain current quality level"]
