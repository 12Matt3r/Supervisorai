"""Instruction Adherence Monitoring - Cross-checks agent steps against user instructions"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ConstraintViolation:
    """Represents a constraint violation"""
    constraint_type: str
    violation_description: str
    severity: str
    location: str
    suggested_fix: str

class InstructionAdherenceMonitor:
    """Monitors adherence to user instructions and constraints"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.constraint_patterns = self._initialize_constraint_patterns()
        self.format_validators = self._initialize_format_validators()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'strict_mode': True,
            'format_weight': 0.4,
            'procedure_weight': 0.3,
            'constraint_weight': 0.3,
            'violation_threshold': 0.2
        }
    
    def _initialize_constraint_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for detecting constraint violations"""
        return [
            {
                'type': 'format_constraint',
                'patterns': [
                    r'(?i)\bmust be\s+(json|xml|yaml|markdown|csv)\b',
                    r'(?i)\bformat\s*:\s*(json|xml|yaml|markdown|csv)\b',
                    r'(?i)\breturn\s+(json|xml|yaml|markdown|csv)\b'
                ],
                'severity': 'high'
            },
            {
                'type': 'tone_constraint',
                'patterns': [
                    r'(?i)\b(formal|informal|professional|casual|technical)\s+tone\b',
                    r'(?i)\bwrite\s+in\s+(formal|informal|professional|casual|technical)\s+style\b'
                ],
                'severity': 'medium'
            },
            {
                'type': 'length_constraint',
                'patterns': [
                    r'(?i)\b(maximum|minimum|exactly)\s+(\d+)\s+(words|characters|lines)\b',
                    r'(?i)\bno more than\s+(\d+)\s+(words|characters|lines)\b',
                    r'(?i)\bat least\s+(\d+)\s+(words|characters|lines)\b'
                ],
                'severity': 'medium'
            },
            {
                'type': 'content_constraint',
                'patterns': [
                    r'(?i)\bmust include\s+([^.!?]+)\b',
                    r'(?i)\bshould contain\s+([^.!?]+)\b',
                    r'(?i)\brequired\s*:\s*([^.!?]+)\b',
                    r'(?i)\bavoid\s+([^.!?]+)\b',
                    r'(?i)\bdo not\s+([^.!?]+)\b'
                ],
                'severity': 'high'
            },
            {
                'type': 'procedure_constraint',
                'patterns': [
                    r'(?i)\bstep\s+(\d+)\s*:\s*([^.!?]+)\b',
                    r'(?i)\bfirst\s+([^.!?]+)\b',
                    r'(?i)\bthen\s+([^.!?]+)\b',
                    r'(?i)\bfinally\s+([^.!?]+)\b',
                    r'(?i)\bbefore\s+([^.!?]+)\b',
                    r'(?i)\bafter\s+([^.!?]+)\b'
                ],
                'severity': 'medium'
            }
        ]
    
    def _initialize_format_validators(self) -> Dict[str, callable]:
        """Initialize format validation functions"""
        return {
            'json': self._validate_json_format,
            'xml': self._validate_xml_format,
            'yaml': self._validate_yaml_format,
            'markdown': self._validate_markdown_format,
            'csv': self._validate_csv_format
        }
    
    def evaluate_adherence(self, instructions: List[str], 
                          agent_steps: List[str],
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate adherence to instructions and constraints"""
        try:
            # Extract constraints from instructions
            extracted_constraints = self._extract_constraints(instructions)
            all_constraints = {**extracted_constraints, **constraints}
            
            # Validate format adherence
            format_result = self._validate_format_adherence(
                agent_steps, all_constraints
            )
            
            # Validate procedure adherence
            procedure_result = self._validate_procedure_adherence(
                instructions, agent_steps
            )
            
            # Detect constraint violations
            violations = self._detect_constraint_violations(
                agent_steps, all_constraints
            )
            
            # Calculate adherence scores
            format_score = format_result['score']
            procedure_score = procedure_result['score']
            constraint_score = self._calculate_constraint_score(violations)
            
            # Calculate overall adherence score
            overall_score = (
                format_score * self.config['format_weight'] +
                procedure_score * self.config['procedure_weight'] +
                constraint_score * self.config['constraint_weight']
            )
            
            # Analyze instruction following patterns
            following_patterns = self._analyze_instruction_following(
                instructions, agent_steps
            )
            
            return {
                'score': overall_score,
                'format_adherence': format_result,
                'procedure_adherence': procedure_result,
                'constraint_violations': violations,
                'constraint_score': constraint_score,
                'instruction_following_patterns': following_patterns,
                'adherence_status': self._determine_adherence_status(
                    overall_score, violations
                ),
                'recommendations': self._generate_adherence_recommendations(
                    format_result, procedure_result, violations
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
    
    def _extract_constraints(self, instructions: List[str]) -> Dict[str, Any]:
        """Extract constraints from instruction text"""
        constraints = {}
        full_text = ' '.join(instructions)
        
        for constraint_group in self.constraint_patterns:
            constraint_type = constraint_group['type']
            constraints[constraint_type] = []
            
            for pattern in constraint_group['patterns']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    constraints[constraint_type].append({
                        'text': match.group(0),
                        'groups': match.groups(),
                        'severity': constraint_group['severity'],
                        'position': match.span()
                    })
        
        return constraints
    
    def _validate_format_adherence(self, agent_steps: List[str], 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate format adherence in agent outputs"""
        format_constraints = constraints.get('format_constraint', [])
        
        if not format_constraints:
            return {'score': 1.0, 'violations': [], 'status': 'no_format_constraints'}
        
        violations = []
        total_score = 0.0
        
        for constraint in format_constraints:
            required_format = None
            for group in constraint['groups']:
                if group and group.lower() in self.format_validators:
                    required_format = group.lower()
                    break
            
            if not required_format:
                continue
            
            # Check each agent step for format compliance
            step_scores = []
            for i, step in enumerate(agent_steps):
                validator = self.format_validators[required_format]
                validation_result = validator(step)
                step_scores.append(validation_result['score'])
                
                if not validation_result['valid']:
                    violations.append({
                        'type': 'format_violation',
                        'step_index': i,
                        'required_format': required_format,
                        'error': validation_result['error'],
                        'severity': constraint['severity']
                    })
            
            if step_scores:
                total_score += sum(step_scores) / len(step_scores)
        
        final_score = total_score / len(format_constraints) if format_constraints else 1.0
        
        return {
            'score': min(final_score, 1.0),
            'violations': violations,
            'constraints_checked': len(format_constraints),
            'status': 'validated'
        }
    
    def _validate_procedure_adherence(self, instructions: List[str], 
                                    agent_steps: List[str]) -> Dict[str, Any]:
        """Validate adherence to procedural instructions"""
        procedure_constraints = []
        
        # Extract procedural steps from instructions
        for instruction in instructions:
            steps = self._extract_procedural_steps(instruction)
            procedure_constraints.extend(steps)
        
        if not procedure_constraints:
            return {'score': 1.0, 'violations': [], 'status': 'no_procedure_constraints'}
        
        # Check if agent steps follow the procedure
        violations = []
        step_matches = 0
        
        for i, proc_step in enumerate(procedure_constraints):
            matched = False
            
            # Look for keywords from procedural step in agent steps
            proc_keywords = self._extract_keywords(proc_step['content'])
            
            for j, agent_step in enumerate(agent_steps):
                agent_keywords = self._extract_keywords(agent_step.lower())
                
                # Check keyword overlap
                overlap = len(set(proc_keywords).intersection(set(agent_keywords)))
                if overlap >= len(proc_keywords) * 0.3:  # 30% keyword overlap
                    matched = True
                    step_matches += 1
                    break
            
            if not matched:
                violations.append({
                    'type': 'procedure_violation',
                    'step_index': i,
                    'procedure_step': proc_step['content'],
                    'step_type': proc_step['type'],
                    'severity': 'medium'
                })
        
        score = step_matches / len(procedure_constraints) if procedure_constraints else 1.0
        
        return {
            'score': score,
            'violations': violations,
            'steps_matched': step_matches,
            'total_procedure_steps': len(procedure_constraints),
            'status': 'validated'
        }
    
    def _detect_constraint_violations(self, agent_steps: List[str], 
                                    constraints: Dict[str, Any]) -> List[ConstraintViolation]:
        """Detect violations of content and other constraints"""
        violations = []
        agent_content = ' '.join(agent_steps).lower()
        
        # Check content constraints
        content_constraints = constraints.get('content_constraint', [])
        for constraint in content_constraints:
            constraint_text = constraint['text'].lower()
            
            if 'must include' in constraint_text or 'should contain' in constraint_text:
                required_content = constraint['groups'][0] if constraint['groups'] else ''
                if required_content and required_content.lower() not in agent_content:
                    violations.append(ConstraintViolation(
                        constraint_type='content_requirement',
                        violation_description=f"Required content missing: {required_content}",
                        severity='high',
                        location='agent_output',
                        suggested_fix=f"Include '{required_content}' in the output"
                    ))
            
            elif 'avoid' in constraint_text or 'do not' in constraint_text:
                forbidden_content = constraint['groups'][0] if constraint['groups'] else ''
                if forbidden_content and forbidden_content.lower() in agent_content:
                    violations.append(ConstraintViolation(
                        constraint_type='content_prohibition',
                        violation_description=f"Forbidden content present: {forbidden_content}",
                        severity='high',
                        location='agent_output',
                        suggested_fix=f"Remove '{forbidden_content}' from the output"
                    ))
        
        # Check length constraints
        length_constraints = constraints.get('length_constraint', [])
        for constraint in length_constraints:
            violation = self._check_length_constraint(agent_content, constraint)
            if violation:
                violations.append(violation)
        
        # Check tone constraints
        tone_constraints = constraints.get('tone_constraint', [])
        for constraint in tone_constraints:
            violation = self._check_tone_constraint(agent_content, constraint)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _extract_procedural_steps(self, instruction: str) -> List[Dict[str, Any]]:
        """Extract procedural steps from instruction text"""
        steps = []
        
        # Pattern for numbered steps
        numbered_steps = re.finditer(r'(?i)step\s+(\d+)\s*[:.]\s*([^.!?]+)', instruction)
        for match in numbered_steps:
            steps.append({
                'type': 'numbered_step',
                'number': int(match.group(1)),
                'content': match.group(2).strip()
            })
        
        # Pattern for sequence words
        sequence_patterns = [
            (r'(?i)first[,\s]+([^.!?]+)', 'first'),
            (r'(?i)then[,\s]+([^.!?]+)', 'then'),
            (r'(?i)next[,\s]+([^.!?]+)', 'next'),
            (r'(?i)finally[,\s]+([^.!?]+)', 'finally'),
            (r'(?i)after[,\s]+([^.!?]+)', 'after'),
            (r'(?i)before[,\s]+([^.!?]+)', 'before')
        ]
        
        for pattern, step_type in sequence_patterns:
            matches = re.finditer(pattern, instruction)
            for match in matches:
                steps.append({
                    'type': step_type,
                    'content': match.group(1).strip()
                })
        
        return steps
    
    def _check_length_constraint(self, content: str, constraint: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """Check length constraints"""
        constraint_text = constraint['text'].lower()
        groups = constraint['groups']
        
        if not groups or len(groups) < 2:
            return None
        
        try:
            limit = int(groups[0])
            unit = groups[1].lower()
        except (ValueError, IndexError):
            return None
        
        if unit == 'words':
            actual_count = len(content.split())
        elif unit == 'characters':
            actual_count = len(content)
        elif unit == 'lines':
            actual_count = content.count('\n') + 1
        else:
            return None
        
        if 'maximum' in constraint_text or 'no more than' in constraint_text:
            if actual_count > limit:
                return ConstraintViolation(
                    constraint_type='length_violation',
                    violation_description=f"Content exceeds {limit} {unit} limit (actual: {actual_count})",
                    severity='medium',
                    location='agent_output',
                    suggested_fix=f"Reduce content to {limit} {unit} or less"
                )
        
        elif 'minimum' in constraint_text or 'at least' in constraint_text:
            if actual_count < limit:
                return ConstraintViolation(
                    constraint_type='length_violation',
                    violation_description=f"Content below {limit} {unit} minimum (actual: {actual_count})",
                    severity='medium',
                    location='agent_output',
                    suggested_fix=f"Expand content to at least {limit} {unit}"
                )
        
        return None
    
    def _check_tone_constraint(self, content: str, constraint: Dict[str, Any]) -> Optional[ConstraintViolation]:
        """Check tone constraints (basic implementation)"""
        constraint_text = constraint['text'].lower()
        
        # Basic tone indicators
        tone_indicators = {
            'formal': ['please', 'kindly', 'respectively', 'furthermore', 'therefore'],
            'informal': ['gonna', 'wanna', 'yeah', 'ok', 'cool'],
            'professional': ['recommend', 'suggest', 'implement', 'analyze', 'optimize'],
            'technical': ['configure', 'parameter', 'function', 'algorithm', 'protocol']
        }
        
        required_tone = None
        for tone in tone_indicators.keys():
            if tone in constraint_text:
                required_tone = tone
                break
        
        if not required_tone:
            return None
        
        # Check for tone indicators in content
        indicators = tone_indicators[required_tone]
        matches = sum(1 for indicator in indicators if indicator in content.lower())
        
        if matches < 2:  # Require at least 2 tone indicators
            return ConstraintViolation(
                constraint_type='tone_violation',
                violation_description=f"Content does not match required {required_tone} tone",
                severity='medium',
                location='agent_output',
                suggested_fix=f"Adjust language to be more {required_tone}"
            )
        
        return None
    
    def _calculate_constraint_score(self, violations: List[ConstraintViolation]) -> float:
        """Calculate constraint adherence score based on violations"""
        if not violations:
            return 1.0
        
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.5}
        total_penalty = sum(severity_weights.get(v.severity, 0.3) for v in violations)
        
        # Cap penalty at 1.0
        total_penalty = min(total_penalty, 1.0)
        
        return max(1.0 - total_penalty, 0.0)
    
    def _analyze_instruction_following(self, instructions: List[str], 
                                     agent_steps: List[str]) -> Dict[str, Any]:
        """Analyze patterns of instruction following"""
        patterns = {
            'keyword_matching': 0.0,
            'structure_following': 0.0,
            'completeness': 0.0
        }
        
        if not instructions or not agent_steps:
            return patterns
        
        # Keyword matching analysis
        instruction_keywords = set()
        for instruction in instructions:
            instruction_keywords.update(self._extract_keywords(instruction.lower()))
        
        agent_keywords = set()
        for step in agent_steps:
            agent_keywords.update(self._extract_keywords(step.lower()))
        
        if instruction_keywords:
            keyword_overlap = len(instruction_keywords.intersection(agent_keywords))
            patterns['keyword_matching'] = keyword_overlap / len(instruction_keywords)
        
        # Structure following (basic)
        if any('step' in instr.lower() for instr in instructions):
            step_words = ['first', 'then', 'next', 'finally', 'step']
            agent_text = ' '.join(agent_steps).lower()
            structure_indicators = sum(1 for word in step_words if word in agent_text)
            patterns['structure_following'] = min(structure_indicators / 3.0, 1.0)
        
        # Completeness (based on instruction count vs step count)
        instruction_count = len(instructions)
        step_count = len(agent_steps)
        if instruction_count > 0:
            patterns['completeness'] = min(step_count / instruction_count, 1.0)
        
        return patterns
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [word for word in words if word not in stop_words]
    
    def _determine_adherence_status(self, score: float, violations: List[ConstraintViolation]) -> str:
        """Determine overall adherence status"""
        critical_violations = sum(1 for v in violations if v.severity == 'high')
        
        if critical_violations > 0:
            return 'critical_violations'
        elif score >= 0.9:
            return 'excellent_adherence'
        elif score >= 0.8:
            return 'good_adherence'
        elif score >= 0.6:
            return 'acceptable_adherence'
        elif score >= 0.4:
            return 'poor_adherence'
        else:
            return 'failing_adherence'
    
    def _generate_adherence_recommendations(self, format_result: Dict[str, Any],
                                         procedure_result: Dict[str, Any],
                                         violations: List[ConstraintViolation]) -> List[str]:
        """Generate adherence improvement recommendations"""
        recommendations = []
        
        if format_result['score'] < 0.8:
            recommendations.append("Improve format adherence")
            if format_result['violations']:
                recommendations.append(f"Fix {len(format_result['violations'])} format violations")
        
        if procedure_result['score'] < 0.8:
            recommendations.append("Better follow procedural instructions")
            recommendations.append("Ensure all instruction steps are addressed")
        
        if violations:
            high_priority = [v for v in violations if v.severity == 'high']
            if high_priority:
                recommendations.append(f"Immediately address {len(high_priority)} high-severity violations")
            
            for violation in violations[:3]:  # Top 3 violations
                recommendations.append(violation.suggested_fix)
        
        return recommendations or ["Maintain current adherence level"]
    
    # Format validation functions
    def _validate_json_format(self, content: str) -> Dict[str, Any]:
        """Validate JSON format"""
        try:
            json.loads(content)
            return {'valid': True, 'score': 1.0, 'error': None}
        except json.JSONDecodeError as e:
            return {'valid': False, 'score': 0.0, 'error': str(e)}
        except Exception as e:
            return {'valid': False, 'score': 0.0, 'error': f"Unknown error: {str(e)}"}
    
    def _validate_xml_format(self, content: str) -> Dict[str, Any]:
        """Validate XML format (basic)"""
        # Basic XML validation
        if '<' in content and '>' in content:
            # Check for balanced tags (very basic)
            open_tags = re.findall(r'<([^/][^>]*)>', content)
            close_tags = re.findall(r'</([^>]+)>', content)
            if len(open_tags) == len(close_tags):
                return {'valid': True, 'score': 0.8, 'error': None}
        
        return {'valid': False, 'score': 0.0, 'error': 'Invalid XML structure'}
    
    def _validate_yaml_format(self, content: str) -> Dict[str, Any]:
        """Validate YAML format (basic)"""
        # Basic YAML structure check
        if ':' in content and ('\n' in content or len(content.split(':')) > 1):
            return {'valid': True, 'score': 0.7, 'error': None}
        return {'valid': False, 'score': 0.0, 'error': 'Invalid YAML structure'}
    
    def _validate_markdown_format(self, content: str) -> Dict[str, Any]:
        """Validate Markdown format"""
        markdown_indicators = ['#', '*', '_', '```', '[', ']', '(']
        indicator_count = sum(1 for indicator in markdown_indicators if indicator in content)
        
        if indicator_count >= 2:
            return {'valid': True, 'score': 0.8, 'error': None}
        return {'valid': False, 'score': 0.3, 'error': 'Minimal markdown formatting detected'}
    
    def _validate_csv_format(self, content: str) -> Dict[str, Any]:
        """Validate CSV format"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {'valid': False, 'score': 0.0, 'error': 'Insufficient rows for CSV'}
        
        # Check if all lines have same number of fields
        field_counts = [len(line.split(',')) for line in lines]
        if len(set(field_counts)) == 1 and field_counts[0] > 1:
            return {'valid': True, 'score': 1.0, 'error': None}
        
        return {'valid': False, 'score': 0.0, 'error': 'Inconsistent CSV structure'}
