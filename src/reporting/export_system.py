"""
Export System for Supervisor Agent Reporting
Handles multiple export formats and external system integration
"""

import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import zipfile
import tempfile
import io


@dataclass
class ExportJob:
    id: str
    export_type: str
    format: str
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    output_path: Optional[str]
    file_size: Optional[int]
    error_message: Optional[str]


class ExportManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Export directory
        self.export_dir = Path(config.get('export_directory', 'exports'))
        self.export_dir.mkdir(exist_ok=True, parents=True)
        
        # Job tracking
        self.jobs: Dict[str, ExportJob] = {}
        self.max_concurrent_jobs = config.get('max_concurrent_jobs', 3)
        
        # Data source references
        self.data_sources = {}
        
    def set_data_sources(self, **sources):
        """Set references to data sources"""
        self.data_sources.update(sources)
        
    def export_audit_logs(self, format: str = 'jsonl', 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         agent_ids: Optional[List[str]] = None,
                         compress: bool = False) -> str:
        """Export audit logs in specified format"""
        
        job_id = self._generate_job_id('audit_logs', format)
        
        job = ExportJob(
            id=job_id,
            export_type='audit_logs',
            format=format,
            parameters={
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None,
                'agent_ids': agent_ids,
                'compress': compress
            },
            status='pending',
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            output_path=None,
            file_size=None,
            error_message=None
        )
        
        self.jobs[job_id] = job
        
        try:
            job.started_at = datetime.now()
            job.status = 'running'
            
            # Get audit trail manager
            audit_manager = self.data_sources.get('audit_manager')
            if not audit_manager:
                raise ValueError("Audit manager not available")
            
            # Query events
            events = audit_manager.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=100000
            )
            
            # Filter by agent IDs if specified
            if agent_ids:
                events = [e for e in events if e.agent_id in agent_ids]
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"audit_logs_{timestamp}.{format}"
            if compress:
                filename += '.gz'
                
            output_path = self.export_dir / filename
            
            # Export based on format
            if format.lower() == 'jsonl':
                self._export_events_jsonl(events, output_path, compress)
            elif format.lower() == 'csv':
                self._export_events_csv(events, output_path, compress)
            elif format.lower() == 'json':
                self._export_events_json(events, output_path, compress)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            job.output_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.completed_at = datetime.now()
            job.status = 'completed'
            
            self.logger.info(f"Exported {len(events)} audit events to {output_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.completed_at = datetime.now()
            self.logger.error(f"Failed to export audit logs: {e}")
            
        return job_id
        
    def export_performance_reports(self, format: str = 'pdf',
                                 period_hours: int = 24,
                                 include_charts: bool = True) -> str:
        """Export performance reports"""
        
        job_id = self._generate_job_id('performance_reports', format)
        
        job = ExportJob(
            id=job_id,
            export_type='performance_reports',
            format=format,
            parameters={
                'period_hours': period_hours,
                'include_charts': include_charts
            },
            status='pending',
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            output_path=None,
            file_size=None,
            error_message=None
        )
        
        self.jobs[job_id] = job
        
        try:
            job.started_at = datetime.now()
            job.status = 'running'
            
            # Get report generator
            report_generator = self.data_sources.get('report_generator')
            if not report_generator:
                raise ValueError("Report generator not available")
            
            # Generate summary
            summary = report_generator.generate_period_summary(hours=period_hours)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_report_{timestamp}.{format}"
            output_path = self.export_dir / filename
            
            # Export based on format
            if format.lower() == 'markdown':
                content = report_generator.generate_markdown_report(summary)
                with open(output_path, 'w') as f:
                    f.write(content)
            elif format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(asdict(summary), f, indent=2, default=str)
            elif format.lower() == 'pdf':
                # Would integrate with PDF generation library
                self._export_report_pdf(summary, output_path, include_charts)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            job.output_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.completed_at = datetime.now()
            job.status = 'completed'
            
            self.logger.info(f"Exported performance report to {output_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.completed_at = datetime.now()
            self.logger.error(f"Failed to export performance report: {e}")
            
        return job_id
        
    def export_confidence_analysis(self, format: str = 'json',
                                 period_hours: int = 24,
                                 include_calibration: bool = True) -> str:
        """Export confidence analysis data"""
        
        job_id = self._generate_job_id('confidence_analysis', format)
        
        job = ExportJob(
            id=job_id,
            export_type='confidence_analysis',
            format=format,
            parameters={
                'period_hours': period_hours,
                'include_calibration': include_calibration
            },
            status='pending',
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            output_path=None,
            file_size=None,
            error_message=None
        )
        
        self.jobs[job_id] = job
        
        try:
            job.started_at = datetime.now()
            job.status = 'running'
            
            # Get confidence reporter
            confidence_reporter = self.data_sources.get('confidence_reporter')
            if not confidence_reporter:
                raise ValueError("Confidence reporter not available")
            
            # Generate metrics
            metrics = confidence_reporter.generate_metrics(hours=period_hours)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"confidence_analysis_{timestamp}.{format}"
            output_path = self.export_dir / filename
            
            # Export based on format
            if format.lower() == 'json':
                data = asdict(metrics)
                # Convert datetime objects to strings
                data['timestamp'] = data['timestamp'].isoformat()
                data['period_start'] = data['period_start'].isoformat()
                data['period_end'] = data['period_end'].isoformat()
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == 'markdown':
                content = confidence_reporter.generate_calibration_report(metrics)
                with open(output_path, 'w') as f:
                    f.write(content)
            elif format.lower() == 'csv':
                self._export_confidence_csv(metrics, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            job.output_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.completed_at = datetime.now()
            job.status = 'completed'
            
            self.logger.info(f"Exported confidence analysis to {output_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.completed_at = datetime.now()
            self.logger.error(f"Failed to export confidence analysis: {e}")
            
        return job_id
        
    def export_patterns_knowledge(self, format: str = 'json',
                                include_examples: bool = True) -> str:
        """Export patterns and knowledge base"""
        
        job_id = self._generate_job_id('patterns_knowledge', format)
        
        job = ExportJob(
            id=job_id,
            export_type='patterns_knowledge',
            format=format,
            parameters={
                'include_examples': include_examples
            },
            status='pending',
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            output_path=None,
            file_size=None,
            error_message=None
        )
        
        self.jobs[job_id] = job
        
        try:
            job.started_at = datetime.now()
            job.status = 'running'
            
            # Get pattern tracker
            pattern_tracker = self.data_sources.get('pattern_tracker')
            if not pattern_tracker:
                raise ValueError("Pattern tracker not available")
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'zip':
                # Create zip with multiple files
                filename = f"patterns_knowledge_{timestamp}.zip"
                output_path = self.export_dir / filename
                
                with zipfile.ZipFile(output_path, 'w') as zf:
                    # Export patterns
                    patterns_data = {
                        'patterns': [asdict(p) for p in pattern_tracker.patterns.values()],
                        'export_timestamp': datetime.now().isoformat()
                    }
                    zf.writestr('patterns.json', json.dumps(patterns_data, indent=2, default=str))
                    
                    # Export knowledge base
                    kb_data = {
                        'knowledge_entries': [asdict(k) for k in pattern_tracker.knowledge_base.values()],
                        'export_timestamp': datetime.now().isoformat()
                    }
                    zf.writestr('knowledge_base.json', json.dumps(kb_data, indent=2, default=str))
                    
            else:
                filename = f"patterns_knowledge_{timestamp}.{format}"
                output_path = self.export_dir / filename
                
                # Combined export
                data = {
                    'patterns': [asdict(p) for p in pattern_tracker.patterns.values()],
                    'knowledge_base': [asdict(k) for k in pattern_tracker.knowledge_base.values()],
                    'export_timestamp': datetime.now().isoformat()
                }
                
                if format.lower() == 'json':
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            job.output_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.completed_at = datetime.now()
            job.status = 'completed'
            
            self.logger.info(f"Exported patterns and knowledge to {output_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.completed_at = datetime.now()
            self.logger.error(f"Failed to export patterns and knowledge: {e}")
            
        return job_id
        
    def export_complete_backup(self, compress: bool = True) -> str:
        """Export complete system backup"""
        
        job_id = self._generate_job_id('complete_backup', 'zip')
        
        job = ExportJob(
            id=job_id,
            export_type='complete_backup',
            format='zip',
            parameters={'compress': compress},
            status='pending',
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            output_path=None,
            file_size=None,
            error_message=None
        )
        
        self.jobs[job_id] = job
        
        try:
            job.started_at = datetime.now()
            job.status = 'running'
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"supervisor_backup_{timestamp}.zip"
            output_path = self.export_dir / filename
            
            with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as zf:
                
                # Export audit logs
                audit_job_id = self.export_audit_logs(format='jsonl', compress=False)
                audit_job = self.jobs[audit_job_id]
                if audit_job.status == 'completed' and audit_job.output_path:
                    zf.write(audit_job.output_path, 'audit_logs.jsonl')
                    
                # Export performance reports
                perf_job_id = self.export_performance_reports(format='json')
                perf_job = self.jobs[perf_job_id]
                if perf_job.status == 'completed' and perf_job.output_path:
                    zf.write(perf_job.output_path, 'performance_report.json')
                    
                # Export confidence analysis
                conf_job_id = self.export_confidence_analysis(format='json')
                conf_job = self.jobs[conf_job_id]
                if conf_job.status == 'completed' and conf_job.output_path:
                    zf.write(conf_job.output_path, 'confidence_analysis.json')
                    
                # Export patterns and knowledge
                pattern_job_id = self.export_patterns_knowledge(format='json')
                pattern_job = self.jobs[pattern_job_id]
                if pattern_job.status == 'completed' and pattern_job.output_path:
                    zf.write(pattern_job.output_path, 'patterns_knowledge.json')
                    
                # Add backup metadata
                metadata = {
                    'backup_timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'components': ['audit_logs', 'performance_reports', 'confidence_analysis', 'patterns_knowledge'],
                    'job_ids': [audit_job_id, perf_job_id, conf_job_id, pattern_job_id]
                }
                zf.writestr('backup_metadata.json', json.dumps(metadata, indent=2))
            
            job.output_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.completed_at = datetime.now()
            job.status = 'completed'
            
            self.logger.info(f"Created complete backup at {output_path}")
            
        except Exception as e:
            job.error_message = str(e)
            job.status = 'failed'
            job.completed_at = datetime.now()
            self.logger.error(f"Failed to create complete backup: {e}")
            
        return job_id
        
    def _export_events_jsonl(self, events, output_path, compress=False):
        """Export events as JSONL"""
        import gzip
        
        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'
        
        with open_func(output_path, mode) as f:
            for event in events:
                event_data = asdict(event)
                event_data['timestamp'] = event.timestamp.isoformat()
                event_data['event_type'] = event.event_type.value
                event_data['level'] = event.level.value
                f.write(json.dumps(event_data) + '\n')
                
    def _export_events_csv(self, events, output_path, compress=False):
        """Export events as CSV"""
        import gzip
        
        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'
        
        with open_func(output_path, mode, newline='') as f:
            if events:
                fieldnames = ['id', 'timestamp', 'event_type', 'level', 'agent_id', 'task_id',
                            'cause', 'action', 'outcome', 'confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in events:
                    writer.writerow({
                        'id': event.id,
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type.value,
                        'level': event.level.value,
                        'agent_id': event.agent_id,
                        'task_id': event.task_id,
                        'cause': event.cause,
                        'action': event.action,
                        'outcome': event.outcome,
                        'confidence': event.confidence
                    })
                    
    def _export_events_json(self, events, output_path, compress=False):
        """Export events as JSON"""
        import gzip
        
        events_data = []
        for event in events:
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            event_data['event_type'] = event.event_type.value
            event_data['level'] = event.level.value
            events_data.append(event_data)
            
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_events': len(events),
            'events': events_data
        }
        
        if compress:
            with gzip.open(output_path, 'wt') as f:
                json.dump(data, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
    def _export_report_pdf(self, summary, output_path, include_charts=True):
        """Export report as PDF (placeholder)"""
        # This would integrate with a PDF generation library like reportlab
        # For now, just create a text file with the report content
        text_path = str(output_path).replace('.pdf', '.txt')
        
        content = f"""
SUPERVISOR AGENT PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Period: {summary.period_start} - {summary.period_end}

SUMMARY:
- Total Agents: {summary.total_agents}
- Total Tasks: {summary.total_tasks}
- Success Rate: {summary.success_rate:.1f}%
- Average Confidence: {summary.avg_confidence:.3f}
- Average Task Duration: {summary.avg_task_duration:.2f}s
- Error Rate: {summary.error_rate:.2f}%

TOP PERFORMING AGENTS:
{chr(10).join(f'- {agent}' for agent in summary.top_performing_agents)}

PROBLEMATIC PATTERNS:
{chr(10).join(f'- {pattern}' for pattern in summary.problematic_patterns)}

TRENDS:
{chr(10).join(f'- {metric}: {data["direction"]} ({data["change"]:+.2f})' for metric, data in summary.trends.items())}
"""
        
        with open(text_path, 'w') as f:
            f.write(content)
            
        # Rename to PDF extension (placeholder)
        Path(text_path).rename(output_path)
        
    def _export_confidence_csv(self, metrics, output_path):
        """Export confidence metrics as CSV"""
        with open(output_path, 'w', newline='') as f:
            # Summary metrics
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Decisions', metrics.total_decisions])
            writer.writerow(['Average Confidence', f'{metrics.avg_confidence:.3f}'])
            writer.writerow(['Accuracy', f'{metrics.accuracy:.3f}'])
            writer.writerow(['Calibration Score', f'{metrics.calibration_score:.3f}'])
            writer.writerow(['Overconfidence Rate', f'{metrics.overconfidence_rate:.3f}'])
            writer.writerow(['Underconfidence Rate', f'{metrics.underconfidence_rate:.3f}'])
            writer.writerow(['Reliability', f'{metrics.reliability:.3f}'])
            
            # Calibration bins
            writer.writerow([])
            writer.writerow(['Calibration Bins'])
            writer.writerow(['Range', 'Count', 'Avg Confidence', 'Accuracy', 'Calibration Error'])
            
            for bin_obj in metrics.bins:
                if bin_obj.count > 0:
                    range_str = f"{bin_obj.bin_range[0]:.1f}-{bin_obj.bin_range[1]:.1f}"
                    writer.writerow([
                        range_str, bin_obj.count, f'{bin_obj.avg_confidence:.3f}',
                        f'{bin_obj.accuracy:.3f}', f'{bin_obj.calibration_error:.3f}'
                    ])
                    
    def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get status of an export job"""
        return self.jobs.get(job_id)
        
    def list_jobs(self, status_filter: Optional[str] = None) -> List[ExportJob]:
        """List export jobs with optional status filter"""
        jobs = list(self.jobs.values())
        
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
            
        return sorted(jobs, key=lambda x: x.created_at, reverse=True)
        
    def cleanup_old_jobs(self, days: int = 7):
        """Clean up old completed jobs and their files"""
        cutoff = datetime.now() - timedelta(days=days)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.completed_at and job.completed_at < cutoff:
                # Remove output file if exists
                if job.output_path and Path(job.output_path).exists():
                    try:
                        Path(job.output_path).unlink()
                        self.logger.info(f"Deleted old export file: {job.output_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete {job.output_path}: {e}")
                        
                jobs_to_remove.append(job_id)
                
        # Remove job records
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            
        self.logger.info(f"Cleaned up {len(jobs_to_remove)} old export jobs")
        
    def _generate_job_id(self, export_type: str, format: str) -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{export_type}_{format}_{timestamp}"
        
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export system statistics"""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == 'completed'])
        failed_jobs = len([j for j in self.jobs.values() if j.status == 'failed'])
        
        total_size = sum(
            j.file_size for j in self.jobs.values() 
            if j.file_size and j.status == 'completed'
        )
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': len([j for j in self.jobs.values() if j.status == 'running']),
            'success_rate': completed_jobs / total_jobs * 100 if total_jobs > 0 else 0,
            'total_exported_size': total_size,
            'export_directory': str(self.export_dir),
            'max_concurrent_jobs': self.max_concurrent_jobs
        }
