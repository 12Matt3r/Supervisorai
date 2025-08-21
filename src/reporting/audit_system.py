#!/usr/bin/env python3
"""
Machine-Readable Audit Trail System for Supervisor Agent
Handles structured logging, audit trails, and log aggregation with searchable metadata.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from queue import Queue
from enum import Enum
import sqlite3
from contextlib import contextmanager

class AuditEventType(Enum):
    """Types of audit events"""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"
    RECOVERY_ATTEMPTED = "recovery_attempted"
    ALERT_GENERATED = "alert_generated"
    SYSTEM_STATE_CHANGE = "system_state_change"
    USER_INTERACTION = "user_interaction"
    PERFORMANCE_METRIC = "performance_metric"

class AuditLevel(Enum):
    """Audit event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Structured audit event"""
    id: str
    timestamp: str
    event_type: str
    level: str
    source: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), default=str, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEvent':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

class AuditLogger:
    """Core audit logging functionality"""
    
    def __init__(self, log_file: str, db_file: Optional[str] = None):
        self.log_file = Path(log_file)
        self.db_file = Path(db_file) if db_file else None
        self.session_id = hashlib.md5(
            datetime.now().isoformat().encode()
        ).hexdigest()[:12]
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.db_file:
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()
        
        # Setup async logging
        self._setup_async_logging()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for structured logging"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    context TEXT,
                    tags TEXT,
                    correlation_id TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better search performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_level ON audit_events(level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source ON audit_events(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_events(correlation_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)"
            )

            # Create feedback log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_log (
                    feedback_id TEXT PRIMARY KEY,
                    original_event_id TEXT NOT NULL,
                    user_correction TEXT NOT NULL,
                    notes TEXT,
                    feedback_timestamp TEXT NOT NULL,
                    FOREIGN KEY (original_event_id) REFERENCES audit_events (id)
                )
            """)
    
    def _setup_async_logging(self):
        """Setup asynchronous logging to avoid blocking"""
        self.log_queue = Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
    
    def _log_worker(self):
        """Background worker for writing logs"""
        while True:
            try:
                event = self.log_queue.get()
                if event is None:  # Shutdown signal
                    break
                
                # Write to JSON log file
                with open(self.log_file, 'a') as f:
                    f.write(event.to_json() + '\n')
                
                # Write to database if enabled
                if self.db_file:
                    self._write_to_database(event)
                
            except Exception as e:
                self.logger.error(f"Error writing audit log: {e}")
    
    def _write_to_database(self, event: AuditEvent):
        """Write event to SQLite database"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_events 
                (id, timestamp, event_type, level, source, message, 
                 metadata, context, tags, correlation_id, session_id, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id, event.timestamp, event.event_type, event.level,
                    event.source, event.message,
                    json.dumps(event.metadata) if event.metadata else None,
                    json.dumps(event.context) if event.context else None,
                    json.dumps(event.tags) if event.tags else None,
                    event.correlation_id, event.session_id, event.user_id
                )
            )
    
    def log_event(self, event_type: Union[AuditEventType, str],
                  level: Union[AuditLevel, str], source: str, message: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  context: Optional[Dict[str, Any]] = None,
                  tags: Optional[List[str]] = None,
                  correlation_id: Optional[str] = None,
                  user_id: Optional[str] = None) -> str:
        """Log an audit event"""
        
        # Generate unique event ID
        event_id = hashlib.sha256(
            f"{datetime.now().isoformat()}:{source}:{message}".encode()
        ).hexdigest()[:16]
        
        # Convert enums to strings
        if isinstance(event_type, AuditEventType):
            event_type = event_type.value
        if isinstance(level, AuditLevel):
            level = level.value
        
        event = AuditEvent(
            id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            level=level,
            source=source,
            message=message,
            metadata=metadata or {},
            context=context or {},
            tags=tags or [],
            correlation_id=correlation_id,
            session_id=self.session_id,
            user_id=user_id
        )
        
        # Queue for async processing
        self.log_queue.put(event)
        
        return event_id
    
    def shutdown(self):
        """Shutdown the audit logger"""
        self.log_queue.put(None)  # Shutdown signal
        self.log_thread.join(timeout=5)

class AuditSearcher:
    """Search and query audit logs"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.logger = audit_logger
        self.db_file = audit_logger.db_file
        self.log_file = audit_logger.log_file
    
    def search_events(self, 
                     event_type: Optional[str] = None,
                     level: Optional[str] = None,
                     source: Optional[str] = None,
                     start_time: Optional[str] = None,
                     end_time: Optional[str] = None,
                     correlation_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     text_search: Optional[str] = None,
                     limit: int = 100) -> List[AuditEvent]:
        """Search audit events with multiple criteria"""
        
        if self.db_file and self.db_file.exists():
            return self._search_database(event_type, level, source, start_time,
                                       end_time, correlation_id, session_id,
                                       tags, text_search, limit)
        else:
            return self._search_file(event_type, level, source, start_time,
                                   end_time, correlation_id, session_id,
                                   tags, text_search, limit)
    
    def _search_database(self, event_type, level, source, start_time,
                        end_time, correlation_id, session_id, tags,
                        text_search, limit) -> List[AuditEvent]:
        """Search using SQLite database"""
        query_parts = ["SELECT * FROM audit_events WHERE 1=1"]
        params = []
        
        if event_type:
            query_parts.append("AND event_type = ?")
            params.append(event_type)
        
        if level:
            query_parts.append("AND level = ?")
            params.append(level)
        
        if source:
            query_parts.append("AND source = ?")
            params.append(source)
        
        if start_time:
            query_parts.append("AND timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            query_parts.append("AND timestamp <= ?")
            params.append(end_time)
        
        if correlation_id:
            query_parts.append("AND correlation_id = ?")
            params.append(correlation_id)
        
        if session_id:
            query_parts.append("AND session_id = ?")
            params.append(session_id)
        
        if text_search:
            query_parts.append("AND (message LIKE ? OR metadata LIKE ?)")
            params.extend([f"%{text_search}%", f"%{text_search}%"])
        
        query_parts.append("ORDER BY timestamp DESC LIMIT ?")
        params.append(limit)
        
        query = " ".join(query_parts)
        
        events = []
        with sqlite3.connect(self.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                event_data = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'event_type': row['event_type'],
                    'level': row['level'],
                    'source': row['source'],
                    'message': row['message'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'context': json.loads(row['context']) if row['context'] else {},
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'correlation_id': row['correlation_id'],
                    'session_id': row['session_id'],
                    'user_id': row['user_id']
                }
                
                # Filter by tags if specified
                if tags:
                    event_tags = set(event_data['tags'])
                    if not set(tags).intersection(event_tags):
                        continue
                
                events.append(AuditEvent(**event_data))
        
        return events
    
    def _search_file(self, event_type, level, source, start_time,
                    end_time, correlation_id, session_id, tags,
                    text_search, limit) -> List[AuditEvent]:
        """Search using JSON log file (fallback)"""
        events = []
        
        if not self.log_file.exists():
            return events
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    event = AuditEvent.from_json(line.strip())
                    
                    # Apply filters
                    if event_type and event.event_type != event_type:
                        continue
                    if level and event.level != level:
                        continue
                    if source and event.source != source:
                        continue
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    if correlation_id and event.correlation_id != correlation_id:
                        continue
                    if session_id and event.session_id != session_id:
                        continue
                    if tags and not set(tags).intersection(set(event.tags)):
                        continue
                    if text_search:
                        if (text_search.lower() not in event.message.lower() and
                            text_search.lower() not in json.dumps(event.metadata).lower()):
                            continue
                    
                    events.append(event)
                    
                    if len(events) >= limit:
                        break
                        
                except Exception as e:
                    continue  # Skip malformed lines
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_event_statistics(self, start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about audit events"""
        events = self.search_events(start_time=start_time, end_time=end_time, limit=10000)
        
        stats = {
            'total_events': len(events),
            'by_type': {},
            'by_level': {},
            'by_source': {},
            'by_hour': {},
            'sessions': set(),
            'correlations': set()
        }
        
        for event in events:
            # Count by type
            stats['by_type'][event.event_type] = stats['by_type'].get(event.event_type, 0) + 1
            
            # Count by level
            stats['by_level'][event.level] = stats['by_level'].get(event.level, 0) + 1
            
            # Count by source
            stats['by_source'][event.source] = stats['by_source'].get(event.source, 0) + 1
            
            # Count by hour
            hour_key = event.timestamp[:13]  # YYYY-MM-DDTHH
            stats['by_hour'][hour_key] = stats['by_hour'].get(hour_key, 0) + 1
            
            # Track sessions and correlations
            if event.session_id:
                stats['sessions'].add(event.session_id)
            if event.correlation_id:
                stats['correlations'].add(event.correlation_id)
        
        # Convert sets to counts
        stats['unique_sessions'] = len(stats['sessions'])
        stats['unique_correlations'] = len(stats['correlations'])
        del stats['sessions']
        del stats['correlations']
        
        return stats

class CorrelationTracker:
    """Tracks event correlations and chains"""
    
    def __init__(self, searcher: AuditSearcher):
        self.searcher = searcher
        self._correlation_cache = {}
    
    def create_correlation_id(self, prefix: str = "corr") -> str:
        """Create a new correlation ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_part = hashlib.md5(f"{timestamp}:{prefix}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{unique_part}"
    
    def get_correlated_events(self, correlation_id: str) -> List[AuditEvent]:
        """Get all events with the same correlation ID"""
        if correlation_id in self._correlation_cache:
            return self._correlation_cache[correlation_id]
        
        events = self.searcher.search_events(correlation_id=correlation_id, limit=1000)
        events.sort(key=lambda e: e.timestamp)
        
        self._correlation_cache[correlation_id] = events
        return events
    
    def trace_event_chain(self, correlation_id: str) -> Dict[str, Any]:
        """Trace the complete chain of correlated events"""
        events = self.get_correlated_events(correlation_id)
        
        if not events:
            return {'correlation_id': correlation_id, 'events': [], 'summary': {}}
        
        # Analyze the event chain
        summary = {
            'start_time': events[0].timestamp,
            'end_time': events[-1].timestamp,
            'duration_seconds': self._calculate_duration(events[0].timestamp, events[-1].timestamp),
            'event_count': len(events),
            'sources_involved': list(set(e.source for e in events)),
            'event_types': list(set(e.event_type for e in events)),
            'levels': list(set(e.level for e in events)),
            'errors': len([e for e in events if e.level in ['error', 'critical']]),
            'warnings': len([e for e in events if e.level == 'warning'])
        }
        
        return {
            'correlation_id': correlation_id,
            'events': events,
            'summary': summary
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between timestamps in seconds"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return (end - start).total_seconds()
        except:
            return 0.0

class ComprehensiveAuditSystem:
    """Main audit system combining all components"""
    
    def __init__(self, log_file: str = "audit.jsonl",
                 db_file: Optional[str] = "audit.db"):
        self.logger = AuditLogger(log_file, db_file)
        self.searcher = AuditSearcher(self.logger)
        self.correlation_tracker = CorrelationTracker(self.searcher)
        
    def log(self, event_type: Union[AuditEventType, str],
            level: Union[AuditLevel, str], source: str, message: str,
            **kwargs) -> str:
        """Main logging interface"""
        return self.logger.log_event(event_type, level, source, message, **kwargs)
    
    def search(self, **kwargs) -> List[AuditEvent]:
        """Search interface"""
        return self.searcher.search_events(**kwargs)
    
    def get_stats(self, **kwargs) -> Dict[str, Any]:
        """Statistics interface"""
        return self.searcher.get_event_statistics(**kwargs)
    
    def create_correlation(self, prefix: str = "corr") -> str:
        """Create correlation ID"""
        return self.correlation_tracker.create_correlation_id(prefix)
    
    def trace_correlation(self, correlation_id: str) -> Dict[str, Any]:
        """Trace correlation chain"""
        return self.correlation_tracker.trace_event_chain(correlation_id)
    
    def export_events(self, output_file: str, format: str = "json",
                     **search_kwargs) -> str:
        """Export events to file"""
        events = self.search(**search_kwargs)
        output_path = Path(output_file)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([asdict(event) for event in events], f, indent=2, default=str)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if events:
                    fieldnames = list(asdict(events[0]).keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for event in events:
                        row = asdict(event)
                        # Convert complex fields to JSON strings
                        for key in ['metadata', 'context', 'tags']:
                            if row[key]:
                                row[key] = json.dumps(row[key])
                        writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)
    
    def log_feedback(self, original_event_id: str, user_correction: str, notes: Optional[str] = None):
        """Logs user feedback for a specific decision event."""
        feedback_id = f"fb_{original_event_id}"
        feedback_timestamp = datetime.now().isoformat()

        if self.logger.db_file:
            with sqlite3.connect(self.logger.db_file) as conn:
                conn.execute(
                    "INSERT INTO feedback_log (feedback_id, original_event_id, user_correction, notes, feedback_timestamp) VALUES (?, ?, ?, ?, ?)",
                    (feedback_id, original_event_id, user_correction, notes, feedback_timestamp)
                )

    def get_feedback_logs(self, limit: int = 100) -> List[Dict]:
        """Retrieves recent feedback logs."""
        if not self.logger.db_file:
            return []

        with sqlite3.connect(self.logger.db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM feedback_log ORDER BY feedback_timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def shutdown(self):
        """Shutdown the audit system"""
        self.logger.shutdown()

# Demo and testing functions
def create_demo_audit_system() -> ComprehensiveAuditSystem:
    """Create demo audit system"""
    return ComprehensiveAuditSystem(
        "demo_audit.jsonl",
        "demo_audit.db"
    )

def generate_demo_events(audit_system: ComprehensiveAuditSystem):
    """Generate demo audit events"""
    # Create correlation for a task execution
    task_correlation = audit_system.create_correlation("task")
    
    # Task started
    audit_system.log(
        AuditEventType.TASK_STARTED,
        AuditLevel.INFO,
        "task_executor",
        "Started processing user request",
        correlation_id=task_correlation,
        metadata={"task_type": "data_analysis", "user_id": "user123"},
        tags=["task", "start"]
    )
    
    # Decision made
    audit_system.log(
        AuditEventType.DECISION_MADE,
        AuditLevel.INFO,
        "decision_engine",
        "Selected algorithm: RandomForest",
        correlation_id=task_correlation,
        metadata={"algorithm": "RandomForest", "confidence": 0.85},
        tags=["decision", "ml"]
    )
    
    # Error occurred
    audit_system.log(
        AuditEventType.ERROR_OCCURRED,
        AuditLevel.ERROR,
        "data_processor",
        "Failed to load dataset: File not found",
        correlation_id=task_correlation,
        metadata={"error_type": "FileNotFoundError", "file": "/data/input.csv"},
        tags=["error", "data"]
    )
    
    # Recovery attempted
    audit_system.log(
        AuditEventType.RECOVERY_ATTEMPTED,
        AuditLevel.WARNING,
        "recovery_system",
        "Attempting to use backup dataset",
        correlation_id=task_correlation,
        metadata={"recovery_method": "backup_dataset", "backup_file": "/data/backup.csv"},
        tags=["recovery", "data"]
    )
    
    # Task completed
    audit_system.log(
        AuditEventType.TASK_COMPLETED,
        AuditLevel.INFO,
        "task_executor",
        "Task completed successfully using backup data",
        correlation_id=task_correlation,
        metadata={"result": "success", "records_processed": 10000},
        tags=["task", "complete", "success"]
    )
    
    # Alert generated
    audit_system.log(
        AuditEventType.ALERT_GENERATED,
        AuditLevel.WARNING,
        "monitor",
        "File access failure detected",
        metadata={"alert_type": "file_access_failure", "frequency": "first_occurrence"},
        tags=["alert", "monitoring"]
    )

if __name__ == '__main__':
    # Demo usage
    audit_system = create_demo_audit_system()
    
    # Generate demo events
    generate_demo_events(audit_system)
    
    # Search examples
    print("\n=== Recent Events ===")
    recent_events = audit_system.search(limit=10)
    for event in recent_events:
        print(f"{event.timestamp} [{event.level.upper()}] {event.source}: {event.message}")
    
    print("\n=== Error Events ===")
    error_events = audit_system.search(level="error", limit=5)
    for event in error_events:
        print(f"{event.timestamp} {event.source}: {event.message}")
    
    print("\n=== Event Statistics ===")
    stats = audit_system.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n=== Correlation Trace ===")
    # Find a correlation ID from recent events
    if recent_events and recent_events[0].correlation_id:
        correlation_id = recent_events[0].correlation_id
        trace = audit_system.trace_correlation(correlation_id)
        print(f"Correlation {correlation_id}:")
        print(f"  Duration: {trace['summary']['duration_seconds']}s")
        print(f"  Events: {trace['summary']['event_count']}")
        print(f"  Sources: {trace['summary']['sources_involved']}")
        print(f"  Errors: {trace['summary']['errors']}")
    
    # Export demo
    audit_system.export_events("demo_audit_export.json", format="json", limit=100)
    print("\nAudit events exported to demo_audit_export.json")
    
    # Shutdown
    audit_system.shutdown()
