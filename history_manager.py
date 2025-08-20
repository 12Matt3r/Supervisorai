"""
History Manager for Supervisor Agent - Versioned history tracking and management.

Maintains comprehensive audit trails of all interventions, state changes,
and recovery attempts with version control and diff generation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import difflib


class HistoryEventType(Enum):
    """Types of events tracked in history."""
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"
    ROLLBACK_EXECUTED = "rollback_executed"
    ESCALATION_CREATED = "escalation_created"
    LOOP_DETECTED = "loop_detected"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    INTERVENTION_APPLIED = "intervention_applied"
    RECOVERY_SUCCESS = "recovery_success"
    RECOVERY_FAILURE = "recovery_failure"
    SNAPSHOT_CREATED = "snapshot_created"
    STATE_CHANGED = "state_changed"


@dataclass
class HistoryEntry:
    """Represents a single entry in the history."""
    entry_id: str
    timestamp: datetime
    event_type: HistoryEventType
    agent_id: str
    task_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: int
    parent_entry_id: Optional[str] = None
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps({
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'data': self.data,
            'metadata': self.metadata
        }, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = HistoryEventType(data['event_type'])
        return cls(**data)


@dataclass
class HistoryVersion:
    """Represents a version in the history timeline."""
    version_id: str
    version_number: int
    created_at: datetime
    entries: List[HistoryEntry]
    summary: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'version_number': self.version_number,
            'created_at': self.created_at.isoformat(),
            'entries': [entry.to_dict() for entry in self.entries],
            'summary': self.summary,
            'tags': self.tags
        }


class HistoryManager:
    """System for managing versioned history and audit trails."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("supervisor_data/history")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage
        self.histories: Dict[str, List[HistoryEntry]] = {}  # history_id -> entries
        self.versions: Dict[str, List[HistoryVersion]] = {}  # history_id -> versions
        
        # Configuration
        self.config = {
            'max_entries_per_history': 1000,
            'max_versions_per_history': 100,
            'retention_days': 30,
            'auto_version_threshold': 50,  # Create version after N entries
            'compression_enabled': True
        }
        
        # Statistics
        self.stats = {
            'total_histories': 0,
            'total_entries': 0,
            'total_versions': 0,
            'by_event_type': {event_type.value: 0 for event_type in HistoryEventType}
        }
    
    def create_history(
        self,
        agent_id: str,
        task_id: str,
        initial_data: Dict[str, Any]
    ) -> str:
        """Create a new history timeline."""
        
        history_id = self._generate_history_id(agent_id, task_id)
        
        # Create initial entry
        initial_entry = HistoryEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.utcnow(),
            event_type=HistoryEventType.STATE_CHANGED,
            agent_id=agent_id,
            task_id=task_id,
            data=initial_data,
            metadata={'event': 'history_created'},
            version=1
        )
        
        self.histories[history_id] = [initial_entry]
        self.versions[history_id] = []
        
        self.stats['total_histories'] += 1
        self.stats['total_entries'] += 1
        self.stats['by_event_type'][HistoryEventType.STATE_CHANGED.value] += 1
        
        self.logger.info(f"Created history {history_id} for agent {agent_id}")
        
        return history_id
    
    def add_entry(
        self,
        history_id: str,
        event_type: HistoryEventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: str = "unknown",
        task_id: str = "unknown"
    ) -> str:
        """Add an entry to a history."""
        
        if history_id not in self.histories:
            self.logger.error(f"History {history_id} not found")
            return ""
        
        # Get current version number
        current_entries = self.histories[history_id]
        version_number = (current_entries[-1].version if current_entries else 0) + 1
        
        # Create entry
        entry = HistoryEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            agent_id=agent_id,
            task_id=task_id,
            data=data,
            metadata=metadata or {},
            version=version_number,
            parent_entry_id=current_entries[-1].entry_id if current_entries else None
        )
        
        # Add to history
        self.histories[history_id].append(entry)
        
        # Check if we need to create a version
        if len(current_entries) % self.config['auto_version_threshold'] == 0:
            self._create_version(history_id, "Auto-versioning after threshold")
        
        # Store to disk
        self._store_entry(history_id, entry)
        
        # Update statistics
        self.stats['total_entries'] += 1
        self.stats['by_event_type'][event_type.value] += 1
        
        self.logger.debug(f"Added {event_type.value} entry to history {history_id}")
        
        return entry.entry_id
    
    def record_intervention(
        self,
        history_id: str,
        intervention_type: str,
        intervention_data: Dict[str, Any],
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record an intervention with before/after states."""
        
        # Generate diff
        diff = self._generate_diff(before_state, after_state)
        
        intervention_entry_data = {
            'intervention_type': intervention_type,
            'intervention_data': intervention_data,
            'before_state': before_state,
            'after_state': after_state,
            'diff': diff,
            'state_change_size': len(json.dumps(diff))
        }
        
        return self.add_entry(
            history_id=history_id,
            event_type=HistoryEventType.INTERVENTION_APPLIED,
            data=intervention_entry_data,
            metadata=metadata
        )
    
    def create_version(
        self,
        history_id: str,
        summary: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """Manually create a version snapshot."""
        return self._create_version(history_id, summary, tags or [])
    
    def _create_version(
        self,
        history_id: str,
        summary: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a version snapshot of the current history state."""
        
        if history_id not in self.histories:
            return ""
        
        entries = self.histories[history_id].copy()
        if not entries:
            return ""
        
        # Generate version ID
        version_id = self._generate_version_id(history_id)
        
        # Get version number
        version_number = len(self.versions.get(history_id, [])) + 1
        
        # Create version
        version = HistoryVersion(
            version_id=version_id,
            version_number=version_number,
            created_at=datetime.utcnow(),
            entries=entries,
            summary=summary,
            tags=tags or []
        )
        
        # Add to versions
        if history_id not in self.versions:
            self.versions[history_id] = []
        
        self.versions[history_id].append(version)
        
        # Store version to disk
        self._store_version(history_id, version)
        
        # Update statistics
        self.stats['total_versions'] += 1
        
        self.logger.info(f"Created version {version_number} for history {history_id}: {summary}")
        
        return version_id
    
    def get_history(
        self,
        history_id: str,
        version: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get history entries, optionally from a specific version."""
        
        if version is not None:
            # Get from specific version
            versions = self.versions.get(history_id, [])
            if version <= len(versions):
                entries = versions[version - 1].entries
            else:
                entries = []
        else:
            # Get current entries
            entries = self.histories.get(history_id, [])
        
        # Apply limit
        if limit:
            entries = entries[-limit:]
        
        return [entry.to_dict() for entry in entries]
    
    def get_diff_between_versions(
        self,
        history_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Generate diff between two versions."""
        
        versions = self.versions.get(history_id, [])
        
        if version1 > len(versions) or version2 > len(versions):
            return {'error': 'Version not found'}
        
        v1_entries = versions[version1 - 1].entries
        v2_entries = versions[version2 - 1].entries
        
        # Generate entry-level diff
        v1_data = [entry.to_dict() for entry in v1_entries]
        v2_data = [entry.to_dict() for entry in v2_entries]
        
        diff = self._generate_diff(v1_data, v2_data)
        
        return {
            'version1': version1,
            'version2': version2,
            'diff': diff,
            'summary': f"Diff between version {version1} and {version2}"
        }
    
    def search_entries(
        self,
        history_id: str,
        event_type: Optional[HistoryEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        search_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for specific entries in history."""
        
        entries = self.histories.get(history_id, [])
        filtered_entries = []
        
        for entry in entries:
            # Apply filters
            if event_type and entry.event_type != event_type:
                continue
            
            if start_time and entry.timestamp < start_time:
                continue
            
            if end_time and entry.timestamp > end_time:
                continue
            
            if search_text:
                entry_str = json.dumps(entry.to_dict()).lower()
                if search_text.lower() not in entry_str:
                    continue
            
            filtered_entries.append(entry.to_dict())
        
        return filtered_entries
    
    def _generate_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Generate a diff between two data structures."""
        
        # Convert to JSON strings for comparison
        str1 = json.dumps(data1, sort_keys=True, indent=2, default=str)
        str2 = json.dumps(data2, sort_keys=True, indent=2, default=str)
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            str1.splitlines(keepends=True),
            str2.splitlines(keepends=True),
            fromfile='before',
            tofile='after'
        ))
        
        return {
            'unified_diff': ''.join(diff_lines),
            'has_changes': len(diff_lines) > 0,
            'line_count': len(diff_lines)
        }
    
    def _generate_history_id(self, agent_id: str, task_id: str) -> str:
        """Generate a unique history ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"hist_{agent_id}_{task_id}_{timestamp}"
    
    def _generate_entry_id(self) -> str:
        """Generate a unique entry ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _generate_version_id(self, history_id: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{history_id}_v{timestamp}"
    
    def _store_entry(self, history_id: str, entry: HistoryEntry):
        """Store entry to disk."""
        
        history_dir = self.storage_path / history_id
        history_dir.mkdir(exist_ok=True)
        
        entry_file = history_dir / f"{entry.entry_id}.json"
        
        try:
            with open(entry_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to store entry {entry.entry_id}: {str(e)}")
    
    def _store_version(self, history_id: str, version: HistoryVersion):
        """Store version to disk."""
        
        version_dir = self.storage_path / "versions" / history_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = version_dir / f"v{version.version_number}.json"
        
        try:
            with open(version_file, 'w') as f:
                json.dump(version.to_dict(), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to store version {version.version_id}: {str(e)}")
    
    async def cleanup_old_entries(self):
        """Clean up old entries based on retention policy."""
        
        cutoff_time = datetime.utcnow() - timedelta(days=self.config['retention_days'])
        
        for history_id in list(self.histories.keys()):
            entries = self.histories[history_id]
            
            # Keep entries newer than cutoff
            filtered_entries = [
                entry for entry in entries
                if entry.timestamp > cutoff_time
            ]
            
            if len(filtered_entries) != len(entries):
                self.histories[history_id] = filtered_entries
                self.logger.info(
                    f"Cleaned up {len(entries) - len(filtered_entries)} "
                    f"old entries from history {history_id}"
                )
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the history manager."""
        return {
            'active_histories': len(self.histories),
            'total_versions': len([v for versions in self.versions.values() for v in versions]),
            'stats': self.stats,
            'storage_path': str(self.storage_path),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the history manager."""
        self.logger.info("Shutting down history manager")
        
        # Cleanup old entries
        await self.cleanup_old_entries()
        
        self.histories.clear()
        self.versions.clear()
        
        self.logger.info("History manager shutdown complete")
