"""
Rollback Manager for Supervisor Agent - State preservation and restoration capabilities.

Adapted from rollback_system.py for integration with the supervisor error handling system.
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class StateSnapshot:
    """Represents a state snapshot that can be restored."""
    snapshot_id: str
    timestamp: datetime
    agent_id: str
    task_id: str
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
    checksum: str
    tags: List[str]
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class RollbackManager:
    """System for creating state snapshots and rolling back to previous states."""
    
    def __init__(self, storage_path: Optional[Path] = None, max_snapshots: int = 50):
        self.storage_path = storage_path or Path("supervisor_data/snapshots")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_snapshots = max_snapshots
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory snapshot cache
        self.snapshot_cache: Dict[str, StateSnapshot] = {}
        
        # Named checkpoints for easy rollback
        self.checkpoints: Dict[str, str] = {}  # name -> snapshot_id
        
        # Statistics
        self.stats = {
            'snapshots_created': 0,
            'rollbacks_executed': 0,
            'rollback_successes': 0,
            'rollback_failures': 0
        }
        
        # Load existing snapshots
        asyncio.create_task(self._load_existing_snapshots())
    
    def create_snapshot(
        self,
        state_data: Dict[str, Any],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: str = "system",
        task_id: str = "unknown"
    ) -> str:
        """Create a state snapshot."""
        
        # Generate snapshot ID
        snapshot_id = self._generate_snapshot_id(agent_id, task_id)
        
        # Calculate checksum
        checksum = self._calculate_checksum(state_data)
        
        # Create snapshot
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            task_id=task_id,
            state_data=state_data,
            metadata=metadata or {},
            checksum=checksum,
            tags=tags or []
        )
        
        # Store and cache snapshot
        self._store_snapshot(snapshot)
        self.snapshot_cache[snapshot_id] = snapshot
        
        # Cleanup old snapshots
        self._cleanup_old_snapshots()
        
        self.stats['snapshots_created'] += 1
        self.logger.info(f"Created snapshot {snapshot_id}")
        
        return snapshot_id
    
    def create_checkpoint(
        self,
        checkpoint_name: str,
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a named checkpoint for easy rollback."""
        
        snapshot_id = self.create_snapshot(
            state_data=state_data,
            tags=["checkpoint", checkpoint_name],
            metadata=metadata
        )
        
        self.checkpoints[checkpoint_name] = snapshot_id
        self.logger.info(f"Created checkpoint '{checkpoint_name}' with snapshot {snapshot_id}")
        
        return snapshot_id
    
    def rollback_to_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Rollback to a specific snapshot."""
        
        try:
            snapshot = self.snapshot_cache.get(snapshot_id)
            if not snapshot:
                snapshot = self._load_snapshot(snapshot_id)
            
            if not snapshot:
                return {
                    'success': False,
                    'error': f'Snapshot {snapshot_id} not found'
                }
            
            # Verify snapshot integrity
            if not self._verify_snapshot_integrity(snapshot):
                return {
                    'success': False,
                    'error': f'Snapshot {snapshot_id} integrity check failed'
                }
            
            self.stats['rollbacks_executed'] += 1
            self.stats['rollback_successes'] += 1
            
            self.logger.info(f"Successfully rolled back to snapshot {snapshot_id}")
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'state_data': snapshot.state_data,
                'metadata': snapshot.metadata,
                'timestamp': snapshot.timestamp.isoformat()
            }
            
        except Exception as e:
            self.stats['rollback_failures'] += 1
            self.logger.error(f"Rollback to snapshot {snapshot_id} failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def rollback_to_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Rollback to a named checkpoint."""
        
        snapshot_id = self.checkpoints.get(checkpoint_name)
        if not snapshot_id:
            return {
                'success': False,
                'error': f'Checkpoint {checkpoint_name} not found'
            }
        
        return self.rollback_to_snapshot(snapshot_id)
    
    def get_snapshots(
        self,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get available snapshots with optional filtering."""
        
        snapshots = []
        
        for snapshot in self.snapshot_cache.values():
            # Apply filters
            if agent_id and snapshot.agent_id != agent_id:
                continue
            if task_id and snapshot.task_id != task_id:
                continue
            if tags and not any(tag in snapshot.tags for tag in tags):
                continue
            
            snapshots.append({
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'agent_id': snapshot.agent_id,
                'task_id': snapshot.task_id,
                'tags': snapshot.tags,
                'metadata': snapshot.metadata
            })
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return snapshots[:limit]
    
    def _generate_snapshot_id(self, agent_id: str, task_id: str) -> str:
        """Generate a unique snapshot ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"{agent_id}_{task_id}_{timestamp}"
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _verify_snapshot_integrity(self, snapshot: StateSnapshot) -> bool:
        """Verify snapshot data integrity."""
        calculated_checksum = self._calculate_checksum(snapshot.state_data)
        return calculated_checksum == snapshot.checksum
    
    def _store_snapshot(self, snapshot: StateSnapshot):
        """Store snapshot to disk."""
        snapshot_file = self.storage_path / f"{snapshot.snapshot_id}.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
    
    def _load_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load snapshot from disk."""
        
        snapshot_file = self.storage_path / f"{snapshot_id}.json"
        if not snapshot_file.exists():
            return None
        
        try:
            with open(snapshot_file, 'r') as f:
                data = json.load(f)
            
            return StateSnapshot.from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to load snapshot {snapshot_id}: {str(e)}")
            return None
    
    async def _load_existing_snapshots(self):
        """Load existing snapshots from storage."""
        
        try:
            for snapshot_file in self.storage_path.glob('*.json'):
                snapshot_id = snapshot_file.stem
                snapshot = self._load_snapshot(snapshot_id)
                if snapshot:
                    self.snapshot_cache[snapshot_id] = snapshot
                    
                    # Restore checkpoints
                    if "checkpoint" in snapshot.tags:
                        for tag in snapshot.tags:
                            if tag != "checkpoint":
                                self.checkpoints[tag] = snapshot_id
            
            self.logger.info(f"Loaded {len(self.snapshot_cache)} existing snapshots")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing snapshots: {str(e)}")
    
    def _cleanup_old_snapshots(self):
        """Clean up old snapshots to maintain size limits."""
        
        if len(self.snapshot_cache) <= self.max_snapshots:
            return
        
        # Sort by timestamp (oldest first)
        sorted_snapshots = sorted(
            self.snapshot_cache.values(),
            key=lambda x: x.timestamp
        )
        
        # Keep only the most recent snapshots
        snapshots_to_remove = sorted_snapshots[:-self.max_snapshots]
        
        for snapshot in snapshots_to_remove:
            # Don't remove checkpoints
            if "checkpoint" not in snapshot.tags:
                self._delete_snapshot(snapshot.snapshot_id)
    
    def _delete_snapshot(self, snapshot_id: str):
        """Delete a specific snapshot."""
        
        try:
            # Remove from cache
            self.snapshot_cache.pop(snapshot_id, None)
            
            # Remove from storage
            snapshot_file = self.storage_path / f"{snapshot_id}.json"
            if snapshot_file.exists():
                snapshot_file.unlink()
            
        except Exception as e:
            self.logger.error(f"Failed to delete snapshot {snapshot_id}: {str(e)}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of the rollback manager."""
        return {
            'cached_snapshots': len(self.snapshot_cache),
            'checkpoints': list(self.checkpoints.keys()),
            'stats': self.stats,
            'storage_path': str(self.storage_path),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the rollback manager."""
        self.logger.info("Shutting down rollback manager")
        self.snapshot_cache.clear()
        self.checkpoints.clear()
        self.logger.info("Rollback manager shutdown complete")
