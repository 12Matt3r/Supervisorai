import json
from typing import Dict, Any, Optional
from datetime import datetime
try:
    import aiofiles
except ImportError:
    aiofiles = None
from pathlib import Path


class AuditLogger:
    """Handles comprehensive audit logging for supervisor activities"""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)

    async def log_event(
        self,
        task_id: str,
        event_type: str,
        details: Dict[str, Any],
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a supervisor event to the audit trail"""
        
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "event_type": event_type,
            "details": details,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        if not aiofiles:
            print("Warning: aiofiles is not installed. Audit logging to file is disabled.")
            return

        try:
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(json.dumps(event_record) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to audit log: {e}")

    async def get_events(
        self,
        task_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> list[Dict[str, Any]]:
        """Retrieve audit events based on filters"""
        
        events = []
        
        if not aiofiles or not self.log_file.exists():
            return events
        
        try:
            async with aiofiles.open(self.log_file, 'r') as f:
                async for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line.strip())
                            
                            # Apply filters
                            if task_id and event.get("task_id") != task_id:
                                continue
                            
                            if event_type and event.get("event_type") != event_type:
                                continue
                            
                            if since:
                                event_time = datetime.fromisoformat(event["timestamp"])
                                if event_time < since:
                                    continue
                            
                            events.append(event)
                            
                            if len(events) >= limit:
                                break
                                
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Could not read audit log: {e}")
        
        return events

    async def get_task_timeline(
        self,
        task_id: str
    ) -> list[Dict[str, Any]]:
        """Get chronological timeline of events for a specific task"""
        
        events = await self.get_events(task_id=task_id, limit=1000)
        
        # Sort by timestamp
        events.sort(key=lambda e: e["timestamp"])
        
        return events

    async def get_statistics(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit log statistics"""
        
        events = await self.get_events(since=since, limit=10000)
        
        if not events:
            return {
                "total_events": 0,
                "events_by_type": {},
                "unique_tasks": 0,
                "time_range": None
            }
        
        # Calculate statistics
        events_by_type = {}
        unique_tasks = set()
        
        for event in events:
            event_type = event.get("event_type", "unknown")
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            if event.get("task_id"):
                unique_tasks.add(event["task_id"])
        
        time_range = None
        if events:
            earliest = min(event["timestamp"] for event in events)
            latest = max(event["timestamp"] for event in events)
            time_range = {"earliest": earliest, "latest": latest}
        
        return {
            "total_events": len(events),
            "events_by_type": events_by_type,
            "unique_tasks": len(unique_tasks),
            "time_range": time_range
        }