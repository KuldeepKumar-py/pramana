"""
Sakshin — Meta Observer

BORING BY DESIGN.

Sakshin is NOT smart. It is a tape recorder with a hash function.
If it feels "smart", you did it wrong.

What Sakshin does:
  - Observe (watch cognitive events)
  - Log (record immutable traces)
  - Hash (detect tampering)
  - Replay (reconstruct history)

What Sakshin does NOT do:
  - Reason (no interpretation)
  - Judge (no evaluation)
  - Interfere (read-only on cognitive system)

See: DESIGN.md for full architecture
"""

from __future__ import annotations
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Optional
from collections import deque


@dataclass
class Event:
    """
    Immutable cognitive event.
    
    Pure data - no interpretation.
    """
    timestamp: float
    module: str  # "manas", "buddhi", "chitta", "hre", "ahankara"
    event_type: str  # "parse", "judgment", "belief_stored", etc.
    data: dict[str, Any]
    event_id: str = ""  # Set in __post_init__
    
    def __post_init__(self):
        """Generate event ID from content."""
        if not self.event_id:
            content = f"{self.timestamp}:{self.module}:{self.event_type}"
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute cryptographic hash for tamper detection."""
        # Sort keys for deterministic hashing
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class Sakshin:
    """
    Meta Observer - the cognitive event logger.
    
    BORING BY DESIGN. This is intentional.
    
    Sakshin is like a flight recorder for MARC's cognitive processes. It watches
    what happens and writes it down, nothing more. No intelligence, no interpretation,
    no decision-making. Just observation and logging.
    
    What it does:
        - Record events: when Manas parses, when Buddhi judges, when Chitta stores
        - Replay sequences: show what happened in what order
        - Verify integrity: detect if the log has been tampered with
        - Reconstruct influences: trace which events led to which beliefs
    
    What it explicitly does NOT do:
        - Reason about events
        - Judge whether events were good or bad
        - Interfere with cognitive processes
        - Optimize or improve anything
        - Learn patterns
    
    If this class starts feeling "smart" or "intelligent", something went wrong.
    It's supposed to be dumb. Reliably, verifiably dumb.
    """
    
    def __init__(self):
        """Initialize empty event log."""
        self.events: list[Event] = []
        self.event_hashes: list[str] = []
        self.event_index: dict[str, Event] = {}  # event_id → Event
        self.belief_to_event: dict[str, str] = {}  # belief_id → event_id that created it
        
        # Statistics (boring counters only)
        self.stats = {
            "total_events": 0,
            "events_by_module": {},
            "events_by_type": {}
        }
    
    def observe(self, module: str, event_type: str, data: dict[str, Any]) -> str:
        """
        Record cognitive event (no interpretation).
        
        Args:
            module: Source module ("manas", "buddhi", etc.)
            event_type: Type of event ("parse", "judgment", etc.)
            data: Event data (as-is, NO processing)
        
        Returns:
            event_id for later reference
        """
        # Create event
        event = Event(
            timestamp=time.time(),
            module=module,
            event_type=event_type,
            data=data
        )
        
        # Compute hash (tamper detection)
        event_hash = event.compute_hash()
        
        # Store
        self.events.append(event)
        self.event_hashes.append(event_hash)
        self.event_index[event.event_id] = event
        
        # Track belief creation
        if event_type == "belief_stored" and "belief_id" in data:
            self.belief_to_event[data["belief_id"]] = event.event_id
        
        # Update statistics (boring counters)
        self.stats["total_events"] += 1
        self.stats["events_by_module"][module] = \
            self.stats["events_by_module"].get(module, 0) + 1
        self.stats["events_by_type"][event_type] = \
            self.stats["events_by_type"].get(event_type, 0) + 1
        
        return event.event_id
    
    def replay(self, from_idx: int = 0, to_idx: Optional[int] = None) -> list[Event]:
        """
        Replay event sequence (verification).
        
        Args:
            from_idx: Start index (inclusive)
            to_idx: End index (exclusive)
        
        Returns:
            List of events in chronological order
        """
        return self.events[from_idx:to_idx]
    
    def verify_integrity(self) -> bool:
        """
        Check if event log was tampered with.
        
        Returns:
            True if log is intact, False if tampered
        """
        for i, event in enumerate(self.events):
            computed_hash = event.compute_hash()
            if computed_hash != self.event_hashes[i]:
                return False
        return True
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        return self.event_index.get(event_id)
    
    def reconstruct_influences(self, output_event_id: str) -> dict[str, Any]:
        """
        Trace backward from output to all inputs.
        
        This is PURE GRAPH TRAVERSAL, not reasoning.
        
        Args:
            output_event_id: Event ID to trace backward from
        
        Returns:
            {
                "output_event": Event,
                "influenced_by": [Event, Event, ...],
                "dependency_graph": {event_id: [parent_ids, ...]}
            }
        """
        # Get output event
        output_event = self.get_event(output_event_id)
        if not output_event:
            return {
                "output_event": None,
                "influenced_by": [],
                "dependency_graph": {}
            }
        
        # BFS to find all ancestors
        influenced_by = []
        dependency_graph = {}
        visited = set()
        queue = deque([output_event_id])
        
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            current_event = self.get_event(current_id)
            if not current_event:
                # Might be a belief_id, not an event_id
                # Try to resolve it
                if current_id.startswith('b_'):
                    actual_event_id = self.belief_to_event.get(current_id)
                    if actual_event_id and actual_event_id not in visited:
                        queue.append(actual_event_id)
                continue
            
            influenced_by.append(current_event)
            
            # Extract parent IDs from event data
            parent_ids = self._extract_parent_ids(current_event)
            dependency_graph[current_id] = parent_ids
            
            # Add parents to queue
            for parent_id in parent_ids:
                if parent_id not in visited:
                    queue.append(parent_id)
        
        return {
            "output_event": output_event,
            "influenced_by": influenced_by,
            "dependency_graph": dependency_graph
        }
    
    def _extract_parent_ids(self, event: Event) -> list[str]:
        """
        Extract parent event IDs from event data.
        
        This is dumb pattern matching, NOT reasoning.
        """
        parent_ids = []
        
        # Recursively search for IDs in data
        def _extract_from_value(value):
            if isinstance(value, str):
                # Check if it's an event ID or belief ID
                if len(value) in [8, 16] and all(c in '0123456789abcdef' for c in value):
                    parent_ids.append(value)
                elif value.startswith('b_'):
                    parent_ids.append(value)
            elif isinstance(value, list):
                for item in value:
                    _extract_from_value(item)
            elif isinstance(value, dict):
                for v in value.values():
                    _extract_from_value(v)
        
        # Common direct patterns (faster)
        if "parent_event" in event.data:
            parent_ids.append(event.data["parent_event"])
        
        if "belief_id" in event.data:
            parent_ids.append(event.data["belief_id"])
        
        if "belief_ids" in event.data:
            parent_ids.extend(event.data["belief_ids"])
        
        if "inputs" in event.data:
            inputs = event.data["inputs"]
            if isinstance(inputs, list):
                parent_ids.extend(inputs)
        
        # Fallback: search all data (slower but comprehensive)
        _extract_from_value(event.data)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_parents = []
        for pid in parent_ids:
            if pid not in seen:
                seen.add(pid)
                unique_parents.append(pid)
        
        return unique_parents
    
    def get_events_by_module(self, module: str) -> list[Event]:
        """Get all events from a specific module."""
        return [e for e in self.events if e.module == module]
    
    def get_events_by_type(self, event_type: str) -> list[Event]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_recent_events(self, n: int = 10) -> list[Event]:
        """Get n most recent events."""
        return self.events[-n:]
    
    def clear(self):
        """Clear all events (for testing only)."""
        self.events = []
        self.event_hashes = []
        self.event_index = {}
        self.belief_to_event = {}
        self.stats = {
            "total_events": 0,
            "events_by_module": {},
            "events_by_type": {}
        }
    
    def export_log(self) -> list[dict]:
        """Export event log as JSON-serializable list."""
        return [event.to_dict() for event in self.events]
    
    def summary(self) -> str:
        """
        Generate boring summary of event log.
        
        No interpretation. Just counts.
        """
        lines = [
            f"Sakshin Event Log",
            f"================",
            f"Total events: {self.stats['total_events']}",
            f"",
            f"Events by module:",
        ]
        
        for module, count in sorted(self.stats['events_by_module'].items()):
            lines.append(f"  {module}: {count}")
        
        lines.append("")
        lines.append("Events by type:")
        
        for event_type, count in sorted(self.stats['events_by_type'].items()):
            lines.append(f"  {event_type}: {count}")
        
        lines.append("")
        lines.append(f"Log integrity: {'✅ INTACT' if self.verify_integrity() else '❌ TAMPERED'}")
        
        return "\n".join(lines)


# Singleton instance (optional - can also inject)
_global_sakshin: Optional[Sakshin] = None


def get_sakshin() -> Optional[Sakshin]:
    """Get global Sakshin instance (if initialized)."""
    return _global_sakshin


def initialize_sakshin() -> Sakshin:
    """Initialize global Sakshin instance."""
    global _global_sakshin
    _global_sakshin = Sakshin()
    return _global_sakshin
