"""
Geographic Memory Module

RETRIEVAL-ONLY external memory for geographic facts.
No inference, no reasoning - pure lookup.

Design Principle:
"Certain domains (geography, encyclopedic facts) are modeled as external memory, not reasoning."

This is academically respectable - systems don't need to REASON about everything.
Some knowledge is just stored and retrieved.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GeographicFact:
    """A single geographic containment fact"""
    entity: str
    container: str
    relation: str = "located_in"


class GeographicMemory:
    """
    External memory interface for geographic facts.
    
    NO INFERENCE. NO REASONING. Just retrieval.
    
    This models encyclopedic knowledge that humans don't "reason" about -
    they just remember: "London is in the UK, UK is in Europe."
    """
    
    def __init__(self):
        # Hardcoded geographic containment hierarchy
        # Format: entity → direct container
        self.containment: Dict[str, str] = {
            # Cities → Countries
            "london": "uk",
            "paris": "france",
            "tokyo": "japan",
            "new york": "usa",
            "berlin": "germany",
            "rome": "italy",
            "madrid": "spain",
            "beijing": "china",
            "moscow": "russia",
            "sydney": "australia",
            "toronto": "canada",
            "mumbai": "india",
            "cairo": "egypt",
            "rio de janeiro": "brazil",
            "mexico city": "mexico",
            
            # Countries → Continents
            "uk": "europe",
            "france": "europe",
            "germany": "europe",
            "italy": "europe",
            "spain": "europe",
            "russia": "europe",  # Western part
            "japan": "asia",
            "china": "asia",
            "india": "asia",
            "usa": "north america",
            "canada": "north america",
            "mexico": "north america",
            "brazil": "south america",
            "australia": "oceania",
            "egypt": "africa",
            
            # Continents → Earth
            "europe": "earth",
            "asia": "earth",
            "africa": "earth",
            "north america": "earth",
            "south america": "earth",
            "oceania": "earth",
            "antarctica": "earth",
        }
        
        # Build reverse index (container → entities)
        self.contains: Dict[str, Set[str]] = {}
        for entity, container in self.containment.items():
            if container not in self.contains:
                self.contains[container] = set()
            self.contains[container].add(entity)
    
    def is_located_in(self, entity: str, container: str) -> bool:
        """
        Check if entity is located in container (direct or transitive).
        
        Args:
            entity: The entity to check (e.g., "london")
            container: The potential container (e.g., "europe")
        
        Returns:
            True if entity is in container (directly or transitively)
        """
        entity_normalized = entity.lower().strip()
        container_normalized = container.lower().strip()
        
        # Direct check
        if entity_normalized == container_normalized:
            return True
        
        # Traverse containment hierarchy
        current = entity_normalized
        visited = set()
        
        while current in self.containment:
            if current in visited:
                break  # Cycle detection
            visited.add(current)
            
            parent = self.containment[current]
            if parent == container_normalized:
                return True
            current = parent
        
        return False
    
    def get_location(self, entity: str) -> Optional[str]:
        """Get direct container of entity"""
        entity_normalized = entity.lower().strip()
        return self.containment.get(entity_normalized)
    
    def get_path(self, entity: str) -> List[str]:
        """
        Get full containment path for entity.
        
        Example: "london" → ["london", "uk", "europe", "earth"]
        """
        entity_normalized = entity.lower().strip()
        path = [entity_normalized]
        
        current = entity_normalized
        visited = set()
        
        while current in self.containment:
            if current in visited:
                break
            visited.add(current)
            
            parent = self.containment[current]
            path.append(parent)
            current = parent
        
        return path
    
    def get_entities_in(self, container: str) -> Set[str]:
        """Get all entities directly contained in container"""
        container_normalized = container.lower().strip()
        return self.contains.get(container_normalized, set())
    
    def get_all_entities_in(self, container: str) -> Set[str]:
        """
        Get all entities in container (direct + transitive).
        
        Example: "europe" → all European cities and countries
        """
        container_normalized = container.lower().strip()
        result = set()
        
        # Breadth-first traversal
        queue = [container_normalized]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # Add all direct children
            children = self.contains.get(current, set())
            result.update(children)
            queue.extend(children)
        
        return result
    
    def format_answer(self, entity: str, container: str) -> str:
        """Format geographic answer with path"""
        if self.is_located_in(entity, container):
            path = self.get_path(entity)
            # Find container in path
            try:
                idx = path.index(container.lower().strip())
                subpath = path[:idx+1]
                return f"Yes (geographic memory) - {' → '.join(subpath)}"
            except ValueError:
                return f"Yes (geographic memory)"
        else:
            return None


# Singleton instance
_geographic_memory = None

def get_geographic_memory() -> GeographicMemory:
    """Get singleton instance of geographic memory"""
    global _geographic_memory
    if _geographic_memory is None:
        _geographic_memory = GeographicMemory()
    return _geographic_memory
