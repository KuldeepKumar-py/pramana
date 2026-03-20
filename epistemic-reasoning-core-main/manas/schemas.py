"""
RELATION SCHEMAS — Taxonomic Inheritance and Property Propagation

Relation schemas are abstract patterns extracted from belief chains.
They enable:
- Taxonomic inheritance (bat → mammal → vertebrate → animal)
- Property propagation (mammals produce milk → bats produce milk)
- Constraint-based analogy (hypothesis generation only, NOT direct inference)

CRITICAL RULE:
Only STRUCTURAL predicates may propagate via schemas.
BEHAVIORAL and ABSTRACT predicates require direct teaching.

Philosophy:
- Schemas are patterns, not beliefs
- Schemas propose, epistemics dispose
- Analogy creates hypotheses, not facts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class SchemaType(Enum):
    """Type of relation schema."""
    TAXONOMIC = "taxonomic"  # X is_a Y, Y is_a Z ⇒ X is_a Z
    PROPERTY = "property"    # X is_a Y, Y has_property P ⇒ X has_property P
    ANALOGICAL = "analogical"  # X:Y :: A:B (structure mapping)


@dataclass
class RelationSchema:
    """
    Abstract relation pattern extracted from belief chains.
    
    Example (Taxonomic):
        pattern: "X is_a Y"
        instances: [(bat, mammal), (whale, mammal), (dog, mammal)]
    
    Example (Property):
        pattern: "X is_a Mammal ⇒ X produces_milk"
        support: [bat, whale, dog, human]
        confidence: 0.95
    
    Example (Analogical):
        pattern: "X:flight :: Y:swimming"
        mapping: {bird: fish, wing: fin, air: water}
    """
    schema_id: str
    schema_type: SchemaType
    pattern: str  # Human-readable pattern description
    
    # Pattern components
    variables: list[str]  # e.g., ['X', 'Y', 'P']
    constraints: dict[str, Any]  # Type/epistemic constraints on variables
    
    # Evidence
    instances: list[tuple[str, ...]]  # Concrete instances that match this pattern
    supporting_beliefs: list[str]  # Belief IDs that support this schema
    
    # Epistemic metadata
    confidence: float = 0.5  # How reliable is this schema?
    epistemic_class: str = "structural"  # Only structural schemas may auto-propagate
    
    # Usage tracking
    successful_applications: int = 0
    failed_applications: int = 0
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def matches(self, entities: tuple[str, ...], predicates: set[str]) -> bool:
        """
        Check if given entities and predicates match this schema pattern.
        
        Args:
            entities: Entity tuple to match
            predicates: Set of predicates to check
        
        Returns:
            True if pattern matches
        """
        # Implementation depends on schema type
        # For now, simple string matching
        return len(entities) == len(self.variables)
    
    def instantiate(self, bindings: dict[str, str]) -> dict[str, Any]:
        """
        Instantiate schema with concrete entity bindings.
        
        Args:
            bindings: Variable → entity mapping (e.g., {'X': 'bat', 'Y': 'mammal'})
        
        Returns:
            Canonical structure for new belief (HYPOTHESIS, not asserted belief)
        """
        # This creates a HYPOTHESIS, not a belief
        # Buddhi must validate before accepting
        return {
            "type": "hypothesis",
            "schema_id": self.schema_id,
            "bindings": bindings,
            "confidence": self.confidence * 0.5,  # Hypotheses start with lower confidence
            "requires_validation": True,
        }
    
    def to_dict(self) -> dict:
        """Serialize to JSON."""
        return {
            "schema_id": self.schema_id,
            "schema_type": self.schema_type.value,
            "pattern": self.pattern,
            "variables": self.variables,
            "constraints": self.constraints,
            "instances": self.instances,
            "supporting_beliefs": self.supporting_beliefs,
            "confidence": self.confidence,
            "epistemic_class": self.epistemic_class,
            "successful_applications": self.successful_applications,
            "failed_applications": self.failed_applications,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> RelationSchema:
        """Deserialize from JSON."""
        return cls(
            schema_id=data["schema_id"],
            schema_type=SchemaType(data["schema_type"]),
            pattern=data["pattern"],
            variables=data["variables"],
            constraints=data["constraints"],
            instances=[tuple(inst) for inst in data["instances"]],
            supporting_beliefs=data["supporting_beliefs"],
            confidence=data["confidence"],
            epistemic_class=data.get("epistemic_class", "structural"),
            successful_applications=data.get("successful_applications", 0),
            failed_applications=data.get("failed_applications", 0),
            metadata=data.get("metadata", {}),
        )


class SchemaExtractor:
    """
    Extract relation schemas from belief graph.
    
    Identifies repeating patterns and creates abstract schemas.
    """
    
    def __init__(self, min_support: int = 3):
        """
        Args:
            min_support: Minimum number of instances to create schema
        """
        self.min_support = min_support
        self.schemas: dict[str, RelationSchema] = {}
    
    def extract_taxonomic_chains(self, graph: Any) -> list[RelationSchema]:
        """
        Extract taxonomic inheritance chains (X is_a Y is_a Z).
        
        Returns:
            List of taxonomic schemas
        """
        schemas = []
        
        # Find all is_a relationships
        is_a_beliefs = [b for b in graph.beliefs.values() 
                        if b.template == "is_a" and b.active]
        
        # Build taxonomy tree using subject → object
        taxonomy = {}  # child → parent mapping
        for belief in is_a_beliefs:
            child = belief.subject
            parent = belief.object
            if child and parent:
                if child not in taxonomy:
                    taxonomy[child] = []
                taxonomy[child].append((parent, belief.id))
        
        # Extract transitive chains
        for child, parents in taxonomy.items():
            for parent, belief_id in parents:
                # Check if parent has parents (transitive chain)
                if parent in taxonomy:
                    for grandparent, parent_belief_id in taxonomy[parent]:
                        # Create schema: X is_a Y, Y is_a Z ⇒ X is_a Z
                        schema = RelationSchema(
                            schema_id=f"tax_{child}_{parent}_{grandparent}",
                            schema_type=SchemaType.TAXONOMIC,
                            pattern=f"{child} is_a {parent} is_a {grandparent}",
                            variables=['X', 'Y', 'Z'],
                            constraints={'epistemic_class': 'structural'},
                            instances=[(child, parent, grandparent)],
                            supporting_beliefs=[belief_id, parent_belief_id],
                            confidence=0.9,
                            epistemic_class="structural",
                        )
                        schemas.append(schema)
        
        return schemas
    
    def extract_property_propagation(self, graph: Any) -> list[RelationSchema]:
        """
        Extract property propagation patterns.
        
        Example:
            Mammals produce milk
            Bats are mammals
            ⇒ Bats produce milk
        
        Returns:
            List of property propagation schemas
        """
        schemas = []
        
        # Find all is_a relationships
        is_a_beliefs = [b for b in graph.beliefs.values() 
                        if b.template == "is_a" and b.active]
        
        # Group by parent concept using subject → object
        by_parent = {}
        for belief in is_a_beliefs:
            child = belief.subject
            parent = belief.object
            if child and parent:
                if parent not in by_parent:
                    by_parent[parent] = []
                by_parent[parent].append((child, belief.id))
        
        # For each parent, find properties attributed to it
        for parent, children in by_parent.items():
            if len(children) < self.min_support:
                continue
            
            # Find properties of parent
            parent_properties = [b for b in graph.beliefs.values()
                                if parent in b.entities and b.active
                                and b.template != "is_a"
                                and b.epistemic_class == "structural"]  # Only structural!
            
            for prop_belief in parent_properties:
                predicates = list(prop_belief.predicates)
                if predicates:
                    predicate = predicates[0]
                    schema = RelationSchema(
                        schema_id=f"prop_{parent}_{predicate}",
                        schema_type=SchemaType.PROPERTY,
                        pattern=f"X is_a {parent} ⇒ X {predicate}",
                        variables=['X'],
                        constraints={'parent': parent, 'predicate': predicate,
                                    'epistemic_class': 'structural'},
                        instances=[(child,) for child, _ in children],
                        supporting_beliefs=[belief_id for _, belief_id in children] + [prop_belief.id],
                        confidence=0.8,
                        epistemic_class="structural",
                    )
                    schemas.append(schema)
        
        return schemas
    
    def extract_all(self, graph: Any) -> list[RelationSchema]:
        """Extract all schemas from graph."""
        schemas = []
        schemas.extend(self.extract_taxonomic_chains(graph))
        schemas.extend(self.extract_property_propagation(graph))
        return schemas
