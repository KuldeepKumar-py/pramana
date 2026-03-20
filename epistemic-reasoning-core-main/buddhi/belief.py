"""
BELIEF MODULE
Reconstructed from usage patterns.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

class Polarity(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()
    
    
    def __str__(self):
        return self.name

class EpistemicType(str, Enum):
    """
    Nature of the belief's origin and strength.
    """
    AXIOM = "AXIOM"               # Hard constraint, universal truth
    DEFAULT = "DEFAULT"           # General tendency, defeasible "Birds can fly"
    OBSERVATION = "OBSERVATION"   # Instance fact "Socrates is a man"
    EXCEPTION = "EXCEPTION"       # Specific override "Penguins cannot fly"
    HYPOTHESIS = "HYPOTHESIS"     # World-fork assumption
    INFERRED = "INFERRED"         # Derived internally (keep for trace)
    UNKNOWN = "UNKNOWN"           # Placeholder or gap in knowledge

    def __str__(self):
        return self.value

@dataclass
class Provenance:
    op: str
    from_source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Belief:
    template: str
    canonical: Dict[str, Any]
    confidence: float
    original_text: Optional[str] = None
    statement_text: Optional[str] = None
    statement_text: Optional[str] = None
    epistemic_state: EpistemicType = EpistemicType.OBSERVATION
    epistemic_class: Optional[str] = None
    epistemic_class: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    # Explicit polarity from parser
    polarity_value: int = 1  # 1 = Positive, -1 = Negative
    
    # Internal state
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Lifecycle
    decay_rate: float = 0.995
    activation: float = 1.0  # Starts fresh
    evidence_count: int = 1
    
    # History
    provenance: List[Provenance] = field(default_factory=list)
    versions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Connectivity
    edges_out: Dict[str, List[str]] = field(default_factory=dict)
    edges_in: Dict[str, List[str]] = field(default_factory=dict)
    
    # Cached properties
    _polarity: Polarity = Polarity.POSITIVE
    
    def __post_init__(self):
        # Ensure epistemic_state is Enum
        if isinstance(self.epistemic_state, str):
            try:
                # Handle legacy mappings
                s_upper = self.epistemic_state.upper()
                if s_upper == "ASSERTED":
                    self.epistemic_state = EpistemicType.OBSERVATION
                elif s_upper == "OVERRIDDEN":
                    self.epistemic_state = EpistemicType.EXCEPTION
                else:
                    self.epistemic_state = EpistemicType(s_upper)
            except ValueError:
                # Default fallback
                self.epistemic_state = EpistemicType.OBSERVATION

        if self.statement_text is None and self.original_text:
            self.statement_text = self.original_text
        
        # Infer polarity from template or explicitly set?
        # Assuming Manas handles polarity detection and puts it in source or template?
        # For now default to POSITIVE
        pass

    @property
    def polarity(self) -> Polarity:
        if self.polarity_value < 0:
            return Polarity.NEGATIVE
        return Polarity.POSITIVE

    @property
    def entities(self) -> set[str]:
        # Extract entities from canonical
        ents = set()
        if 'entities' in self.canonical and isinstance(self.canonical['entities'], list):
            ents.update(self.canonical['entities'])
        if 'subject' in self.canonical: ents.add(self.canonical['subject'])
        if 'object' in self.canonical: ents.add(self.canonical['object'])
        return ents

    @property
    def predicates(self) -> set[str]:
        preds = set()
        if 'predicate' in self.canonical:
            preds.add(self.canonical['predicate'])
        if 'predicate_type' in self.canonical:
            preds.add(self.canonical['predicate_type'])
        if self.template == 'is_a':
            preds.add('is_a')
        if 'relation_type' in self.canonical:
            preds.add(self.canonical['relation_type'])
        if 'attribute' in self.canonical:
            preds.add(self.canonical['attribute'])
        return preds
    
    @property
    def subject(self) -> Optional[str]:
        return self.canonical.get('subject')
        
    @property
    def object(self) -> Optional[str]:
        return self.canonical.get('object')

    def add_provenance(self, op: str, from_source: str, score: float, metadata: Dict[str, Any] = None):
        self.provenance.append(Provenance(op, from_source, score, metadata or {}))

    def deactivate(self):
        self.active = False
        self.updated_at = datetime.now(timezone.utc)

    def apply_decay(self, current_time: datetime) -> float:
        # Simple decay simulation
        # In real implementation this might be time-based
        self.confidence *= self.decay_rate
        return self.confidence

    def _add_edge_out(self, relation: str, target_id: str):
        if relation not in self.edges_out:
            self.edges_out[relation] = []
        if target_id not in self.edges_out[relation]:
            self.edges_out[relation].append(target_id)

    def _add_edge_in(self, relation: str, source_id: str):
        if relation not in self.edges_in:
            self.edges_in[relation] = []
        if source_id not in self.edges_in[relation]:
            self.edges_in[relation].append(source_id)

    def _remove_edge_out(self, relation: str, target_id: str):
        if relation in self.edges_out and target_id in self.edges_out[relation]:
            self.edges_out[relation].remove(target_id)

    def _remove_edge_in(self, relation: str, source_id: str):
        if relation in self.edges_in and source_id in self.edges_in[relation]:
            self.edges_in[relation].remove(source_id)
            
    # For versioning test support via setter interception?
    # Python dataclasses don't support property setters easily for fields defined in init.
    # But test_chitta.py does: belief.confidence = 0.6
    # and expects versions to update.
    # To support this, we would need to wrap confidence in a property.
    # But for now, let's just add a method `update_confidence` if needed or ignore versioning test failure/fix test.
    # Actually, `test_chitta.py` explicitly checks `len(belief.versions)`.
    # So `confidence` MUST be a property that updates versions on set.
    
    # We can't easily change `confidence` to property in dataclass without renaming the field.
    # Let's pivot: Use `_confidence` field and `confidence` property.
    
    def __setattr__(self, name, value):
        if name == 'confidence':
            # Record version before change (if initialized)
            if hasattr(self, 'versions') and hasattr(self, 'confidence'):
                self.versions.append({
                    'confidence': self.confidence,
                    'epistemic_state': self.epistemic_state,
                    'timestamp': datetime.now(timezone.utc)
                })
        super().__setattr__(name, value)

    def reinforce(self, boost: float = 0.05, success: bool = True):
        """Reinforce belief confidence based on usage success."""
        if success:
            self.confidence = min(1.0, self.confidence + boost)
        else:
            self.confidence = max(0.01, self.confidence - boost)
        self.evidence_count += 1
        self.updated_at = datetime.now(timezone.utc)

