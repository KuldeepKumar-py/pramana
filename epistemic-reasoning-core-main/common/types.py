"""
MARC Core Types - Quantifier Logic, Relation Frames & Structured Answers

This module defines the core types needed for proper reasoning:
- Relation Frames (TAXONOMIC, SPATIAL, MATERIAL, FUNCTIONAL, etc.)
- Quantifiers (ALL, SOME, MOST, FEW, NONE)
- Polarity (POSITIVE, NEGATIVE)
- Structured Answer type
- Verdict enum

These types enforce correct logical semantics throughout MARC.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# RELATION FRAMES — Structural Properties of Relations
# ═══════════════════════════════════════════════════════════════════

class RelationKind(str, Enum):
    """
    Categorical types of relations with distinct structural properties.
    
    These are NOT arbitrary - they reflect how humans structure knowledge:
    - TAXONOMIC: Type membership ("bat is-a mammal")
    - SPATIAL: Containment/location ("London in Europe")
    - MATERIAL: Physical properties ("copper conducts")
    - FUNCTIONAL: Capabilities/affordances ("birds fly")
    - STATE: Conditions ("water is liquid")
    - TEMPORAL: Time-based relations ("occurs before")
    """
    TAXONOMIC = "taxonomic"      # is_a, subclass_of, instance_of
    SPATIAL = "spatial"          # located_in, part_of, contains
    MATERIAL = "material"        # made_of, composed_of, property_of
    FUNCTIONAL = "functional"    # can_do, affords, enables
    STATE = "state"             # is_state, has_state
    TEMPORAL = "temporal"        # before, after, during
    GENERIC = "generic"          # Fallback for untyped relations


@dataclass
class RelationFrame:
    """
    Structural properties of a relation type.
    
    Defines HOW a relation behaves, not just what it's called.
    This enables principled reasoning without hardcoded hacks.
    """
    kind: RelationKind
    transitive: bool = False      # If A→B and B→C, then A→C?
    symmetric: bool = False       # If A→B, then B→A?
    inherits: bool = False        # Properties propagate down hierarchy?
    negation_blocks: bool = False # Negative on parent blocks positive?
    
    @staticmethod
    def for_predicate(predicate: str) -> 'RelationFrame':
        """
        Map predicate to its relation frame.
        
        This is where we define the structural semantics of predicates.
        NOT hardcoded answers - structural inference rules.
        
        Uses pattern matching for flexibility (e.g., "produces_milk" matches "produces").
        """
        pred_lower = predicate.lower()
        
        # TAXONOMIC relations (transitive, inheritable)
        if pred_lower in {'is_a', 'subclass_of', 'instance_of', 'type_of'}:
            return RelationFrame(
                kind=RelationKind.TAXONOMIC,
                transitive=True,
                inherits=True,
                negation_blocks=True
            )
        
        # SPATIAL relations (transitive containment)
        elif (pred_lower in {'located_in', 'part_of', 'contains', 'within', 'habitat'} or
              'located' in pred_lower or 'contain' in pred_lower):
            return RelationFrame(
                kind=RelationKind.SPATIAL,
                transitive=True,
                inherits=False,  # Location doesn't inherit
                negation_blocks=False
            )
        
        # MATERIAL properties (inheritable)
        elif (pred_lower in {'made_of', 'composed_of', 'has_property', 'conducts', 'is_conductive'} or
              'made' in pred_lower or 'composed' in pred_lower or 'conduct' in pred_lower):
            return RelationFrame(
                kind=RelationKind.MATERIAL,
                transitive=False,
                inherits=True,
                negation_blocks=True
            )
        
        # FUNCTIONAL capabilities (inheritable)
        # This is the KEY category - most biological/behavioral predicates
        elif (pred_lower in {'can_fly', 'can_swim', 'lays_eggs', 'breathes'} or
              'produces' in pred_lower or  # produces_milk, produces_sound
              'breathe' in pred_lower or   # breathes_air
              'has_' in pred_lower or      # has_gills, has_beaks, has_backbones, has_wings
              'lays' in pred_lower or
              'can_' in pred_lower):       # can_fly, can_swim
            return RelationFrame(
                kind=RelationKind.FUNCTIONAL,
                transitive=False,
                inherits=True,
                negation_blocks=True
            )
        
        # STATE predicates (context-dependent, not inheritable)
        elif (pred_lower in {'is_liquid', 'is_solid', 'is_gaseous', 'is_warm', 'is_cold'} or
              pred_lower.startswith('is_')):  # is_X usually means state
            return RelationFrame(
                kind=RelationKind.STATE,
                transitive=False,
                inherits=False,
                negation_blocks=False
            )
        
        # TEMPORAL (transitive ordering)
        elif pred_lower in {'before', 'after', 'during', 'simultaneous'}:
            return RelationFrame(
                kind=RelationKind.TEMPORAL,
                transitive=True,
                inherits=False,
                negation_blocks=False
            )

        # COMPARATIVE / ORDERING (transitive)
        elif (pred_lower in {'larger_than', 'smaller_than', 'taller_than', 'shorter_than', 'more_than', 'less_than'} or
              '_than' in pred_lower):
            return RelationFrame(
                kind=RelationKind.GENERIC, # Or added RelationKind.COMPARATIVE if avoiding schema change
                transitive=True,
                inherits=False,
                negation_blocks=False
            )
        
        # GENERIC fallback (assume inheritable for biological/physical predicates)
        else:
            # Default: assume predicates CAN inherit (conservative for reasoning)
            return RelationFrame(
                kind=RelationKind.GENERIC,
                transitive=False,
                inherits=True,  # Changed from False - allow inheritance by default
                negation_blocks=True  # Changed from False - respect negations
            )


# ═══════════════════════════════════════════════════════════════════
# QUANTIFIERS (Universal/Existential Logic)
# ═══════════════════════════════════════════════════════════════════

class Quantifier(str, Enum):
    """
    Quantifier type for beliefs.
    
    Maps natural language quantifiers to logical operators:
    - ALL: ∀ (universal quantification)
    - SOME: ∃ (existential quantification)
    - MOST: probabilistic majority (>50%)
    - FEW: probabilistic minority (<50%)
    - NONE: ¬∃ (negated existential)
    - UNSPECIFIED: default (treated as SOME when positive, NONE when negative)
    """
    ALL = "ALL"           # ∀x P(x) - "All mammals breathe air"
    SOME = "SOME"         # ∃x P(x) - "Some birds can swim"
    MOST = "MOST"         # Majority - "Most cats are independent"
    FEW = "FEW"           # Minority - "Few mammals lay eggs"
    NONE = "NONE"         # ¬∃x P(x) - "No mammals have gills"
    UNSPECIFIED = "UNSPECIFIED"  # Default when not explicitly stated
    
    @classmethod
    def from_text(cls, text: str) -> "Quantifier":
        """
        Extract quantifier from natural language text.
        
        Examples:
            "All birds fly" → ALL
            "Some fish walk" → SOME
            "Most dogs are social" → MOST
            "Few mammals lay eggs" → FEW
            "No reptiles produce milk" → NONE
            "Birds have feathers" → UNSPECIFIED
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["all ", "every ", "each "]):
            return cls.ALL
        elif any(word in text_lower for word in ["some ", "certain ", "a few", "several "]):
            return cls.SOME
        elif any(word in text_lower for word in ["most ", "many ", "majority "]):
            return cls.MOST
        elif any(word in text_lower for word in ["few ", "rarely ", "seldom "]):
            return cls.FEW
        elif any(word in text_lower for word in ["no ", "none ", "never "]):
            return cls.NONE
        else:
            return cls.UNSPECIFIED


# ═══════════════════════════════════════════════════════════════════
# POLARITY (Positive/Negative Assertion)
# ═══════════════════════════════════════════════════════════════════

class Polarity(str, Enum):
    """
    Polarity of a belief (affirmative or negative).
    
    Examples:
        POSITIVE: "Mammals breathe air"
        NEGATIVE: "Mammals do not have gills"
    """
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    
    @classmethod
    def from_text(cls, text: str) -> "Polarity":
        """
        Extract polarity from natural language text.
        
        Negative indicators:
            - "do not", "does not", "did not"
            - "cannot", "can't"
            - "are not", "is not", "was not"
            - "have no", "has no"
            - "never", "nothing", "nobody"
        
        Examples:
            "Mammals breathe air" → POSITIVE
            "Mammals do not have gills" → NEGATIVE
            "Birds cannot breathe underwater" → NEGATIVE
        """
        text_lower = text.lower()
        
        negative_indicators = [
            "do not", "does not", "did not", "don't", "doesn't", "didn't",
            "cannot", "can't", "can not",
            "are not", "is not", "was not", "were not", "aren't", "isn't", "wasn't", "weren't",
            "have not", "has not", "had not", "haven't", "hasn't", "hadn't",
            "have no", "has no", "had no",
            "never", "nothing", "nobody", "nowhere", "no one",
            "without",
        ]
        
        for indicator in negative_indicators:
            if indicator in text_lower:
                return cls.NEGATIVE
        
        return cls.POSITIVE


# ═══════════════════════════════════════════════════════════════════
# VERDICT (Three-Valued Logic)
# ═══════════════════════════════════════════════════════════════════

class Verdict(str, Enum):
    """
    Three-valued logic for query answers.
    
    YES: Affirmative answer with supporting evidence
    NO: Negative answer with blocking evidence OR explicit negation
    UNKNOWN: Insufficient evidence or contradictory evidence
    """
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    INVALID = "invalid"  # Malformed query, namespace violation, or non-epistemic input
    CONFLICT = "conflict" # Mutually defeating information (e.g. Nixon Diamond: Quaker vs Republican)


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED ANSWER TYPE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Answer:
    """
    Structured answer with full reasoning trace.
    
    Separates reasoning (verdict/support/blockers) from presentation.
    UI layer converts this to natural language.
    
    Fields:
        verdict: YES/NO/UNKNOWN
        support: list of belief IDs that support the answer
        blockers: list of belief IDs that block/contradict the answer
        confidence: overall confidence score [0.0, 1.0]
        reasoning_steps: list of derivation steps
        query: original query text
        conflicts: contradictions detected during reasoning
    """
    verdict: Verdict
    query: str
    support: list[str] = field(default_factory=list)  # Belief IDs
    blockers: list[str] = field(default_factory=list)  # Belief IDs
    confidence: float = 0.0
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, rule: str, inputs: list[str], output: str, confidence: float | None = None):
        """Add a reasoning step to the trace."""
        self.reasoning_steps.append({
            "rule": rule,
            "inputs": inputs,
            "output": output,
            "confidence": confidence,
        })
    
    def add_blocker(self, belief_id: str, reason: str):
        """Add a blocking belief."""
        self.blockers.append(belief_id)
        self.add_step(
            rule="blocker",
            inputs=[belief_id],
            output=f"Blocked: {reason}",
            confidence=None
        )
    
    def add_support(self, belief_id: str, reason: str, conf: float):
        """Add a supporting belief."""
        self.support.append(belief_id)
        self.add_step(
            rule="support",
            inputs=[belief_id],
            output=f"Supports: {reason}",
            confidence=conf
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "verdict": self.verdict.value,
            "query": self.query,
            "support": self.support,
            "blockers": self.blockers,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "conflicts": self.conflicts,
            "metadata": self.metadata,
        }
    
    def to_natural_language(self) -> str:
        """
        Convert structured answer to natural language.
        
        This is the ONLY place where verdict becomes "Yes" or "No" string.
        All internal reasoning uses Verdict enum.
        """
        if self.verdict == Verdict.YES:
            return "Yes."
        elif self.verdict == Verdict.NO:
            return "No."
        elif self.verdict == Verdict.INVALID:
            return "The question is invalid or malformed."
        elif self.verdict == Verdict.CONFLICT:
            return "I have found conflicting information."
        
        return "I do not know."


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def extract_quantifier_and_polarity(text: str) -> tuple[Quantifier, Polarity]:
    """
    Extract both quantifier and polarity from text in one pass.
    
    Returns:
        (quantifier, polarity) tuple
    
    Examples:
        "All mammals breathe air" → (ALL, POSITIVE)
        "No reptiles produce milk" → (NONE, POSITIVE)  # NONE includes negation
        "Most mammals do not lay eggs" → (MOST, NEGATIVE)
        "Birds have feathers" → (UNSPECIFIED, POSITIVE)
    """
    quantifier = Quantifier.from_text(text)
    polarity = Polarity.from_text(text)
    
    # Special case: NONE quantifier implies negation is in the quantifier, not polarity
    # "No mammals have gills" → (NONE, POSITIVE) not (UNSPECIFIED, NEGATIVE)
    if quantifier == Quantifier.NONE:
        polarity = Polarity.POSITIVE
    
    return (quantifier, polarity)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def validate_quantifier(q: Any) -> Quantifier:
    """Validate and normalize quantifier."""
    if isinstance(q, Quantifier):
        return q
    elif isinstance(q, str):
        try:
            return Quantifier(q.upper())
        except ValueError:
            return Quantifier.UNSPECIFIED
    else:
        return Quantifier.UNSPECIFIED


def validate_polarity(p: Any) -> Polarity:
    """Validate and normalize polarity."""
    if isinstance(p, Polarity):
        return p
    elif isinstance(p, str):
        try:
            return Polarity(p.upper())
        except ValueError:
            return Polarity.POSITIVE
    else:
        return Polarity.POSITIVE


def validate_verdict(v: Any) -> Verdict:
    """Validate and normalize verdict."""
    if isinstance(v, Verdict):
        return v
    elif isinstance(v, str):
        v_upper = v.upper()
        if v_upper in ["YES", "Y", "TRUE", "T", "1"]:
            return Verdict.YES
        elif v_upper in ["NO", "N", "FALSE", "F", "0"]:
            return Verdict.NO
        elif v_upper == "INVALID":
            return Verdict.INVALID
        elif v_upper == "CONFLICT":
            return Verdict.CONFLICT
        else:
            return Verdict.UNKNOWN
    else:
        return Verdict.UNKNOWN
