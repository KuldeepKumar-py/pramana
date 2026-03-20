"""
CHITTA UTILITIES

Helper functions for ID generation, validation, math, and timestamps.
"""

import math
import uuid
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# ID GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_belief_id() -> str:
    """Generate unique belief ID."""
    return f"b_{uuid.uuid4().hex[:12]}"


def generate_edge_id() -> str:
    """Generate unique edge ID."""
    return f"e_{uuid.uuid4().hex[:12]}"


# ═══════════════════════════════════════════════════════════════════
# TIMESTAMP UTILITIES
# ═══════════════════════════════════════════════════════════════════

def now_utc() -> datetime:
    """Get current UTC timestamp with timezone awareness."""
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string."""
    return dt.isoformat()


def from_iso(iso_str: str) -> datetime:
    """Parse ISO 8601 string to datetime."""
    return datetime.fromisoformat(iso_str)


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE MATHEMATICS (LOG-ODDS)
# ═══════════════════════════════════════════════════════════════════

def logit(p: float) -> float:
    """
    Convert probability to log-odds.
    
    logit(p) = ln(p / (1 - p))
    
    Args:
        p: probability in [0, 1]
    
    Returns:
        log-odds in (-∞, +∞)
    
    Raises:
        ValueError: if p not in (0, 1) exclusive
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """
    Convert log-odds to probability.
    
    sigmoid(x) = 1 / (1 + exp(-x))
    
    Args:
        x: log-odds in (-∞, +∞)
    
    Returns:
        probability in (0, 1)
    """
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def update_confidence_logodds(
    current_confidence: float,
    evidence_score: float,
    min_conf: float = 0.01,
    max_conf: float = 0.99,
) -> float:
    """
    Update confidence using log-odds addition.
    
    logit(p_new) = logit(p_old) + evidence_score
    p_new = sigmoid(logit(p_new))
    
    Args:
        current_confidence: current probability [0, 1]
        evidence_score: evidence in log-odds space
        min_conf: minimum allowed confidence
        max_conf: maximum allowed confidence
    
    Returns:
        updated confidence clamped to [min_conf, max_conf]
    """
    # Clamp input to valid range
    p = max(min_conf, min(max_conf, current_confidence))
    
    # Convert to log-odds, add evidence, convert back
    current_logodds = logit(p)
    new_logodds = current_logodds + evidence_score
    new_conf = sigmoid(new_logodds)
    
    # Clamp output
    return max(min_conf, min(max_conf, new_conf))


def decay_confidence(
    confidence: float,
    lambda_decay: float,
    delta_time_seconds: float,
) -> float:
    """
    Apply exponential decay to confidence.
    
    p(t) = p₀ * exp(-λ * Δt)
    
    Args:
        confidence: current confidence
        lambda_decay: decay rate (e.g., 1e-6 for slow decay)
        delta_time_seconds: time elapsed in seconds
    
    Returns:
        decayed confidence
    """
    return confidence * math.exp(-lambda_decay * delta_time_seconds)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

VALID_EPISTEMIC_STATES = {"asserted", "unknown", "hypothetical"}

VALID_TEMPLATES = {
    "relation",
    "event",
    "has_attr",
    "is_a",
    "temporal",
    "moral",
    "action",
    "epistemic",
    "spatial",
    "causal",
    "part_of",
}

VALID_EDGE_TYPES = {
    "supports",
    "contradicts",
    "derived_from",
    "refines",
    "causes",
    "answers",
    "related",
    "part_of",
    "is_a",
    "temporal_before",
    "temporal_after",
}


def validate_confidence(confidence: float) -> float:
    """Validate and clamp confidence to [0.0, 1.0]."""
    if not isinstance(confidence, (int, float)):
        raise TypeError(f"Confidence must be numeric, got {type(confidence)}")
    return max(0.0, min(1.0, float(confidence)))


def validate_epistemic_state(state: str) -> str:
    """Validate epistemic state."""
    if state not in VALID_EPISTEMIC_STATES:
        raise ValueError(
            f"Invalid epistemic state '{state}'. "
            f"Must be one of {VALID_EPISTEMIC_STATES}"
        )
    return state


def validate_template(template: str) -> str:
    """Validate template type."""
    if template not in VALID_TEMPLATES:
        raise ValueError(
            f"Invalid template '{template}'. "
            f"Must be one of {VALID_TEMPLATES}"
        )
    return template


def validate_edge_type(edge_type: str) -> str:
    """Validate edge/relation type."""
    if edge_type not in VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge type '{edge_type}'. "
            f"Must be one of {VALID_EDGE_TYPES}"
        )
    return edge_type


def validate_canonical(canonical: dict) -> dict:
    """Validate canonical structure is a dictionary."""
    if not isinstance(canonical, dict):
        raise TypeError(f"Canonical must be dict, got {type(canonical)}")
    return canonical


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION UTILITIES
# ═══════════════════════════════════════════════════════════════════

def extract_entities(canonical: dict) -> set[str]:
    """
    Extract all entity names from canonical structure.
    
    Looks for:
    - canonical["entities"] (list)
    - canonical["subject"], canonical["object"] (str)
    - canonical["X"], canonical["Y"] (str)
    
    Returns:
        set of entity strings
    """
    entities = set()
    
    # Direct entities list
    if "entities" in canonical and isinstance(canonical["entities"], list):
        entities.update(str(e) for e in canonical["entities"])
    
    # Common keys
    for key in ["subject", "object", "agent", "patient", "X", "Y", "entity"]:
        if key in canonical and canonical[key]:
            entities.add(str(canonical[key]))
    
    return entities


def extract_predicates(canonical: dict) -> set[str]:
    """
    Extract predicates/relations from canonical structure.
    
    Returns:
        set of predicate strings
    """
    predicates = set()
    
    for key in ["relation_type", "predicate", "action", "event_type"]:
        if key in canonical and canonical[key]:
            predicates.add(str(canonical[key]))
    
    return predicates
