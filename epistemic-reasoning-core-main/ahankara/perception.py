"""
Perceptual Priors Module

NON-INFERABLE, NON-INHERITABLE perceptual priors.
These represent learned associations from direct observation, not logical derivation.

Design Principles:
- Lower confidence than logical beliefs
- Cannot be inherited taxonomically
- Explicitly labeled as "perceptual"
- Mirrors human perceptual knowledge: "gold is shiny because I saw it"

This is NOT perception - it's perceptual PRIORS.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PriorConfidence(Enum):
    """Confidence levels for perceptual priors (always < logical beliefs)"""
    HIGH = 0.85      # Strong perceptual association (water → liquid)
    MEDIUM = 0.70    # Moderate association (gold → shiny)
    LOW = 0.50       # Weak association (clouds → white)


@dataclass
class PerceptualPrior:
    """A single perceptual prior"""
    entity: str
    property: str
    confidence: PriorConfidence
    source: str = "perceptual"  # Always labeled as perceptual


class PerceptualPriors:
    """
    Manages perceptual priors - knowledge from observation, not inference.
    
    Critical constraints:
    - NON-INFERABLE: Cannot be derived logically
    - NON-INHERITABLE: Do not propagate taxonomically
    - LOWER CONFIDENCE: Always < 1.0, explicitly uncertain
    - EXPLICITLY LABELED: Marked as "perceptual" not "logical"
    """
    
    def __init__(self):
        # Hardcoded perceptual priors (could be learned from observation)
        self.priors: Dict[str, List[PerceptualPrior]] = {
            "gold": [
                PerceptualPrior("gold", "shiny", PriorConfidence.HIGH),
                PerceptualPrior("gold", "yellow", PriorConfidence.HIGH),
                PerceptualPrior("gold", "metallic", PriorConfidence.HIGH),
            ],
            "water": [
                PerceptualPrior("water", "liquid", PriorConfidence.HIGH),
                PerceptualPrior("water", "transparent", PriorConfidence.MEDIUM),
                PerceptualPrior("water", "flows", PriorConfidence.HIGH),
            ],
            "copper": [
                PerceptualPrior("copper", "conductive", PriorConfidence.HIGH),
                PerceptualPrior("copper", "reddish", PriorConfidence.MEDIUM),
                PerceptualPrior("copper", "metallic", PriorConfidence.HIGH),
            ],
            "ice": [
                PerceptualPrior("ice", "solid", PriorConfidence.HIGH),
                PerceptualPrior("ice", "cold", PriorConfidence.HIGH),
                PerceptualPrior("ice", "transparent", PriorConfidence.MEDIUM),
            ],
            "fire": [
                PerceptualPrior("fire", "hot", PriorConfidence.HIGH),
                PerceptualPrior("fire", "bright", PriorConfidence.HIGH),
                PerceptualPrior("fire", "dangerous", PriorConfidence.MEDIUM),
            ],
            "sky": [
                PerceptualPrior("sky", "blue", PriorConfidence.MEDIUM),
                PerceptualPrior("sky", "above", PriorConfidence.HIGH),
            ],
            "grass": [
                PerceptualPrior("grass", "green", PriorConfidence.MEDIUM),
            ],
            "blood": [
                PerceptualPrior("blood", "red", PriorConfidence.HIGH),
                PerceptualPrior("blood", "liquid", PriorConfidence.HIGH),
            ],
        }
    
    def query(self, entity: str, property: str) -> Optional[PerceptualPrior]:
        """
        Query for a perceptual prior.
        
        Returns:
            PerceptualPrior if found, None otherwise
            
        Note: This is RETRIEVAL ONLY. No inference, no inheritance.
        """
        entity_normalized = entity.lower().strip()
        property_normalized = property.lower().strip()
        
        if entity_normalized not in self.priors:
            return None
        
        for prior in self.priors[entity_normalized]:
            if prior.property.lower() == property_normalized:
                return prior
        
        return None
    
    def has_property(self, entity: str, property: str) -> bool:
        """Check if entity has perceptual property"""
        return self.query(entity, property) is not None
    
    def get_properties(self, entity: str) -> List[PerceptualPrior]:
        """Get all perceptual properties for an entity"""
        entity_normalized = entity.lower().strip()
        return self.priors.get(entity_normalized, [])
    
    def get_confidence(self, entity: str, property: str) -> float:
        """
        Get confidence for a perceptual prior.
        
        Returns:
            Confidence value (0.0-1.0), or 0.0 if not found
        """
        prior = self.query(entity, property)
        if prior is None:
            return 0.0
        return prior.confidence.value
    
    def format_answer(self, entity: str, property: str) -> Optional[str]:
        """
        Format answer with epistemic qualification.
        
        Returns formatted answer like:
        "Yes (perceptual: 85% confidence) - gold appears shiny"
        """
        prior = self.query(entity, property)
        if prior is None:
            return None
        
        confidence_pct = int(prior.confidence.value * 100)
        return f"Yes (perceptual: {confidence_pct}% confidence) - {entity} appears {property}"


# Singleton instance
_perceptual_priors = None

def get_perceptual_priors() -> PerceptualPriors:
    """Get singleton instance of perceptual priors"""
    global _perceptual_priors
    if _perceptual_priors is None:
        _perceptual_priors = PerceptualPriors()
    return _perceptual_priors
