"""
PREDICATE NORMALIZER - Semantic Normalization for Perception

This module fixes MARC's perception layer by normalizing predicates to
canonical forms. This ensures:
- "Fish live in water" → habitat(water)
- "Do fish live in water?" → habitat(water)
- Both map to SAME representation

This is NOT reasoning. This is PERCEPTION.
Perception must be accurate before reasoning can be evaluated.

EPISTEMIC CLASSES:
- STRUCTURAL: anatomy, taxonomy, physics (safe to propagate via inheritance)
- BEHAVIORAL: actions, tendencies (require direct teaching)
- ABSTRACT: evaluative, subjective (require direct teaching, block propagation)
"""

from typing import Tuple, Set
import re
from enum import Enum


class EpistemicClass(Enum):
    """
    Epistemic classification of predicates.
    
    Controls inference safety:
    - STRUCTURAL predicates may propagate via inheritance (is_a, has_gills)
    - BEHAVIORAL predicates require direct teaching (hunts, migrates)
    - ABSTRACT predicates require direct teaching (intelligent, dangerous)
    
    This prevents semantic bleed-through:
        "Whales are mammals" + "Some mammals are intelligent" ≠> "Whales are intelligent"
    """
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    ABSTRACT = "abstract"


# ═══════════════════════════════════════════════════════════════════
# CANONICAL PREDICATE MAPPINGS
# ═══════════════════════════════════════════════════════════════════

# Habitat/Location predicates
HABITAT_PATTERNS = {
    r'\blive[sd]?\s+in\b': 'habitat',
    r'\bexist[s]?\s+in\b': 'habitat',
    r'\binhabit[s]?\b': 'habitat',
    r'\bresides?\s+in\b': 'habitat',
    r'\bfound\s+in\b': 'habitat',
    r'\boccup(?:y|ies)\b': 'habitat',
}

# Capability predicates
CAPABILITY_PATTERNS = {
    r'\b(\w+)\s+can\s+(\w+)': 'can_{1}',  # birds can fly → can_fly (capture entity then action)
    r'\bcan\s+(\w+)\s+(\w+)': 'can_{1}',  # can birds fly → can_fly (question form)
    r'\bcannot\s+(\w+)': 'can_{0}',  # cannot fly → can_fly
    r'\bable\s+to\s+(\w+)': 'can_{0}',  # able to swim → can_swim
    r'\bbreathe[s]?\s+(\w+)': 'breathes_{0}',  # breathe air → breathes_air
}

# Possession/Attribute predicates
ATTRIBUTE_PATTERNS = {
    r'\bha(?:s|ve)\s+(\w+)': 'has_{0}',  # has gills → has_gills
    r'\bpossess(?:es)?\s+(\w+)': 'has_{0}',
    r'\bown[s]?\s+(\w+)': 'has_{0}',
}

# Taxonomy predicates
TAXONOMY_PATTERNS = {
    r'\bis\s+a\s+(\w+)': 'is_a',
    r'\bare\s+(\w+)': 'is_a',
    r'\bkind\s+of\s+(\w+)': 'is_a',
    r'\btype\s+of\s+(\w+)': 'is_a',
    r'\binstance\s+of\s+(\w+)': 'is_a',
}

# Production predicates
PRODUCTION_PATTERNS = {
    r'\bproduce[sd]?\s+(\w+)': 'produces_{0}',  # produce milk → produces_milk
    r'\bcreate[sd]?\s+(\w+)': 'produces_{0}',
    r'\bgenerate[sd]?\s+(\w+)': 'produces_{0}',
    r'\bmake[s]?\s+(\w+)': 'produces_{0}',
}

# Action predicates
ACTION_PATTERNS = {
    r'\beat[s]?\s+(\w+)': 'eats_{0}',
    r'\bfeed[s]?\s+on\s+(\w+)': 'eats_{0}',
    r'\bconsume[s]?\s+(\w+)': 'eats_{0}',
    r'\bhunt[s]?\s+(\w+)': 'hunts_{0}',
}

# State predicates
STATE_PATTERNS = {
    r'\bactive\b': 'state_active',
    r'\basleep\b': 'state_sleeping',
    r'\balive\b': 'state_alive',
    r'\bdead\b': 'state_dead',
}

# Abstract/Evaluative predicates (subjective, opinion-based)
ABSTRACT_PATTERNS = {
    r'\bintelligent\b': 'abstract_intelligent',
    r'\bsmart\b': 'abstract_intelligent',
    r'\bclever\b': 'abstract_intelligent',
    r'\bdangerous\b': 'abstract_dangerous',
    r'\bharmful\b': 'abstract_dangerous',
    r'\baggressive\b': 'abstract_aggressive',
    r'\bbeautiful\b': 'abstract_beautiful',
}

# Behavioral predicates (require explicit observation)
BEHAVIORAL_PATTERNS = {
    r'\bmigrate[s]?\b': 'behavior_migrates',
    r'\bpredator[s]?\b': 'behavior_predator',  # Simplified: just detect word
    r'\bhunt[s]?\b': 'behavior_hunts',
    r'\bswim[s]?\b': 'behavior_swims',
    r'\bfly\b|flies\b': 'behavior_flies',
    r'\bwalk[s]?\b': 'behavior_walks',
}


# ═══════════════════════════════════════════════════════════════════
# EPISTEMIC CLASS MAPPINGS
# ═══════════════════════════════════════════════════════════════════

# Maps predicate types to epistemic classes
PREDICATE_EPISTEMIC_CLASS = {
    # STRUCTURAL: Safe to propagate via inheritance
    'habitat': EpistemicClass.STRUCTURAL,
    'taxonomy': EpistemicClass.STRUCTURAL,
    'attribute': EpistemicClass.STRUCTURAL,  # has_gills, has_lungs
    'production': EpistemicClass.STRUCTURAL,  # produces_milk
    'state': EpistemicClass.STRUCTURAL,  # alive, dead
    
    # BEHAVIORAL: Require direct teaching or explicit rules
    'capability': EpistemicClass.BEHAVIORAL,  # can_fly
    'action': EpistemicClass.BEHAVIORAL,  # eats_fish
    'behavior': EpistemicClass.BEHAVIORAL,  # migrates, hunts
    
    # ABSTRACT: Require direct teaching, block all propagation
    'abstract': EpistemicClass.ABSTRACT,  # intelligent, dangerous
    
    # FALLBACK
    'generic': EpistemicClass.BEHAVIORAL,  # Conservative: treat unknowns as behavioral
}


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class PredicateNormalizer:
    """
    Normalizes natural language predicates to canonical forms.
    
    This is perception-layer semantic grounding, not reasoning.
    """
    
    def __init__(self):
        """Initialize normalizer with pattern mappings."""
        self.pattern_groups = [
            # Check ABSTRACT first
            ('abstract', ABSTRACT_PATTERNS),
            # Capability is more specific ("can X") than simple behavior ("X")
            ('capability', CAPABILITY_PATTERNS),
            ('behavior', BEHAVIORAL_PATTERNS),
            # Then STRUCTURAL patterns
            ('habitat', HABITAT_PATTERNS),
            ('taxonomy', TAXONOMY_PATTERNS),
            ('attribute', ATTRIBUTE_PATTERNS),
            ('production', PRODUCTION_PATTERNS),
            ('action', ACTION_PATTERNS),
            ('state', STATE_PATTERNS),
        ]
    
    def normalize(self, text: str, entities: list[str] = None) -> Tuple[str, str, list[str]]:
        """
        Normalize predicate from natural language text.
        
        Args:
            text: Natural language text
            entities: Extracted entities (optional, for context)
        
        Returns:
            (canonical_predicate, predicate_type, arguments)
        
        Examples:
            "Fish live in water" → ("habitat", "habitat", ["water"])
            "Birds can fly" → ("can_fly", "capability", ["fly"])
            "Mammals have gills" → ("has_gills", "attribute", ["gills"])
        """
        text_lower = text.lower().strip()
        
        # Try each pattern group
        for pred_type, patterns in self.pattern_groups:
            for pattern, canonical_template in patterns.items():
                match = re.search(pattern, text_lower)
                if match:
                    # Extract arguments from pattern groups
                    if match.groups():
                        args = [g.strip() for g in match.groups() if g]
                        # Format template with arguments
                        if '{1}' in canonical_template and len(args) > 1:
                            # Use second argument (for "can X Y" → use Y)
                            canonical = canonical_template.format(args[0], args[1])
                        elif '{0}' in canonical_template and args:
                            canonical = canonical_template.format(args[0])
                        else:
                            canonical = canonical_template
                    else:
                        canonical = canonical_template
                        args = []
                    
                    # For habitat, extract location argument
                    if pred_type == 'habitat':
                        # Get what comes after "in"
                        in_match = re.search(r'\bin\s+(\w+)', text_lower)
                        if in_match:
                            args = [in_match.group(1)]
                    
                    return (canonical, pred_type, args)
        
        # Fallback: generic
        return ('generic', 'generic', [])
    
    def normalize_with_confidence(self, text: str, entities: list[str] = None) -> dict:
        """
        Normalize with confidence score, entity extraction, and epistemic class.
        
        Returns:
            {
                "canonical": str,
                "type": str,
                "arguments": list,
                "entities": list,
                "epistemic_class": EpistemicClass,  # NEW: STRUCTURAL/BEHAVIORAL/ABSTRACT
                "confidence": float,
                "matched_pattern": str
            }
        """
        canonical, pred_type, args = self.normalize(text, entities)
        extracted_entities = self._extract_entities(text, pred_type)
        
        # Get epistemic class
        epistemic_class = PREDICATE_EPISTEMIC_CLASS.get(pred_type, EpistemicClass.BEHAVIORAL)
        
        # Confidence based on specificity
        confidence_map = {
            'habitat': 0.90,
            'capability': 0.90,
            'attribute': 0.85,
            'taxonomy': 0.95,
            'production': 0.85,
            'action': 0.85,
            'state': 0.80,
            'abstract': 0.70,  # Lower confidence for subjective
            'behavior': 0.75,  # Moderate for behavioral
            'generic': 0.50,
        }
        
        return {
            "canonical": canonical,
            "type": pred_type,
            "arguments": args,
            "entities": extracted_entities,
            "epistemic_class": epistemic_class,
            "confidence": confidence_map.get(pred_type, 0.50),
            "matched_pattern": f"{pred_type}:{canonical}",
        }
    
    def _extract_entities(self, text: str, pred_type: str) -> list[str]:
        """
        Extract entities based on predicate type.
        
        CRITICAL FIX: Extract interrogative subject for all question types.
        Questions of form "Do X verb Y?" must extract X as subject.
        """
        text_lower = text.lower().strip()
        entities = []
        
        # ═══════════════════════════════════════════════════════════════
        # 🔧 INTERROGATIVE SUBJECT RECOVERY (UNIVERSAL)
        # ═══════════════════════════════════════════════════════════════
        # For questions: "Do <X> <verb> <Y>?"
        # Always extract <X> as the subject entity
        # 
        # Examples:
        #   "Do bats produce milk?" → subject=bat
        #   "Do dolphins breathe air?" → subject=dolphin
        #   "Are whales warm-blooded?" → subject=whale
        # 
        # This runs BEFORE type-specific extraction to ensure we always
        # have a subject for inheritance to anchor to.
        # ═══════════════════════════════════════════════════════════════
        
        # Check if this is a question
        is_question = text_lower.startswith(('do ', 'does ', 'did ', 'can ', 'is ', 'are ', 'was ', 'were ', 'will '))
        
        if is_question:
            # Pattern: "Do/Does/Can/Is/Are <SUBJECT> ..."
            question_match = re.match(r'(?:do|does|did|can|is|are|was|were|will)\s+(\w+)', text_lower)
            if question_match:
                subject = question_match.group(1).strip()
                # Don't add if it's a verb itself
                if subject not in ['live', 'lives', 'breathe', 'breathes', 'fly', 'swim', 'exist', 'produce', 'produces', 'have', 'has']:
                    entities.append(subject)
        
        # ═══════════════════════════════════════════════════════════════
        # Type-specific extraction (supplements interrogative extraction)
        # ═══════════════════════════════════════════════════════════════
        
        if pred_type == 'habitat':
            # Remove question words first
            cleaned = re.sub(r'\b(do|does|did|will|can|is|are)\b', '', text_lower).strip()
            
            # Get subject (first noun before verb)
            words = cleaned.split()
            subject = words[0].strip(".,!?") if words else None
            
            # Get location (after "in/on/at")
            location_match = re.search(r'\b(?:in|on|at)\s+(\w+)', text_lower)
            location = location_match.group(1) if location_match else None
            
            if subject and subject not in ['live', 'lives', 'exist', 'exists']:
                entities.append(subject)
            if location and location not in entities:
                entities.append(location)
        
        elif pred_type == 'capability':
            # Remove question words
            cleaned = re.sub(r'\b(can|do|does|did|will|is|are)\b', '', text_lower).strip()
            
            # Get subject (first word after cleaning)
            words = cleaned.split()
            subject = words[0].strip(".,!?") if words else None
            
            if subject and subject not in ['fly', 'swim', 'walk']:
                entities.append(subject)
        
        elif pred_type == 'attribute':
            # "X has Y" → extract X
            has_match = re.search(r'(\w+)\s+(?:has?|have|possess)', text_lower)
            if has_match:
                entities.append(has_match.group(1))
        
        elif pred_type == 'generic':
            # Extract first 2-3 meaningful words
            words = text.split()
            stop_words = {'is', 'are', 'was', 'were', 'be', 'been', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'do', 'does'}
            entities = [w.lower().strip(".,!?") for w in words[:4] if w.lower() not in stop_words][:2]
        
        return entities



# ═══════════════════════════════════════════════════════════════════
# ENTITY NORMALIZER (Simple for now)
# ═══════════════════════════════════════════════════════════════════

class EntityNormalizer:
    """
    Normalizes entity references to canonical forms.
    
    Examples:
        "fish" → "fish"
        "fishes" → "fish"
        "mammals" → "mammal"
    """
    
    # Simple pluralization rules
    PLURAL_MAP = {
        'fishes': 'fish',
        'mammals': 'mammal',
        'birds': 'bird',
        'reptiles': 'reptile',
        'amphibians': 'amphibian',
        'insects': 'insect',
        'animals': 'animal',
        'plants': 'plant',
        'trees': 'tree',
        'flowers': 'flower',
        'penguins': 'penguin',
        'lions': 'lion',
        'eagles': 'eagle',
        'cats': 'cat',
        'dogs': 'dog',
    }
    
    def normalize(self, entity: str) -> str:
        """Normalize entity to singular canonical form."""
        # CRITICAL: Strip punctuation FIRST before normalization
        entity_clean = entity.lower().strip().rstrip('.,!?;:')
        
        # Check explicit mappings
        if entity_clean in self.PLURAL_MAP:
            return self.PLURAL_MAP[entity_clean]
            
        # Protected words that end in 's' but are singular
        PROTECTED_ENDS = {'socrates', 'species', 'series', 'corpus', 'status', 'chaos', 'lens', 'mathematics', 'physics'}
        if entity_clean in PROTECTED_ENDS:
            return entity_clean
        
        # English plural rules (ordered by specificity)
        # -es endings (e.g., "boxes", "wishes", "platypuses")
        if entity_clean.endswith('ses') and len(entity_clean) > 4:
            return entity_clean[:-2]  # "platypuses" → "platypus"
        if entity_clean.endswith(('xes', 'ches', 'shes')) and len(entity_clean) > 4:
            return entity_clean[:-2]  # "boxes" → "box"
        
        # -ies ending (e.g., "stories", "countries")
        if entity_clean.endswith('ies') and len(entity_clean) > 4:
            return entity_clean[:-3] + 'y'  # "countries" → "country"
        
        # -ves ending (e.g., "wolves", "knives")
        if entity_clean.endswith('ves') and len(entity_clean) > 4:
            return entity_clean[:-3] + 'f'  # "wolves" → "wolf"
        
        # Simple trailing 's' (most common)
        if entity_clean.endswith('s') and len(entity_clean) > 3:
            # Don't remove 's' from words like "grass", "glass", "cannabis"
            if not entity_clean.endswith('ss'):
                return entity_clean[:-1]
        
        return entity_clean
    
    def normalize_list(self, entities: list[str]) -> list[str]:
        """Normalize list of entities."""
        return [self.normalize(e) for e in entities]


# ═══════════════════════════════════════════════════════════════════
# GLOBAL NORMALIZERS
# ═══════════════════════════════════════════════════════════════════

_predicate_normalizer = None
_entity_normalizer = None

def get_predicate_normalizer() -> PredicateNormalizer:
    """Get singleton predicate normalizer."""
    global _predicate_normalizer
    if _predicate_normalizer is None:
        _predicate_normalizer = PredicateNormalizer()
    return _predicate_normalizer

def get_entity_normalizer() -> EntityNormalizer:
    """Get singleton entity normalizer."""
    global _entity_normalizer
    if _entity_normalizer is None:
        _entity_normalizer = EntityNormalizer()
    return _entity_normalizer


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def normalize_predicate(text: str, entities: list[str] = None) -> str:
    """
    Normalize predicate from text.
    
    Returns canonical predicate string.
    """
    normalizer = get_predicate_normalizer()
    canonical, _, _ = normalizer.normalize(text, entities)
    return canonical

def normalize_entity(entity: str) -> str:
    """Normalize single entity."""
    normalizer = get_entity_normalizer()
    return normalizer.normalize(entity)

def normalize_entities(entities: list[str]) -> list[str]:
    """Normalize list of entities."""
    normalizer = get_entity_normalizer()
    return normalizer.normalize_list(entities)
