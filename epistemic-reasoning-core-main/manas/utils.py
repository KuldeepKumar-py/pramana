"""
Manas utility functions for normalization and detection.
"""

import re


def normalize_entity(entity: str) -> str:
    """
    Convert an entity to its canonical form.
    
    Without this, the belief graph gets fragmented — you end up with separate nodes
    for "Birds", "bird", "birds", and "BIRDS" when they all mean the same thing.
    
    The normalization process:
    1. Make it lowercase ("Birds" → "birds")
    2. Remove punctuation
    3. Convert to singular form ("birds" → "bird")
    
    Args:
        entity: the raw entity string from parsing
    
    Returns:
        normalized entity (lowercase, singular, no punctuation)
    """
    if not entity:
        return ""
    
    # Step 1: lowercase everything
    entity = entity.lower()
    
    # Step 2: remove punctuation (commas, periods, etc.)
    entity = re.sub(r'[^\w\s-]', '', entity)
    
    # Step 3: trim whitespace
    entity = entity.strip()
    
    # Step 4: convert plural to singular using English heuristics
    # This is approximate but works for most common cases
    if entity.endswith('ies') and len(entity) > 4:
        # "berries" → "berry", "studies" → "study"
        entity = entity[:-3] + 'y'
    elif entity.endswith('es') and len(entity) > 3:
        # Handle different -es endings
        if entity.endswith('sses'):
            # "classes" → "class"
            entity = entity[:-2]
        elif entity.endswith('xes') or entity.endswith('ches') or entity.endswith('shes'):
            # "boxes" → "box", "watches" → "watch"
            entity = entity[:-2]
        else:
            entity = entity[:-1]
    elif entity.endswith('s') and len(entity) > 2:
        # Simple plural: "birds" → "bird", "dogs" → "dog"
        # But don't touch words that end in double-s or -us
        if not entity.endswith('ss') and not entity.endswith('us'):
            entity = entity[:-1]
    
    return entity


def detect_intent(text: str) -> str:
    """
    Figure out if the text is asking a question or making a statement.
    
    Questions usually have these markers:
    - Ends with a question mark: "Do birds fly?"
    - Starts with a question word: "What do birds eat?"
    - Uses auxiliary inversion: "Can penguins fly?" (verb comes before subject)
    
    Args:
        text: the input text
    
    Returns:
        "query" if it's a question, "assertion" if it's a statement
    """
    text_lower = text.lower().strip()
    
    # Easiest case: ends with a question mark
    if text.endswith('?'):
        return "query"
    
    # Check if it starts with a question word
    question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which']
    first_word = text_lower.split()[0] if text_lower.split() else ""
    if first_word in question_words:
        return "query"
    
    # Check for auxiliary verb inversion (typical in yes/no questions)
    # "Can birds fly?" or "Is water wet?" — verb comes before the subject
    aux_verbs = ['is', 'are', 'was', 'were', 'can', 'could', 'will', 'would', 'do', 'does', 'did']
    if first_word in aux_verbs:
        return "query"
    
    # If none of the above, assume it's a statement
    return "assertion"


def detect_modality(text: str) -> str:
    """
    Detect how strongly the claim is being made.
    
    This picks up on hedging or emphasis words:
    - Strong: "Penguins definitely cannot fly" — high certainty
    - Weak: "Birds probably can fly" — uncertain or qualified
    - Default: "Birds fly" — neutral statement
    
    We use this to adjust confidence scores:
    - strong modality: boost confidence by 10%
    - weak modality: reduce confidence by 30%
    - default: no adjustment
    
    Args:
        text: input text
    
    Returns:
        "strong" | "weak" | "default"
    """
    text_lower = text.lower()
    
    # Strong modality markers
    strong_markers = [
        'always', 'definitely', 'certainly', 'surely', 'must',
        'absolutely', 'undoubtedly', 'never', 'invariably',
    ]
    
    for marker in strong_markers:
        if marker in text_lower:
            return "strong"
    
    # Weak modality markers
    weak_markers = [
        'might', 'maybe', 'possibly', 'could', 'probably',
        'perhaps', 'sometimes', 'often', 'usually', 'may',
        'seem', 'appears', 'likely',
    ]
    
    for marker in weak_markers:
        if marker in text_lower:
            return "weak"
    
    return "default"
