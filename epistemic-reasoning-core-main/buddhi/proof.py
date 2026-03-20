"""
HRE Output Type: HypotheticalAnswerProof

This is the ONLY output type HRE is allowed to produce.

It is explicitly marked as hypothetical and contains NO:
  - belief IDs (from Chitta)
  - confidence updates
  - edges
  - promotions
  - learning signals

It IS allowed to contain:
  - assumptions (temporary beliefs for this reasoning)
  - derivation steps (how conclusion was reached)
  - counterfactual conclusion (verdict)
  - conflicts (detected within hypothetical world)
"""

from dataclasses import dataclass, field


@dataclass
class HypotheticalAssumption:
    """
    A temporary belief assumed for hypothetical reasoning.
    
    This is NOT a real belief - it exists only within HRE's sandbox.
    """
    statement: str                 # Natural language assumption
    canonical: dict                # Canonical structure (from Manas)
    confidence: float = 1.0        # Assumed confidence (default: certain)
    
    def __post_init__(self):
        """Validate assumption."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class HypotheticalDerivationStep:
    """
    A single reasoning step in hypothetical world.
    
    Similar to DerivationStep but explicitly marked as hypothetical.
    """
    step_id: int
    rule: str                      # Rule applied (e.g., "taxonomic_identity", "direct_match")
    inputs: list[str]              # Assumption/belief identifiers used
    output: str                    # Intermediate conclusion
    confidence: float | None = None
    hypothetical: bool = True      # ALWAYS True (locked)
    
    def __post_init__(self):
        """Enforce hypothetical flag."""
        if not self.hypothetical:
            raise ValueError("HypotheticalDerivationStep MUST have hypothetical=True")


@dataclass
class HypotheticalConflict:
    """
    A conflict detected within hypothetical world.
    
    This does NOT affect real Chitta - it's purely informational.
    """
    predicate: str                 # Conflicting predicate
    assumption_a: str              # First conflicting assumption
    assumption_b: str              # Second conflicting assumption
    delta: float                   # Confidence difference


@dataclass
class HypotheticalAnswerProof:
    """
    The ONLY output type for HRE.
    
    This is explicitly marked as hypothetical and creates NO side effects.
    
    Invariants:
      - hypothetical flag is ALWAYS True
      - NO belief IDs from Chitta (assumptions are temporary)
      - NO confidence updates to real beliefs
      - NO writes to Chitta
      - NO learning signals
    
    If you violate any invariant, STOP immediately.
    """
    query: str                                          # Original query
    assumptions: list[HypotheticalAssumption]          # Temporary beliefs
    verdict: str                                       # "yes" | "no" | "uncertain" | "unknown"
    steps: list[HypotheticalDerivationStep] = field(default_factory=list)
    conflicts: list[HypotheticalConflict] = field(default_factory=list)
    hypothetical: bool = True                          # ALWAYS True (locked)
    
    def __post_init__(self):
        """Enforce invariants."""
        # Invariant 1: hypothetical flag MUST be True
        if not self.hypothetical:
            raise ValueError("HypotheticalAnswerProof MUST have hypothetical=True")
        
        # Invariant 2: verdict MUST be valid
        valid_verdicts = {"yes", "no", "uncertain", "unknown"}
        if self.verdict not in valid_verdicts:
            raise ValueError(f"Invalid verdict: {self.verdict}. Must be one of {valid_verdicts}")
        
        # Invariant 3: All steps MUST be hypothetical
        for step in self.steps:
            if not step.hypothetical:
                raise ValueError(f"Step {step.step_id} is not marked hypothetical!")
    
    def add_step(
        self,
        rule: str,
        inputs: list[str],
        output: str,
        confidence: float | None = None
    ):
        """
        Add a hypothetical derivation step.
        
        Args:
            rule: reasoning rule applied
            inputs: assumption/belief identifiers
            output: intermediate conclusion
            confidence: optional confidence value
        """
        step = HypotheticalDerivationStep(
            step_id=len(self.steps) + 1,
            rule=rule,
            inputs=inputs,
            output=output,
            confidence=confidence,
            hypothetical=True,  # Explicit
        )
        self.steps.append(step)
    
    def add_conflict(
        self,
        predicate: str,
        assumption_a: str,
        assumption_b: str,
        delta: float
    ):
        """
        Add a conflict record (within hypothetical world).
        
        Args:
            predicate: conflicting predicate
            assumption_a: first assumption
            assumption_b: second assumption
            delta: confidence difference
        """
        conflict = HypotheticalConflict(
            predicate=predicate,
            assumption_a=assumption_a,
            assumption_b=assumption_b,
            delta=delta,
        )
        self.conflicts.append(conflict)
    
    def to_natural_language(self) -> str:
        """
        Convert to natural language (for Ahankara rendering).
        
        This MUST be linguistically marked as hypothetical.
        """
        # Prefix MUST indicate hypothetical nature
        prefix = "Hypothetically"
        
        if self.assumptions:
            assumptions_text = ", ".join([a.statement for a in self.assumptions])
            prefix = f"Hypothetically (assuming {assumptions_text})"
        
        # Verdict mapping
        verdict_map = {
            "yes": "yes",
            "no": "no",
            "uncertain": "uncertain",
            "unknown": "unknown"
        }
        
        verdict_text = verdict_map.get(self.verdict, "unknown")
        
        # Construct answer
        if self.verdict == "unknown":
            return f"{prefix}: I do not know."
        elif self.verdict == "uncertain":
            return f"{prefix}: I am not certain."
        else:
            return f"{prefix}: {verdict_text}."
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"HypotheticalAnswerProof(\n"
            f"  query='{self.query}',\n"
            f"  assumptions={len(self.assumptions)},\n"
            f"  verdict='{self.verdict}',\n"
            f"  steps={len(self.steps)},\n"
            f"  conflicts={len(self.conflicts)},\n"
            f"  hypothetical={self.hypothetical}\n"
            f")"
        )
