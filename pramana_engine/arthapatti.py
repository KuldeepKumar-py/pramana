"""
Arthāpatti (postulation / abduction) engine.

Implements abductive reasoning to:
- Detect missing premises in an argument chain.
- Generate hypotheses that would explain an observed fact.
- Trigger when contradictory logic states occur or a required premise
  is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Hypothesis:
    """A generated hypothesis from abductive reasoning."""

    claim: str
    """The hypothetical claim that would resolve the contradiction."""

    confidence: float
    """Estimated probability that this hypothesis is correct."""

    supporting_facts: List[str] = field(default_factory=list)
    """Facts that support this hypothesis."""

    explanation: str = ""
    """Human-readable explanation of why this hypothesis was generated."""


@dataclass
class ArthappattiResult:
    """Result from an Arthāpatti (abductive) reasoning pass."""

    triggered: bool
    """Whether abductive reasoning was required."""

    missing_premises: List[str] = field(default_factory=list)
    """Identified missing premises in the argument."""

    hypotheses: List[Hypothesis] = field(default_factory=list)
    """Generated hypotheses to resolve contradictions or gaps."""

    contradiction_detected: bool = False
    """Whether a logical contradiction was the trigger."""

    explanation: str = ""
    """Summary explanation of what Arthāpatti did."""


class ArthappattiEngine:
    """Abductive reasoning / missing-value imputation engine.

    Triggered when:
    - A required premise is absent from the knowledge base.
    - Contradictory logic states are detected in the inference chain.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialise the engine.

        Parameters
        ----------
        confidence_threshold:
            Minimum confidence for a hypothesis to be considered viable.
        """
        self.confidence_threshold = confidence_threshold

    def detect_missing_premises(
        self, required: List[str], available: List[str]
    ) -> List[str]:
        """Identify which of the *required* premises are absent from *available*.

        Parameters
        ----------
        required:
            List of premise claims that the argument needs.
        available:
            List of premise claims currently known.

        Returns
        -------
        List[str]
            Missing premise claims.
        """
        available_lower = {a.lower() for a in available}
        return [r for r in required if r.lower() not in available_lower]

    def generate_hypotheses(
        self,
        missing_premises: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Hypothesis]:
        """Generate hypotheses for each missing premise.

        For each missing premise a best-effort hypothesis is created with
        a confidence derived from context (falls back to a default value).

        Parameters
        ----------
        missing_premises:
            Claims that are absent from the knowledge base.
        context:
            Optional dict of contextual facts that may raise/lower
            hypothesis confidence.

        Returns
        -------
        List[Hypothesis]
            One hypothesis per missing premise.
        """
        context = context or {}
        hypotheses: List[Hypothesis] = []
        for premise in missing_premises:
            # Base confidence heuristic: scale with available context clues
            related_facts = [
                v for k, v in context.items() if premise.lower() in k.lower()
            ]
            base_conf = 0.6 if related_facts else 0.4
            hypotheses.append(
                Hypothesis(
                    claim=premise,
                    confidence=base_conf,
                    supporting_facts=related_facts,
                    explanation=(
                        f"Hypothesised via Arthāpatti: '{premise}' is assumed "
                        f"to hold because its absence would leave the argument "
                        f"unexplained."
                    ),
                )
            )
        return hypotheses

    def resolve_contradiction(
        self,
        claim_a: str,
        claim_b: str,
        confidence_a: float,
        confidence_b: float,
    ) -> ArthappattiResult:
        """Attempt to resolve a contradiction between two claims.

        Generates a defeater hypothesis for the lower-confidence claim.

        Parameters
        ----------
        claim_a, claim_b:
            The two contradicting claims.
        confidence_a, confidence_b:
            Their respective confidence values.

        Returns
        -------
        ArthappattiResult
            Triggered result with defeater hypothesis.
        """
        if confidence_a >= confidence_b:
            weaker, stronger = claim_b, claim_a
            weak_conf = confidence_b
        else:
            weaker, stronger = claim_a, claim_b
            weak_conf = confidence_a

        defeater = Hypothesis(
            claim=f"NOT ({weaker})",
            confidence=1.0 - weak_conf,
            supporting_facts=[stronger],
            explanation=(
                f"Contradiction detected between '{claim_a}' and '{claim_b}'. "
                f"Arthāpatti postulates the defeater of the weaker claim "
                f"'{weaker}' (conf={weak_conf:.2f})."
            ),
        )
        return ArthappattiResult(
            triggered=True,
            missing_premises=[],
            hypotheses=[defeater],
            contradiction_detected=True,
            explanation=(
                f"Abductive defeater generated: '{defeater.claim}' "
                f"with confidence {defeater.confidence:.2f}."
            ),
        )

    def run(
        self,
        required_premises: List[str],
        available_premises: List[str],
        contradictions: Optional[List[tuple]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ArthappattiResult:
        """Full Arthāpatti reasoning pass.

        Parameters
        ----------
        required_premises:
            Claims the argument requires.
        available_premises:
            Claims currently known / asserted.
        contradictions:
            Optional list of ``(claim_a, claim_b, conf_a, conf_b)`` tuples
            representing detected contradictions.
        context:
            Optional context dict for hypothesis generation.

        Returns
        -------
        ArthappattiResult
            Complete abductive result.
        """
        missing = self.detect_missing_premises(required_premises, available_premises)
        hypotheses = self.generate_hypotheses(missing, context)

        contradiction_flag = False
        if contradictions:
            for claim_a, claim_b, conf_a, conf_b in contradictions:
                res = self.resolve_contradiction(claim_a, claim_b, conf_a, conf_b)
                hypotheses.extend(res.hypotheses)
                contradiction_flag = True

        triggered = bool(missing or contradiction_flag)
        explanation = ""
        if missing:
            explanation += (
                f"Missing premises identified: {missing}. "
                f"Hypotheses generated via postulation. "
            )
        if contradiction_flag:
            explanation += "Contradictions detected and defeater hypotheses generated."

        return ArthappattiResult(
            triggered=triggered,
            missing_premises=missing,
            hypotheses=hypotheses,
            contradiction_detected=contradiction_flag,
            explanation=explanation.strip(),
        )
