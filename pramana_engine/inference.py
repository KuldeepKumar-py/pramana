"""
Pramāṇa checker and inference engine.

Routes validated inputs through:
- Pramāṇa-type constraint verification.
- Modus ponens rule application.
- Nyāya 5-step syllogism (via anumana module).
- Epistemic confidence threshold gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pramana_engine.anumana import (
    SyllogismResult,
    assert_paksha,
    assert_vyapti,
    nyaya_syllogism,
    search_sadhya,
)
from pramana_engine.models import Evidence, PramanaType, Proposition

# Default confidence required for a claim to pass the epistemic gate
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5


@dataclass
class CheckResult:
    """Result of the Pramāṇa constraint checker."""

    pramana_type: PramanaType
    passed: bool
    violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Aggregated output of the inference engine for a single proposition."""

    proposition: Proposition
    check: CheckResult
    modus_ponens_valid: bool = False
    syllogism: Optional[SyllogismResult] = None
    epistemic_gate_passed: bool = False
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pramāṇa checker
# ---------------------------------------------------------------------------


def check_pramana(item: Any) -> CheckResult:
    """Verify epistemic constraints for a knowledge unit.

    Each pramāṇa type carries its own validity constraints:

    - **Pratyakṣa**: confidence must be ≥ 0.7 (direct observation is
      expected to be highly reliable).
    - **Anumāna**: source must not be empty and must not claim direct
      observation.
    - **Upamāna**: claim should mention an analogical comparison.
    - **Śabda**: source should reference a named authority.
    - **Arthāpatti**: confidence may be lower (abductive); no extra
      constraints beyond base validation.

    Parameters
    ----------
    item:
        A :class:`~pramana_engine.models.Proposition` or
        :class:`~pramana_engine.models.Evidence`.

    Returns
    -------
    CheckResult
    """
    ptype = item.pramana_type
    violations: List[str] = []
    notes: List[str] = []

    if ptype == PramanaType.PRATYAKSHA:
        if item.confidence < 0.7:
            violations.append(
                f"Pratyakṣa claims require confidence ≥ 0.7; got {item.confidence:.2f}."
            )
        notes.append("Pratyakṣa: direct-perception constraint verified.")

    elif ptype == PramanaType.ANUMANA:
        if "direct" in item.source.lower():
            violations.append(
                "Anumāna source must not claim to be direct observation."
            )
        notes.append("Anumāna: inference-source constraint verified.")

    elif ptype == PramanaType.UPAMANA:
        keywords = {"like", "similar", "analogous", "compared", "resembles"}
        if not any(kw in item.claim.lower() for kw in keywords):
            notes.append(
                "Upamāna: claim does not explicitly mention analogical comparison "
                "(this is a soft warning)."
            )
        else:
            notes.append("Upamāna: analogical comparison marker found in claim.")

    elif ptype == PramanaType.SHABDA:
        if len(item.source.split()) < 2:
            notes.append(
                "Śabda: source should ideally reference a named authority "
                "(soft suggestion)."
            )
        else:
            notes.append("Śabda: authority reference found in source.")

    elif ptype == PramanaType.ARTHAPATTI:
        notes.append(
            "Arthāpatti: abductive claim – no additional structural constraints."
        )

    return CheckResult(
        pramana_type=ptype,
        passed=len(violations) == 0,
        violations=violations,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Modus ponens
# ---------------------------------------------------------------------------


def modus_ponens(
    major_premise: str,
    minor_premise: str,
    conclusion: str,
    evidence_items: List[Evidence],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[bool, List[str]]:
    """Evaluate a classical modus ponens argument.

    Returns ``(valid, notes)`` where *valid* is ``True`` iff both premises
    are supported by evidence above the confidence threshold.

    Parameters
    ----------
    major_premise:
        The general rule (e.g. ``"all smoke implies fire"``).
    minor_premise:
        The specific case (e.g. ``"there is smoke on the hill"``).
    conclusion:
        The conclusion to be drawn.
    evidence_items:
        Evidence available to support the premises.
    confidence_threshold:
        Minimum confidence to accept a premise.
    """
    notes: List[str] = []
    claims_lower = {e.claim.lower(): e.confidence for e in evidence_items}

    major_supported = any(
        major_premise.lower() in c or c in major_premise.lower()
        for c in claims_lower
    )
    minor_supported = any(
        minor_premise.lower() in c or c in minor_premise.lower()
        for c in claims_lower
    )

    def _max_conf(premise: str) -> float:
        return max(
            (conf for c, conf in claims_lower.items() if premise.lower() in c or c in premise.lower()),
            default=0.0,
        )

    major_conf = _max_conf(major_premise)
    minor_conf = _max_conf(minor_premise)

    valid = (
        major_supported
        and minor_supported
        and major_conf >= confidence_threshold
        and minor_conf >= confidence_threshold
    )

    if not major_supported:
        notes.append(f"Major premise not found in evidence: '{major_premise}'.")
    if not minor_supported:
        notes.append(f"Minor premise not found in evidence: '{minor_premise}'.")
    if major_supported and major_conf < confidence_threshold:
        notes.append(
            f"Major premise confidence {major_conf:.2f} below threshold {confidence_threshold:.2f}."
        )
    if minor_supported and minor_conf < confidence_threshold:
        notes.append(
            f"Minor premise confidence {minor_conf:.2f} below threshold {confidence_threshold:.2f}."
        )
    if valid:
        notes.append(f"Modus ponens valid → '{conclusion}'.")

    return valid, notes


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Routes validated propositions through the full inference pipeline.

    Parameters
    ----------
    confidence_threshold:
        Epistemic gate – claims below this level are not acted upon.
    """

    def __init__(
        self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> None:
        self.confidence_threshold = confidence_threshold

    def _epistemic_gate(self, proposition: Proposition) -> bool:
        """Return ``True`` if the proposition passes the confidence gate."""
        return proposition.confidence >= self.confidence_threshold

    def evaluate(
        self,
        proposition: Proposition,
        evidence: List[Evidence],
        subject: Optional[str] = None,
        hetu: Optional[str] = None,
        sadhya: Optional[str] = None,
        example_subject: Optional[str] = None,
    ) -> InferenceResult:
        """Evaluate a proposition against available evidence.

        Parameters
        ----------
        proposition:
            The claim to evaluate.
        evidence:
            Supporting evidence set.
        subject, hetu, sadhya, example_subject:
            Positional arguments for the Nyāya syllogism when the
            proposition is of *Anumāna* type.  ``subject`` defaults to
            the proposition's source.

        Returns
        -------
        InferenceResult
        """
        check = check_pramana(proposition)
        gate_passed = self._epistemic_gate(proposition)
        notes: List[str] = list(check.notes)

        if not gate_passed:
            notes.append(
                f"Epistemic gate FAILED: confidence {proposition.confidence:.2f} "
                f"< threshold {self.confidence_threshold:.2f}."
            )

        # Modus ponens across evidence
        mp_notes: List[str] = []
        mp_valid = False
        if evidence:
            mp_valid, mp_notes = modus_ponens(
                major_premise=proposition.claim,
                minor_premise=evidence[0].claim if evidence else "",
                conclusion=proposition.claim,
                evidence_items=evidence,
                confidence_threshold=self.confidence_threshold,
            )
        notes.extend(mp_notes)

        # Nyāya syllogism for Anumāna propositions
        syllogism_result: Optional[SyllogismResult] = None
        if proposition.pramana_type == PramanaType.ANUMANA and sadhya:
            _subject = subject or proposition.source
            _hetu = hetu

            # If hetu is provided explicitly, directly assert the facts
            if _hetu is not None:
                assert_paksha(_subject, _hetu)
                assert_vyapti(_hetu, sadhya)
            else:
                # Auto-assert relationships from Anumāna evidence if hetu unknown
                for ev in evidence:
                    if ev.pramana_type == PramanaType.ANUMANA and ev.confidence >= self.confidence_threshold:
                        assert_paksha(_subject, ev.claim)
                        assert_vyapti(ev.claim, sadhya)
                        if _hetu is None:
                            _hetu = ev.claim

            syllogism_result = nyaya_syllogism(
                subject=_subject,
                hetu_claim=_hetu,
                sadhya=sadhya,
                example_subject=example_subject,
            )

        return InferenceResult(
            proposition=proposition,
            check=check,
            modus_ponens_valid=mp_valid,
            syllogism=syllogism_result,
            epistemic_gate_passed=gate_passed,
            notes=notes,
        )
