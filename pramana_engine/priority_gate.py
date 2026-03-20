"""
Priority gate and conflict resolution module.

Implements the Nyāya rule:
    Pratyakṣa (direct perception) **completely overrides** Anumāna (inference).

When a contradiction between Pratyakṣa and any derived claim is detected,
a Bādhita state is triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pramana_engine.models import Evidence, PramanaType, Proposition

KnowledgeUnit = Any  # Proposition | Evidence


@dataclass
class ConflictRecord:
    """Description of a detected conflict between knowledge units."""

    dominant: KnowledgeUnit
    dominated: KnowledgeUnit
    reason: str
    badhita: bool = False


@dataclass
class PriorityGateResult:
    """Output of the priority gate pass."""

    accepted: List[KnowledgeUnit] = field(default_factory=list)
    """Knowledge units that survived the gate."""

    rejected: List[KnowledgeUnit] = field(default_factory=list)
    """Knowledge units that were overridden / rejected."""

    conflicts: List[ConflictRecord] = field(default_factory=list)
    """All detected conflicts."""

    badhita_triggered: bool = False
    """True when a Pratyakṣa–Anumāna contradiction was found."""


def _claims_contradict(claim_a: str, claim_b: str) -> bool:
    """Naïve contradiction check: one claim negates the other.

    Checks for explicit negation keywords and whether the two claims
    overlap significantly in vocabulary (same topic, opposite polarity).
    """
    a_low = claim_a.lower()
    b_low = claim_b.lower()

    negation_words = {"not", "no", "never", "false", "deny", "contradict", "opposite"}

    a_neg = any(w in a_low.split() for w in negation_words)
    b_neg = any(w in b_low.split() for w in negation_words)

    # If exactly one is negated and both share common key words → contradiction
    a_words = set(a_low.split())
    b_words = set(b_low.split())
    overlap = a_words & b_words - {"the", "a", "an", "is", "are", "has", "have"}

    if overlap and (a_neg != b_neg):
        return True

    # Direct substring negation: "X" vs "not X" or "no X"
    if b_low.startswith("not ") and b_low[4:] == a_low:
        return True
    if a_low.startswith("not ") and a_low[4:] == b_low:
        return True

    return False


def run_priority_gate(
    units: List[KnowledgeUnit],
) -> PriorityGateResult:
    """Apply the Pratyakṣa-override priority rule to a mixed knowledge set.

    Algorithm
    ---------
    1. Separate Pratyakṣa units from others.
    2. For each non-Pratyakṣa unit check whether any Pratyakṣa unit
       contradicts it (using :func:`_claims_contradict`).
    3. Contradicted units are moved to *rejected*; a :class:`ConflictRecord`
       with ``badhita=True`` is appended.
    4. Remaining units are returned in *accepted*.

    Parameters
    ----------
    units:
        Mixed list of validated Proposition and Evidence objects.

    Returns
    -------
    PriorityGateResult
    """
    pratyaksha_units = [u for u in units if u.pramana_type == PramanaType.PRATYAKSHA]
    other_units = [u for u in units if u.pramana_type != PramanaType.PRATYAKSHA]

    accepted: List[KnowledgeUnit] = list(pratyaksha_units)
    rejected: List[KnowledgeUnit] = []
    conflicts: List[ConflictRecord] = []
    badhita = False

    for unit in other_units:
        overridden = False
        for px in pratyaksha_units:
            if _claims_contradict(px.claim, unit.claim):
                conflicts.append(
                    ConflictRecord(
                        dominant=px,
                        dominated=unit,
                        reason=(
                            f"Pratyakṣa claim '{px.claim}' contradicts "
                            f"{unit.pramana_type.value} claim '{unit.claim}'. "
                            f"Bādhita state triggered."
                        ),
                        badhita=True,
                    )
                )
                rejected.append(unit)
                overridden = True
                badhita = True
                break

        if not overridden:
            # Also check for Anumāna–Anumāna contradictions (Satpratipakṣa)
            for acc in accepted:
                if (
                    acc.pramana_type == unit.pramana_type
                    and _claims_contradict(acc.claim, unit.claim)
                ):
                    conflicts.append(
                        ConflictRecord(
                            dominant=acc,
                            dominated=unit,
                            reason=(
                                f"Mutual contradiction between two "
                                f"{unit.pramana_type.value} claims: "
                                f"'{acc.claim}' vs '{unit.claim}' "
                                f"(Satpratipakṣa)."
                            ),
                            badhita=False,
                        )
                    )
            accepted.append(unit)

    return PriorityGateResult(
        accepted=accepted,
        rejected=rejected,
        conflicts=conflicts,
        badhita_triggered=badhita,
    )
