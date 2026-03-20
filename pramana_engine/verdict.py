"""
Verdict classifier and Abhāva/Bādhita negation-defeater module.

Classifies the final epistemic state of a proposition as one of:
- ``valid``                 – justified, no defeaters
- ``unjustified``           – Asiddha (hetu / premises not established)
- ``suspended``             – Satpratipakṣa (equally weighted counter-evidence)
- ``rejected``              – Bādhita (directly overridden by Pratyakṣa or
                              contradicted by a stronger source)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pramana_engine.inference import InferenceResult
from pramana_engine.priority_gate import ConflictRecord, PriorityGateResult


class VerdictLabel(str, Enum):
    """The four canonical verdict states."""

    VALID = "valid"
    UNJUSTIFIED = "unjustified (Asiddha)"
    SUSPENDED = "suspended (Satpratipakṣa)"
    REJECTED = "rejected/bādhita"


@dataclass
class Verdict:
    """Full verdict for a proposition."""

    label: VerdictLabel
    confidence: float
    justification: str
    defeaters: List[str] = field(default_factory=list)
    badhita: bool = False


def _abhava_check(
    conflicts: List[ConflictRecord],
    proposition_claim: str,
) -> List[str]:
    """Return textual defeaters (Abhāva / Bādhita notations) for the claim.

    An *Abhāva* (absence) defeater is triggered when the proposition is
    listed as *dominated* in a conflict record.
    """
    defeaters: List[str] = []
    for cr in conflicts:
        if cr.dominated.claim == proposition_claim:
            defeaters.append(cr.reason)
    return defeaters


def classify_verdict(
    inference_result: InferenceResult,
    gate_result: Optional[PriorityGateResult] = None,
) -> Verdict:
    """Classify the final verdict for a proposition.

    Decision logic
    --------------
    1. If the proposition was rejected by the priority gate (Bādhita) →
       **rejected/bādhita**.
    2. If the proposition's epistemic gate failed, or the Nyāya syllogism
       explicitly failed with an Asiddha reason, or the pramāṇa check has
       unresolved violations and neither modus ponens nor syllogism succeeded
       → **unjustified (Asiddha)**.
    3. If there are Satpratipakṣa conflicts (same-strength counter-evidence)
       without Bādhita override → **suspended (Satpratipakṣa)**.
    4. Otherwise → **valid**.

    Parameters
    ----------
    inference_result:
        Output of :class:`~pramana_engine.inference.InferenceEngine`.
    gate_result:
        Optional output of :func:`~pramana_engine.priority_gate.run_priority_gate`.

    Returns
    -------
    Verdict
    """
    prop = inference_result.proposition
    gate_result = gate_result or PriorityGateResult()

    # ------------------------------------------------------------------ #
    # 1. Bādhita check – was this proposition explicitly rejected?         #
    # ------------------------------------------------------------------ #
    rejected_claims = {r.claim for r in gate_result.rejected}
    if prop.claim in rejected_claims:
        defeaters = _abhava_check(gate_result.conflicts, prop.claim)
        return Verdict(
            label=VerdictLabel.REJECTED,
            confidence=0.0,
            justification=(
                "Proposition overridden by Pratyakṣa (direct perception). "
                "Bādhita state triggered."
            ),
            defeaters=defeaters,
            badhita=True,
        )

    # ------------------------------------------------------------------ #
    # 2. Asiddha check – premise / hetu not established                    #
    # ------------------------------------------------------------------ #
    syllogism_failed_asiddha = (
        inference_result.syllogism is not None
        and not inference_result.syllogism.success
        and "Asiddha" in (inference_result.syllogism.failure_reason or "")
    )
    gate_failed = not inference_result.epistemic_gate_passed
    check_violations = not inference_result.check.passed
    no_positive_inference = (
        not inference_result.modus_ponens_valid
        and (
            inference_result.syllogism is None
            or not inference_result.syllogism.success
        )
    )

    if syllogism_failed_asiddha or gate_failed or (check_violations and no_positive_inference):
        reason_parts: List[str] = []
        if syllogism_failed_asiddha:
            reason_parts.append(
                inference_result.syllogism.failure_reason or "Syllogism failed."
            )
        if gate_failed:
            reason_parts.append("Epistemic confidence threshold not met.")
        if check_violations:
            reason_parts.extend(inference_result.check.violations)
        return Verdict(
            label=VerdictLabel.UNJUSTIFIED,
            confidence=prop.confidence,
            justification=" | ".join(reason_parts) or "Hetu / premises not established.",
        )

    # ------------------------------------------------------------------ #
    # 3. Satpratipakṣa check – suspended by counter-evidence              #
    # ------------------------------------------------------------------ #
    satpratipaksha_conflicts = [
        cr for cr in gate_result.conflicts if not cr.badhita
    ]
    if satpratipaksha_conflicts:
        reasons = [cr.reason for cr in satpratipaksha_conflicts]
        return Verdict(
            label=VerdictLabel.SUSPENDED,
            confidence=prop.confidence * 0.5,
            justification="Counter-evidence of equal force detected (Satpratipakṣa).",
            defeaters=reasons,
        )

    # ------------------------------------------------------------------ #
    # 4. Valid                                                             #
    # ------------------------------------------------------------------ #
    return Verdict(
        label=VerdictLabel.VALID,
        confidence=prop.confidence,
        justification="Proposition is epistemically justified and undefeated.",
    )
