"""
Anumāna (inference) core using kanren (miniKanren) relational logic.

Implements:
- Vyāpti (invariable concomitance) as a kanren relation
- The 5-step Nyāya syllogism:
    1. Pratijñā   – the thesis
    2. Hetu       – the reason / mark
    3. Udāharaṇa  – the example
    4. Upanaya    – the application
    5. Nigamana   – the conclusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kanren import eq, lall, run, var
from kanren.facts import fact, Relation

# ---------------------------------------------------------------------------
# Relational environment
# ---------------------------------------------------------------------------

#: Relation: vyapti(hetu, sadhya) – "wherever hetu is, sadhya is also"
vyapti: Relation = Relation()

#: Relation: paksha(subject, hetu) – "subject has the hetu mark"
paksha: Relation = Relation()


def assert_vyapti(hetu: str, sadhya: str) -> None:
    """Assert an invariable concomitance between *hetu* and *sadhya*.

    Parameters
    ----------
    hetu:   The inferential mark (reason).
    sadhya: The property to be inferred (target).
    """
    fact(vyapti, hetu, sadhya)


def assert_paksha(subject: str, hetu: str) -> None:
    """Assert that *subject* possesses the *hetu* mark.

    Parameters
    ----------
    subject: The locus (pakṣa) under investigation.
    hetu:    The inferential mark that the subject exhibits.
    """
    fact(paksha, subject, hetu)


# ---------------------------------------------------------------------------
# Truth search
# ---------------------------------------------------------------------------


def search_sadhya(subject: str, limit: int = 1) -> List[str]:
    """Return the list of *sadhya* properties inferable for *subject*.

    Uses kanren to resolve: paksha(subject, H) ∧ vyapti(H, S) → S.

    Parameters
    ----------
    subject: The locus to query.
    limit:   Maximum number of solutions to return (0 = all).

    Returns
    -------
    List[str]
        Inferred sadhya values (may be empty if none can be derived).
    """
    h = var()
    s = var()
    results = run(
        limit or 0,
        s,
        lall(paksha(subject, h), vyapti(h, s)),
    )
    return list(results)


def search_hetu(subject: str, limit: int = 1) -> List[str]:
    """Return hetu marks possessed by *subject*.

    Parameters
    ----------
    subject: The locus to query.
    limit:   Maximum number of solutions to return.

    Returns
    -------
    List[str]
        Known hetu marks for the subject.
    """
    h = var()
    results = run(limit or 0, h, paksha(subject, h))
    return list(results)


# ---------------------------------------------------------------------------
# 5-step Nyāya syllogism
# ---------------------------------------------------------------------------


@dataclass
class SyllogismResult:
    """Structured output of a 5-step Nyāya syllogism."""

    pratijña: str
    """Step 1 – The thesis being argued."""

    hetu: str
    """Step 2 – The reason / mark cited."""

    udaharana: str
    """Step 3 – The universal example backing the vyāpti."""

    upanaya: str
    """Step 4 – The application of the example to the subject."""

    nigamana: str
    """Step 5 – The final conclusion (mirrors pratijñā)."""

    success: bool = True
    """Whether the syllogism could be completed successfully."""

    failure_reason: Optional[str] = None
    """Populated when *success* is ``False``."""

    trace_nodes: List[Dict[str, Any]] = field(default_factory=list)
    """Ordered list of trace-friendly step dicts for DAG construction."""


def nyaya_syllogism(
    subject: str,
    hetu_claim: Optional[str],
    sadhya: str,
    example_subject: Optional[str] = None,
) -> SyllogismResult:
    """Execute a 5-step Nyāya syllogism using kanren relations.

    Parameters
    ----------
    subject:
        The pakṣa (locus) – e.g. ``"hill"``.
    hetu_claim:
        The asserted hetu mark – e.g. ``"smoke"``.  Pass ``None`` to let
        the engine derive it; if it cannot be found the syllogism fails
        with *Asiddha*.
    sadhya:
        The property to be inferred – e.g. ``"fire"``.
    example_subject:
        A known locus used for the Udāharaṇa step (defaults to
        ``"kitchen"``).

    Returns
    -------
    SyllogismResult
        Fully populated result including trace nodes.
    """
    example_subject = example_subject or "kitchen"

    # ------------------------------------------------------------------
    # Step 1 – Pratijñā: articulate the thesis
    # ------------------------------------------------------------------
    pratijña = f"{subject} has {sadhya}"
    trace_nodes: List[Dict[str, Any]] = [
        {"step": 1, "name": "Pratijñā", "content": pratijña}
    ]

    # ------------------------------------------------------------------
    # Step 2 – Hetu: verify the reason mark
    # ------------------------------------------------------------------
    if hetu_claim is not None:
        # Check that the subject has this hetu via kanren
        h_var = var()
        known_hetu_list = run(0, h_var, lall(paksha(subject, h_var), eq(h_var, hetu_claim)))
        has_hetu = len(known_hetu_list) > 0
    else:
        # Attempt automated discovery of hetu
        discovered = search_hetu(subject, limit=1)
        if discovered:
            hetu_claim = str(discovered[0])
            has_hetu = True
        else:
            has_hetu = False

    if not has_hetu:
        trace_nodes.append(
            {
                "step": 2,
                "name": "Hetu",
                "content": f"FAILED – no hetu mark found for subject '{subject}'",
            }
        )
        return SyllogismResult(
            pratijña=pratijña,
            hetu="<missing>",
            udaharana="<missing>",
            upanaya="<missing>",
            nigamana="<missing>",
            success=False,
            failure_reason="Asiddha – hetu not established for the pakṣa.",
            trace_nodes=trace_nodes,
        )

    hetu_text = f"{subject} has {hetu_claim}"
    trace_nodes.append({"step": 2, "name": "Hetu", "content": hetu_text})

    # ------------------------------------------------------------------
    # Step 3 – Udāharaṇa: universal example
    # ------------------------------------------------------------------
    # Verify vyāpti holds: wherever hetu is, sadhya is
    s_var = var()
    vyapti_results = run(1, s_var, lall(vyapti(hetu_claim, s_var), eq(s_var, sadhya)))
    if not vyapti_results:
        trace_nodes.append(
            {
                "step": 3,
                "name": "Udāharaṇa",
                "content": f"FAILED – vyāpti not established: {hetu_claim}→{sadhya}",
            }
        )
        return SyllogismResult(
            pratijña=pratijña,
            hetu=hetu_text,
            udaharana="<missing>",
            upanaya="<missing>",
            nigamana="<missing>",
            success=False,
            failure_reason="Asiddha – vyāpti (universal concomitance) not found.",
            trace_nodes=trace_nodes,
        )

    udaharana = (
        f"wherever there is {hetu_claim}, there is {sadhya} (e.g. {example_subject})"
    )
    trace_nodes.append({"step": 3, "name": "Udāharaṇa", "content": udaharana})

    # ------------------------------------------------------------------
    # Step 4 – Upanaya: application
    # ------------------------------------------------------------------
    upanaya = f"{subject} has {hetu_claim}, and all {hetu_claim} correlates with {sadhya}"
    trace_nodes.append({"step": 4, "name": "Upanaya", "content": upanaya})

    # ------------------------------------------------------------------
    # Step 5 – Nigamana: conclusion
    # ------------------------------------------------------------------
    nigamana = f"therefore {subject} has {sadhya}"
    trace_nodes.append({"step": 5, "name": "Nigamana", "content": nigamana})

    return SyllogismResult(
        pratijña=pratijña,
        hetu=hetu_text,
        udaharana=udaharana,
        upanaya=upanaya,
        nigamana=nigamana,
        success=True,
        trace_nodes=trace_nodes,
    )
