"""
Ingestion module for the Pramāṇa-Constrained Inference Engine.

Accepts raw dicts or pre-built model instances, validates them, and
returns normalised Proposition / Evidence objects.
"""

from __future__ import annotations

from typing import List, Union

from pramana_engine.models import Evidence, Proposition

KnowledgeUnit = Union[Proposition, Evidence]


def ingest(
    items: List[Union[dict, KnowledgeUnit]],
    kind: str = "evidence",
) -> List[KnowledgeUnit]:
    """Validate and normalise a list of knowledge units.

    Parameters
    ----------
    items:
        Each element may be a raw ``dict`` or an already-constructed
        :class:`Proposition` / :class:`Evidence` instance.
    kind:
        ``"evidence"`` (default) or ``"proposition"`` — governs which model
        is used when parsing raw dicts.

    Returns
    -------
    List[KnowledgeUnit]
        Validated, normalised objects.

    Raises
    ------
    ValueError
        If *kind* is not recognised.
    pydantic.ValidationError
        If any item fails validation.
    """
    model_cls: type
    if kind == "evidence":
        model_cls = Evidence
    elif kind == "proposition":
        model_cls = Proposition
    else:
        raise ValueError(f"kind must be 'evidence' or 'proposition', got {kind!r}.")

    normalised: List[KnowledgeUnit] = []
    for item in items:
        if isinstance(item, (Proposition, Evidence)):
            # Re-validate to enforce invariants even if object was mutated.
            normalised.append(model_cls.model_validate(item.model_dump()))
        else:
            normalised.append(model_cls.model_validate(item))
    return normalised
