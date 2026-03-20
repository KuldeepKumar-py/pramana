"""
Pydantic data models for the Pramāṇa-Constrained Inference Engine.

Defines strict data models for Proposition and Evidence with
Nyāya epistemic source (pramāṇa) typing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class PramanaType(str, Enum):
    """Nyāya epistemological sources of valid knowledge (pramāṇa)."""

    PRATYAKSHA = "Pratyakṣa"   # Direct perception
    ANUMANA = "Anumāna"         # Inference
    UPAMANA = "Upamāna"         # Comparison / analogy
    SHABDA = "Śabda"            # Testimony / verbal knowledge
    ARTHAPATTI = "Arthāpatti"   # Postulation / abduction


class _BaseKnowledgeUnit(BaseModel):
    """Shared strict base model for all knowledge units."""

    claim: str = Field(..., description="The knowledge claim being asserted.")
    source: str = Field(..., description="Identifier of the knowledge source.")
    pramana_type: PramanaType = Field(
        ..., description="Epistemological source category (pramāṇa)."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score in the range [0, 1]."
    )
    timestamp: datetime = Field(
        ..., description="ISO-8601 timestamp of when the claim was recorded."
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_claim_non_empty(self) -> "_BaseKnowledgeUnit":
        if not self.claim.strip():
            raise ValueError("claim must not be blank.")
        if not self.source.strip():
            raise ValueError("source must not be blank.")
        return self


class Proposition(_BaseKnowledgeUnit):
    """A logical proposition to be evaluated by the inference engine."""


class Evidence(_BaseKnowledgeUnit):
    """A piece of evidence supporting or contesting a proposition."""
