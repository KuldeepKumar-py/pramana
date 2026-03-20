"""
Pramāṇa-Constrained Inference Engine.

A Python-based inference engine that enforces epistemic justification
based on Nyāya philosophy.
"""

from pramana_engine.models import Evidence, PramanaType, Proposition
from pramana_engine.ingestion import ingest

__all__ = [
    "Evidence",
    "Proposition",
    "PramanaType",
    "ingest",
]
