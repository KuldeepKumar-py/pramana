"""
Test suite for the Pramāṇa-Constrained Inference Engine.

Covers:
1. Nyāya syllogism – success case
2. Nyāya syllogism – missing hetu → Asiddha
3. Contradictory evidence → Satpratipakṣa
4. Pratyakṣa overrides Anumāna → Bādhita verdict
5. Upamāna similarity used to support inference
6. Pydantic model validation (extra fields, confidence range)
7. Ingestion module normalisation
8. Arthāpatti missing-premise detection
9. Priority gate Bādhita trigger
10. Full engine pipeline (valid verdict)
11. Execution timing – elapsed_time_ms present and non-negative
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from pramana_engine.anumana import (
    assert_paksha,
    assert_vyapti,
    nyaya_syllogism,
    paksha,
    search_sadhya,
    vyapti,
)
from pramana_engine.arthapatti import ArthappattiEngine
from pramana_engine.engine import PramanaEngine
from pramana_engine.ingestion import ingest
from pramana_engine.models import Evidence, PramanaType, Proposition
from pramana_engine.priority_gate import run_priority_gate
from pramana_engine.upamana import (
    cosine_similarity_matrix,
    jaccard_similarity_matrix,
    top_k_analogies,
)
from pramana_engine.verdict import VerdictLabel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _prop(
    claim: str,
    source: str = "test",
    ptype: PramanaType = PramanaType.ANUMANA,
    confidence: float = 0.8,
) -> Proposition:
    return Proposition(
        claim=claim,
        source=source,
        pramana_type=ptype,
        confidence=confidence,
        timestamp=_NOW,
    )


def _ev(
    claim: str,
    source: str = "test",
    ptype: PramanaType = PramanaType.ANUMANA,
    confidence: float = 0.8,
) -> Evidence:
    return Evidence(
        claim=claim,
        source=source,
        pramana_type=ptype,
        confidence=confidence,
        timestamp=_NOW,
    )


# ---------------------------------------------------------------------------
# Test 1 – Kanren: Nyāya syllogism success case
# ---------------------------------------------------------------------------

def test_syllogism_success():
    """Syllogism succeeds when vyāpti and paksha facts are asserted."""
    # Arrange: assert facts into the relational DB
    assert_vyapti("smoke", "fire")
    assert_paksha("hill", "smoke")

    # Act
    result = nyaya_syllogism(
        subject="hill",
        hetu_claim="smoke",
        sadhya="fire",
        example_subject="kitchen",
    )

    # Assert
    assert result.success is True, f"Expected success but got: {result.failure_reason}"
    assert "fire" in result.nigamana
    assert "hill" in result.pratijña
    assert len(result.trace_nodes) == 5


# ---------------------------------------------------------------------------
# Test 2 – Kanren: syllogism with missing hetu → Asiddha
# ---------------------------------------------------------------------------

def test_syllogism_missing_hetu_asiddha():
    """Syllogism fails with Asiddha when the paksha has no hetu mark."""
    # Use a subject with no asserted paksha relation
    result = nyaya_syllogism(
        subject="desert_plain_xyz",  # unique to avoid side-effects from test 1
        hetu_claim="ice",            # subject does not have this hetu
        sadhya="cold",
    )

    assert result.success is False
    assert "Asiddha" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# Test 3 – Kanren: contradictory evidence → Satpratipakṣa
# ---------------------------------------------------------------------------

def test_contradictory_evidence_satpratipaksha():
    """Two mutually contradicting Anumāna claims trigger Satpratipakṣa."""
    prop = _prop("hill has fire", ptype=PramanaType.ANUMANA, confidence=0.75)
    evidence_a = _ev("hill has fire", ptype=PramanaType.ANUMANA)
    evidence_b = _ev("hill has not fire", ptype=PramanaType.ANUMANA)

    gate_result = run_priority_gate([prop, evidence_a, evidence_b])

    # Both Anumāna items accepted (no Pratyakṣa override), but conflict recorded
    satpratipaksha_conflicts = [c for c in gate_result.conflicts if not c.badhita]
    assert len(satpratipaksha_conflicts) >= 1, (
        "Expected at least one Satpratipakṣa conflict record"
    )
    assert gate_result.badhita_triggered is False


# ---------------------------------------------------------------------------
# Test 4 – Kanren: Pratyakṣa overrides Anumāna → Bādhita
# ---------------------------------------------------------------------------

def test_pratyaksha_overrides_anumana_badhita():
    """Pratyakṣa claim directly contradicting Anumāna triggers Bādhita."""
    perception = _prop(
        "hill has fire",
        ptype=PramanaType.PRATYAKSHA,
        confidence=0.95,
    )
    # Anumāna claim contradicts perception
    inference_claim = _ev(
        "hill has not fire",
        ptype=PramanaType.ANUMANA,
        confidence=0.7,
    )

    gate_result = run_priority_gate([perception, inference_claim])

    assert gate_result.badhita_triggered is True
    assert any(
        "Bādhita" in cr.reason for cr in gate_result.conflicts
    )
    assert inference_claim in gate_result.rejected

    # Now run the full engine and check verdict label
    engine = PramanaEngine(confidence_threshold=0.5)
    output = engine.run(
        proposition_data=inference_claim,
        evidence_data=[perception],
    )
    assert output.verdict.label == VerdictLabel.REJECTED


# ---------------------------------------------------------------------------
# Test 5 – Upamāna: similarity used to support inference
# ---------------------------------------------------------------------------

def test_upamana_similarity_supports_inference():
    """Upamāna cosine/jaccard similarity matrices are computed correctly."""
    # ── cosine ──
    vectors = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],   # identical → cosine = 1.0
        [0.0, 1.0, 0.0],   # orthogonal → cosine = 0.0
    ])
    cos_mat = cosine_similarity_matrix(vectors)
    assert np.isclose(cos_mat[0, 1], 1.0, atol=1e-6)
    assert np.isclose(cos_mat[0, 2], 0.0, atol=1e-6)

    # ── jaccard ──
    binary = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],   # identical → jaccard = 1.0
        [0, 0, 1, 1],   # disjoint  → jaccard = 0.0
    ])
    jac_mat = jaccard_similarity_matrix(binary)
    assert np.isclose(jac_mat[0, 1], 1.0)
    assert np.isclose(jac_mat[0, 2], 0.0)

    # ── top-k ──
    labels = ["A", "B", "C"]
    top = top_k_analogies(0, cos_mat, labels=labels, k=2)
    assert top[0][2] == "B"   # most similar to A is B


# ---------------------------------------------------------------------------
# Test 6 – Pydantic: invalid model raises ValidationError
# ---------------------------------------------------------------------------

def test_pydantic_validation_rejects_extra_fields():
    """Pydantic model with extra fields raises ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Evidence(
            claim="test",
            source="tester",
            pramana_type=PramanaType.SHABDA,
            confidence=0.9,
            timestamp=_NOW,
            extra_field="should_fail",  # type: ignore[call-arg]
        )


def test_pydantic_confidence_out_of_range():
    """Confidence outside [0,1] is rejected."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Proposition(
            claim="x",
            source="src",
            pramana_type=PramanaType.PRATYAKSHA,
            confidence=1.5,   # invalid
            timestamp=_NOW,
        )


# ---------------------------------------------------------------------------
# Test 7 – Ingestion normalisation
# ---------------------------------------------------------------------------

def test_ingestion_normalises_dict_input():
    """Ingestion module correctly validates and returns Evidence from dicts."""
    raw = [
        {
            "claim": "smoke observed",
            "source": "sensor-1",
            "pramana_type": "Pratyakṣa",
            "confidence": 0.95,
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
    ]
    items = ingest(raw, kind="evidence")
    assert len(items) == 1
    assert isinstance(items[0], Evidence)
    assert items[0].pramana_type == PramanaType.PRATYAKSHA


# ---------------------------------------------------------------------------
# Test 8 – Arthāpatti missing-premise detection
# ---------------------------------------------------------------------------

def test_arthapatti_detects_missing_premise():
    """ArthappattiEngine identifies missing premises and generates hypotheses."""
    engine = ArthappattiEngine()
    result = engine.run(
        required_premises=["smoke present", "fire causes smoke"],
        available_premises=["smoke present"],  # "fire causes smoke" missing
    )
    assert result.triggered is True
    assert "fire causes smoke" in result.missing_premises
    assert len(result.hypotheses) == 1
    assert "fire causes smoke" in result.hypotheses[0].claim


# ---------------------------------------------------------------------------
# Test 9 – Priority gate: no false Bādhita for non-contradicting claims
# ---------------------------------------------------------------------------

def test_priority_gate_no_false_badhita():
    """Non-contradicting Pratyakṣa and Anumāna do NOT trigger Bādhita."""
    perception = _prop(
        "hill has smoke",
        ptype=PramanaType.PRATYAKSHA,
        confidence=0.95,
    )
    inference_claim = _ev(
        "hill has fire",  # compatible claim, not a contradiction
        ptype=PramanaType.ANUMANA,
        confidence=0.8,
    )
    gate_result = run_priority_gate([perception, inference_claim])
    assert gate_result.badhita_triggered is False
    assert inference_claim in gate_result.accepted


# ---------------------------------------------------------------------------
# Test 10 – Full engine pipeline → valid verdict
# ---------------------------------------------------------------------------

def test_full_engine_valid_verdict():
    """Full pipeline produces a valid verdict for a well-supported Pratyakṣa claim."""
    engine = PramanaEngine(confidence_threshold=0.5)
    output = engine.run(
        proposition_data={
            "claim": "the temperature is high",
            "source": "thermometer",
            "pramana_type": "Pratyakṣa",
            "confidence": 0.92,
            "timestamp": "2025-01-01T00:00:00+00:00",
        },
        evidence_data=[
            {
                "claim": "thermometer reads 40C",
                "source": "sensor-42",
                "pramana_type": "Pratyakṣa",
                "confidence": 0.90,
                "timestamp": "2025-01-01T00:00:00+00:00",
            }
        ],
    )

    assert output.verdict.label == VerdictLabel.VALID
    # Trace must be a valid JSON and contain nodes
    trace = json.loads(output.trace_json)
    assert "nodes" in trace
    assert len(trace["nodes"]) >= 1


# ---------------------------------------------------------------------------
# Test 11 – Execution timing fields are present and non-negative
# ---------------------------------------------------------------------------

def test_engine_output_contains_timing():
    """EngineOutput.elapsed_time_ms is non-negative and included in to_dict()."""
    engine = PramanaEngine(confidence_threshold=0.5)
    output = engine.run(
        proposition_data={
            "claim": "river flows downhill",
            "source": "observation",
            "pramana_type": "Pratyakṣa",
            "confidence": 0.99,
            "timestamp": "2025-01-01T00:00:00+00:00",
        },
    )

    # Total elapsed time must be a non-negative float
    assert isinstance(output.elapsed_time_ms, float)
    assert output.elapsed_time_ms >= 0.0

    # Must appear in the serialised dict
    summary = output.to_dict()
    assert "elapsed_time_ms" in summary
    assert summary["elapsed_time_ms"] >= 0.0

    # Each trace node must carry an elapsed_ms timestamp
    trace = json.loads(output.trace_json)
    for node in trace["nodes"]:
        assert "elapsed_ms" in node, f"Node {node.get('id')} missing elapsed_ms"
        assert node["elapsed_ms"] >= 0.0
