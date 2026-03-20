"""
Top-level Pramāṇa inference engine that orchestrates the full pipeline.

Pipeline
--------
1. Ingest & validate (Pydantic)
2. Pramāṇa check
3. Priority gate (Pratyakṣa override)
4. Inference (modus ponens + Nyāya syllogism)
5. Verdict classification
6. Reasoning trace (NetworkX DAG)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from pramana_engine.ingestion import ingest
from pramana_engine.inference import InferenceEngine, InferenceResult
from pramana_engine.models import Evidence, PramanaType, Proposition
from pramana_engine.priority_gate import PriorityGateResult, run_priority_gate
from pramana_engine.trace import ReasoningTrace
from pramana_engine.verdict import Verdict, VerdictLabel, classify_verdict


@dataclass
class EngineOutput:
    """Full output of one engine run."""

    proposition: Proposition
    verdict: Verdict
    inference_result: InferenceResult
    gate_result: PriorityGateResult
    trace_json: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary dict."""
        return {
            "claim": self.proposition.claim,
            "source": self.proposition.source,
            "pramana_type": self.proposition.pramana_type.value,
            "confidence": self.proposition.confidence,
            "verdict": self.verdict.label.value,
            "verdict_confidence": self.verdict.confidence,
            "justification": self.verdict.justification,
            "defeaters": self.verdict.defeaters,
            "badhita": self.verdict.badhita,
            "check_passed": self.inference_result.check.passed,
            "check_violations": self.inference_result.check.violations,
            "modus_ponens_valid": self.inference_result.modus_ponens_valid,
            "syllogism_success": (
                self.inference_result.syllogism.success
                if self.inference_result.syllogism
                else None
            ),
            "notes": self.notes,
        }


class PramanaEngine:
    """Main orchestrator for the Pramāṇa-Constrained Inference Engine.

    Parameters
    ----------
    confidence_threshold:
        Minimum confidence required to pass the epistemic gate.
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self._inference_engine = InferenceEngine(confidence_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        proposition_data: Union[dict, Proposition],
        evidence_data: Optional[List[Union[dict, Evidence]]] = None,
        subject: Optional[str] = None,
        hetu: Optional[str] = None,
        sadhya: Optional[str] = None,
        example_subject: Optional[str] = None,
    ) -> EngineOutput:
        """Run the full inference pipeline for a single proposition.

        Parameters
        ----------
        proposition_data:
            A raw dict or validated :class:`Proposition`.
        evidence_data:
            Optional list of raw dicts or :class:`Evidence` objects.
        subject:
            The pakṣa (locus of inference). Defaults to ``proposition.source``
            when not provided.
        hetu:
            Inferential mark for the Nyāya syllogism (optional).
        sadhya:
            Property to be inferred (required for Anumāna syllogism).
        example_subject:
            Known locus for the Udāharaṇa example step.

        Returns
        -------
        EngineOutput
            Full pipeline result including verdict and reasoning trace.
        """
        trace = ReasoningTrace()

        # ── 1. Ingest ──────────────────────────────────────────────────
        [prop] = ingest([proposition_data], kind="proposition")
        evidence: List[Evidence] = (
            ingest(evidence_data, kind="evidence") if evidence_data else []
        )
        trace.add_step(
            "ingestion",
            "Ingestion",
            {"proposition_claim": prop.claim, "evidence_count": len(evidence)},
        )

        # ── 2. Priority gate ────────────────────────────────────────────
        all_units = [prop] + evidence
        gate_result = run_priority_gate(all_units)
        trace.add_step(
            "priority_gate",
            "PriorityGate",
            {
                "accepted_count": len(gate_result.accepted),
                "rejected_count": len(gate_result.rejected),
                "badhita": gate_result.badhita_triggered,
            },
            parent="ingestion",
        )

        # ── 3. Inference ────────────────────────────────────────────────
        inference_result = self._inference_engine.evaluate(
            proposition=prop,
            evidence=evidence,
            subject=subject,
            hetu=hetu,
            sadhya=sadhya,
            example_subject=example_subject,
        )
        trace.add_step(
            "pramana_check",
            "PramanaCheck",
            {
                "pramana_type": prop.pramana_type.value,
                "check_passed": inference_result.check.passed,
                "violations": inference_result.check.violations,
            },
            parent="priority_gate",
        )
        trace.add_step(
            "inference",
            "Inference",
            {
                "epistemic_gate_passed": inference_result.epistemic_gate_passed,
                "modus_ponens_valid": inference_result.modus_ponens_valid,
            },
            parent="pramana_check",
        )

        # Add syllogism trace nodes if available
        if inference_result.syllogism:
            trace.add_syllogism_steps(
                inference_result.syllogism.trace_nodes,
                parent="inference",
            )
            last_syl_step = (
                f"syllogism_step_{len(inference_result.syllogism.trace_nodes)}"
                if inference_result.syllogism.trace_nodes
                else "inference"
            )
        else:
            last_syl_step = "inference"

        # ── 4. Verdict ──────────────────────────────────────────────────
        verdict = classify_verdict(inference_result, gate_result)
        trace.add_step(
            "verdict",
            "Verdict",
            {
                "label": verdict.label.value,
                "confidence": verdict.confidence,
                "justification": verdict.justification,
                "badhita": verdict.badhita,
            },
            parent=last_syl_step,
        )

        return EngineOutput(
            proposition=prop,
            verdict=verdict,
            inference_result=inference_result,
            gate_result=gate_result,
            trace_json=trace.to_json(),
            notes=inference_result.notes,
        )
