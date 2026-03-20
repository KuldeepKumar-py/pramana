"""
HRE — Hypothetical Reasoning Engine

🔒 FROZEN as of 2025-12-15 - DO NOT MODIFY 🔒

Epistemic sandbox for counterfactual reasoning.

CRITICAL INVARIANT:
  HRE answers are epistemically sterile.
  They create NO evidence, NO confidence, NO memory.
  
If you violate this even once, STOP.

Architecture:
  HRE = Buddhi ∘ Chitta_clone
  
  1. Clone Chitta (sandbox)
  2. Add assumptions to clone
  3. Run Buddhi on clone
  4. Return HypotheticalAnswerProof
  5. Discard clone (garbage collect)
  
  Base Chitta is NEVER modified.

IMMUTABILITY POLICY:
  This module is FROZEN. No features. No tweaks.
  If acceptance ratio drops, DO NOT use HRE to "improve" it.
  HRE is for CONDITIONALS, not INTELLIGENCE AUGMENTATION.
  
  Violations of epistemic sterility = IMMEDIATE STOP.
  
See: EPISTEMIC_STERILITY.md, CONTRACTS.md (Contract 3)
"""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buddhi.layer import Buddhi
    from chitta.graph import ChittaGraph
    from manas.layer import Manas

from buddhi.proof import (
    HypotheticalAnswerProof,
    HypotheticalAssumption,
)
from buddhi.layer import Buddhi


class HRE:
    """
    Hypothetical Reasoning Engine - for "what if" questions.
    
    The big idea: sometimes you want to explore hypotheticals without committing
    to them as beliefs. "What if penguins could fly?" shouldn't actually make
    MARC believe penguins can fly.
    
    How it works:
      HRE = Buddhi ∘ Chitta_clone
      
    The process:
      1. Make a deep copy of Chitta (creates a sandbox)
      2. Parse the assumptions using Manas
      3. Add those assumptions to the sandbox (as temporary beliefs)
      4. Let Buddhi reason in the sandbox
      5. Return the results (marked as hypothetical)
      6. Throw away the sandbox (Python garbage collects it)
      
    The key invariant (NEVER break this):
      The base Chitta stays READ-ONLY. HRE never modifies real beliefs,
      never updates confidences, never creates edges in the real graph.
      Everything happens in a disposable copy.
    """
    
    def __init__(self, buddhi: Buddhi, chitta: ChittaGraph, manas: Manas):
        """
        Set up HRE with references to the core components.
        
        Args:
            buddhi: the reasoning engine (we'll use it in the sandbox)
            chitta: the belief graph (READ ONLY - we never touch this directly)
            manas: the parser (for understanding hypothetical assumptions)
        """
        self.buddhi = buddhi
        self.base_chitta = chitta  # READ ONLY - never modify
        self.manas = manas
        
        # Statistics (for debugging)
        self.stats = {
            "queries": 0,
            "sandboxes_created": 0,
            "assumptions_added": 0,
        }
    
    def answer(
        self,
        query: str,
        assumptions: list[str] | None = None
    ) -> HypotheticalAnswerProof:
        """
        Answer query under hypothetical assumptions.
        
        This is the ONLY public API for HRE.
        
        Process:
          1. Create sandbox (clone base Chitta)
          2. Parse and add assumptions
          3. Query Buddhi using sandbox
          4. Wrap in HypotheticalAnswerProof
          5. Discard sandbox
          
        Args:
            query: natural language question
            assumptions: list of temporary beliefs (optional)
        
        Returns:
            HypotheticalAnswerProof (marked hypothetical=True)
            
        Invariants:
            - Base Chitta unchanged after this call
            - NO side effects on real memory
            - Output always has hypothetical=True
        """
        self.stats["queries"] += 1
        
        assumptions = assumptions or []
        
        # ================================================================
        # PHASE 1: Create sandbox (clone base Chitta)
        # ================================================================
        sandbox = self._create_sandbox()
        
        # ================================================================
        # PHASE 2: Parse and add assumptions to sandbox
        # ================================================================
        parsed_assumptions = []
        
        for assumption_text in assumptions:
            # Parse via Manas
            proposal = self.manas.parse(assumption_text)
            
            # Add to sandbox as asserted belief (confidence=1.0)
            
            # ----------------------------------------------------------------
            # WORLD FORKING (Option A): Suspend conflicting axioms
            # If assumption is "S is_a T", we disable existing "S is_a X"
            # This allows "If bats were fish" to stop bat->mammal inheritance
            # ----------------------------------------------------------------
            if proposal.get('template') == 'is_a':
                subject = proposal.get('canonical', {}).get('subject')
                if subject:
                    # Find and suspend existing is_a beliefs for this subject
                    for b in sandbox.beliefs.values():
                        if b.active and b.template == 'is_a' and b.subject == subject:
                            b.active = False
                            # We don't log this suspension yet, but it's crucial for logic
            
            # This is OK because sandbox is ephemeral
            sandbox.add_belief_from_proposal(
                proposal,
                confidence=1.0,  # Assumptions are treated as certain
            )
            
            # Record for proof
            parsed_assumptions.append(
                HypotheticalAssumption(
                    statement=assumption_text,
                    canonical=proposal.get("canonical", {}),
                    confidence=1.0,
                )
            )
            
            self.stats["assumptions_added"] += 1
        
        # ================================================================
        # PHASE 3: Query Buddhi using sandbox
        # ================================================================
        # Create temporary Buddhi instance pointing to sandbox
        sandbox_buddhi = self._create_sandbox_buddhi(sandbox)
        
        # Parse query
        query_proposal = self.manas.parse(query)
        
        # Get answer from sandbox Buddhi
        base_proof = sandbox_buddhi.answer(query_proposal)
        
        # ================================================================
        # PHASE 4: Wrap in HypotheticalAnswerProof
        # ================================================================
        hypothetical_proof = HypotheticalAnswerProof(
            query=query,
            assumptions=parsed_assumptions,
            verdict=base_proof.verdict,
            hypothetical=True,  # LOCKED
        )
        
        # Copy derivation steps (marked as hypothetical)
        for step in base_proof.steps:
            hypothetical_proof.add_step(
                rule=step.rule,
                inputs=step.inputs,
                output=step.output,
                confidence=step.confidence,
            )
        
        # Copy conflicts (if any)
        for conflict in base_proof.conflicts:
            hypothetical_proof.add_conflict(
                predicate=conflict.predicate,
                assumption_a=conflict.positive,
                assumption_b=conflict.negative,
                delta=conflict.delta,
            )
        
        # ================================================================
        # PHASE 5: Discard sandbox (garbage collection)
        # ================================================================
        # Python will automatically GC sandbox and sandbox_buddhi
        # No explicit cleanup needed
        
        return hypothetical_proof
    
    def _create_sandbox(self) -> ChittaGraph:
        """
        Create ephemeral sandbox by deep-copying base Chitta.
        
        Returns:
            ChittaGraph clone (completely independent)
        """
        self.stats["sandboxes_created"] += 1
        
        # Deep copy creates completely independent instance
        # Changes to sandbox do NOT affect base_chitta
        sandbox = copy.deepcopy(self.base_chitta)
        
        return sandbox
    
    def _create_sandbox_buddhi(self, sandbox: ChittaGraph) -> Buddhi:
        """
        Create temporary Buddhi instance pointing to sandbox.
        
        Args:
            sandbox: sandboxed ChittaGraph
            
        Returns:
            Buddhi instance (using sandbox, not base Chitta)
        """
        # Create new Buddhi pointing to sandbox
        sandbox_buddhi = Buddhi(chitta=sandbox)
        
        return sandbox_buddhi
    
    # ====================================================================
    # FORBIDDEN OPERATIONS (do not implement)
    # ====================================================================
    # The following methods are explicitly FORBIDDEN:
    #
    # def process(self, statement: str) -> None:
    #     """FORBIDDEN: HRE must not process statements"""
    #
    # def teach(self, facts: list[str]) -> None:
    #     """FORBIDDEN: HRE must not learn"""
    #
    # def store(self, belief: Belief) -> str:
    #     """FORBIDDEN: HRE must not write to Chitta"""
    #
    # def update_confidence(self, belief_id: str, confidence: float) -> None:
    #     """FORBIDDEN: HRE must not modify beliefs"""
    #
    # If you add any of these, STOP immediately.
    # ====================================================================
