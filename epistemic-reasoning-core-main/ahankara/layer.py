"""
AHANKARA — Self and Execution Orchestrator

Ahankara is Sanskrit for "I-maker" or "ego."

In MARC, Ahankara is:
- The SUPREME DOER
- The execution controller
- The scheduler/orchestrator  
- The owner of the cognitive loop
- NOT a reasoner
- NOT a storage

Philosophy:
- SELF is not data
- SELF is not memory
- SELF is not a belief
- SELF is the DOER
- WITHOUT AHANKARA'S WILL, NOTHING HAPPENS
- Manas cannot receive input without Ahankara
- Buddhi cannot judge without Ahankara
- Chitta cannot store without Ahankara (via Buddhi)

Responsibilities:
- Receive input (ONLY entry point)
- Trigger Manas (parsing) - Ahankara decides WHEN
- Trigger Buddhi (reasoning) - Ahankara decides WHEN
- Trigger Chitta (storage via Buddhi) - Ahankara decides WHEN
- Return output
- Log to Sakshin
- Manage cognitive cycle phases
- Enforce execution budgets

Cognitive Cycle Phases:
1. PERCEIVE: Ahankara → Manas (understanding)
2. JUDGE: Ahankara → Buddhi (reasoning)
3. COMMIT: Ahankara → Buddhi → Chitta (storage)
4. REFLECT: Ahankara → HRE (hypothetical reasoning) [future]
5. OBSERVE: Ahankara → Sakshin (logging) [future]

Core Loop:
text → Ahankara.perceive() → proposal → Ahankara.judge() → result
"""

from __future__ import annotations

from manas.layer import Manas
from buddhi.layer import Buddhi, AnswerProof
from chitta.graph import ChittaGraph
from hre.layer import HRE
from buddhi.proof import HypotheticalAnswerProof
from common.types import Answer, Verdict
from ahankara.perception import get_perceptual_priors
from buddhi.geography import get_geographic_memory
from common.utils import now_utc
import os
import json


# ═══════════════════════════════════════════════════════════════
# AHANKARA CLASS
# ═══════════════════════════════════════════════════════════════

# Cognitive phases (what stage of thinking are we in?)
PHASE_IDLE = "IDLE"           # waiting for input
PHASE_PERCEIVE = "PERCEIVE"   # parsing text into structured meaning
PHASE_JUDGE = "JUDGE"
PHASE_COMMIT = "COMMIT"
PHASE_REFLECT = "REFLECT"
PHASE_OBSERVE = "OBSERVE"


class Ahankara:
    """
    Execution orchestrator for MARC.
    
    The SUPREME DOER - the SELF that binds all components together.
    
    WITHOUT AHANKARA, NOTHING HAPPENS.
    
    Components:
    - Manas: understanding (triggered by Ahankara)
    - Buddhi: reasoning (triggered by Ahankara)
    - Chitta: memory (written via Buddhi, triggered by Ahankara)
    - Sakshin: observer (future)
    
    Invariants:
    - No logic in Ahankara
    - No intelligence in Ahankara
    - Pure orchestration
    - All component calls go through Ahankara
    - Phase assertions enforced (MUST be in correct phase)
    """
    
    def __init__(
        self,
        manas: Manas | None = None,
        buddhi: Buddhi | None = None,
        chitta: ChittaGraph | None = None,
        max_cycles_per_input: int = 10,
        max_query_depth: int = 5,
        persistence_dir: str | None = None,
    ):
        """
        Initialize Ahankara with MARC components.
        
        Args:
            manas: Manas instance (understanding)
            buddhi: Buddhi instance (reasoning)
            chitta: ChittaGraph instance (memory)
            max_cycles_per_input: execution budget (prevent infinite reasoning)
            max_query_depth: deliberation depth limit for queries
            persistence_dir: Directory to save/load state (Persistent Live KB)
        """
        # Persistence Configuration
        self.persistence_dir = persistence_dir
        if self.persistence_dir:
            os.makedirs(self.persistence_dir, exist_ok=True)
            
        # Initialize components if not provided
        self.chitta = chitta or ChittaGraph()
        
        # AUTO-LOAD: If persistence enabled, load graph immediately
        if self.persistence_dir:
            graph_path = os.path.join(self.persistence_dir, "graph.json")
            if os.path.exists(graph_path):
                print(f"[Ahankara] Loading memory from {graph_path}...")
                try:
                    report = self.chitta.load(graph_path)
                    print(f"[Ahankara] Loaded {report['loaded']} / {report['total_in_file']} beliefs.")
                    if report['errors'] > 0:
                         print(f"[Ahankara] ⚠️ Skipped {report['errors']} beliefs due to errors: {dict(report['reasons'])}")
                except Exception as e:
                    print(f"[Ahankara] Failed to load memory: {e}")
        self.manas = manas or Manas(llm_backend="mock")
        self.buddhi = buddhi or Buddhi(self.chitta)
        
        # Initialize HRE (Hypothetical Reasoning Engine)
        # HRE sits BELOW Ahankara's authority
        # It can only be invoked via Ahankara, never directly
        if HRE is not None:
            self.hre = HRE(
                buddhi=self.buddhi,
                chitta=self.chitta,  # READ ONLY
                manas=self.manas,
            )
        else:
            self.hre = None
        
        # Initialize perceptual priors (optional, explicit)
        # NON-INFERABLE, NON-INHERITABLE observational knowledge
        if get_perceptual_priors is not None:
            self.perceptual_priors = get_perceptual_priors()
        else:
            self.perceptual_priors = None
        
        # Initialize geographic memory (external memory, no inference)
        # Retrieval-only for encyclopedic facts
        if get_geographic_memory is not None:
            self.geographic_memory = get_geographic_memory()
        else:
            self.geographic_memory = None
        
        # Execution budget
        self.max_cycles_per_input = max_cycles_per_input
        self.max_query_depth = max_query_depth
        
        # Current phase (for assertions)
        self.phase = PHASE_IDLE
        
        # Query depth tracking (for deliberation budget)
        self.query_depth = 0
        
        # Deferred queries (for retry loop)
        self.deferred_queries: list[dict] = []
        
        # Statistics
        self.stats = {
            "cycles": 0,
            "inputs": 0,
            "perceive_calls": 0,
            "judge_calls": 0,
            "commit_calls": 0,
            "query_depth_exceeded": 0,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # COGNITIVE CYCLE PHASES (AHANKARA AS SUPREME DOER)
    # ═══════════════════════════════════════════════════════════════
    
    def perceive(self, text: str) -> dict:
        """
        PHASE 1: PERCEIVE
        
        Ahankara triggers Manas to understand input.
        
        WITHOUT AHANKARA'S WILL, MANAS CANNOT RECEIVE INPUT.
        
        Args:
            text: raw input
        
        Returns:
            BeliefProposal from Manas
        """
        # PHASE ASSERTION: Must be in PERCEIVE or IDLE phase
        assert self.phase in (PHASE_IDLE, PHASE_PERCEIVE), \
            f"Cannot perceive in phase {self.phase}. Must be IDLE or PERCEIVE."
        
        self.phase = PHASE_PERCEIVE
        self.stats["perceive_calls"] += 1
        
        # Ahankara commands Manas: "Understand this"
        proposal = self.manas.parse(text)
        
        return proposal
    
    def judge(self, proposal: dict) -> dict:
        """
        PHASE 2: JUDGE
        
        Ahankara triggers Buddhi to reason about proposal.
        
        WITHOUT AHANKARA'S WILL, BUDDHI CANNOT JUDGE.
        
        Args:
            proposal: BeliefProposal from Manas
        
        Returns:
            judgment result from Buddhi
        """
        # PHASE ASSERTION: Must be in PERCEIVE phase (just finished perceiving)
        assert self.phase == PHASE_PERCEIVE, \
            f"Cannot judge in phase {self.phase}. Must PERCEIVE first."
        
        self.phase = PHASE_JUDGE
        self.stats["judge_calls"] += 1
        
        # Ahankara commands Buddhi: "Judge this"
        result = self.buddhi.think(proposal)
        
        return result
    
    def commit(self, result: dict) -> dict:
        """
        PHASE 3: COMMIT
        
        Ahankara acknowledges the commit to Chitta (already done by Buddhi).
        
        Chitta is ONLY written via Buddhi, and Buddhi is ONLY triggered by Ahankara.
        
        Args:
            result: judgment result (already committed)
        
        Returns:
            result (unchanged)
        """
        # PHASE ASSERTION: Must be in JUDGE phase (just finished judging)
        assert self.phase == PHASE_JUDGE, \
            f"Cannot commit in phase {self.phase}. Must JUDGE first."
        
        self.phase = PHASE_COMMIT
        self.stats["commit_calls"] += 1
        
        # Commit already happened in Buddhi.think()
        # Ahankara just acknowledges it
        
        # Return to IDLE after commit
        self.phase = PHASE_IDLE
        
        return result
    
    def query_answer(self, question: str, depth: int = 0, explain_conflicts: bool = False) -> str:
        """
        Query answering through Ahankara's will.
        
        WITHOUT AHANKARA, BUDDHI CANNOT ANSWER.
        
        Query Resolution Order (Epistemic Discipline):
        1. Check perceptual priors (observational knowledge)
        2. Check geographic memory (external memory, retrieval-only)
        3. Call Buddhi for logical reasoning (inference)
        
        Phases:
        1. Ahankara.perceive() → parse question
        2. Check external knowledge sources (perceptual/geographic)
        3. Ahankara triggers Buddhi.answer() → AnswerProof (if needed)
        4. Ahankara renders AnswerProof to natural language
        
        Args:
            question: natural language question
            depth: current deliberation depth (for budget enforcement)
            explain_conflicts: If True, surface known contradictions in answer
        
        Returns:
            answer string (rendered by Ahankara)
        """
        # DELIBERATION BUDGET: Check depth limit
        if depth >= self.max_query_depth:
            self.stats["query_depth_exceeded"] += 1
            return self._render_depth_limit_exceeded(question, depth)
        
        # Track query depth
        old_depth = self.query_depth
        self.query_depth = depth
        
        # PHASE 1: PERCEIVE (Ahankara → Manas)
        query_proposal = self.perceive(question)
        
        # PHASE 1.5: CHECK EXTERNAL KNOWLEDGE SOURCES
        # (before calling Buddhi for logical reasoning)
        
        # Try perceptual priors first (if available)
        if self.perceptual_priors is not None:
            perceptual_answer = self._check_perceptual_priors(query_proposal)
            if perceptual_answer is not None:
                self.query_depth = old_depth
                self.phase = PHASE_IDLE
                return perceptual_answer
        
        # Try geographic memory second (if available)
        if self.geographic_memory is not None:
            geographic_answer = self._check_geographic_memory(query_proposal)
            if geographic_answer is not None:
                self.query_depth = old_depth
                self.phase = PHASE_IDLE
                return geographic_answer
        
        # PHASE 2: JUDGE (Ahankara → Buddhi)
        # Ahankara commands Buddhi: "Answer this query"
        # Buddhi now returns AnswerProof (verdict + derivation trace)
        self.phase = PHASE_JUDGE
        proof = self.buddhi.answer(query_proposal)  # Returns AnswerProof
        
        # PHASE 3: RENDER (Ahankara converts proof to language)
        # ONLY Ahankara speaks to the external world
        rendered_answer = self._render_proof(proof, query_proposal, depth, explain_conflicts)
        
        # Check if uncertain - add to deferred queries
        if proof.verdict in ["uncertain", "unknown"]:
            self.deferred_queries.append({
                "question": question,
                "proposal": query_proposal,
                "timestamp": self.stats["cycles"],
                "depth": depth,
            })
        
        # Restore depth
        self.query_depth = old_depth
        
        # Return to IDLE
        self.phase = PHASE_IDLE
        
        return rendered_answer
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN EXECUTION LOOP (FULL COGNITIVE CYCLE)
    # ═══════════════════════════════════════════════════════════════
    
    def step(self, text: str) -> dict:
        """
        Execute one cognitive cycle through Ahankara.
        
        PHASES:
        1. PERCEIVE: Ahankara → Manas (understanding)
        2. JUDGE: Ahankara → Buddhi (reasoning)
        3. COMMIT: Ahankara acknowledges storage
        4. REFLECT: Ahankara → HRE (future)
        5. OBSERVE: Ahankara → Sakshin (future)
        
        WITHOUT AHANKARA, NOTHING HAPPENS.
        
        Args:
            text: raw natural language input
        
        Returns:
            result dict with belief_id and action
        """
        self.stats["cycles"] += 1
        self.stats["inputs"] += 1
        
        # PHASE 1: PERCEIVE (Ahankara → Manas)
        proposal = self.perceive(text)
        
        # PHASE 2: JUDGE (Ahankara → Buddhi)
        result = self.judge(proposal)
        
        # PHASE 3: COMMIT (acknowledge)
        result = self.commit(result)
        
        # PERSISTENCE & LOGGING
        if result.get("action") == "added_belief" or result.get("action") == "updated_belief":
            self._log_event("TEACH", {"input": text, "result": result})
            self._persist_state()
        elif result.get("action") == "hypothetical":
             # Still log hypotheticals? Yes, for audit.
             self._log_event("HYPOTHESIS", {"input": text, "result": result})
             
        # PHASE 4: REFLECT (future - HRE)
        # TODO: Ahankara → HRE for hypothetical reasoning
        
        # PHASE 5: OBSERVE (future - Sakshin)
        # TODO: Ahankara → Sakshin for logging
        
        return result
    
    def run(self, text: str) -> dict:
        """
        Alias for step() with more descriptive name.
        
        Args:
            text: input text
        
        Returns:
            result dict
        """
        return self.step(text)
    
    def process(self, text: str) -> dict:
        """
        Process input and return human-readable result.
        
        Args:
            text: input text
        
        Returns:
            dict with action, belief_id, and message
        """
        result = self.step(text)
        
        # Add human-readable message
        action = result.get("action", "unknown")
        
        if action == "added_belief":
            result["message"] = "Belief accepted and stored"
        elif action == "existing_belief":
            result["message"] = "Existing belief confirmed"
        elif action == "hypothetical":
            result["message"] = "Uncertain - stored as hypothesis"
        elif action == "unknown":
            result["message"] = "I don't know - insufficient information"
        else:
            result["message"] = "Processed"
        
        return result
    
    def set_reasoning_mode(self):
        """Switch Buddhi to reasoning mode (enables decay/promotion/demotion)."""
        self.buddhi.learning_mode = False
    
    def set_learning_mode(self):
        """Switch Buddhi to learning mode (disables lifecycle management)."""
        self.buddhi.learning_mode = True
    
    # ═══════════════════════════════════════════════════════════════
    # QUERY INTERFACE
    # ═══════════════════════════════════════════════════════════════
    
    def ask(self, question: str, explain_conflicts: bool = False) -> str:
        """
        Ask a question through Ahankara's orchestration.
        
        AHANKARA IS THE SUPREME DOER.
        Without Ahankara's will, Buddhi cannot answer.
        
        Args:
            question: natural language question
            explain_conflicts: If True, surface known contradictions in answer
        
        Returns:
            answer string
        """
        # Reset phase to IDLE before each question (prevents phase lock)
        self.phase = PHASE_IDLE
        
        return self.query_answer(question, explain_conflicts=explain_conflicts)
    
    def ask_hypothetically(
        self,
        question: str,
        assumptions: list[str] | None = None
    ) -> str:
        """
        Ask a hypothetical question (via HRE).
        
        CRITICAL: HRE answers are epistemically sterile.
        They create NO evidence, NO confidence, NO memory.
        
        Ahankara enforces:
          - HRE sits BELOW authority
          - Answers are linguistically marked as hypothetical
          - NO side effects on Chitta
        
        Args:
            question: natural language question
            assumptions: list of temporary beliefs (optional)
        
        Returns:
            answer string (marked as hypothetical)
        """
        if self.hre is None:
            return "Hypothetical reasoning not available."
        
        # Record Chitta state BEFORE HRE (for invariant check)
        belief_count_before = len(self.chitta.beliefs)
        
        # Query HRE (creates ephemeral sandbox)
        hypothetical_proof = self.hre.answer(
            query=question,
            assumptions=assumptions,
        )
        
        # INVARIANT CHECK: Chitta MUST be unchanged
        belief_count_after = len(self.chitta.beliefs)
        if belief_count_before != belief_count_after:
            raise RuntimeError(
                f"🚨 EPISTEMIC STERILITY VIOLATED! "
                f"Chitta modified by HRE: {belief_count_before} → {belief_count_after}"
            )
        
        # Render to natural language (linguistically marked)
        answer = self._render_hypothetical_proof(hypothetical_proof)
        
        return answer
    
    def _render_hypothetical_proof(self, proof: HypotheticalAnswerProof) -> str:
        """
        Convert HypotheticalAnswerProof to natural language.
        
        CRITICAL: Answer MUST be linguistically marked as hypothetical.
        HRE must NEVER speak with Buddhi's voice.
        
        Args:
            proof: HypotheticalAnswerProof from HRE
        
        Returns:
            natural language answer (marked as hypothetical)
        """
        # Use proof's built-in rendering (already marked)
        return proof.to_natural_language()
    
    def retry_deferred_queries(self) -> list[dict]:
        """
        Retry previously uncertain queries.
        
        After new beliefs are added, previously uncertain
        queries may now have answers.
        
        Returns:
            list of retry results
        """
        if not self.deferred_queries:
            return []
        
        results = []
        remaining = []
        
        for query_info in self.deferred_queries:
            # Check if query is old enough to retry
            age = self.stats["cycles"] - query_info["timestamp"]
            if age < 3:  # Wait at least 3 cycles
                remaining.append(query_info)
                continue
            
            # Retry the query with incremented depth
            depth = query_info.get("depth", 0) + 1
            answer = self.query_answer(query_info["question"], depth=depth)
            
            # If still uncertain, keep in deferred list
            if "not certain" in answer.lower() or "don't know" in answer.lower():
                remaining.append(query_info)
            else:
                # Resolved!
                results.append({
                    "question": query_info["question"],
                    "answer": answer,
                    "resolved": True,
                })
        
        self.deferred_queries = remaining
        return results
    
    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE RENDERING (ONLY AHANKARA SPEAKS)
    # ═══════════════════════════════════════════════════════════════
    
    def _render_proof(self, proof: AnswerProof, proposal: dict, depth: int, explain_conflicts: bool = False) -> str:
        """
        Convert AnswerProof to natural language.
        
        This is THE KEY ARCHITECTURAL FEATURE:
        - Buddhi produces structured proofs
        - Ahankara converts them to natural language
        - Clean separation of reasoning vs rendering
        
        Args:
            proof: AnswerProof from Buddhi
            proposal: query proposal
            depth: deliberation depth
            explain_conflicts: If True, append conflict warnings to answer
        
        Returns:
            natural language answer
        """
        # Handle unknown
        if proof.verdict == "unknown":
            return "I do not know."
        
        # Handle conflicts with detailed explanation
        if proof.conflicts:
            # Find the general rule and exceptions
            general = None
            exception = None
            
            for step in proof.steps:
                if step.rule == "direct_match":
                    if step.confidence and step.confidence > 0.7:
                        general = step.output
                    elif step.confidence and step.confidence < 0.5:
                        exception = step.output
            
            # If we have both general and exception, show uncertainty with context
            if general and exception:
                answer = (
                    f"I am not certain. "
                    f"General rule: {general}. "
                    f"Exception hypothesis: {exception}."
                )
            else:
                answer = "I am not certain."
            
            # Add depth info if needed
            if depth > 0:
                answer = f"[Depth {depth}] {answer}"
            
            return answer
        
        # Simple yes/no answer
        if proof.verdict == "yes":
            base_answer = "Yes."
            if depth > 0:
                base_answer = f"[Depth {depth}] {base_answer}"
            
            # Add conflict warning if requested
            if explain_conflicts and proof.conflicts:
                conflict_note = self._render_conflicts(proof.conflicts)
                return f"{base_answer}\n{conflict_note}"
            
            return base_answer
        
        elif proof.verdict == "no":
            # Check for hard_negative_gate rule
            for step in proof.steps:
                if step.rule == "hard_negative_gate":
                    # Return simple "No." for hard blocks
                    base_answer = "No."
                    if depth > 0:
                        base_answer = f"[Depth {depth}] {base_answer}"
                    return base_answer
            
            # Default no answer
            base_answer = "No."
            if depth > 0:
                base_answer = f"[Depth {depth}] {base_answer}"
            return base_answer
        
        elif proof.verdict == "conflict":
            base = "I found conflicting information."
            if proof.conflicts:
                note = self._render_conflicts(proof.conflicts)
                return f"{base} {note}"
            return base

        elif proof.verdict == "uncertain":
            answer = "I am not certain."
            if depth > 0:
                answer = f"[Depth {depth}] {answer}"
            return answer
        
        return "I do not know."
    
    def _render_conflicts(self, conflicts: list) -> str:
        """
        Render conflict warnings in natural language.
        
        Args:
            conflicts: List of ConflictRecord from AnswerProof
        
        Returns:
            Human-readable conflict warning
        """
        if not conflicts:
            return ""
        
        if len(conflicts) == 1:
            conflict = conflicts[0]
            return f"⚠️ Note: Conflicting belief exists ({conflict.statement})."
        else:
            conflict_statements = [c.statement for c in conflicts[:3]]  # Limit to 3
            conflict_list = ", ".join(conflict_statements)
            if len(conflicts) > 3:
                return f"⚠️ Note: {len(conflicts)} conflicting beliefs exist (e.g., {conflict_list}, ...)."
            return f"⚠️ Note: Conflicting beliefs exist ({conflict_list})."
    
    def _render_answer(self, raw_answer: str, proposal: dict, depth: int) -> str:
        """
        Render answer in natural language.
        
        ONLY Ahankara speaks to the external world.
        Buddhi returns judgments, Ahankara renders them.
        
        Args:
            raw_answer: raw answer from Buddhi
            proposal: query proposal
            depth: deliberation depth
        
        Returns:
            rendered answer string
        """
        # Currently Buddhi.answer() already returns natural language
        # In future refactor, Buddhi should return structured judgments
        # and this method would convert them to language
        
        # For now, just pass through (but could add depth info, etc.)
        if depth > 0:
            return f"[Depth {depth}] {raw_answer}"
        return raw_answer
    
    def _render_depth_limit_exceeded(self, question: str, depth: int) -> str:
        """
        Render message when deliberation depth limit is exceeded.
        
        Args:
            question: the question that exceeded depth
            depth: current depth
        
        Returns:
            rendered message
        """
        return (
            f"I am not certain. "
            f"The question requires too much deliberation (depth {depth} >= {self.max_query_depth}). "
            f"I need more direct evidence to answer: {question}"
        )
    
    # ═══════════════════════════════════════════════════════════════
    # PERSISTENCE HELPERS
    # ═══════════════════════════════════════════════════════════════
    
    def _persist_state(self):
        """Save current graph state to disk."""
        if not self.persistence_dir:
            return
            
        graph_path = os.path.join(self.persistence_dir, "graph.json")
        try:
            self.chitta.save(graph_path)
        except Exception as e:
            print(f"[Ahankara] Failed to save state: {e}")
            
    def _log_event(self, event_type: str, payload: dict):
        """Append event to event log."""
        if not self.persistence_dir:
            return
            
        log_path = os.path.join(self.persistence_dir, "event_log.jsonl")
        event = {
            "timestamp": str(now_utc()),
            "type": event_type,
            "payload": payload,
            "stats": self.stats
        }
        
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            print(f"[Ahankara] Failed to log event: {e}")

    # ═══════════════════════════════════════════════════════════════
    # EXTERNAL KNOWLEDGE SOURCES (NON-INFERABLE)
    # ═══════════════════════════════════════════════════════════════
    
    def _check_perceptual_priors(self, query_proposal: dict) -> str | None:
        """
        Check perceptual priors for answer (non-inferable observational knowledge).
        
        Perceptual priors:
        - NON-INFERABLE: Cannot be derived logically
        - NON-INHERITABLE: Do not propagate taxonomically
        - LOWER CONFIDENCE: Always < logical beliefs
        - EXPLICITLY LABELED: Marked as "perceptual"
        
        Examples: "Is gold shiny?", "Is water liquid?", "Is copper conductive?"
        
        Args:
            query_proposal: parsed query from Manas
        
        Returns:
            Answer string if found in priors, None otherwise
        """
        if self.perceptual_priors is None:
            return None
        
        # Extract entities and properties from query
        # For queries like "Is gold shiny?", entities = ['gold', 'shiny']
        entities = query_proposal.get('entities', [])
        predicates = query_proposal.get('predicates', [])
        
        if not entities:
            return None
        
        # Try entity-predicate combinations
        for entity in entities:
            for predicate in predicates:
                if self.perceptual_priors.has_property(entity, predicate):
                    return self.perceptual_priors.format_answer(entity, predicate)
                
                # Try normalized forms (e.g., "is_liquid" → "liquid")
                predicate_normalized = predicate.replace('is_', '').replace('_', ' ')
                if self.perceptual_priors.has_property(entity, predicate_normalized):
                    return self.perceptual_priors.format_answer(entity, predicate_normalized)
        
        # Also try entity pairs (e.g., "gold" + "shiny" → check if gold has property shiny)
        for i, entity in enumerate(entities):
            for property_candidate in entities[i+1:]:
                if self.perceptual_priors.has_property(entity, property_candidate):
                    return self.perceptual_priors.format_answer(entity, property_candidate)
                # Try reverse
                if self.perceptual_priors.has_property(property_candidate, entity):
                    return self.perceptual_priors.format_answer(property_candidate, entity)
        
        return None
    
    def _check_geographic_memory(self, query_proposal: dict) -> str | None:
        """
        Check geographic memory for answer (retrieval-only, no inference).
        
        Geographic memory:
        - NO INFERENCE: Pure lookup
        - NO REASONING: Just retrieval
        - EXTERNAL MEMORY: Not part of logical reasoning
        
        Examples: "Is London in Europe?", "Is Tokyo in Asia?"
        
        Args:
            query_proposal: parsed query from Manas
        
        Returns:
            Answer string if found in geographic memory, None otherwise
        """
        if self.geographic_memory is None:
            return None
        
        # Extract entities from query
        # For queries like "Is London in Europe?", entities = ['london', 'europe']
        entities = query_proposal.get('entities', [])
        predicates = query_proposal.get('predicates', [])
        
        if len(entities) < 2:
            return None
        
        # Check for location queries (located_in, in, contains, etc.)
        # OR if predicates is ['generic'] (typical for "Is X in Y?" queries)
        location_predicates = ['located_in', 'in', 'contains', 'part_of']
        has_location_predicate = (
            any(any(loc in pred.lower() for loc in location_predicates) for pred in predicates) or
            'generic' in predicates  # Generic often used for location queries
        )
        
        if not has_location_predicate:
            return None
        
        # Check each entity pair for geographic containment
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Try entity1 in entity2
                if self.geographic_memory.is_located_in(entity1, entity2):
                    path = self.geographic_memory.get_path(entity1)
                    if entity2.lower() in [p.lower() for p in path]:
                        idx = [p.lower() for p in path].index(entity2.lower())
                        subpath = path[:idx+1]
                        return f"Yes (geographic memory) - {' → '.join(subpath)}"
                
                # Try entity2 in entity1
                if self.geographic_memory.is_located_in(entity2, entity1):
                    path = self.geographic_memory.get_path(entity2)
                    if entity1.lower() in [p.lower() for p in path]:
                        idx = [p.lower() for p in path].index(entity1.lower())
                        subpath = path[:idx+1]
                        return f"Yes (geographic memory) - {' → '.join(subpath)}"
                
                # Check if both entities exist in geographic memory but are NOT related
                # This allows us to say "No" for "Is Tokyo in Europe?"
                path1 = self.geographic_memory.get_path(entity1)
                path2 = self.geographic_memory.get_path(entity2)
                
                # If both entities exist in memory (non-trivial paths)
                if len(path1) > 1 and len(path2) > 1:
                    # Neither is in the other
                    if not self.geographic_memory.is_located_in(entity1, entity2) and \
                       not self.geographic_memory.is_located_in(entity2, entity1):
                        # They're in different branches
                        return f"No (geographic memory) - {entity1} and {entity2} are in different locations"
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # INTROSPECTION
    # ═══════════════════════════════════════════════════════════════
    
    def introspect(self) -> dict:
        """
        Get system state and statistics.
        
        Returns:
            dict with all component stats
        """
        return {
            "ahankara": self.stats,
            "manas": self.manas.get_stats(),
            "buddhi": self.buddhi.get_stats(),
            "chitta": self.chitta.get_stats(),
        }
    
    def summary(self) -> str:
        """
        Human-readable system summary.
        
        Returns:
            summary string
        """
        intro = self.introspect()
        
        lines = [
            "MARC System Summary",
            "=" * 50,
            f"Cognitive cycles: {intro['ahankara']['cycles']}",
            f"Inputs processed: {intro['ahankara']['inputs']}",
            "",
            "Manas (Understanding):",
            f"  Parses: {intro['manas']['parses']}",
            f"  Success rate: {intro['manas']['successes']/(intro['manas']['parses'] or 1):.1%}",
            "",
            "Buddhi (Reasoning):",
            f"  Judgments: {intro['buddhi']['judgments']}",
            f"  Accepts: {intro['buddhi']['accepts']}",
            f"  Uncertains: {intro['buddhi']['uncertains']}",
            f"  Unknowns: {intro['buddhi']['unknowns']}",
            "",
            "Chitta (Memory):",
            f"  Total beliefs: {intro['chitta']['total_beliefs']}",
            f"  Active beliefs: {intro['chitta']['active_beliefs']}",
            f"  Total edges: {intro['chitta']['total_edges']}",
        ]
        
        return "\n".join(lines)
    
    # ═══════════════════════════════════════════════════════════════
    # REPR
    # ═══════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return (
            f"<Ahankara "
            f"cycles={self.stats['cycles']} "
            f"beliefs={self.chitta.count_beliefs()}>"
        )
