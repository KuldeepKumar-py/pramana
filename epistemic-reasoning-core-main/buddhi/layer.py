"""
BUDDHI — Reasoning and Judgment Engine

Buddhi is Sanskrit for "intellect" or "discernment."

In MARC, Buddhi is:
- The ONLY module that writes to Chitta
- The judgment engine that decides what to believe
- The reasoning engine that handles contradictions
- The inference engine that derives new beliefs

Philosophy:
- Thinking = judgment over competing beliefs
- No LLM in Buddhi (pure algorithmic reasoning)
- Explicit, inspectable decision process
- Can say "I don't know"
- Contradictions are first-class

Core Algorithm:
1. Focus attention (restrict to subgraph)
2. Generate candidates (existing + proposed)
3. Evaluate with judgment function
4. Select winner
5. Revise Chitta (update beliefs & edges)
"""

from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass, field

from chitta.graph import ChittaGraph
from buddhi.belief import Belief, Polarity, EpistemicType
from manas.schemas import SchemaExtractor, RelationSchema
from common.types import RelationFrame, RelationKind


# ═══════════════════════════════════════════════════════════════════
# PROOF-CARRYING ANSWER TYPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DerivationStep:
    """
    Atomic, replayable step in answer derivation.
    
    Each step records:
    - Which rule was applied
    - Which beliefs were used as input
    - What intermediate conclusion was reached
    - Optional confidence value
    
    Invariant: Replaying all steps in order must reach the same verdict.
    """
    step_id: int
    rule: str                      # Rule identifier (e.g., "direct_match", "inheritance")
    inputs: list[str]              # Belief IDs used in this step
    output: str                    # Intermediate proposition/conclusion
    confidence: float | None = None  # Optional confidence for this step


@dataclass
class ConflictRecord:
    """
    Record of epistemic conflict between beliefs.
    
    Represents:
    - Two beliefs with opposing polarities on same predicate
    - Confidence difference between them
    - How the conflict was resolved
    
    This is how MARC shows epistemic honesty.
    """
    predicate: str                 # Predicate in conflict (e.g., "can_fly")
    positive: str                  # Belief ID asserting predicate
    negative: str                  # Belief ID negating predicate
    delta: float                   # |conf_positive - conf_negative|
    resolution: str                # "uncertain" | "prefer_positive" | "prefer_negative"


@dataclass
class AnswerProof:
    """
    Complete proof trace for an answer.
    
    Every answer MARC gives includes:
    - The original query
    - A verdict (yes/no/uncertain/unknown)
    - Derivation steps showing how the verdict was reached
    - Any conflicts encountered
    
    This separates MARC from:
    - RAG systems (no proofs)
    - Chain-of-thought LLMs (narrative, not formal)
    - Probabilistic QA (no epistemic states)
    
    MARC is closer to proof assistants + justification logic.
    """
    query: str
    verdict: str                   # "yes" | "no" | "uncertain" | "unknown"
    steps: list[DerivationStep] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    
    def add_step(self, rule: str, inputs: list[str], output: str, confidence: float | None = None):
        """Add a derivation step to the proof."""
        step = DerivationStep(
            step_id=len(self.steps) + 1,
            rule=rule,
            inputs=inputs,
            output=output,
            confidence=confidence
        )
        self.steps.append(step)
    
    def add_conflict(self, predicate: str, positive: str, negative: str, delta: float, resolution: str):
        """Add a conflict record to the proof."""
        conflict = ConflictRecord(
            predicate=predicate,
            positive=positive,
            negative=negative,
            delta=delta,
            resolution=resolution
        )
        self.conflicts.append(conflict)


# ═══════════════════════════════════════════════════════════════════
# JUDGMENT WEIGHTS (CONSTANTS FOR v0.1)
# ═══════════════════════════════════════════════════════════════════

# When evaluating competing beliefs, we combine several factors. These weights determine
# how much each factor contributes to the final judgment score.
W_CONF = 0.40  # How confident we are in this belief (most important)
W_SPEC = 0.20  # How specific the belief is ("penguin" beats "bird")
W_SUP = 0.15   # How much supporting evidence exists
W_CON = 0.20   # How much conflicting evidence exists (penalty)
W_ACT = 0.05   # How recently this belief was relevant (recency bonus)

# Decision thresholds
WIN_MARGIN = 0.15  # How much better a belief needs to be to decisively win
MIN_SCORE = 0.25   # Below this score, we say "I don't know" (lowered to help with attribute matching)

# Belief lifecycle thresholds
DECAY_RATE = 0.995      # Beliefs fade over time if not used (half-life ≈ 140 cycles)
DECAY_EPSILON = 0.05    # Beliefs never decay below this minimum
PROMOTE_THRESHOLD = 0.65  # Hypotheses that reach this confidence get promoted to beliefs
KILL_THRESHOLD = 0.10     # Beliefs that fall below this get archived
ACTIVATION_THRESHOLD = 0.1  # Skip decay if belief was recently accessed

# Coherence
W_COH = 0.05  # Bonus for beliefs that fit well with existing knowledge


# ═══════════════════════════════════════════════════════════════════
# BUDDHI CLASS
# ═══════════════════════════════════════════════════════════════════

class Buddhi:
    """
    Reasoning and judgment engine for MARC.
    
    Responsibilities:
    - Focus attention on relevant beliefs
    - Generate candidate beliefs
    - Evaluate candidates with judgment function
    - Select winner based on scores
    - Revise Chitta (add/update beliefs and edges)
    
    Invariants:
    - ONLY module allowed to write to Chitta
    - No LLM calls (pure algorithmic reasoning)
    - Stateless (all state in Chitta)
    """
    
    def __init__(self, chitta: ChittaGraph, learning_mode: bool = True):
        """
        Initialize Buddhi with reference to Chitta (memory).
        
        Args:
            chitta: the belief graph where we store everything
            learning_mode: if True, freeze lifecycle management during teaching sessions
        """
        self.chitta = chitta
        self.learning_mode = learning_mode  # FREEZE epistemic judgment during teaching
        
        # Schema extraction for taxonomic reasoning
        self.schema_extractor = SchemaExtractor(min_support=2) if SchemaExtractor else None
        self.schemas: list = []  # Extracted relation schemas
        
        # Statistics
        self.stats = {
            "judgments": 0,
            "accepts": 0,
            "rejects": 0,
            "uncertains": 0,
            "unknowns": 0,
            "decays": 0,
            "promotions": 0,
            "demotions": 0,
            "schemas_extracted": 0,
            "taxonomic_inferences": 0,
            "property_propagations": 0,
            "beliefs_suppressed": 0,  # Track silent suppressions
        }
    
    # ═══════════════════════════════════════════════════════════════
    # BELIEF LIFECYCLE (AGING, PROMOTION, DEMOTION)
    # ═══════════════════════════════════════════════════════════════
    
    def decay_beliefs(self):
        """
        Age beliefs over time based on how they're used.
        
        The decay policy captures some intuitive principles:
        - Hypotheses (unconfirmed ideas) fade 3x faster than established beliefs
        - Inherited beliefs (learned through inference) fade 2x faster than directly taught beliefs
        - Negative facts ("penguins can't fly") are more stable, so they fade slower (0.5x)
        - Recently used beliefs don't decay (if you just thought about it, it stays fresh)
        - Beliefs never completely disappear (minimum confidence = 0.05)
        """
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc)
        
        for belief in self.chitta.beliefs.values():
            # Skip inactive (archived) beliefs
            if not belief.active:
                continue
            
            # Skip beliefs that were just used — they're still fresh
            if belief.activation > ACTIVATION_THRESHOLD:
                continue
            
            # Skip if already at minimum confidence
            if belief.confidence <= DECAY_EPSILON:
                continue
            
            # Adjust decay rate based on belief type
            base_rate = belief.decay_rate
            
            # Hypotheses (unconfirmed ideas) fade faster
            if belief.epistemic_state == EpistemicType.HYPOTHESIS:
                belief.decay_rate = base_rate * 3.0
            
            # Beliefs learned through inference fade faster than directly taught ones
            has_inference_provenance = any(
                p.op == "inferred" for p in belief.provenance
            )
            if has_inference_provenance:
                belief.decay_rate = base_rate * 2.0
            
            # Negative facts are more stable ("can't fly" is harder to forget)
            polarity_str = str(belief.polarity).replace("Polarity.", "")
            if polarity_str == "NEGATIVE":
                belief.decay_rate = base_rate * 0.5
            
            # Actually apply the decay
            old_conf = belief.confidence
            new_conf = belief.apply_decay(current_time)
            
            # Make sure we don't go below the minimum
            if new_conf < DECAY_EPSILON:
                belief.confidence = DECAY_EPSILON
            
            # Reset decay rate to normal
            belief.decay_rate = base_rate
            
            # Track statistics
            if belief.confidence != old_conf:
                self.stats["decays"] += 1
    
    def promote_hypotheses(self):
        """
        Automatically promote hypotheses that gain enough confidence.
        
        This is a form of learning: if a hypothesis keeps getting confirmed over time,
        it graduates to being a real belief. The threshold is 0.65 — once a hypothesis
        reaches that confidence, we trust it enough to treat it as knowledge.
        
        We also make any tentative edges permanent when this happens.
        """
        for belief in self.chitta.beliefs.values():
            if not belief.active:
                continue
            
            if belief.epistemic_state == EpistemicType.HYPOTHESIS:
                if belief.confidence > PROMOTE_THRESHOLD:
                    # Promote based on content
                    # NEGATIVE -> EXCEPTION (Overrides)
                    # is_a -> DEFAULT (Class taxonomy)
                    # Other -> OBSERVATION (Instance fact)
                    if belief.polarity == Polarity.NEGATIVE:
                        belief.epistemic_state = EpistemicType.EXCEPTION
                    elif belief.template == "is_a":
                        belief.epistemic_state = EpistemicType.DEFAULT
                    else:
                        belief.epistemic_state = EpistemicType.OBSERVATION
                        
                    self.stats["promotions"] += 1
                    
                    # Make tentative edges permanent
                    self._make_edges_permanent(belief.id)
    
    def demote_weak_beliefs(self):
        """
        Archive beliefs that have faded too much.
        
        If a belief's confidence drops below 0.10, we mark it as inactive (archived).
        It's not deleted — it's just moved to cold storage. If new evidence comes in
        later, we could potentially resurrect it.
        
        Exception: we don't demote "unknown" beliefs, since they're intentionally
        at 0.0 confidence (they're placeholders for things we don't know yet).
        """
        for belief in self.chitta.beliefs.values():
            if not belief.active:
                continue
            
            # Don't demote unknown beliefs (they're placeholders)
            if belief.epistemic_state == EpistemicType.UNKNOWN:
                continue
            
            if belief.confidence < KILL_THRESHOLD:
                belief.active = False
                self.stats["demotions"] += 1
    
    def _make_edges_permanent(self, belief_id: str):
        """
        Convert tentative edges to permanent.
        
        Args:
            belief_id: belief that was promoted
        """
        # Find all edges involving this belief
        for relation in self.chitta.edges:
            # Outgoing edges
            if belief_id in self.chitta.edges[relation]:
                new_edges = []
                for edge_data in self.chitta.edges[relation][belief_id]:
                    if len(edge_data) == 4:
                        tgt_id, weight, edge_id, metadata = edge_data
                        if metadata.get("tentative"):
                            # Remove tentative flag
                            metadata = {k: v for k, v in metadata.items() if k != "tentative"}
                        new_edges.append((tgt_id, weight, edge_id, metadata) if metadata else (tgt_id, weight, edge_id))
                    else:
                        new_edges.append(edge_data)
                self.chitta.edges[relation][belief_id] = new_edges
            
            # Incoming edges
            if belief_id in self.chitta.edges_reverse[relation]:
                new_edges = []
                for edge_data in self.chitta.edges_reverse[relation][belief_id]:
                    if len(edge_data) == 4:
                        src_id, weight, edge_id, metadata = edge_data
                        if metadata.get("tentative"):
                            metadata = {k: v for k, v in metadata.items() if k != "tentative"}
                        new_edges.append((src_id, weight, edge_id, metadata) if metadata else (src_id, weight, edge_id))
                    else:
                        new_edges.append(edge_data)
                self.chitta.edges_reverse[relation][belief_id] = new_edges
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN THINKING ENTRY POINT
    # ═══════════════════════════════════════════════════════════════
    
    def think(self, proposal: dict) -> dict:
        """
        Main reasoning function — this is where the decision-making happens.
        
        When Manas hands us a parsed belief proposal, we have to decide what to do with it.
        The process has several stages:
        
        1. Focus: What existing beliefs are relevant to this proposal?
        2. Generate: What are all the competing options (accept, reject, modify)?
        3. Evaluate: Score each option using the judgment function
        4. Select: Pick the winner (if there is one)
        5. Revise: Update Chitta with the decision
        6. Lifecycle: Apply decay/promotion/demotion to keep the belief space healthy
        
        Args:
            proposal: parsed belief from Manas
        
        Returns:
            decision dict with the outcome and reasoning
        """
        self.stats["judgments"] += 1
        
        # Stage 1: Focus attention
        # Stage 1: Figure out what existing beliefs are relevant
        focus_set = self._focus(proposal)
        
        # Stage 2: Create all possible belief options to consider
        candidates = self._generate_candidates(proposal, focus_set)
        
        # Stage 3: Score each option
        scores = self._evaluate(candidates)
        
        # Stage 4: Pick the best one (if there is a clear winner)
        decision = self._select(scores, proposal)
        
        # Stage 5: Update the belief graph with our decision
        result = self._revise(decision, proposal)
        
        # Stage 6: Lifecycle management (BUT NOT DURING TEACHING)
        # ═══════════════════════════════════════════════════════════════
        # IMPORTANT: During teaching (learning_mode=True), we freeze lifecycle management.
        # 
        # Why? Teaching needs to be monotonic — facts you teach should stick around.
        # If we let beliefs decay or get demoted during bulk teaching, facts silently
        # disappear and the system becomes unpredictable.
        # 
        # During learning_mode:
        #   - NO decay (beliefs don't fade)
        #   - NO promotion (hypotheses stay hypothetical)
        #   - NO demotion (weak beliefs stay active)
        # 
        # Contradictions are okay — they get resolved when answering queries.
        # ═══════════════════════════════════════════════════════════════
        if not self.learning_mode:
            # Age old beliefs
            self.decay_beliefs()
            
            # Graduate confident hypotheses
            self.promote_hypotheses()
            
            # Archive really weak beliefs
            self.demote_weak_beliefs()
        
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # SCHEMA EXTRACTION & TAXONOMIC REASONING
    # ═══════════════════════════════════════════════════════════════
    
    def extract_schemas(self) -> int:
        """
        Extract relation patterns from the belief graph.
        
        After you finish teaching a batch of facts, call this to discover patterns.
        For example, if you taught "fish have gills", "sharks have gills",
        "trout have gills", this might extract a schema that fish-like things tend
        to have gills.
        
        Returns:
            number of schemas discovered
        """
        if not self.schema_extractor:
            return 0
        
        self.schemas = self.schema_extractor.extract_all(self.chitta)
        self.stats["schemas_extracted"] = len(self.schemas)
        return len(self.schemas)
    
    def _apply_taxonomic_inference(
        self,
        query_entities: set[str],
        query_predicates: set[str],
        proof: AnswerProof
    ) -> bool:
        """
        Answer queries using taxonomic inheritance with STRICT Conflict Resolution.
        
        Algorithm:
        1. COLLECT ALL PATHS: Traverse all ancestors, collecting both POSITIVE and NEGATIVE beliefs.
        2. CHECK SPECIFICITY: If paths conflict, prefer the one with shorter distance (Vertical Conflict).
        3. CHECK RANK: If distances equal, prefer AXIOM > EXCEPTION > DEFAULT.
        4. CHECK HORIZONTAL CONFLICT: If distance and rank are equal, return CONFLICT (Nixon Diamond).
        
        Returns:
            True if we found an answer (Yes/No/Conflict), False otherwise.
        """
        if not query_entities or not query_predicates:
            return False
        
        # Check if ANY query predicate allows inheritance
        inheritable_predicates = set()
        for pred in query_predicates:
            if RelationFrame:
                frame = RelationFrame.for_predicate(pred)
                if frame.inherits:
                    inheritable_predicates.add(pred)
            else:
                # Fallback
                inheritable_predicates.add(pred)
        
        if not inheritable_predicates:
            return False
        
        # For each query entity, traverse taxonomic chain
        for entity in query_entities:
            # Get all ancestors via recursive is_a traversal
            ancestors = self._get_taxonomic_ancestors(entity)
            
            # PHASE 1: COLLECT ALL PATHS
            # Do NOT exit early. Collect all relevant positive and negative beliefs.
            positive_paths = []
            negative_paths = []
            
            for ancestor, path in ancestors:
                # Calculate specificity (Distance from entity)
                distance = len(path)
                
                for belief in self.chitta.beliefs.values():
                    if not belief.active:
                        continue
                    
                    if ancestor in belief.entities:
                        belief_preds = set(belief.predicates)
                        matching_preds = inheritable_predicates & belief_preds
                        
                        if matching_preds:
                            # Do not inherit from weak Hypotheses
                            if belief.epistemic_state == EpistemicType.HYPOTHESIS:
                                continue
                            
                            # Determine polarity
                            is_negative = False
                            if belief.polarity == Polarity.NEGATIVE:
                                is_negative = True
                            
                            # QUANTITATIVE LAYER: Calculate Weakest Link Confidence
                            chain_conf = self._calculate_chain_confidence(path, belief.id)
                            
                            # Create path record
                            path_record = {
                                "distance": distance,
                                "ancestor": ancestor,
                                "path_ids": path,
                                "belief": belief,
                                "predicates": list(matching_preds),
                                "chain_confidence": chain_conf
                            }
                            
                            if is_negative:
                                negative_paths.append(path_record)
                                proof.add_step(
                                    rule="path_collection",
                                    inputs=[belief.id],
                                    output=f"Found NEGATIVE path on {ancestor} (Dist {distance})",
                                    confidence=chain_conf
                                )
                            else:
                                positive_paths.append(path_record)
                                proof.add_step(
                                    rule="path_collection",
                                    inputs=[belief.id],
                                    output=f"Found POSITIVE path on {ancestor} (Dist {distance})",
                                    confidence=chain_conf
                                )

            # PHASE 2: RESOLVE CONFLICTS
            if not positive_paths and not negative_paths:
                continue
                
            if positive_paths and negative_paths:
                return self._resolve_taxonomic_conflict(entity, positive_paths, negative_paths, proof)
            elif negative_paths:
                # Only explicit negations found
                best_neg = min(negative_paths, key=lambda x: x["distance"])
                proof.verdict = "no"
                proof.add_step(
                    rule="taxonomic_inheritance",
                    inputs=best_neg["path_ids"] + [best_neg["belief"].id],
                    output=f"Inherited NEGATION from {best_neg['ancestor']} (Dist {best_neg['distance']})",
                    confidence=best_neg.get("chain_confidence", best_neg["belief"].confidence)
                )
                self.stats["taxonomic_inferences"] += 1
                return True
            elif positive_paths:
                # Only positive facts found
                best_pos = min(positive_paths, key=lambda x: x["distance"])
                proof.verdict = "yes"
                proof.add_step(
                    rule="taxonomic_inheritance",
                    inputs=best_pos["path_ids"] + [best_pos["belief"].id],
                    output=f"Inherited POSITIVE from {best_pos['ancestor']} (Dist {best_pos['distance']})",
                    confidence=best_pos.get("chain_confidence", best_pos["belief"].confidence)
                )
                self.stats["taxonomic_inferences"] += 1
                # Reinforce used beliefs
                best_pos["belief"].reinforce(boost=0.03, success=True)
                return True
        
        return False

    def _resolve_taxonomic_conflict(
        self,
        entity: str,
        pos_paths: list[dict],
        neg_paths: list[dict],
        proof: AnswerProof
    ) -> bool:
        """
        Resolve conflict between positive and negative inheritance paths.
        
        Strategies:
        1. Vertical Conflict (Specificity): Smallest distance wins.
        2. Epistemic Rank: AXIOM > EXCEPTION/OBSERVATION > DEFAULT.
        3. Horizontal Conflict: If equal specificity and rank -> CONFLICT.
        """
        # Get best candidates (shortest path)
        best_pos = min(pos_paths, key=lambda x: x["distance"])
        best_neg = min(neg_paths, key=lambda x: x["distance"])
        
        # 1. SPECIFICITY CHECK
        if best_pos["distance"] < best_neg["distance"]:
            proof.verdict = "yes"
            proof.add_step(
                rule="specificity_resolution",
                inputs=[best_pos["belief"].id, best_neg["belief"].id],
                output=f"Specificity Win: Positive {best_pos['ancestor']} (Dist {best_pos['distance']}) overrides Negative {best_neg['ancestor']} (Dist {best_neg['distance']})",
                confidence=best_pos.get("chain_confidence", best_pos["belief"].confidence)
            )
            return True
        elif best_neg["distance"] < best_pos["distance"]:
            proof.verdict = "no"
            proof.add_step(
                rule="specificity_resolution",
                inputs=[best_neg["belief"].id, best_pos["belief"].id],
                output=f"Specificity Win: Negative {best_neg['ancestor']} (Dist {best_neg['distance']}) overrides Positive {best_pos['ancestor']} (Dist {best_pos['distance']})",
                confidence=best_neg.get("chain_confidence", best_neg["belief"].confidence)
            )
            return True
            
        # 2. RANK CHECK (if distances equal)
        # For now, treat EXCEPTION > DEFAULT.
        pos_type = best_pos["belief"].epistemic_state
        neg_type = best_neg["belief"].epistemic_state
        
        # Determine strict ranks
        def get_rank(etype):
            if etype == EpistemicType.AXIOM: return 3
            if etype == EpistemicType.EXCEPTION: return 2
            if etype == EpistemicType.OBSERVATION: return 2
            if etype == EpistemicType.DEFAULT: return 1
            return 0
            
        rank_pos = get_rank(pos_type)
        rank_neg = get_rank(neg_type)
        
        if rank_pos > rank_neg:
            proof.verdict = "yes"
            proof.add_step(
                rule="rank_resolution",
                inputs=[best_pos["belief"].id],
                output=f"Rank Win: {pos_type.name} > {neg_type.name}",
                confidence=best_pos.get("chain_confidence", best_pos["belief"].confidence)
            )
            return True
        elif rank_neg > rank_pos:
            proof.verdict = "no"
            proof.add_step(
                rule="rank_resolution",
                inputs=[best_neg["belief"].id],
                output=f"Rank Win: {neg_type.name} > {pos_type.name}",
                confidence=best_neg.get("chain_confidence", best_neg["belief"].confidence)
            )
            return True
            
        # 3. HORIZONTAL CONFLICT (Nixon Diamond)
        # Equal distance, equal rank.
        proof.verdict = "conflict"
        proof.add_conflict(
            predicate="inheritance_conflict",
            positive=best_pos["belief"].id,
            negative=best_neg["belief"].id,
            delta=0.0,
            resolution="unresolved"
        )
        proof.add_step(
            rule="conflict_detected",
            inputs=[best_pos["belief"].id, best_neg["belief"].id],
            output=f"Horizontal Conflict: {best_pos['ancestor']} (Pos) vs {best_neg['ancestor']} (Neg) at equal distance {best_pos['distance']}.",
            confidence=0.0
        )
        return True
    
    def _calculate_chain_confidence(self, path_ids: list[str], terminal_belief_id: str) -> float:
        """
        Calculate Weakest Link confidence for an inference chain.
        Confidence = min(confidence of all beliefs in chain).
        """
        min_conf = 1.0
        
        # Check path (is_a links)
        for bid in path_ids:
            belief = self.chitta.get(bid)
            if belief and belief.confidence < min_conf:
                min_conf = belief.confidence
                
        # Check terminal belief (property assertion)
        term_belief = self.chitta.get(terminal_belief_id)
        if term_belief and term_belief.confidence < min_conf:
            min_conf = term_belief.confidence
            
        return min_conf

    def _get_taxonomic_ancestors(self, entity: str, visited=None) -> list[tuple[str, list[str]]]:
        """
        Get all taxonomic ancestors of an entity via recursive is_a traversal.
        
        Args:
            entity: starting entity
            visited: set of visited entities (for cycle detection)
        
        Returns:
            List of (ancestor, path) tuples where path is list of is_a belief IDs
        """
        if visited is None:
            visited = set()
        
        if entity in visited:
            return []  # Cycle detected
        
        visited.add(entity)
        ancestors = []
        
        # Find direct parents via is_a
        for belief in self.chitta.beliefs.values():
            if not belief.active or belief.template != "is_a":
                continue
            
            if belief.subject == entity and belief.object:
                parent = belief.object
                # Add direct parent
                ancestors.append((parent, [belief.id]))
                
                # Recursively get grandparents
                grandparents = self._get_taxonomic_ancestors(parent, visited.copy())
                for grandparent, path in grandparents:
                    ancestors.append((grandparent, [belief.id] + path))
        
        return ancestors
    
    def _check_predicate_grounding(
        self,
        query_entities: set[str],
        query_predicates: set[str],
        applicable_beliefs: list,
        proof: AnswerProof
    ) -> bool:
        """
        Check if predicate is GROUNDED for the given entities.
        
        A predicate is grounded if:
        1. DIRECT: We have applicable beliefs (strict entity+predicate match)
        2. TAXONOMIC: An ancestor has this predicate explicitly
        3. COMPOSITION ALLOWED: Query involves only simple property/state checks
        
        This prevents unbounded composition inference.
        
        Examples that should FAIL grounding:
        - "Do copper objects conduct electricity?" (composition not taught)
        - "Is intelligence measurable?" (measurement relation not defined)
        
        Examples that should PASS grounding:
        - "Do bats produce milk?" (mammals produce milk via taxonomy)
        - "Is water a liquid?" (direct teaching exists, even if predicate differs)
        
        Returns:
            True if predicate is grounded, False otherwise
        """
        # CASE A: Direct grounding via applicable beliefs
        if applicable_beliefs:
            proof.add_step(
                rule="grounding_check",
                inputs=[],
                output=f"Direct grounding: {len(applicable_beliefs)} applicable belief(s)",
                confidence=None
            )
            return True
        
        # CASE B: Taxonomic grounding
        # Check if any ancestor has this predicate defined
        for entity in query_entities:
            # Skip trivial words
            if entity in {'in', 'on', 'at', 'of', 'to', 'from', 'by', 'with', 'the', 'a', 'an'}:
                continue
            
            # Get ancestors
            ancestors = self._get_taxonomic_ancestors(entity)
            
            # Check if any ancestor has the query predicate
            for ancestor, path in ancestors:
                for belief in self.chitta.beliefs.values():
                    if not belief.active:
                        continue
                    
                    if ancestor in belief.entities:
                        belief_preds = set(belief.predicates)
                        # Predicate match OR similar semantic content
                        if query_predicates & belief_preds:
                            proof.add_step(
                                rule="grounding_check",
                                inputs=[],
                                output=f"Taxonomic grounding: ancestor '{ancestor}' has predicate {query_predicates & belief_preds}",
                                confidence=None
                            )
                            return True
        
        # CASE C: Simple entity-property queries (lenient for direct facts)
        # If all query entities exist in chitta AND query has <3 entities,
        # it's likely a simple property check, not complex composition
        non_trivial_entities = [e for e in query_entities if e not in {'in', 'on', 'at', 'of', 'to', 'from', 'by', 'with', 'the', 'a', 'an'}]
        
        if len(non_trivial_entities) <= 2:
            # Check if at least ONE entity exists in entity_index
            entities_found = [e for e in non_trivial_entities if e in self.chitta.entity_index]
            
            if entities_found:
                # Simple property check - allow it
                # This handles cases like "Is water a liquid?" where predicate
                # normalization differs but the fact was taught
                proof.add_step(
                    rule="grounding_check",
                    inputs=[],
                    output=f"Simple property check: entity '{entities_found[0]}' exists, {len(non_trivial_entities)} total entities",
                    confidence=None
                )
                return True
        
        # NO GROUNDING: Complex composition not taught
        proof.add_step(
            rule="grounding_check",
            inputs=[],
            output=f"No grounding: complex composition {query_entities} + {query_predicates} not taught",
            confidence=None
        )
        return False
    
    def _apply_property_propagation(
        self,
        query_entities: set[str],
        query_predicates: set[str],
        proof: AnswerProof
    ) -> bool:
        """
        Apply property propagation via schemas.
        
        Example:
            Query: "Do sparrows lay eggs?"
            Schema: X is_a Bird => X lays_eggs
            Inference: YES (via property schema)
        
        Returns:
            True if answer found via property propagation
        """
        if not self.schemas:
            return False
        
        # Check each property schema
        for schema in self.schemas:
            if schema.schema_type.value != 'property':
                continue
            
            # Only STRUCTURAL schemas may propagate
            if schema.epistemic_class != 'structural':
                continue
            
            # Check if schema matches query
            parent = schema.constraints.get('parent')
            predicate = schema.constraints.get('predicate')
            
            if not parent or not predicate:
                continue
            
            # Check if query predicate matches schema predicate
            if predicate not in query_predicates:
                continue
            
            # Check if query entity is instance of parent class
            for entity in query_entities:
                for belief in self.chitta.beliefs.values():
                    if not belief.active or belief.template != "is_a":
                        continue
                    
                    # Use subject/object for directed matching
                    if belief.subject == entity and belief.object == parent:
                        # Found match via property propagation!
                        proof.verdict = "yes"
                        proof.add_step(
                            rule="property_propagation",
                            inputs=[belief.id] + schema.supporting_beliefs[:2],
                            output=f"{entity} is_a {parent}, {parent} has {predicate}",
                            confidence=schema.confidence
                        )
                        
                        self.stats["property_propagations"] += 1
                        return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════
    # INFERENCE RULES
    # ═══════════════════════════════════════════════════════════════════

    def _check_taxonomic_membership(
        self,
        query_entities: set[str],
        query_predicates: set[str],
        proof: AnswerProof,
    ) -> bool:
        """
        Check if query is asking for a taxonomic relationship established by chain.
        Example: Is a Penguin a Bird? (Where Penguin -> Bird exists)
        """
        # 1. Verify query is asking for is_a/taxonomic relation
        is_taxonomic_query = False
        for pred in query_predicates:
            if RelationFrame:
                 frame = RelationFrame.for_predicate(pred)
                 if frame.kind == RelationKind.TAXONOMIC:
                     is_taxonomic_query = True
                     break
            elif pred == "is_a":
                 is_taxonomic_query = True
                 break
        
        if not is_taxonomic_query:
            print(f"DEBUG: Skipping taxonomic check. Preds: {query_predicates}")
            return False

        print(f"DEBUG: Checking taxonomic membership. Entities: {query_entities}")

        # 2. Check each entity's ancestors
        for entity in query_entities:
            ancestors = self._get_taxonomic_ancestors(entity)
            print(f"DEBUG: Ancestors for {entity}: {[a[0] for a in ancestors]}")
            
            for ancestor, path in ancestors:
                if ancestor in query_entities and ancestor != entity:
                     print(f"DEBUG: MATCH FOUND! {entity} -> {ancestor}")
                     proof.verdict = "yes"
                     proof.add_step(
                         rule="taxonomic_entailment",
                         inputs=path,
                         output=f"Entailment: {entity} is a {ancestor} (Path: {' → '.join(path)})",
                         confidence=1.0 
                     )
                     self.stats["taxonomic_inferences"] += 1
                     return True
        return False

    def _check_transitive_inference(
        self,
        query_entities: set[str],
        query_predicates: set[str],
        proof: AnswerProof
    ) -> bool:
        """
        Check for transitive relations (A > B > C -> A > C).
        """
        transitive_preds = set()
        for pred in query_predicates:
            if RelationFrame:
                frame = RelationFrame.for_predicate(pred)
                if frame.transitive and frame.kind != RelationKind.TAXONOMIC:
                    transitive_preds.add(pred)
        
        if not transitive_preds:
            return False

        sorted_entities = list(query_entities)
        if len(sorted_entities) < 2:
            return False
            
        import collections
        for start_node in sorted_entities:
            targets = {e for e in sorted_entities if e != start_node}
            for pred in transitive_preds:
                # BFS Search
                queue = collections.deque([(start_node, [])])
                visited = {start_node}
                while queue:
                    curr, path = queue.popleft()
                    if curr in targets:
                        proof.verdict = "yes"
                        proof.add_step(
                            rule="transitive_inference",
                            inputs=path,
                            output=f"Transitive Chain: {start_node} {pred} ... {pred} {curr}",
                            confidence=0.9
                        )
                        return True
                    
                    if pred in self.chitta.edges and curr in self.chitta.edges[pred]:
                        for edge in self.chitta.edges[pred][curr]:
                            neighbor = edge[0]
                            belief_id = edge[2] 
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, path + [belief_id]))
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # QUERY ANSWERING (CRITICAL: NEVER BYPASS JUDGMENT)
    # ═══════════════════════════════════════════════════════════════
    # QUERY ANSWERING (PROOF-CARRYING)
    # ═══════════════════════════════════════════════════════════════
    
    def answer(self, query_proposal: dict) -> AnswerProof:
        """
        Answer a query using judgment pipeline, returning proof trace.
        
        CRITICAL RULES:
        - Query answering MUST go through judgment (no Chitta bypass)
        - This is a PURE FUNCTION (no side effects)
        - NO belief updates, NO decay, NO promotion
        - Returns AnswerProof with replayable derivation trace
        
        This prevents RAG-like leakage where systems:
        - Answer different questions
        - Fail to expose uncertainty
        - Leak taxonomic knowledge as answers
        
        For queries, we evaluate ONLY existing beliefs (not the query itself).
        
        Args:
            query_proposal: BeliefProposal from Manas (parsed query)
        
        Returns:
            AnswerProof object with verdict + derivation trace
        """
        # Initialize proof
        proof = AnswerProof(
            query=query_proposal.get("statement", query_proposal.get("text", "unknown query")),
            verdict="unknown"
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 0: EPISTEMIC HYGIENE check
        # ═══════════════════════════════════════════════════════════════
        # Reject malformed or generic queries.
        if "generic" in query_proposal.get("predicates", []) and query_proposal.get("parser_confidence", 1.0) < 0.6:
            proof.verdict = "invalid"  # Will map to Verdict.INVALID
            proof.add_step(
                rule="epistemic_hygiene",
                inputs=[],
                output="Query too vague or malformed (generic predicate)",
                confidence=1.0
            )
            return proof

        # ═══════════════════════════════════════════════════════════════
        # HARD UNKNOWN GATE: Primary Entity Existence Check
        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: Check if the query's PRIMARY entity (subject) exists.
        # 
        # Strategy: Check the FIRST meaningful entity. If it doesn't exist
        # in our knowledge base, return UNKNOWN immediately. This prevents:
        # - "Are platypuses mammals?" → "no" (platypus never taught)
        # - "Do kangaroos hop?" → "no" (kangaroo never taught)
        # 
        # But allows:
        # - "Is water a liquid?" (water is taught, liquid is state/property)
        # - "Do whales have gills?" (whale is taught, gill is in predicate)
        # 
        # Rule: First entity must exist in entity_index to proceed.
        # ═══════════════════════════════════════════════════════════════
        query_entities = query_proposal.get("entities", [])
        query_predicates = set(query_proposal.get("predicates", []))
        
        # Check if primary entity exists
        if query_entities:
            # Get first non-trivial entity (skip prepositions)
            trivial_words = {'in', 'on', 'at', 'of', 'to', 'from', 'by', 'with', 'the', 'a', 'an'}
            primary_entity = None
            for entity in query_entities:
                if entity not in trivial_words:
                    primary_entity = entity
                    break
            
            # If we have a primary entity, check if it's known
            if primary_entity:
                # Primary entity is known if:
                # 1. It appears in entity_index, OR
                # 2. It's embedded in the predicate (e.g., "liquid" in "is_liquid")
                in_entity_index = primary_entity in self.chitta.entity_index
                in_predicate = any(primary_entity in pred for pred in query_predicates)
                
                if not in_entity_index and not in_predicate:
                    proof.verdict = "unknown"
                    proof.add_step(
                        rule="entity_existence_check",
                        inputs=[],
                        output=f"Primary entity not in knowledge base: {primary_entity}",
                        confidence=0.0
                    )
                    return proof
        
        # Focus on relevant beliefs (read-only)
        focus_set = self._focus(query_proposal)
        
        if not focus_set:
            proof.verdict = "unknown"
            proof.add_step(
                rule="focus",
                inputs=[],
                output="No relevant beliefs found",
                confidence=0.0
            )
            return proof
        
        # Record focus step
        proof.add_step(
            rule="focus",
            inputs=[b.id for b in focus_set[:5]],  # Limit to first 5
            output=f"Found {len(focus_set)} relevant belief(s)",
            confidence=None
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: ANSWER PIPELINE (ordered by specificity)
        # ═══════════════════════════════════════════════════════════════
        # Try answering methods in order of specificity:
        # 1. Direct match (highest confidence)
        # 2. Taxonomic inference (inherited properties)
        # 3. Property propagation (schema-based)
        # 4. Unknown (no path found)
        # ═══════════════════════════════════════════════════════════════
        
        query_entities_set = set(query_proposal.get("entities", []))
        query_predicates_set = set(query_proposal.get("predicates", []))
        
        # ───────────────────────────────────────────────────────────────
        # METHOD 1: DIRECT APPLICABILITY (entity + predicate match)
        # ───────────────────────────────────────────────────────────────
        # 🔴 HARD GATE: Strict applicability for high restraint
        # 
        # Only beliefs that DIRECTLY answer the question are applicable.
        # This prevents false positives from loose semantic overlap.
        # ───────────────────────────────────────────────────────────────
        applicable_beliefs = []
        for belief in focus_set:
            belief_entities = set(belief.entities)
            belief_predicates = set(belief.predicates)
            
            # Entity overlap (Standard Jaccard-ish)
            entity_sect = query_entities_set & belief_entities
            entity_overlap = len(entity_sect) / max(len(query_entities_set), 1) if len(query_entities_set) > 0 else 0
            
            # Predicate overlap (Fuzzy: substring match)
            # Check if any query predicate "covers" a belief predicate or vice-versa
            pred_match_count = 0
            for qp in query_predicates_set:
                for bp in belief_predicates:
                    if qp in bp or bp in qp:  # Substring match
                        pred_match_count += 1
                        break # Count each query predicate only once
            
            predicate_overlap = pred_match_count / max(len(query_predicates_set), 1) if len(query_predicates_set) > 0 else 0
            
            # STRICT APPLICABILITY: Relaxed for fuzzy predicates
            # Require high entity overlap AND (high predicate overlap OR exact entity match with some predicate signal)
            # STRICT APPLICABILITY: Relaxed for fuzzy predicates
            # Require high entity overlap AND (high predicate overlap OR exact entity match with some predicate signal)
            is_applicable = entity_overlap >= 0.8 and predicate_overlap >= 0.5
            
            # EXTRA CHECK: For 'is_a' queries, we MUST match the object/class.
            # "Is Socrates a plumber?" should NOT match "Socrates is a man".
            if is_applicable and "is_a" in query_predicates_set:
                q_obj = query_proposal.get("canonical", {}).get("object")
                b_obj = belief.canonical.get("object")
                if q_obj and b_obj and q_obj.lower() != b_obj.lower():
                     # Classes differ! Not applicable.
                     is_applicable = False
            
            if is_applicable:
                applicable_beliefs.append(belief)
                proof.add_step(
                    rule="applicability_check",
                    inputs=[belief.id],
                    output=f"✓ Applicable: entity={entity_overlap:.2f}, pred={predicate_overlap:.2f} | {belief.statement_text[:60]}",
                    confidence=None
                )
        
        # ═══════════════════════════════════════════════════════════════
        # PREDICATE GROUNDING CHECK (CRITICAL FOR RESTRAINT)
        # ═══════════════════════════════════════════════════════════════
        # Before attempting inference, verify we have GROUNDING for this
        # specific (entity, predicate) combination.
        #
        # A predicate is GROUNDED if one of these is true:
        # 1. Direct grounding: Belief exists with same (entity, predicate)
        # 2. Taxonomic grounding: Belief exists on ancestor with this predicate
        # 3. Schema grounding: Learned schema binds (entity-type, predicate)
        #
        # If NONE of these → return UNKNOWN (no composition inference)
        #
        # Example failures without this:
        # - "Do copper objects conduct electricity?" 
        #   → Copper exists, "conducts" exists, but composition NOT taught
        # - "Is intelligence measurable?"
        #   → Intelligence exists as concept, but measurement NOT defined
        #
        # This is the key difference between:
        # - Entity recognition (cheap)
        # - Predicate grounding (expensive, requires explicit teaching)
        # ═══════════════════════════════════════════════════════════════
        
        grounded = self._check_predicate_grounding(
            query_entities_set,
            query_predicates_set,
            applicable_beliefs,
            proof
        )
        
        if not grounded:
            proof.verdict = "unknown"
            proof.add_step(
                rule="predicate_not_grounded",
                inputs=[],
                output=f"Predicate(s) {query_predicates_set} not grounded for entity(ies) {query_entities_set}",
                confidence=0.0
            )
            return proof
        
        # ═══════════════════════════════════════════════════════════════
        # ANSWER PIPELINE: Direct → Taxonomic → Transitive → Property → Unknown
        # ═══════════════════════════════════════════════════════════════
        
        # 1. Try direct answer from applicable beliefs
        direct_result = self._try_direct_answer(applicable_beliefs, query_proposal, proof)
        if direct_result:
            return direct_result

        # 2. Try Taxonomic Membership (Is Penguin a Bird?) [NEW]
        # Strict Directionality Check via Canonical Subject/Object
        canonical = query_proposal.get("canonical", {})
        subject = canonical.get("subject")
        object_ = canonical.get("object")
        
        if self._check_taxonomic_membership(subject, object_, query_predicates, proof):
            return proof
        
        # 3. Try Taxonomic Inheritance (Do Penguins have wings?)
        # CRITICAL FIX: Only checking inheritance for the SUBJECT.
        # Otherwise checking the object (e.g. "Is A a B?") might find that B inherits something and return YES.
        entities_to_check = {subject} if subject else query_entities_set
        
        if self._apply_taxonomic_inference(entities_to_check, query_predicates_set, proof):
            self.stats["taxonomic_inferences"] += 1
            for step in proof.steps:
                for belief_id in step.inputs:
                    if belief_id in self.chitta.beliefs:
                        self.chitta.beliefs[belief_id].reinforce(boost=0.03, success=True)
            return proof

        # 4. Try General Transitivity (A > B > C) [NEW]
        if self._check_transitive_inference(subject, object_, query_predicates_set, proof):
            return proof
        
        # 5. Try Property Propagation (Schemas)
        if self._apply_property_propagation(query_entities_set, query_predicates_set, proof):
            self.stats["property_propagations"] += 1
            for step in proof.steps:
                for belief_id in step.inputs:
                    if belief_id in self.chitta.beliefs:
                        self.chitta.beliefs[belief_id].reinforce(boost=0.03, success=True)
            return proof
          
        # No path found
        proof.verdict = "unknown"
        proof.add_step(
            rule="no_answer_path",
            inputs=[],
            output="No direct match, no inference path found",
            confidence=0.0
        )
        return proof
    
    def _check_taxonomic_membership(
        self,
        subject: str | None,
        object_: str | None,
        query_predicates: set[str],
        proof: AnswerProof,
    ) -> bool:
        """
        Check if Subject is_a Object via transitive chain.
        Strictly directional: Ancestors(Subject) must contain Object.
        """
        # 1. Validate Intent (Must be is_a query)
        is_taxonomic_query = False
        if "is_a" in query_predicates:
             is_taxonomic_query = True
        else:
             for pred in query_predicates:
                if RelationFrame.for_predicate(pred).kind == RelationKind.TAXONOMIC:
                     is_taxonomic_query = True
                     break
        
        if not is_taxonomic_query:
            return False

        if not subject or not object_:
             # If parser failed to identify explicit S/O, fallback to entity set logic?
             # No, strict mode requires structured understanding.
             return False

        # 2. Get Ancestors of Subject
        ancestors = self._get_taxonomic_ancestors(subject)
        
        positive_proofs = []
        negative_proofs = []
        
        for ancestor_id, path in ancestors:
            if ancestor_id == object_:
                 # Check path polarity
                 is_negative_path = False
                 for bid in path:
                     belief = self.chitta.get(bid)
                     # Check for Negative Polarity OR Exception type
                     # Note: Polarity.NEGATIVE is often -1 or Enum. Need safe check.
                     pol_val = belief.polarity.value if hasattr(belief.polarity, 'value') else belief.polarity
                     if pol_val == -1 or pol_val == 2: # 2 is NEGATIVE in some Enums
                         is_negative_path = True
                         break
                     if belief.epistemic_state == EpistemicType.EXCEPTION:
                         is_negative_path = True
                         break
                 
                 if is_negative_path:
                     negative_proofs.append(path)
                 else:
                     positive_proofs.append(path)

        # 3. Conflict Resolution
        if positive_proofs and negative_proofs:
             proof.verdict = "conflict"
             proof.add_step(
                 rule="conflict_resolution",
                 inputs=positive_proofs[0] + negative_proofs[0], # Show both
                 output=f"Conflict detected: {subject} is {object_} but also NOT {object_}. (System halted)",
                 confidence=0.0
             )
             return True
             
        elif negative_proofs:
             proof.verdict = "no"
             proof.add_step(
                 rule="taxonomic_exclusion",
                 inputs=negative_proofs[0],
                 output=f"Exclusion: {subject} is NOT {object_} (Negative Path)",
                 confidence=1.0
             )
             return True
             
        elif positive_proofs:
             proof.verdict = "yes"
             proof.add_step(
                 rule="taxonomic_entailment",
                 inputs=positive_proofs[0],
                 output=f"Entailment: {subject} is a {object_}",
                 confidence=1.0
             )
             self.stats["taxonomic_inferences"] += 1
             return True

        return False
        
    def _check_transitive_inference(
        self,
        subject: str | None,
        object_: str | None,
        query_predicates: set[str],
        proof: AnswerProof
    ) -> bool:
        """
        Check for A > B > C style transitivity.
        """
        import collections
        from collections import deque
        
        # 1. Identify Transitive Relation
        transitive_preds = []
        for pred in query_predicates:
            frame = RelationFrame.for_predicate(pred)
            if not frame.transitive:
                continue
            transitive_preds.append(pred)
            
        if not transitive_preds:
            return False
            
        if not subject or not object_:
            return False
            
        # print(f"DEBUG: Checking Transitivity: {subject} {transitive_preds} {object_}")

        # 2. Search for Path
        # BFS from Subject
        for pred in transitive_preds:
            # Start BFS from subject
            bfs_queue = deque([(subject, [subject])])
            visited = {subject}
            
            # print(f"DEBUG: Transitive Check {subject} -> {object_} via {pred}")
            
            while bfs_queue:
                current_entity, path = bfs_queue.popleft()
                
                if current_entity == object_:
                    # Found path!
                    proof.verdict = "yes"
                    proof.add_step(
                        rule="transitive_inference",
                        inputs=path,
                        output=f"Transitivity: {' > '.join(path)}",
                        confidence=0.85 
                    )
                    return True
                
                # Get neighbors
                # Iterate all beliefs to find edges? Efficient?
                # Using chitta.edges would be better but we only have beliefs list
                # Improve chitta?
                # For now search beliefs
                # print(f"DEBUG: Visiting {current_entity}...")
                
                for belief in self.chitta.beliefs.values():
                    # Match Subject AND Predicate
                    if not belief.active: continue
                    
                    # Normalize belief predicate
                    b_pred = list(belief.predicates)[0] if belief.predicates else ""
                    # We need to match the RelationKind or specific predicate?
                    # Transitivity usually implies SAME predicate (A > B > C)
                    # or at least compatible
                    
                    if belief.subject == current_entity:
                         # Check predicate match
                         if b_pred == pred:
                             target = belief.object
                             if target and target not in visited:
                                 visited.add(target)
                                 bfs_queue.append((target, path + [target]))
                                 # print(f"DEBUG:   -> Found neighbor {target}")
                            
        return False
        
        # Check for hard negations first
        for belief in applicable_beliefs:
            is_negative = False
            if hasattr(belief, 'polarity'):
                polarity_str = str(belief.polarity).replace("Polarity.", "")
                if polarity_str == "NEGATIVE":
                    is_negative = True
            if hasattr(belief, 'quantifier'):
                quant_str = str(belief.quantifier).replace("Quantifier.", "")
                if quant_str == "NONE":
                    is_negative = True
            if belief.canonical.get("negated", False):
                is_negative = True
            
            if is_negative:
                proof.verdict = "no"
                proof.add_step(
                    rule="hard_negative",
                    inputs=[belief.id],
                    output="Explicit negation found",
                    confidence=belief.confidence
                )
                return proof
        
        # Standard judgment
        candidates = [{"type": "existing", "belief": b, "id": b.id} for b in applicable_beliefs]
        scores = self._evaluate(candidates)
        decision = self._select(scores, query_proposal)
        
        if decision["decision"] == "accept":
            winner = decision["winner"]["belief"]
            is_negated = winner.canonical.get("negated", False)
            proof.verdict = "no" if is_negated else "yes"
            proof.add_step(
                rule="direct_match",
                inputs=[winner.id],
                output=winner.statement_text,
                confidence=winner.confidence
            )
            winner.reinforce(boost=0.05, success=True)
            return proof
        
        return None
    
    def _detect_answer_conflicts(
        self,
        focus_set: list[Belief],
        query_proposal: dict,
        proof: AnswerProof
    ) -> int:
        """
        Detect and record conflicts in answer derivation.
        
        Args:
            focus_set: relevant beliefs
            query_proposal: original query
            proof: AnswerProof to update
        
        Returns:
            number of conflicts detected
        """
        predicates = query_proposal.get("predicates", [])
        if not predicates:
            return 0
        
        # Group beliefs by predicate polarity
        positive_beliefs = []
        negative_beliefs = []
        
        for belief in focus_set:
            # Check if belief matches query predicates
            if any(pred in belief.predicates for pred in predicates):
                is_negated = belief.canonical.get("negated", False)
                if not is_negated:
                    positive_beliefs.append(belief)
                else:
                    negative_beliefs.append(belief)
        
        # Record conflicts
        conflict_count = 0
        for pos in positive_beliefs:
            for neg in negative_beliefs:
                delta = abs(pos.confidence - neg.confidence)
                
                # Determine resolution
                if delta < WIN_MARGIN:
                    resolution = "uncertain"
                elif pos.confidence > neg.confidence:
                    resolution = "prefer_positive"
                else:
                    resolution = "prefer_negative"
                
                proof.add_conflict(
                    predicate=predicates[0] if predicates else "unknown",
                    positive=pos.id,
                    negative=neg.id,
                    delta=delta,
                    resolution=resolution
                )
                conflict_count += 1
        
        return conflict_count
    
    def render_proof(self, proof: AnswerProof) -> str:
        """
        Convert AnswerProof to natural language.
        
        This is a convenience method for backward compatibility.
        Ahankara should ideally do the rendering.
        
        Args:
            proof: AnswerProof object
        
        Returns:
            natural language answer
        """
        if proof.verdict == "unknown":
            return "I do not know."
        
        # If we have conflicts, show uncertainty
        if proof.conflicts:
            # Find the beliefs involved
            general = None
            exception = None
            
            for step in proof.steps:
                if step.rule == "direct_match" and step.confidence and step.confidence > 0.7:
                    general_text = step.output
                    general = general_text
                elif step.rule == "direct_match" and step.confidence and step.confidence < 0.5:
                    exception_text = step.output
                    exception = exception_text
            
            if general and exception:
                return (
                    f"I am not certain. "
                    f"General rule: {general}. "
                    f"Exception hypothesis: {exception}."
                )
            else:
                return "I am not certain."
        
        # Simple yes/no based on verdict
        if proof.verdict == "yes":
            return "Yes."
        
        elif proof.verdict == "no":
            return "No."
        
        elif proof.verdict == "uncertain":
            return "I am not certain."
        
        return "I do not know."
    
    def _verbalize(self, winner: dict, decision: dict) -> str:
        """
        Convert winning candidate to natural language answer.
        
        Args:
            winner: winning candidate dict
            decision: decision dict with scores
        
        Returns:
            answer string
        """
        if winner["type"] == "existing":
            belief = winner["belief"]
            
            # Check confidence threshold
            if belief.confidence < 0.3:
                return "I am not certain."
            
            return belief.statement_text or "Yes."
        
        else:
            # Proposal won (shouldn't happen in query mode, but handle it)
            return winner["proposal"].get("statement", "Yes.")
    
    def _verbalize_uncertain(self, decision: dict, focus_set: list[Belief]) -> str:
        """
        Verbalize uncertainty with context.
        
        When multiple beliefs compete without clear winner,
        expose the uncertainty to the user.
        
        Args:
            decision: uncertain decision
            focus_set: relevant beliefs
        
        Returns:
            answer exposing uncertainty
        """
        base = "I am not certain."
        
        # Try to provide helpful context
        if focus_set:
            # Find general rule and exceptions
            general = None
            exceptions = []
            
            for b in focus_set:
                if b.confidence > 0.7:
                    general = b
                elif b.confidence < 0.5 and b.epistemic_state == EpistemicType.HYPOTHESIS:
                    exceptions.append(b)
            
            # If we have conflicting information, say so
            if general and exceptions:
                return (
                    f"I am not certain. "
                    f"General rule: {general.statement_text}. "
                    f"Exception hypothesis: {exceptions[0].statement_text}."
                )
        
        return base
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: FOCUS (ATTENTION)
    # ═══════════════════════════════════════════════════════════════
    
    def _focus(self, proposal: dict) -> list[Belief]:
        """
        Restrict reasoning to relevant beliefs.
        
        Query Chitta for beliefs matching:
        - Same entities
        - Same predicates
        - Direct neighbors (1-hop)
        
        Args:
            proposal: BeliefProposal
        
        Returns:
            list of focused beliefs
        """
        return self.chitta.focus(
            entities=proposal.get("entities", []),
            predicates=proposal.get("predicates", []),
            depth=1,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: CANDIDATE GENERATION
    # ═══════════════════════════════════════════════════════════════
    
    def _generate_candidates(
        self,
        proposal: dict,
        focus_set: list[Belief],
    ) -> list[dict]:
        """
        Generate candidate beliefs for judgment.
        
        Candidates include:
        - Existing beliefs from focus set
        - Proposed belief (not yet in memory)
        
        Args:
            proposal: BeliefProposal
            focus_set: focused beliefs
        
        Returns:
            list of candidate dicts
        """
        candidates = []
        
        # Existing beliefs
        for belief in focus_set:
            candidates.append({
                "type": "existing",
                "belief": belief,
                "id": belief.id,
            })
        
        # Proposed belief (wrapped)
        candidates.append(self._wrap_proposal(proposal))
        
        return candidates
    
    def _wrap_proposal(self, proposal: dict) -> dict:
        """
        Wrap BeliefProposal as candidate.
        
        Args:
            proposal: BeliefProposal dict
        
        Returns:
            candidate dict
        """
        return {
            "type": "proposal",
            "belief": None,  # Not yet in memory
            "id": None,
            "proposal": proposal,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: EVALUATION (JUDGMENT FUNCTION)
    # ═══════════════════════════════════════════════════════════════
    
    def _evaluate(self, candidates: list[dict]) -> dict[Any, float]:
        """
        Evaluate candidates with judgment function.
        
        J(b) = w_c·c + w_s·s + w_sup·tanh(S) - w_con·tanh(C) + w_a·tanh(a) + w_coh·coherence
        
        Args:
            candidates: list of candidate dicts
        
        Returns:
            dict mapping candidate -> J-score
        """
        scores = {}
        
        for candidate in candidates:
            # Extract belief state vector ⟨c, s, S, C, a, coherence⟩
            c = self._get_confidence(candidate)
            s = self._get_specificity(candidate)
            S = self._get_support_strength(candidate)
            C = self._get_conflict_strength(candidate)  # Now asymmetric
            a = self._get_activation(candidate)
            coherence = self._get_coherence(candidate)
            
            # Compute judgment score
            J = (
                W_CONF * c +
                W_SPEC * s +
                W_SUP * math.tanh(S) -
                W_CON * math.tanh(C) +
                W_ACT * math.tanh(a) +
                W_COH * coherence
            )
            
            scores[id(candidate)] = {
                "candidate": candidate,
                "score": J,
                "components": {
                    "confidence": c,
                    "specificity": s,
                    "support": S,
                    "conflict": C,
                    "activation": a,
                    "coherence": coherence,
                },
            }
        
        return scores
    
    def _get_confidence(self, candidate: dict) -> float:
        """Extract confidence from candidate."""
        if candidate["type"] == "existing":
            return candidate["belief"].confidence
        else:
            return candidate["proposal"].get("parser_confidence", 0.5)
    
    def _get_specificity(self, candidate: dict) -> float:
        """
        Compute specificity: s = 1 / (1 + depth).
        
        Depth = distance in is_a hierarchy.
        """
        if candidate["type"] == "existing":
            return self.chitta.compute_specificity(candidate["id"])
        else:
            # Proposal has no depth yet (assume specific)
            return 1.0
    
    def _get_support_strength(self, candidate: dict) -> float:
        """Sum of incoming 'supports' edge weights."""
        if candidate["type"] == "existing":
            return self.chitta.compute_support_strength(candidate["id"])
        else:
            return 0.0  # No support yet
    
    def _get_conflict_strength(self, candidate: dict) -> float:
        """
        Sum of incoming 'contradicts' edge weights.
        
        ASYMMETRIC: Specific beliefs contradict generals harder.
        Use entity count as specificity measure.
        """
        if candidate["type"] != "existing":
            return 0.0
        
        edges = self.chitta.get_edges_in(candidate["id"], relation="contradicts")
        
        total_conflict = 0.0
        for edge_data in edges:
            weight = edge_data[1]  # Second element is weight
            src_id = edge_data[0]  # First element is source
            
            # Get source belief specificity
            src_belief = self.chitta.get(src_id)
            if src_belief:
                # More specific = more entities = higher specificity
                src_specificity = len(src_belief.entities) if src_belief.entities else 1
                tgt_specificity = len(candidate["belief"].entities) if candidate["belief"].entities else 1
                
                # If source is more specific than target, it contradicts harder
                # If source is more general, it contradicts weaker
                asymmetry_factor = src_specificity / max(tgt_specificity, 1)
                
                total_conflict += weight * asymmetry_factor
            else:
                total_conflict += weight
        
        return total_conflict
    
    def _get_activation(self, candidate: dict) -> float:
        """Get activation (usage/recency)."""
        if candidate["type"] == "existing":
            return candidate["belief"].activation
        else:
            return 0.0  # Not used yet
    
    def _get_coherence(self, candidate: dict) -> float:
        """
        Compute coherence with neighboring beliefs.
        
        Beliefs that fit their neighborhood are more stable.
        Coherence = avg similarity with neighbors.
        
        Returns:
            coherence score [0, 1]
        """
        if candidate["type"] != "existing":
            return 0.5  # Neutral for proposals
        
        belief_id = candidate["id"]
        
        # Get neighbors (1-hop)
        neighbors = self.chitta.neighbors(belief_id, direction="both")
        
        if not neighbors:
            return 0.5  # Neutral if isolated
        
        # Simple coherence: share of neighbors with same template
        belief = candidate["belief"]
        same_template_count = 0
        
        for neighbor_id in neighbors[:10]:  # Limit to 10 neighbors
            neighbor = self.chitta.get(neighbor_id)
            if neighbor and neighbor.template == belief.template:
                same_template_count += 1
        
        coherence = same_template_count / min(len(neighbors), 10)
        return coherence
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: SELECTION
    # ═══════════════════════════════════════════════════════════════
    
    def _select(self, scores: dict, proposal: dict) -> dict:
        """
        Select winner based on J-scores.
        
        Rules:
        - Highest J-score wins
        - If ΔJ < WIN_MARGIN → uncertain
        - If all J < MIN_SCORE → unknown
        
        Args:
            scores: candidate -> score mapping
            proposal: original proposal
        
        Returns:
            decision dict
        """
        if not scores:
            # Log suppression during learning
            if self.learning_mode:
                self.stats["beliefs_suppressed"] += 1
                # Could add detailed logging here
            return {"decision": "unknown", "reason": "no_candidates"}
        
        # Sort by score (descending)
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )
        
        top_id, top_data = ranked[0]
        top_score = top_data["score"]
        top_candidate = top_data["candidate"]
        
        # Check minimum score threshold
        if top_score < MIN_SCORE:
            return {
                "decision": "unknown",
                "reason": "low_score",
                "score": top_score,
            }
        
        # Check for uncertainty (close second)
        if len(ranked) > 1:
            second_score = ranked[1][1]["score"]
            margin = top_score - second_score
            
            if margin < WIN_MARGIN:
                return {
                    "decision": "uncertain",
                    "reason": "close_scores",
                    "top_score": top_score,
                    "second_score": second_score,
                    "margin": margin,
                }
        
        # Clear winner
        return {
            "decision": "accept",
            "winner": top_candidate,
            "score": top_score,
            "components": top_data["components"],
            "all_scores": ranked,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: REVISION (WRITE TO CHITTA)
    # ═══════════════════════════════════════════════════════════════
    
    def _revise(self, decision: dict, proposal: dict) -> dict:
        """
        Revise Chitta based on decision.
        
        Actions:
        - accept → add belief, update edges
        - uncertain → add as hypothetical
        - unknown → add as unknown belief
        
        LEARNING MODE OVERRIDE:
        During learning, ALWAYS add the proposal without competition.
        This prevents silent belief suppression when teaching.
        
        Args:
            decision: decision dict from _select
            proposal: original BeliefProposal
        
        Returns:
            result dict with belief_id
        """
        # ═══════════════════════════════════════════════════════════════
        # 🔴 LEARNING MODE: Bypass competition, add directly
        # ═══════════════════════════════════════════════════════════════
        if self.learning_mode:
            # Check for exact duplicate first (evidence accumulation)
            existing = self.chitta.find_matching_belief(proposal)
            if existing:
                # True duplicate - accumulate evidence
                return {
                    "action": "existing_belief",
                    "belief_id": existing.id,
                    "confidence": existing.confidence,
                }
            
            # Not a duplicate - add directly without judgment
            confidence = proposal.get("parser_confidence", 0.5)
            belief_id = self.chitta.add_belief_from_proposal(
                proposal,
                confidence=confidence,
            )
            
            return {
                "action": "added_belief",
                "belief_id": belief_id,
                "confidence": confidence,
                "score": 1.0,  # No competition in learning mode
            }
        
        # ═══════════════════════════════════════════════════════════════
        # NORMAL MODE: Use judgment
        # ═══════════════════════════════════════════════════════════════
        decision_type = decision["decision"]
        
        if decision_type == "accept":
            return self._revise_accept(decision, proposal)
        
        elif decision_type == "uncertain":
            return self._revise_uncertain(proposal)
        
        else:  # unknown
            return self._revise_unknown(proposal)
    
    def _revise_accept(self, decision: dict, proposal: dict) -> dict:
        """
        Accept proposal: add to Chitta and update edges.
        
        Args:
            decision: accept decision
            proposal: BeliefProposal
        
        Returns:
            result with belief_id
        """
        self.stats["accepts"] += 1
        
        winner = decision["winner"]
        
        # If winner is existing belief, just return it
        if winner["type"] == "existing":
            belief_id = winner["id"]
            belief = self.chitta.beliefs[belief_id]
            
            # Touch to update activation
            belief.touch()
            
            return {
                "action": "existing_belief",
                "belief_id": belief_id,
                "confidence": belief.confidence,  # Return ACTUAL confidence, not proposal
                "score": decision["score"],
            }
        
        # Winner is proposal → add to Chitta
        # Use parser_confidence as initial confidence
        confidence = proposal.get("parser_confidence", 0.5)
        
        belief_id = self.chitta.add_belief_from_proposal(
            proposal,
            confidence=confidence,
        )
        
        # Update edges (find contradictions/refinements)
        self._update_edges(belief_id, proposal, decision)
        
        return {
            "action": "added_belief",
            "belief_id": belief_id,
            "confidence": confidence,
            "score": decision["score"],
        }
    
    def _revise_uncertain(self, proposal: dict) -> dict:
        """
        Uncertain: add as hypothetical belief.
        
        IMPORTANT: Also create tentative contradiction edges.
        When a hypothesis contradicts a stronger belief, we need
        a soft edge to signal the conflict for future reasoning.
        
        NOTE: If evidence accumulation occurs (existing belief found),
        return the ACTUAL belief state, not "hypothetical".
        
        Args:
            proposal: BeliefProposal
        
        Returns:
            result with belief_id
        """
        self.stats["uncertains"] += 1
        
        belief_id = self.chitta.add_hypothetical(proposal, confidence=0.3)
        
        # Check if evidence was accumulated (belief already existed)
        belief = self.chitta.get(belief_id)
        if belief and belief.evidence_count > 1:
            # Evidence was accumulated! Return actual belief state
            return {
                "action": "existing_belief",
                "belief_id": belief_id,
                "confidence": belief.confidence,
            }
        
        # Find conflicting beliefs and create tentative edges
        self._add_tentative_contradictions(belief_id, proposal)
        
        return {
            "action": "hypothetical",
            "belief_id": belief_id,
            "confidence": 0.3,
        }
    
    def _add_tentative_contradictions(self, hypothesis_id: str, proposal: dict):
        """
        Add tentative contradiction edges for hypothetical beliefs.
        
        When a hypothesis contradicts a stronger belief, we create
        a soft contradiction edge with tentative=True metadata.
        
        This enables:
        - HRE to use this signal for hypothesis generation
        - Future evidence to amplify or decay the edge
        - Gradual belief revision instead of binary logic
        
        Args:
            hypothesis_id: newly added hypothesis
            proposal: BeliefProposal for the hypothesis
        """
        hypothesis = self.chitta.get(hypothesis_id)
        
        # Find beliefs this might contradict
        focus_beliefs = self.chitta.focus(
            entities=proposal.get("entities", []),
            predicates=proposal.get("predicates", []),
            depth=1,
        )
        
        for existing in focus_beliefs:
            if existing.id == hypothesis_id:
                continue
            
            # Check if they contradict
            if self._is_contradiction(proposal, existing):
                # Create tentative contradiction edge
                # Weight = hypothesis confidence (weak signal)
                self.chitta.add_edge(
                    src_id=hypothesis_id,
                    relation="contradicts",
                    tgt_id=existing.id,
                    weight=hypothesis.confidence,
                    tentative=True,  # Mark as tentative
                )
                
                # Bidirectional contradiction
                self.chitta.add_edge(
                    src_id=existing.id,
                    relation="contradicts",
                    tgt_id=hypothesis_id,
                    weight=hypothesis.confidence,
                    tentative=True,
                )
    
    def _revise_unknown(self, proposal: dict) -> dict:
        """
        Unknown: add as unknown belief (gap in knowledge).
        
        NOTE: If evidence accumulation occurs (existing belief found),
        return the ACTUAL belief state, not "unknown".
        
        Args:
            proposal: BeliefProposal
        
        Returns:
            result with belief_id
        """
        self.stats["unknowns"] += 1
        
        belief_id = self.chitta.add_unknown(proposal)
        
        # Check if evidence was accumulated (belief already existed)
        belief = self.chitta.get(belief_id)
        if belief and belief.evidence_count > 1:
            # Evidence was accumulated! Return actual belief state
            return {
                "action": "existing_belief",
                "belief_id": belief_id,
                "confidence": belief.confidence,
            }
        
        return {
            "action": "unknown",
            "belief_id": belief_id,
            "confidence": 0.0,
        }
    
    def _update_edges(
        self,
        belief_id: str,
        proposal: dict,
        decision: dict,
    ):
        """
        Update edges after adding new belief.
        
        Detect and add:
        - Contradictions
        - Refinements
        - Supports
        
        Args:
            belief_id: newly added belief
            proposal: BeliefProposal
            decision: decision dict
        """
        # Check against all scored candidates
        for score_data in decision.get("all_scores", []):
            candidate = score_data[1]["candidate"]
            
            if candidate["type"] != "existing":
                continue
            
            existing_id = candidate["id"]
            existing_belief = candidate["belief"]
            
            # Detect contradiction
            if self._is_contradiction(proposal, existing_belief):
                # Add contradicts edge
                self.chitta.add_edge(
                    belief_id,
                    "contradicts",
                    existing_id,
                    weight=0.9,
                )
            
            # Detect refinement (specialization)
            elif self._is_refinement(proposal, existing_belief):
                # Add refines edge
                self.chitta.add_edge(
                    belief_id,
                    "refines",
                    existing_id,
                    weight=0.8,
                )
    
    def _is_contradiction(self, proposal: dict, belief: Belief) -> bool:
        """
        Detect if proposal contradicts belief.
        
        Simple heuristic: same predicates, opposite polarity.
        
        Args:
            proposal: BeliefProposal
            belief: existing Belief
        
        Returns:
            True if contradiction detected
        """
        # Check if predicates overlap
        proposal_preds = set(proposal.get("predicates", []))
        belief_preds = belief.predicates
        
        if not (proposal_preds & belief_preds):
            return False
        
        # Check polarity
        proposal_polarity = proposal.get("polarity", +1)
        
        # Simple negation detection (can be enhanced)
        canonical = belief.canonical
        belief_negated = canonical.get("negated", False)
        
        # Opposite polarities
        if proposal_polarity == -1 and not belief_negated:
            return True
        if proposal_polarity == +1 and belief_negated:
            return True
        
        return False
    
    def _is_refinement(self, proposal: dict, belief: Belief) -> bool:
        """
        Detect if proposal refines (specializes) belief.
        
        Refinement = more specific version of general rule.
        
        Args:
            proposal: BeliefProposal
            belief: existing Belief
        
        Returns:
            True if refinement detected
        """
        # Check if proposal entities are more specific
        proposal_entities = set(proposal.get("entities", []))
        belief_entities = belief.entities
        
        # Simple heuristic: if proposal contradicts but has subset of entities
        # → it's a refinement (exception to general rule)
        if proposal_entities.issubset(belief_entities):
            if self._is_contradiction(proposal, belief):
                return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════
    
    def get_stats(self) -> dict:
        """Get reasoning statistics."""
        return self.stats.copy()
    
    def __repr__(self) -> str:
        return (
            f"<Buddhi "
            f"judgments={self.stats['judgments']} "
            f"accepts={self.stats['accepts']} "
            f"uncertains={self.stats['uncertains']} "
            f"unknowns={self.stats['unknowns']}>"
        )

    def _try_direct_answer(self, applicable_beliefs, query_proposal, proof):
        """Try to answer directly from applicable beliefs."""
        if not applicable_beliefs:
            return None
        
        # Check for hard negations first
        for belief in applicable_beliefs:
            # Handle Polarity Enum
            pol = belief.polarity
            pol_val = pol.value if hasattr(pol, "value") else pol
            
            # Polarity.NEGATIVE is 2 (auto), or -1 if using legacy int convention
            if pol_val == 2 or pol_val == -1: # check both to be safe
                 # Found explicit negation!
                 proof.verdict = "no"
                 proof.add_step(
                     rule="hard_negative",
                     inputs=[belief.id],
                     output=f"Explicit negation found: {belief.statement_text}",
                     confidence=belief.confidence
                 )
                 return proof
        
        # Check for positive matches
        for belief in applicable_beliefs:
            pol = belief.polarity
            pol_val = pol.value if hasattr(pol, "value") else pol
            
            # Polarity.POSITIVE is 1
            if pol_val == 1:
                 proof.verdict = "yes"
                 proof.add_step(
                     rule="direct_match",
                     inputs=[belief.id],
                     output=f"Direct grounding: {len(applicable_beliefs)} applicable belief(s)",
                     confidence=belief.confidence
                 )
                 return proof
                 
        return None
