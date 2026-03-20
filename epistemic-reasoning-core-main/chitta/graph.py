"""
CHITTA GRAPH — Belief Hypergraph Storage

ChittaGraph is the core storage engine for MARC's belief memory.

Responsibilities:
- Store beliefs (single node type)
- Manage relations between beliefs (hyperedges)
- Index beliefs for fast retrieval
- Track provenance and updates
- Support querying and traversal

Philosophy:
- Edges encode epistemic relations, NOT logic
- Logic lives in Buddhi, not in the graph
- Graph is pure data structure
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
import uuid
import heapq
import uuid
import heapq
import json
import os
import shutil

from buddhi.belief import Belief, EpistemicType
from common.utils import (
    decay_confidence,
    generate_edge_id,
    now_utc,
    update_confidence_logodds,
    validate_edge_type,
)


class ChittaGraph:
    """
    The storage engine for MARC's beliefs and how they relate to each other.
    
    Think of this as a graph database optimized for beliefs:
    - Beliefs are nodes
    - Relations between beliefs are edges
    - Everything is indexed for fast lookup
    
    Main operations:
    - add_belief / remove_belief: manage nodes
    - add_edge / remove_edge: manage relations
    - Query: find beliefs by template, entity, predicate
    - Traverse: walk the graph following connections
    - Update: adjust confidence, track activation
    - Serialize: save/load the entire graph
    """
    
    def __init__(self):
        # ═══════════════════════════════════════════════════════════
        # CORE STORAGE
        # ═══════════════════════════════════════════════════════════
        
        # Main belief storage: belief_id → Belief object
        self.beliefs: dict[str, Belief] = {}
        
        # Edge storage: relation_type → source_id → [(target_id, weight, edge_id)]
        # This lets us quickly find all outgoing edges from a belief
        self.edges: dict[str, dict[str, list[tuple[str, float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Reverse edge index for finding incoming edges
        # Same structure but reversed direction
        self.edges_reverse: dict[str, dict[str, list[tuple[str, float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # ═══════════════════════════════════════════════════════════
        # INDEXES
        # ═══════════════════════════════════════════════════════════
        
        # Template index: template -> {belief_ids}
        self.template_index: dict[str, set[str]] = defaultdict(set)
        
        # Entity index: entity -> {belief_ids}
        self.entity_index: dict[str, set[str]] = defaultdict(set)
        
        # Predicate index: predicate -> {belief_ids}
        self.predicate_index: dict[str, set[str]] = defaultdict(set)
        
        # Epistemic state index: state -> {belief_ids}
        self.state_index: dict[str, set[str]] = defaultdict(set)
        
        # QUERY INDEX: Track unanswered/answered queries
        # Format: predicate -> [(query_proposal, answer_belief_id, status)]
        self.query_index: dict[str, list[dict]] = defaultdict(list)
        
        # SOFT-MERGE CANDIDATES: Track near-duplicates
        # Format: [(belief_id_1, belief_id_2, similarity_score)]
        self.merge_candidates: list[tuple[str, str, float]] = []
        
        # ═══════════════════════════════════════════════════════════
        # METADATA
        # ═══════════════════════════════════════════════════════════
        
        self.created_at = now_utc()
        self.updated_at = self.created_at
        self.metadata: dict[str, Any] = {}
        
        # Statistics
        # Statistics
        self.stats = {
            "beliefs_added": 0,
            "beliefs_removed": 0,
            "edges_added": 0,
            "edges_removed": 0,
            "confidence_updates": 0,
            "evidence_accumulated": 0
        }
    
    # ═══════════════════════════════════════════════════════════════
    # PERSISTENCE (SAVE / LOAD)
    # ═══════════════════════════════════════════════════════════════
    
    def clear(self):
        """Reset the graph state (For testing)."""
        self.beliefs = {}
        self.edges = defaultdict(lambda: defaultdict(list))
        self.edges_reverse = defaultdict(lambda: defaultdict(list))
        
        self.template_index = defaultdict(set)
        self.entity_index = defaultdict(set)
        self.predicate_index = defaultdict(set)
        self.state_index = defaultdict(set)
        
        self.query_index = defaultdict(list)
        self.merge_candidates = []
        
        self.stats = {
            "beliefs_added": 0,
            "beliefs_removed": 0,
            "edges_added": 0,
            "edges_removed": 0,
            "confidence_updates": 0,
            "evidence_accumulated": 0
        }

    def save(self, filepath: str):
        """
        Save the entire graph to a JSON file.
        """
        # 1. Serialize Global Edges (to preserve weights/ids)
        # Structure: list of {relation, source, target, weight, edge_id, metadata}
        serialized_edges = []
        for relation, src_map in self.edges.items():
            for src_id, edge_list in src_map.items():
                for (tgt_id, weight, edge_id, *rest) in edge_list:
                    meta = rest[0] if rest else {}
                    serialized_edges.append({
                        "relation": relation,
                        "source": src_id,
                        "target": tgt_id,
                        "weight": weight,
                        "edge_id": edge_id,
                        "metadata": meta
                    })

        # 2. Serialize Beliefs
        serialized_beliefs = {}
        for bid, belief in self.beliefs.items():
            # Handle Enum serialization safely
            e_state = belief.epistemic_state
            if hasattr(e_state, "value"):
                e_state = e_state.value
                
            serialized_beliefs[bid] = {
                "id": belief.id,
                "template": belief.template,
                "canonical": belief.canonical,
                "active": belief.active,
                "confidence": belief.confidence,
                "epistemic_state": e_state,
                "entities": list(belief.entities),
                "predicates": list(belief.predicates),
                "original_text": belief.original_text,
                "statement_text": belief.statement_text,
                "polarity_value": belief.polarity_value,
                "source": belief.source,
                "provenance": belief.provenance,
                "evidence_count": belief.evidence_count,
                "created_at": belief.created_at,
                "updated_at": belief.updated_at,
                "epistemic_class": getattr(belief, "epistemic_class", None)
            }

        data = {
            "version": "1.0",
            "metadata": self.metadata,
            "stats": self.stats,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "beliefs": serialized_beliefs,
            "edges": serialized_edges
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Atomic write
        temp_path = filepath + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        shutil.move(temp_path, filepath)
        
    def load(self, filepath: str):
        """
        Load graph from JSON file.
        Rebuilds indexes automatically.
        """
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Graph file not found: {filepath}")
             
        with open(filepath, "r") as f:
            data = json.load(f)
            
        # Clear current state
        self.clear()
        
        self.metadata = data.get("metadata", {})
        self.stats = data.get("stats", {})
        self.created_at = data.get("created_at", now_utc())
        self.updated_at = data.get("updated_at", now_utc())
        
        
        load_report = {
            "total_in_file": len(data.get("beliefs", {})),
            "loaded": 0,
            "errors": 0,
            "reasons": defaultdict(int)
        }
        
        # 1. Restore Beliefs
        for bid, b_dict in data["beliefs"].items():
            try:
                belief = Belief(
                    template=b_dict["template"],
                    canonical=b_dict["canonical"],
                    confidence=b_dict["confidence"],
                    epistemic_state=b_dict["epistemic_state"],
                    original_text=b_dict["original_text"],
                    statement_text=b_dict["statement_text"],
                    source=b_dict["source"],
                    polarity_value=b_dict.get("polarity_value", 1)
                )
                # Restore Internals
                belief.id = b_dict["id"]
                belief.created_at = b_dict.get("created_at", now_utc())
                belief.updated_at = b_dict.get("updated_at", now_utc())
                belief.active = b_dict.get("active", True)
                belief.evidence_count = b_dict.get("evidence_count", 1)
                belief.provenance = b_dict.get("provenance", [])
                
                if b_dict.get("epistemic_class"):
                    belief.epistemic_class = b_dict["epistemic_class"]
                
                # Add to graph dict
                self.beliefs[bid] = belief
                
                # Rebuild Indexes
                self.template_index[belief.template].add(bid)
                
                # Enum Key Handling for Index
                e_state_key = belief.epistemic_state
                if hasattr(e_state_key, "value"):
                    e_state_key = e_state_key.value
                self.state_index[e_state_key].add(bid)
                
                for entity in belief.entities:
                    self.entity_index[entity].add(bid)
                for predicate in belief.predicates:
                    self.predicate_index[predicate].add(bid)
                    
                load_report["loaded"] += 1
                
            except Exception as e:
                load_report["errors"] += 1
                load_report["reasons"][str(e)] += 1
                # print(f"[ChittaGraph] Failed to load belief {bid}: {e}")

        # 2. Restore Edges
        raw_edges = data.get("edges", [])
        for e in raw_edges:
            relation = e["relation"]
            src = e["source"]
            tgt = e["target"]
            weight = e["weight"]
            eid = e["edge_id"]
            meta = e.get("metadata", {})
            
            # Skip if nodes missing (integrity check)
            if src not in self.beliefs or tgt not in self.beliefs:
                continue
                
            # Populate Global Index
            tuple_data = (tgt, weight, eid, meta) if meta else (tgt, weight, eid)
            self.edges[relation][src].append(tuple_data)
            
            # Populate Reverse Index
            rev_data = (src, weight, eid, meta) if meta else (src, weight, eid)
            self.edges_reverse[relation][tgt].append(rev_data)
            
            # Populate Belief Internal Lists
            self.beliefs[src]._add_edge_out(relation, tgt)
            self.beliefs[tgt]._add_edge_in(relation, src)
            
        return load_report

    
    # ═══════════════════════════════════════════════════════════════
    # BELIEF MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def add_belief(self, belief: Belief) -> str:
        """
        Add a new belief to the graph.
        
        This stores the belief and updates all indexes so we can find it later.
        
        Args:
            belief: the Belief object to store
        
        Returns:
            the belief's ID
        
        Raises:
            ValueError: if a belief with this ID already exists
        """
        # 1. Check for ID collision (Exact Same Object)
        if belief.id in self.beliefs:
            existing = self.beliefs[belief.id]
            # Reinforce existing belief
            existing.reinforce(boost=0.05, success=True)
            existing.add_provenance("reinforcement", "add_belief", 0.05, {"method": "id_match"})
            self.updated_at = now_utc()
            return existing.id
            
        # 2. Check for Semantic Collision (Different ID, Same Content)
        # Construct quasi-proposal to reuse matching logic
        proposal_proxy = {
            "template": belief.template,
            "canonical": belief.canonical,
            "raw_text": belief.statement_text
        }
        semantic_match = self.find_matching_belief(proposal_proxy)
        
        if semantic_match:
            # Reinforce existing belief
            semantic_match.reinforce(boost=0.05, success=True)
            semantic_match.add_provenance("reinforcement", "add_belief", 0.05, {"method": "semantic_match", "merged_id": belief.id})
            self.updated_at = now_utc()
            return semantic_match.id

        # 3. New Belief - Add to Graph
        self.beliefs[belief.id] = belief
        
        # Update indexes
        self.template_index[belief.template].add(belief.id)
        self.state_index[belief.epistemic_state].add(belief.id)
        
        for entity in belief.entities:
            self.entity_index[entity].add(belief.id)
        
        for predicate in belief.predicates:
            self.predicate_index[predicate].add(belief.id)
        
        # Update metadata
        self.updated_at = now_utc()
        self.stats["beliefs_added"] += 1
        
        
        # ═══════════════════════════════════════════════════════════
        # LOGIC HARDENING: GLOBAL BELIEF REVISION
        # ═══════════════════════════════════════════════════════════
        if belief.epistemic_state == EpistemicType.EXCEPTION:
            candidates = set()
            first = True
            
            # 1. Filter by entities
            for ent in belief.entities:
                if ent in self.entity_index:
                    if first:
                        candidates = self.entity_index[ent].copy()
                        first = False
                    else:
                        candidates &= self.entity_index[ent]
            
            # 2. Filter by predicates
            for pred in belief.predicates:
                if pred in self.predicate_index:
                    if first:
                         candidates = self.predicate_index[pred].copy()
                         first = False
                    else:
                         candidates &= self.predicate_index[pred]
            
            # 3. Check for DEFAULTS to supersede
            for cid in candidates:
                cand = self.beliefs[cid]
                if (cand.active and 
                    cand.epistemic_state == EpistemicType.DEFAULT and 
                    cand.polarity_value != belief.polarity_value):
                    
                    # Supersede
                    print(f"[Chitta] ⚠️ Revision: '{cand.statement_text}' superseded by '{belief.statement_text}'")
                    cand.deactivate()
                    cand.add_provenance("superseded", belief.id, 1.0, {"reason": "Global Revision by EXCEPTION"})

        return belief.id
    
    def remove_belief(self, belief_id: str, hard_delete: bool = False):
        """
        Remove a belief from the graph.
        
        Args:
            belief_id: ID of belief to remove
            hard_delete: if True, delete completely; if False, deactivate
        
        Raises:
            KeyError: if belief not found
        """
        if belief_id not in self.beliefs:
            raise KeyError(f"Belief {belief_id} not found")
        
        belief = self.beliefs[belief_id]
        
        if hard_delete:
            # Remove from indexes
            self.template_index[belief.template].discard(belief_id)
            self.state_index[belief.epistemic_state].discard(belief_id)
            
            for entity in belief.entities:
                self.entity_index[entity].discard(belief_id)
            
            for predicate in belief.predicates:
                self.predicate_index[predicate].discard(belief_id)
            
            # Remove all edges involving this belief
            self._remove_all_edges_for_belief(belief_id)
            
            # Delete belief
            del self.beliefs[belief_id]
            self.stats["beliefs_removed"] += 1
        else:
            # Soft delete (deactivate)
            belief.deactivate()
        
        self.updated_at = now_utc()
    
    def get(self, belief_id: str) -> Belief | None:
        """Get belief by ID."""
        return self.beliefs.get(belief_id)
    
    def get_active(self, belief_id: str) -> Belief | None:
        """Get belief by ID only if active."""
        belief = self.beliefs.get(belief_id)
        return belief if belief and belief.active else None
    
    def has_belief(self, belief_id: str) -> bool:
        """Check if belief exists."""
        return belief_id in self.beliefs
    
    # ═══════════════════════════════════════════════════════════════
    # EDGE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def add_edge(
        self,
        src_id: str,
        relation: str,
        tgt_id: str,
        weight: float = 1.0,
        edge_id: str | None = None,
        tentative: bool = False,
        metadata: dict | None = None,
    ) -> str:
        """
        Add an edge between two beliefs.
        
        Args:
            src_id: source belief ID
            relation: relation type (supports, contradicts, etc.)
            tgt_id: target belief ID
            weight: edge weight (default 1.0)
            edge_id: optional edge ID (generated if None)
            tentative: if True, edge is tentative/uncertain (default False)
            metadata: optional metadata dict for edge
        
        Returns:
            edge ID
        
        Raises:
            ValueError: if belief IDs invalid or relation type invalid
        """
        # Validate
        if src_id not in self.beliefs:
            raise ValueError(f"Source belief {src_id} not found")
        if tgt_id not in self.beliefs:
            raise ValueError(f"Target belief {tgt_id} not found")
        
        validate_edge_type(relation)
        
        # Generate edge ID
        edge_id = edge_id or generate_edge_id()
        
        # Build edge metadata
        edge_meta = metadata or {}
        if tentative:
            edge_meta["tentative"] = True
        
        # Store as tuple: (tgt_id, weight, edge_id, metadata)
        edge_data = (tgt_id, weight, edge_id, edge_meta) if edge_meta else (tgt_id, weight, edge_id)
        
        # Add forward edge
        self.edges[relation][src_id].append(edge_data)
        
        # Add reverse edge
        self.edges_reverse[relation][tgt_id].append(edge_data)
        
        # Update belief edge tracking
        self.beliefs[src_id]._add_edge_out(relation, tgt_id)
        self.beliefs[tgt_id]._add_edge_in(relation, src_id)
        
        # Update metadata
        self.updated_at = now_utc()
        self.stats["edges_added"] += 1
        
        return edge_id
    
    def remove_edge(
        self,
        src_id: str,
        relation: str,
        tgt_id: str,
    ):
        """
        Remove an edge between two beliefs.
        
        Args:
            src_id: source belief ID
            relation: relation type
            tgt_id: target belief ID
        """
        # Remove from forward index
        if relation in self.edges and src_id in self.edges[relation]:
            self.edges[relation][src_id] = [
                edge_data for edge_data in self.edges[relation][src_id]
                if edge_data[0] != tgt_id  # First element is always target ID
            ]
        
        # Remove from reverse index
        if relation in self.edges_reverse and tgt_id in self.edges_reverse[relation]:
            self.edges_reverse[relation][tgt_id] = [
                edge_data for edge_data in self.edges_reverse[relation][tgt_id]
                if edge_data[0] != src_id  # First element is always source ID
            ]
        
        # Update belief edge tracking
        if src_id in self.beliefs:
            self.beliefs[src_id]._remove_edge_out(relation, tgt_id)
        if tgt_id in self.beliefs:
            self.beliefs[tgt_id]._remove_edge_in(relation, src_id)
        
        self.stats["edges_removed"] += 1
        self.updated_at = now_utc()
    
    def _remove_all_edges_for_belief(self, belief_id: str):
        """Remove all edges involving a belief (internal)."""
        # Remove outgoing edges
        for relation in list(self.edges.keys()):
            if belief_id in self.edges[relation]:
                for edge_data in self.edges[relation][belief_id]:
                    tgt_id = edge_data[0]  # First element is always target ID
                    if tgt_id in self.beliefs:
                        self.beliefs[tgt_id]._remove_edge_in(relation, belief_id)
                del self.edges[relation][belief_id]
        
        # Remove incoming edges
        for relation in list(self.edges_reverse.keys()):
            if belief_id in self.edges_reverse[relation]:
                for edge_data in self.edges_reverse[relation][belief_id]:
                    src_id = edge_data[0]  # First element is always source ID
                    if src_id in self.beliefs:
                        self.beliefs[src_id]._remove_edge_out(relation, belief_id)
                del self.edges_reverse[relation][belief_id]
    
    def get_edges_out(
        self,
        belief_id: str,
        relation: str | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Get outgoing edges from a belief.
        
        Args:
            belief_id: source belief ID
            relation: filter by relation type (None = all)
        
        Returns:
            list of (target_id, weight, edge_id) tuples
        """
        if relation:
            return list(self.edges.get(relation, {}).get(belief_id, []))
        
        # All relations
        result = []
        for rel_edges in self.edges.values():
            result.extend(rel_edges.get(belief_id, []))
        return result
    
    def get_edges_in(
        self,
        belief_id: str,
        relation: str | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Get incoming edges to a belief.
        
        Args:
            belief_id: target belief ID
            relation: filter by relation type (None = all)
        
        Returns:
            list of (source_id, weight, edge_id) tuples
        """
        if relation:
            return list(self.edges_reverse.get(relation, {}).get(belief_id, []))
        
        # All relations
        result = []
        for rel_edges in self.edges_reverse.values():
            result.extend(rel_edges.get(belief_id, []))
        return result
    
    def neighbors(
        self,
        belief_id: str,
        relation: str | None = None,
        direction: str = "out",
    ) -> list[str]:
        """
        Get neighbor belief IDs.
        
        Args:
            belief_id: center belief ID
            relation: filter by relation type
            direction: "out", "in", or "both"
        
        Returns:
            list of neighbor belief IDs
        """
        neighbors = []
        
        if direction in ("out", "both"):
            edges = self.get_edges_out(belief_id, relation)
            # Handle both old format (3-tuple) and new format (4-tuple with metadata)
            for edge_data in edges:
                tgt = edge_data[0]  # First element is always target ID
                neighbors.append(tgt)
        
        if direction in ("in", "both"):
            edges = self.get_edges_in(belief_id, relation)
            # Handle both old format (3-tuple) and new format (4-tuple with metadata)
            for edge_data in edges:
                src = edge_data[0]  # First element is always source ID
                neighbors.append(src)
        
        return neighbors
    
    # ═══════════════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════════════
    
    def find_by_template(
        self,
        template: str,
        active_only: bool = True,
    ) -> list[Belief]:
        """Find beliefs by template type."""
        belief_ids = self.template_index.get(template, set())
        beliefs = [self.beliefs[bid] for bid in belief_ids if bid in self.beliefs]
        if active_only:
            beliefs = [b for b in beliefs if b.active]
        return beliefs
    
    def find_by_entity(
        self,
        entity: str,
        active_only: bool = True,
    ) -> list[Belief]:
        """Find beliefs mentioning an entity."""
        belief_ids = self.entity_index.get(entity, set())
        beliefs = [self.beliefs[bid] for bid in belief_ids if bid in self.beliefs]
        if active_only:
            beliefs = [b for b in beliefs if b.active]
        return beliefs
    
    def find_by_predicate(
        self,
        predicate: str,
        active_only: bool = True,
    ) -> list[Belief]:
        """Find beliefs with a predicate."""
        belief_ids = self.predicate_index.get(predicate, set())
        beliefs = [self.beliefs[bid] for bid in belief_ids if bid in self.beliefs]
        if active_only:
            beliefs = [b for b in beliefs if b.active]
        return beliefs
    
    def find_by_state(
        self,
        epistemic_state: str,
        active_only: bool = True,
    ) -> list[Belief]:
        """Find beliefs by epistemic state."""
        belief_ids = self.state_index.get(epistemic_state, set())
        beliefs = [self.beliefs[bid] for bid in belief_ids if bid in self.beliefs]
        if active_only:
            beliefs = [b for b in beliefs if b.active]
        return beliefs
    
    def query(
        self,
        *,
        template: str | None = None,
        entity: str | None = None,
        predicate: str | None = None,
        epistemic_state: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        active_only: bool = True,
    ) -> list[Belief]:
        """
        Flexible query with multiple filters (intersection).
        
        Returns beliefs matching ALL specified criteria.
        """
        # Start with all beliefs
        result_ids: set[str] | None = None
        
        if template:
            result_ids = self.template_index.get(template, set()).copy()
        
        if entity:
            entity_ids = self.entity_index.get(entity, set())
            result_ids = entity_ids if result_ids is None else result_ids & entity_ids
        
        if predicate:
            pred_ids = self.predicate_index.get(predicate, set())
            result_ids = pred_ids if result_ids is None else result_ids & pred_ids
        
        if epistemic_state:
            state_ids = self.state_index.get(epistemic_state, set())
            result_ids = state_ids if result_ids is None else result_ids & state_ids
        
        # If no filters, return all
        if result_ids is None:
            result_ids = set(self.beliefs.keys())
        
        # Filter by confidence and active status
        beliefs = []
        for bid in result_ids:
            if bid not in self.beliefs:
                continue
            belief = self.beliefs[bid]
            
            if active_only and not belief.active:
                continue
            
            if min_confidence is not None and belief.confidence < min_confidence:
                continue
            
            if max_confidence is not None and belief.confidence > max_confidence:
                continue
            
            beliefs.append(belief)
        
        return beliefs
    
    # ═══════════════════════════════════════════════════════════════
    # BUDDHI INTEGRATION APIs
    # ═══════════════════════════════════════════════════════════════
    
    def focus(
        self,
        entities: list[str] | None = None,
        predicates: list[str] | None = None,
        depth: int = 1,
    ) -> list[Belief]:
        """
        Focus attention on relevant beliefs (for Buddhi).
        
        Returns beliefs matching entities/predicates + neighbors up to depth.
        This restricts reasoning to a small subgraph.
        
        Args:
            entities: entities to focus on
            predicates: predicates to focus on
            depth: neighbor depth (0 = exact match only, 1 = include 1-hop neighbors)
        
        Returns:
            list of focused beliefs
        """
        # Start with exact matches
        focus_ids: set[str] = set()
        
        if entities:
            for entity in entities:
                focus_ids.update(self.entity_index.get(entity, set()))
        
        if predicates:
            for predicate in predicates:
                focus_ids.update(self.predicate_index.get(predicate, set()))
        
        # If no filters, return empty (focused search required)
        if not focus_ids:
            return []
        
        # Expand by depth (neighbor traversal)
        current_ids = focus_ids.copy()
        for _ in range(depth):
            next_ids = set()
            for bid in current_ids:
                # Add neighbors (both directions)
                neighbors = self.neighbors(bid, direction="both")
                next_ids.update(neighbors)
            focus_ids.update(next_ids)
            current_ids = next_ids
        
        # Return active beliefs only
        return [
            self.beliefs[bid]
            for bid in focus_ids
            if bid in self.beliefs and self.beliefs[bid].active
        ]
    
    def find_matching_belief(self, proposal: dict) -> Belief | None:
        """
        Find existing belief that matches this proposal semantically.
        
        Used for evidence accumulation - if we see the same assertion multiple times,
        we should boost its confidence rather than create duplicates.
        
        Args:
            proposal: BeliefProposal from Manas
        
        Returns:
            Matching belief or None
        """
        template = proposal.get("template")
        canonical = proposal.get("canonical", {})
        raw_text = proposal.get("raw_text", "").lower().strip()
        
        # Look for beliefs with same template
        candidates = self.template_index.get(template, set())
        
        for bid in candidates:
            belief = self.beliefs.get(bid)
            if not belief or not belief.active:
                continue
            
            # Try exact canonical match first
            if belief.canonical == canonical:
                return belief
            
            # Fallback: fuzzy match on statement text
            # (handles cases where Manas parses slightly differently)
            if belief.statement_text and raw_text:
                belief_text = belief.statement_text.lower().strip()
                # Normalize: remove punctuation
                import string
                belief_norm = belief_text.translate(str.maketrans('', '', string.punctuation))
                raw_norm = raw_text.translate(str.maketrans('', '', string.punctuation))
                
                if belief_norm == raw_norm:
                    return belief
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # QUANTITATIVE LAYER: DECAY
    # ═══════════════════════════════════════════════════════════════

    def apply_decay(self, steps: int = 1, threshold: float = 0.1) -> int:
        """
        Apply temporal decay to all non-AXIOM beliefs.
        
        Simulates memory fading. If confidence drops below threshold,
        the belief becomes INACTIVE.
        
        Args:
            steps: number of decay steps to apply
            threshold: confidence threshold for deactivation
            
        Returns:
            count of beliefs deactivated
        """
        deactivated_count = 0
        
        for belief in self.beliefs.values():
            if not belief.active:
                continue
            
            # Immunity for AXIOMS
            if belief.epistemic_state == EpistemicType.AXIOM:
                continue
                
            # Apply decay
            decay_factor = belief.decay_rate ** steps
            belief.confidence *= decay_factor
            
            # Check threshold
            if belief.confidence < threshold:
                print(f"[Chitta] 📉 Deactivating '{belief.statement_text}' (Conf {belief.confidence:.3f} < {threshold})")
                belief.deactivate()
                deactivated_count += 1
            
        if deactivated_count > 0:
            self.updated_at = now_utc()
            
        return deactivated_count

    def add_belief_from_proposal(
        self,
        proposal: dict,
        confidence: float | None = None,
    ) -> str:
        """
        Create and add belief from BeliefProposal (from Manas).
        
        This is the bridge between Manas output and Chitta storage.
        Only called by Buddhi after judgment.
        
        EVIDENCE ACCUMULATION (Problem #1 Fix):
        If this exact assertion already exists, increment evidence_count
        and boost confidence instead of creating duplicate.
        
        FRAGILITY ZONE A: Evidence ≠ Grounding
        ────────────────────────────────────────
        evidence = repetition (NOT observation)
        
        This is acceptable ONLY because:
        - confidence grows slowly (ε = 0.05)
        - promotion thresholds exist
        - cap at 0.9 (never reaches certainty)
        
        CONSTRAINTS (DO NOT VIOLATE):
        - DO NOT increase ε (no faster learning)
        - DO NOT auto-assert observational facts
        - DO NOT bypass promotion thresholds
        
        HRE will handle true grounding via hypothetical worlds.
        
        Args:
            proposal: BeliefProposal dict from Manas
            confidence: override confidence (default: use parser_confidence)
        
        Returns:
            belief ID
        """
        # Check if this belief already exists
        existing = self.find_matching_belief(proposal)
        
        if existing:
            # EVIDENCE ACCUMULATION: Repeated assertion
            existing.evidence_count += 1
            
            # Confidence boost: ε * log(evidence_count)
            # CRITICAL: ε = 0.05 is calibrated, DO NOT increase
            # Cap at 0.9 prevents false certainty from repetition alone
            import math
            epsilon = 0.05  # LOCKED: calibrated for slow evidence growth
            boost = epsilon * math.log(existing.evidence_count)
            new_confidence = min(0.9, existing.confidence + boost)
            
            # Update confidence
            existing.confidence = new_confidence
            existing.updated_at = now_utc()
            
            # Add provenance
            existing.add_provenance(
                op="evidence_accumulation",
                from_source="repeated_assertion",
                score=boost,
                metadata={"evidence_count": existing.evidence_count}
            )
            
            # Update stats
            self.stats["evidence_accumulated"] = self.stats.get("evidence_accumulated", 0) + 1
            self.updated_at = now_utc()
            
            return existing.id
        
        # Extract confidence
        if confidence is None:
            confidence = proposal.get("parser_confidence", 0.5)
        
        # Create NEW belief
        belief = Belief(
            template=proposal["template"],
            canonical=proposal["canonical"],
            confidence=confidence,
            epistemic_state=proposal.get("epistemic_type", "asserted"),
            original_text=proposal.get("raw_text"),
            statement_text=proposal.get("raw_text"),
            epistemic_class=proposal.get("epistemic_class"),  # NEW: Pass through
            source={
                "input": "manas",
                "parser": "manas_v0.1",
                "parser_confidence": proposal.get("parser_confidence", 0.5),
            },
            polarity_value=proposal.get("polarity", 1),
        )
        
        # Add provenance
        belief.add_provenance(
            op="parsed",
            from_source="manas",
            score=proposal.get("parser_confidence", 0.5),
        )
        
        # Add to graph
        return self.add_belief(belief)
    
    def add_hypothetical(
        self,
        proposal: dict,
        confidence: float = 0.3,
    ) -> str:
        """
        Add belief as hypothetical (uncertain state).
        
        Used when Buddhi cannot decide between competing beliefs.
        
        EVIDENCE ACCUMULATION: Check for existing match first.
        
        Args:
            proposal: BeliefProposal dict
            confidence: low confidence for hypothetical
        
        Returns:
            belief ID
        """
        # Check if this belief already exists
        existing = self.find_matching_belief(proposal)
        
        if existing:
            # EVIDENCE ACCUMULATION
            existing.evidence_count += 1
            
            import math
            epsilon = 0.05
            boost = epsilon * math.log(existing.evidence_count)
            new_confidence = min(0.9, existing.confidence + boost)
            
            existing.confidence = new_confidence
            existing.updated_at = now_utc()
            
            existing.add_provenance(
                op="evidence_accumulation",
                from_source="repeated_assertion",
                score=boost,
                metadata={"evidence_count": existing.evidence_count}
            )
            
            self.stats["evidence_accumulated"] = self.stats.get("evidence_accumulated", 0) + 1
            self.updated_at = now_utc()
            
            return existing.id
        
        # Create new hypothetical belief
        belief = Belief(
            template=proposal["template"],
            canonical=proposal["canonical"],
            confidence=confidence,
            epistemic_state="HYPOTHESIS",
            original_text=proposal.get("raw_text"),
            statement_text=proposal.get("raw_text"),
            source={"input": "manas", "status": "uncertain"},
        )
        
        belief.add_provenance(
            op="hypothetical",
            from_source="buddhi",
            score=confidence,
        )
        
        return self.add_belief(belief)
    
    def add_unknown(
        self,
        proposal: dict,
    ) -> str:
        """
        Add belief as unknown (question/gap in knowledge).
        
        Used when Buddhi has no information to judge.
        
        EVIDENCE ACCUMULATION: Repeated unknowns should promote to hypothetical.
        
        Args:
            proposal: BeliefProposal dict
        
        Returns:
            belief ID
        """
        # Check if this belief already exists
        existing = self.find_matching_belief(proposal)
        
        if existing:
            # EVIDENCE ACCUMULATION for unknowns
            existing.evidence_count += 1
            
            import math
            epsilon = 0.05
            boost = epsilon * math.log(existing.evidence_count)
            new_confidence = min(0.9, existing.confidence + boost)
            
            existing.confidence = new_confidence
            existing.updated_at = now_utc()
            
            # If confidence crosses threshold, promote to hypothetical
            if new_confidence >= 0.3 and existing.epistemic_state == "UNKNOWN":
                existing.epistemic_state = "HYPOTHESIS"
            
            existing.add_provenance(
                op="evidence_accumulation",
                from_source="repeated_assertion",
                score=boost,
                metadata={"evidence_count": existing.evidence_count}
            )
            
            self.stats["evidence_accumulated"] = self.stats.get("evidence_accumulated", 0) + 1
            self.updated_at = now_utc()
            
            return existing.id
        
        # Create new unknown belief
        belief = Belief(
            template=proposal["template"],
            canonical=proposal["canonical"],
            confidence=0.0,
            epistemic_state="UNKNOWN",
            original_text=proposal.get("raw_text"),
            statement_text=proposal.get("raw_text"),
            source={"input": "manas", "status": "unknown"},
        )
        
        belief.add_provenance(
            op="unknown",
            from_source="buddhi",
            score=0.0,
        )
        
        return self.add_belief(belief)
    
    def compute_specificity(self, belief_id: str) -> float:
        """
        Compute specificity of a belief based on is_a depth.
        
        specificity = 1 / (1 + depth)
        
        Args:
            belief_id: belief to compute specificity for
        
        Returns:
            specificity score in [0, 1]
        """
        if belief_id not in self.beliefs:
            return 0.5
        
        belief = self.beliefs[belief_id]
        
        # Find is_a parents (generalization chain)
        visited = {belief_id}
        depth = 0
        current_level = {belief_id}
        
        while current_level and depth < 10:  # max depth 10
            next_level = set()
            for bid in current_level:
                # Get is_a edges (outgoing)
                parents = self.neighbors(bid, relation="is_a", direction="out")
                for parent in parents:
                    if parent not in visited:
                        visited.add(parent)
                        next_level.add(parent)
            
            if next_level:
                depth += 1
            current_level = next_level
        
        # specificity = 1 / (1 + depth)
        return 1.0 / (1.0 + depth)
    
    def compute_support_strength(self, belief_id: str) -> float:
        """
        Compute total support strength for a belief.
        
        Sum of weights from all incoming 'supports' edges.
        
        Args:
            belief_id: belief to compute support for
        
        Returns:
            support strength (>= 0)
        """
        edges = self.get_edges_in(belief_id, relation="supports")
        return sum(edge_data[1] for edge_data in edges)  # Second element is always weight
    
    def compute_conflict_strength(self, belief_id: str) -> float:
        """
        Compute total conflict strength for a belief.
        
        Sum of weights from all incoming 'contradicts' edges.
        
        Args:
            belief_id: belief to compute conflict for
        
        Returns:
            conflict strength (>= 0)
        """
        edges = self.get_edges_in(belief_id, relation="contradicts")
        return sum(edge_data[1] for edge_data in edges)  # Second element is always weight
    
    # ═══════════════════════════════════════════════════════════════
    # CONFIDENCE UPDATES
    # ═══════════════════════════════════════════════════════════════
    
    def update_confidence(
        self,
        belief_id: str,
        new_confidence: float,
    ):
        """
        Directly set confidence (use for manual updates).
        
        For Bayesian updates, use update_confidence_evidence.
        """
        if belief_id not in self.beliefs:
            raise KeyError(f"Belief {belief_id} not found")
        
        belief = self.beliefs[belief_id]
        belief.confidence = new_confidence
        self.stats["confidence_updates"] += 1
        self.updated_at = now_utc()
    
    def update_confidence_evidence(
        self,
        belief_id: str,
        evidence_score: float,
        min_conf: float = 0.01,
        max_conf: float = 0.99,
    ):
        """
        Update confidence using log-odds Bayesian update.
        
        Args:
            belief_id: belief to update
            evidence_score: evidence in log-odds space
            min_conf: minimum confidence bound
            max_conf: maximum confidence bound
        """
        if belief_id not in self.beliefs:
            raise KeyError(f"Belief {belief_id} not found")
        
        belief = self.beliefs[belief_id]
        new_conf = update_confidence_logodds(
            belief.confidence,
            evidence_score,
            min_conf,
            max_conf,
        )
        belief.confidence = new_conf
        self.stats["confidence_updates"] += 1
        self.updated_at = now_utc()
    
    # ═══════════════════════════════════════════════════════════════
    # ACTIVATION & DECAY
    # ═══════════════════════════════════════════════════════════════
    
    def decay_all_confidence(
        self,
        lambda_decay: float = 1e-6,
        reference_time: datetime | None = None,
    ):
        """
        Apply exponential confidence decay to all beliefs.
        
        Args:
            lambda_decay: decay rate (e.g., 1e-6)
            reference_time: reference time (defaults to now)
        """
        ref_time = reference_time or now_utc()
        
        for belief in self.beliefs.values():
            if not belief.active:
                continue
            
            delta_seconds = (ref_time - belief.updated_at).total_seconds()
            new_conf = decay_confidence(
                belief.confidence,
                lambda_decay,
                delta_seconds,
            )
            belief.confidence = new_conf
    
    def decay_all_activation(self, factor: float = 0.9):
        """
        Apply multiplicative decay to all activation levels.
        
        Args:
            factor: decay factor in [0, 1]
        """
        for belief in self.beliefs.values():
            belief.decay_activation(factor)
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════
    
    def count_beliefs(self, active_only: bool = True) -> int:
        """Count total beliefs."""
        if not active_only:
            return len(self.beliefs)
        return sum(1 for b in self.beliefs.values() if b.active)
    
    def count_edges(self, relation: str | None = None) -> int:
        """Count total edges."""
        if relation:
            return sum(
                len(targets)
                for targets in self.edges.get(relation, {}).values()
            )
        return sum(
            len(targets)
            for rel_dict in self.edges.values()
            for targets in rel_dict.values()
        )
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            **self.stats,
            "total_beliefs": len(self.beliefs),
            "active_beliefs": self.count_beliefs(active_only=True),
            "total_edges": self.count_edges(),
            "templates": len(self.template_index),
            "entities": len(self.entity_index),
            "predicates": len(self.predicate_index),
        }
    
    # ═══════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════════════════════════════
    
    def to_dict(self) -> dict:
        """Export graph to dictionary."""
        return {
            "beliefs": {
                bid: belief.to_dict()
                for bid, belief in self.beliefs.items()
            },
            "edges": {
                relation: {
                    src: [(tgt, w, eid) for tgt, w, eid in targets]
                    for src, targets in src_dict.items()
                }
                for relation, src_dict in self.edges.items()
            },
            "metadata": self.metadata,
            "stats": self.get_stats(),
        }
    

    
    # ═══════════════════════════════════════════════════════════════
    # DEBUG & REPR
    # ═══════════════════════════════════════════════════════════════
    
    def __repr__(self) -> str:
        return (
            f"<ChittaGraph "
            f"beliefs={self.count_beliefs(active_only=False)} "
            f"(active={self.count_beliefs(active_only=True)}) "
            f"edges={self.count_edges()}>"
        )
    
    # ═══════════════════════════════════════════════════════════════
    # QUERY INDEX (DEFERRED THINKING)
    # ═══════════════════════════════════════════════════════════════
    
    def register_query(self, query_proposal: dict, answer_belief_id: str | None = None):
        """
        Register a query in the index.
        
        When new beliefs arrive, system can re-evaluate matching queries.
        
        Args:
            query_proposal: BeliefProposal for the query
            answer_belief_id: ID of belief that answered it (None if unanswered)
        """
        predicates = query_proposal.get("predicates", [])
        
        for predicate in predicates:
            self.query_index[predicate].append({
                "proposal": query_proposal,
                "answer_id": answer_belief_id,
                "status": "answered" if answer_belief_id else "open",
                "timestamp": now_utc(),
            })
    
    def get_open_queries(self, predicate: str | None = None) -> list[dict]:
        """
        Get unanswered queries.
        
        Args:
            predicate: filter by predicate (None = all)
        
        Returns:
            list of open query dicts
        """
        if predicate:
            return [q for q in self.query_index.get(predicate, []) if q["status"] == "open"]
        
        # All open queries
        open_queries = []
        for queries in self.query_index.values():
            open_queries.extend(q for q in queries if q["status"] == "open")
        return open_queries
    
    def resolve_query(self, query_proposal: dict, answer_belief_id: str):
        """
        Mark a query as resolved.
        
        Args:
            query_proposal: the query that was resolved
            answer_belief_id: belief that answers it
        """
        predicates = query_proposal.get("predicates", [])
        
        for predicate in predicates:
            for query in self.query_index.get(predicate, []):
                if query["proposal"] == query_proposal and query["status"] == "open":
                    query["status"] = "answered"
                    query["answer_id"] = answer_belief_id
                    break
    
    # ═══════════════════════════════════════════════════════════════
    # SOFT-MERGE DETECTION (NOT AUTO-MERGE)
    # ═══════════════════════════════════════════════════════════════
    
    def detect_soft_merges(self, similarity_threshold: float = 0.85):
        """
        Detect near-duplicate beliefs (DO NOT auto-merge).
        
        Let Buddhi decide whether to merge.
        
        Args:
            similarity_threshold: threshold for flagging [0, 1]
        """
        self.merge_candidates.clear()
        
        belief_list = list(self.beliefs.values())
        
        for i, b1 in enumerate(belief_list):
            if not b1.active:
                continue
            
            for b2 in belief_list[i + 1:]:
                if not b2.active:
                    continue
                
                # Simple similarity: same template + overlapping entities
                if b1.template == b2.template:
                    entities1 = b1.entities
                    entities2 = b2.entities
                    
                    if entities1 and entities2:
                        overlap = len(entities1 & entities2)
                        union = len(entities1 | entities2)
                        jaccard = overlap / union if union > 0 else 0.0
                        
                        if jaccard >= similarity_threshold:
                            self.merge_candidates.append((b1.id, b2.id, jaccard))
    
    def get_merge_candidates(self) -> list[tuple[str, str, float]]:
        """
        Get list of near-duplicate belief pairs.
        
        Returns:
            list of (belief_id_1, belief_id_2, similarity_score)
        """
        return self.merge_candidates.copy()
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    
    def summary(self) -> str:
        """Human-readable summary."""
        stats = self.get_stats()
        return (
            f"ChittaGraph Summary\n"
            f"  Total beliefs: {stats['total_beliefs']}\n"
            f"  Active beliefs: {stats['active_beliefs']}\n"
            f"  Total edges: {stats['total_edges']}\n"
            f"  Templates: {stats['templates']}\n"
            f"  Entities: {stats['entities']}\n"
            f"  Predicates: {stats['predicates']}\n"
            f"  Beliefs added: {stats['beliefs_added']}\n"
            f"  Edges added: {stats['edges_added']}\n"
            f"  Confidence updates: {stats['confidence_updates']}"
        )
