"""
INDEXES — Advanced Indexing for Chitta

Provides specialized indexes for efficient belief retrieval:
- Composite indexes (multi-field queries)
- Range queries (confidence, time)
- Full-text search preparation
- Semantic similarity hooks

Philosophy:
- Indexes are read-optimized, write-tolerant
- All indexes are derived from beliefs (can be rebuilt)
- Extensible for future ML/embedding integration
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

try:
    from .belief import Belief
except ImportError:
    from belief import Belief


class CompositeIndex:
    """
    Multi-field composite index for complex queries.
    
    Example: (template, entity) -> {belief_ids}
    Enables fast queries like "all is_a relations involving 'bird'"
    """
    
    def __init__(self, fields: tuple[str, ...]):
        """
        Initialize composite index.
        
        Args:
            fields: tuple of field names to index on
        """
        self.fields = fields
        # key_tuple -> {belief_ids}
        self.index: dict[tuple, set[str]] = defaultdict(set)
    
    def add(self, belief: Belief):
        """Add belief to index."""
        key = self._extract_key(belief)
        if key:
            for k in key:
                self.index[k].add(belief.id)
    
    def remove(self, belief: Belief):
        """Remove belief from index."""
        key = self._extract_key(belief)
        if key:
            for k in key:
                self.index[k].discard(belief.id)
    
    def query(self, **field_values) -> set[str]:
        """
        Query index with field values.
        
        Example:
            index.query(template="is_a", entity="bird")
        
        Returns:
            set of belief IDs
        """
        # Build key tuple from provided fields
        key_parts = []
        for field in self.fields:
            if field in field_values:
                key_parts.append(field_values[field])
            else:
                return set()  # Missing required field
        
        return self.index.get(tuple(key_parts), set()).copy()
    
    def _extract_key(self, belief: Belief) -> list[tuple] | None:
        """Extract index keys from belief."""
        keys = []
        
        # Build key based on fields
        if self.fields == ("template", "entity"):
            # For each entity in belief
            for entity in belief.entities:
                keys.append((belief.template, entity))
        
        elif self.fields == ("template", "predicate"):
            # For each predicate in belief
            for predicate in belief.predicates:
                keys.append((belief.template, predicate))
        
        elif self.fields == ("epistemic_state", "template"):
            keys.append((belief.epistemic_state, belief.template))
        
        # Add more combinations as needed
        
        return keys if keys else None


class RangeIndex:
    """
    Range index for numeric/temporal queries.
    
    Supports:
    - Confidence ranges
    - Time ranges
    - Activation ranges
    """
    
    def __init__(self, field: str, extractor: Callable[[Belief], float | datetime]):
        """
        Initialize range index.
        
        Args:
            field: field name
            extractor: function to extract value from belief
        """
        self.field = field
        self.extractor = extractor
        # Sorted list of (value, belief_id) tuples
        self.entries: list[tuple[Any, str]] = []
        self._sorted = True
    
    def add(self, belief: Belief):
        """Add belief to index."""
        value = self.extractor(belief)
        self.entries.append((value, belief.id))
        self._sorted = False
    
    def remove(self, belief: Belief):
        """Remove belief from index."""
        value = self.extractor(belief)
        self.entries = [(v, bid) for v, bid in self.entries if bid != belief.id]
    
    def _ensure_sorted(self):
        """Ensure entries are sorted."""
        if not self._sorted:
            self.entries.sort(key=lambda x: x[0])
            self._sorted = True
    
    def range_query(
        self,
        min_value: Any | None = None,
        max_value: Any | None = None,
        include_min: bool = True,
        include_max: bool = True,
    ) -> set[str]:
        """
        Query for beliefs in range.
        
        Args:
            min_value: minimum value (None = no lower bound)
            max_value: maximum value (None = no upper bound)
            include_min: include minimum boundary
            include_max: include maximum boundary
        
        Returns:
            set of belief IDs
        """
        self._ensure_sorted()
        
        result = set()
        for value, belief_id in self.entries:
            # Check lower bound
            if min_value is not None:
                if include_min and value < min_value:
                    continue
                if not include_min and value <= min_value:
                    continue
            
            # Check upper bound
            if max_value is not None:
                if include_max and value > max_value:
                    break
                if not include_max and value >= max_value:
                    break
            
            result.add(belief_id)
        
        return result
    
    def top_k(self, k: int, reverse: bool = False) -> list[str]:
        """
        Get top-k beliefs by value.
        
        Args:
            k: number of results
            reverse: if True, get lowest k instead of highest
        
        Returns:
            list of belief IDs
        """
        self._ensure_sorted()
        
        if reverse:
            return [bid for _, bid in self.entries[:k]]
        else:
            return [bid for _, bid in self.entries[-k:]]


class FullTextIndex:
    """
    Simple full-text index for text search.
    
    Indexes:
    - original_text
    - statement_text
    
    Supports basic keyword search (can be extended to TF-IDF, embeddings).
    """
    
    def __init__(self):
        # token -> {belief_ids}
        self.token_index: dict[str, set[str]] = defaultdict(set)
    
    def add(self, belief: Belief):
        """Add belief to full-text index."""
        tokens = self._tokenize(belief)
        for token in tokens:
            self.token_index[token].add(belief.id)
    
    def remove(self, belief: Belief):
        """Remove belief from index."""
        tokens = self._tokenize(belief)
        for token in tokens:
            self.token_index[token].discard(belief.id)
    
    def search(self, query: str, mode: str = "any") -> set[str]:
        """
        Search for beliefs matching query.
        
        Args:
            query: search query string
            mode: "any" (OR) or "all" (AND)
        
        Returns:
            set of matching belief IDs
        """
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            return set()
        
        if mode == "any":
            # OR: any token matches
            result = set()
            for token in query_tokens:
                result.update(self.token_index.get(token, set()))
            return result
        
        elif mode == "all":
            # AND: all tokens must match
            result = None
            for token in query_tokens:
                token_matches = self.token_index.get(token, set())
                if result is None:
                    result = token_matches.copy()
                else:
                    result &= token_matches
            return result or set()
        
        return set()
    
    def _tokenize(self, belief: Belief) -> set[str]:
        """Extract tokens from belief."""
        text = " ".join(filter(None, [
            belief.original_text,
            belief.statement_text,
        ]))
        return self._tokenize_text(text)
    
    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        """Tokenize text into normalized tokens."""
        # Simple whitespace tokenization + lowercasing
        # Can be extended to stemming, lemmatization, etc.
        tokens = text.lower().split()
        # Remove punctuation
        tokens = [
            "".join(c for c in token if c.isalnum())
            for token in tokens
        ]
        return {t for t in tokens if t}


class SemanticIndex:
    """
    Placeholder for semantic/embedding-based similarity search.
    
    Future integration points:
    - Sentence embeddings (e.g., sentence-transformers)
    - Vector similarity search (cosine, dot product)
    - Approximate nearest neighbors (FAISS, Annoy)
    
    For now, provides stub interface.
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize semantic index.
        
        Args:
            embedding_dim: dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        # belief_id -> embedding vector
        self.embeddings: dict[str, list[float]] = {}
        self._enabled = False  # disabled until embeddings available
    
    def add(self, belief: Belief, embedding: list[float] | None = None):
        """
        Add belief with embedding.
        
        Args:
            belief: belief to add
            embedding: precomputed embedding (None = compute on-demand)
        """
        if embedding is not None:
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch: {len(embedding)} != {self.embedding_dim}"
                )
            self.embeddings[belief.id] = embedding
            self._enabled = True
    
    def remove(self, belief_id: str):
        """Remove belief from semantic index."""
        self.embeddings.pop(belief_id, None)
    
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find most similar beliefs.
        
        Args:
            query_embedding: query vector
            top_k: number of results
        
        Returns:
            list of (belief_id, similarity_score) tuples
        """
        if not self._enabled:
            return []
        
        # Compute cosine similarity
        similarities = []
        for belief_id, embedding in self.embeddings.items():
            sim = self._cosine_similarity(query_embedding, embedding)
            similarities.append((belief_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


class IndexManager:
    """
    Centralized index management for ChittaGraph.
    
    Provides:
    - Automatic index updates
    - Query routing
    - Index rebuilding
    """
    
    def __init__(self):
        # Composite indexes
        self.template_entity = CompositeIndex(("template", "entity"))
        self.template_predicate = CompositeIndex(("template", "predicate"))
        self.state_template = CompositeIndex(("epistemic_state", "template"))
        
        # Range indexes
        self.confidence_range = RangeIndex(
            "confidence",
            lambda b: b.confidence,
        )
        self.activation_range = RangeIndex(
            "activation",
            lambda b: b.activation,
        )
        self.time_range = RangeIndex(
            "updated_at",
            lambda b: b.updated_at,
        )
        
        # Full-text index
        self.fulltext = FullTextIndex()
        
        # Semantic index (stub)
        self.semantic = SemanticIndex()
        
        # All indexes (for batch operations)
        self.all_indexes = [
            self.template_entity,
            self.template_predicate,
            self.state_template,
            self.confidence_range,
            self.activation_range,
            self.time_range,
            self.fulltext,
            self.semantic,
        ]
    
    def add_belief(self, belief: Belief):
        """Add belief to all indexes."""
        for index in self.all_indexes:
            index.add(belief)
    
    def remove_belief(self, belief: Belief):
        """Remove belief from all indexes."""
        for index in self.all_indexes:
            if hasattr(index, 'remove'):
                index.remove(belief)
    
    def rebuild_all(self, beliefs: dict[str, Belief]):
        """Rebuild all indexes from scratch."""
        # Clear all indexes
        self.__init__()
        
        # Re-add all beliefs
        for belief in beliefs.values():
            self.add_belief(belief)
