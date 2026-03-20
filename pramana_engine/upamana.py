"""
Upamāna (analogy/comparison) mapping module.

Uses NumPy to build similarity matrices for analogical reasoning.
Supports:
- Cosine similarity for dense vector inputs.
- Jaccard similarity for binary/set inputs.
- Convenience method to extract top-k most similar analogies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute an n×n cosine similarity matrix.

    Parameters
    ----------
    vectors:
        2-D array of shape (n, d) where each row is a feature vector.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (n, n) with values in [-1, 1].
    """
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1e-10, norms)
    normalised = vectors / norms
    return normalised @ normalised.T


def jaccard_similarity_matrix(binary_matrix: np.ndarray) -> np.ndarray:
    """Compute an n×n Jaccard similarity matrix for binary feature vectors.

    Jaccard similarity = |A ∩ B| / |A ∪ B|.

    Parameters
    ----------
    binary_matrix:
        2-D array of shape (n, d) with values in {0, 1}.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (n, n) with values in [0, 1].
    """
    mat = np.asarray(binary_matrix, dtype=bool)
    n = mat.shape[0]
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            intersection = np.logical_and(mat[i], mat[j]).sum()
            union = np.logical_or(mat[i], mat[j]).sum()
            val = float(intersection) / float(union) if union > 0 else 0.0
            sim[i, j] = val
            sim[j, i] = val
    return sim


def top_k_analogies(
    query_idx: int,
    similarity_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    k: int = 3,
) -> List[Tuple[int, float, str]]:
    """Return the top-k most similar items to *query_idx*.

    Parameters
    ----------
    query_idx:
        Row index of the query item in *similarity_matrix*.
    similarity_matrix:
        Pre-computed similarity matrix.
    labels:
        Optional list of string labels for each row.  Falls back to
        stringified indices when ``None``.
    k:
        Number of analogies to return (excluding the query itself).

    Returns
    -------
    List[Tuple[int, float, str]]
        Sorted list of ``(index, similarity_score, label)`` tuples,
        most-similar first.
    """
    n = similarity_matrix.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]

    row = similarity_matrix[query_idx].copy()
    row[query_idx] = -np.inf  # exclude self

    top_indices = np.argsort(row)[::-1][:k]
    return [(int(idx), float(row[idx]), labels[idx]) for idx in top_indices]
