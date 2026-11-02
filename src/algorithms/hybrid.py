import numpy as np
from typing import List
from src.algorithms.pagerank import min_max_scale


def compute_linear_fusion(
    pagerank_scores: np.ndarray,
    rag_scores: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Computes weighted linear score fusion: Score = alpha * PR + (1 - alpha) * RAG.
    """
    combined = alpha * pagerank_scores + (1.0 - alpha) * rag_scores
    return min_max_scale(combined)


def compute_rrf_fusion(
    pr_scores: np.ndarray,
    rag_scores: np.ndarray,
    k: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).
    Returns: (normalized_rrf_scores, raw_rrf_scores)
    """
    n = len(pr_scores)
    pr_ranks = np.empty(n, dtype=int)
    rag_ranks = np.empty(n, dtype=int)

    pr_order = np.argsort(-pr_scores)
    rag_order = np.argsort(-rag_scores)

    for rank, idx in enumerate(pr_order, start=1):
        pr_ranks[idx] = rank

    for rank, idx in enumerate(rag_order, start=1):
        rag_ranks[idx] = rank

    raw_rrf_scores = np.zeros(n, dtype=float)
    for i in range(n):
        raw_rrf_scores[i] = (1.0 / (k + pr_ranks[i])) + (1.0 / (k + rag_ranks[i]))

    norm_rrf_scores = min_max_scale(raw_rrf_scores)
    return norm_rrf_scores, raw_rrf_scores