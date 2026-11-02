import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models import NodeModel, EdgeModel


def build_adjacency_matrix(nodes: List[NodeModel], edges: List[EdgeModel]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Constructs a row-stochastic transition matrix from node and edge definitions.
    Rows represent source nodes; columns represent destination targets.
    """
    n = len(nodes)
    node_to_idx = {node.id: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((n, n), dtype=float)

    for edge in edges:
        if edge.source in node_to_idx and edge.target in node_to_idx:
            src_idx = node_to_idx[edge.source]
            tgt_idx = node_to_idx[edge.target]
            adj_matrix[src_idx, tgt_idx] = 1.0

    # Convert to stochastic transition matrix (handle dead-ends with uniform teleportation)
    transition_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_sum = np.sum(adj_matrix[i])
        if row_sum == 0:
            transition_matrix[i] = np.ones(n) / n
        else:
            transition_matrix[i] = adj_matrix[i] / row_sum

    return transition_matrix, node_to_idx


def power_iteration(
    transition_matrix: np.ndarray,
    personalization_vector: np.ndarray,
    damping_factor: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Computes stationary distribution vector pi = pi * P using Power Iteration.
    Supports arbitrary personalization (teleportation) vectors.
    """
    n = transition_matrix.shape[0]
    # Google Matrix formulation: G = d * P + (1 - d) * E
    # where E has rows equal to personalization_vector
    teleport_matrix = np.tile(personalization_vector, (n, 1))
    google_matrix = damping_factor * transition_matrix + (1.0 - damping_factor) * teleport_matrix

    pi = np.copy(personalization_vector)
    for _ in range(max_iter):
        pi_next = np.dot(pi, google_matrix)
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            break
        pi = pi_next

    return pi


def compute_tfidf_scores(query: str, nodes: List[NodeModel]) -> np.ndarray:
    """
    Calculates TF-IDF lexical relevance scores between query and document contents (Title + Abstract).
    """
    corpus = [f"{node.title}. {node.abstract}" for node in nodes]
    vectorizer = TfidfVectorizer(stop_words="english")
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    except ValueError:
        # Handles empty vocabularies or out-of-vocabulary queries gracefully
        similarities = np.zeros(len(nodes))

    return similarities


def compute_classical_pagerank_lexical(
    query: str,
    nodes: List[NodeModel],
    edges: List[EdgeModel],
    damping_factor: float = 0.85
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes classical 1998 Google search ranking: Lexical TF-IDF match modulated by PageRank authority.
    Returns: (normalized_combined, raw_combined, raw_pagerank, raw_tfidf)
    """
    n = len(nodes)
    transition_matrix, _ = build_adjacency_matrix(nodes, edges)
    uniform_teleport = np.ones(n) / n
    raw_pr = power_iteration(transition_matrix, uniform_teleport, damping_factor)
    raw_tfidf = compute_tfidf_scores(query, nodes)

    # Multiplicative combination (Brin & Page, 1998)
    combined_scores = raw_tfidf * raw_pr
    norm_combined = min_max_scale(combined_scores)

    return norm_combined, combined_scores, raw_pr, raw_tfidf


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    """Scales numeric values strictly into [0.0, 1.0]."""
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val > 0:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr)