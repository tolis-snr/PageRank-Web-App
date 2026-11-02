import pytest
import numpy as np
from src.models import NodeModel, EdgeModel
from src.algorithms.pagerank import (
    build_adjacency_matrix,
    power_iteration,
    compute_classical_pagerank_lexical,
    min_max_scale
)
from src.algorithms.hybrid import compute_linear_fusion, compute_rrf_fusion
from src.algorithms.benchmark import precision_at_k, reciprocal_rank, ndcg_at_k


# --- Fixtures for reusable synthetic graph data ---

@pytest.fixture
def sample_nodes():
    return [
        NodeModel(id="N1", title="Transformers and Self-Attention", abstract="Deep attention mechanisms for translation.", year=2017),
        NodeModel(id="N2", title="BERT Language Representations", abstract="Bidirectional pre-training across layers.", year=2018),
        NodeModel(id="N3", title="Retrieval-Augmented Generation", abstract="Dense external memory indices for knowledge tasks.", year=2020),
        NodeModel(id="N4", title="General Relativity Physics", abstract="Curvature of spacetime field equations.", year=1916)
    ]


@pytest.fixture
def sample_edges():
    return [
        EdgeModel(source="N2", target="N1"),
        EdgeModel(source="N3", target="N1"),
        EdgeModel(source="N3", target="N2")
    ]


# ============================================================================
# 1. MATHEMATICAL VERIFICATION OF PAGERANK & MARKOV PROPERTIES
# ============================================================================

def test_pagerank_probability_conservation(sample_nodes, sample_edges):
    """
    Perron-Frobenius Guarantee:
    Verifies that the stationary distribution vector sums strictly to 1.0 (within float tolerance).
    """
    transition_matrix, _ = build_adjacency_matrix(sample_nodes, sample_edges)
    n = len(sample_nodes)
    uniform_teleport = np.ones(n) / n

    pi = power_iteration(transition_matrix, uniform_teleport, damping_factor=0.85)

    assert np.isclose(np.sum(pi), 1.0, atol=1e-5), "PageRank distribution must sum exactly to 1.0"
    assert np.all(pi > 0), "All stationary probabilities must be strictly positive (pi_i > 0)"


def test_pagerank_empty_graph_fallback(sample_nodes):
    """
    Edge Case: Fully disconnected graph (0 edges).
    Verifies that Power Iteration converges to a uniform distribution without division errors.
    """
    empty_edges = []
    transition_matrix, _ = build_adjacency_matrix(sample_nodes, empty_edges)
    n = len(sample_nodes)
    uniform_teleport = np.ones(n) / n

    pi = power_iteration(transition_matrix, uniform_teleport, damping_factor=0.85)

    expected_uniform = np.ones(n) / n
    assert np.allclose(pi, expected_uniform, atol=1e-5), "Disconnected graph must converge to uniform distribution"


def test_pagerank_directed_cycle_convergence(sample_nodes):
    """
    Topological Stress Test: Directed Cycle (N1 -> N2 -> N3 -> N1).
    Verifies that cyclic components converge symmetrically under uniform teleportation.
    """
    cycle_edges = [
        EdgeModel(source="N1", target="N2"),
        EdgeModel(source="N2", target="N3"),
        EdgeModel(source="N3", target="N1")
    ]
    sub_nodes = sample_nodes[:3]
    transition_matrix, _ = build_adjacency_matrix(sub_nodes, cycle_edges)
    uniform_teleport = np.ones(3) / 3

    pi = power_iteration(transition_matrix, uniform_teleport, damping_factor=0.85)
    assert np.isclose(pi[0], pi[1], atol=1e-4) and np.isclose(pi[1], pi[2], atol=1e-4), "Symmetric directed cycles must yield equal stationary probabilities"


# ============================================================================
# 2. LEXICAL WEIGHTING & NORMALIZATION VERIFICATION
# ============================================================================

def test_pagerank_lexical_zero_multiplication(sample_nodes, sample_edges):
    """
    Verifies the hard lexical filter: An irrelevant document with 0 keyword match
    must evaluate strictly to 0.0, regardless of topological link authority.
    """
    query = "astronomical telescopes and planetary orbits"
    norm_combined, raw_combined, raw_pr, raw_tfidf = compute_classical_pagerank_lexical(
        query, sample_nodes, sample_edges
    )
    assert np.all(norm_combined == 0.0)


def test_min_max_scale_edge_cases():
    """
    Verifies that Min-Max scaling safely handles uniform arrays without ZeroDivisionError.
    """
    # Case A: Array of identical values
    flat_arr = np.array([0.0, 0.0, 0.0])
    scaled_flat = min_max_scale(flat_arr)
    assert np.all(scaled_flat == 0.0), "Array of identical zeros must map safely to zeros"

    # Case B: Standard array
    arr = np.array([10.0, 20.0, 30.0])
    scaled = min_max_scale(arr)
    assert scaled[0] == 0.0
    assert scaled[2] == 1.0
    assert np.all((scaled >= 0.0) & (scaled <= 1.0)), "Values must lie strictly in [0.0, 1.0]"


# ============================================================================
# 3. RECIPROCAL RANK FUSION (RRF) PROPERTIES
# ============================================================================

def test_rrf_permutation_symmetry():
    """
    Symmetry Property:
    Verifies that RRF assigns identical scores to symmetric rank permutations: (1, 2) == (2, 1).
    """
    # System A scores: doc0 is Rank 1, doc1 is Rank 2
    pr_scores = np.array([0.95, 0.40])
    rag_scores = np.array([0.30, 0.88])
    norm_rrf, raw_rrf = compute_rrf_fusion(pr_scores, rag_scores, k=60)
    assert np.isclose(raw_rrf[0], raw_rrf[1], atol=1e-5)
    assert np.isclose(norm_rrf[0], norm_rrf[1], atol=1e-5)


def test_rrf_monotonicity():
    """
    Monotonicity Property:
    A document ranked (1, 1) across both engines must strictly outscore a document ranked (2, 2).
    """
    pr_scores = np.array([1.0, 0.5, 0.1])
    rag_scores = np.array([1.0, 0.5, 0.1])
    norm_rrf, raw_rrf = compute_rrf_fusion(pr_scores, rag_scores, k=60)
    assert raw_rrf[0] > raw_rrf[1] > raw_rrf[2]


# ============================================================================
# 4. INFORMATION RETRIEVAL EVALUATION METRICS VERIFICATION
# ============================================================================

def test_ir_metrics_calculations():
    """
    Verifies accurate mathematical computation of Precision@1, MRR, and NDCG@3.
    """
    retrieved = ["P1", "P2", "P3", "P4"]
    ground_truth = ["P2"]

    # P@1: First retrieved is P1 (not in ground truth) -> 0.0
    assert precision_at_k(retrieved, ground_truth, k=1) == 0.0

    # MRR: First relevant document (P2) is at rank position 2 -> 1/2 = 0.5
    assert reciprocal_rank(retrieved, ground_truth) == 0.5

    # NDCG@3: DCG = 0 + 1/log2(2+1) = 1/log2(3). IDCG = 1/log2(2). Ratio ~ 0.6309
    expected_ndcg = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
    assert np.isclose(ndcg_at_k(retrieved, ground_truth, k=3), expected_ndcg, atol=1e-4)