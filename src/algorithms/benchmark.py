import time
import numpy as np
from typing import List, Callable, Dict
from src.models import DatasetModel, MetricScore


def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 1) -> float:
    """Computes fraction of relevant documents in the top-k results."""
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return relevant_retrieved / k


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Computes Reciprocal Rank: 1 / rank position of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 3) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank position k."""
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0

    # Ideal DCG calculation
    ideal_rel = sorted([1.0 if doc_id in ground_truth_ids else 0.0 for doc_id in retrieved_ids], reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rel))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def run_benchmark_suite(dataset: DatasetModel, search_executor: Callable) -> List[MetricScore]:
    """
    Evaluates all 5 retrieval paradigms across standard ground-truth benchmark queries.
    """
    modes = [
        ("PageRank (Lexical + Authority)", "pagerank_lexical"),
        ("Dense Semantic RAG", "rag_semantic"),
        ("Hybrid (Linear Score Fusion)", "hybrid_linear"),
        ("Hybrid (Reciprocal Rank Fusion)", "hybrid_rrf")
    ]

    benchmark_results = []
    queries = dataset.benchmark_queries

    for display_name, mode in modes:
        p1_list, mrr_list, ndcg_list, latency_list = [], [], [], []

        for b_query in queries:
            start_time = time.perf_counter()
            ranked_nodes = search_executor(dataset, b_query.query, mode)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            retrieved_ids = [node.id for node in ranked_nodes]
            gt_ids = b_query.ground_truth_relevant_ids

            p1_list.append(precision_at_k(retrieved_ids, gt_ids, k=1))
            mrr_list.append(reciprocal_rank(retrieved_ids, gt_ids))
            ndcg_list.append(ndcg_at_k(retrieved_ids, gt_ids, k=3))
            latency_list.append(elapsed_ms)

        benchmark_results.append(MetricScore(
            algorithm_name=display_name,
            precision_at_1=round(float(np.mean(p1_list)), 3),
            mrr=round(float(np.mean(mrr_list)), 3),
            ndcg_at_3=round(float(np.mean(ndcg_list)), 3),
            latency_ms=round(float(np.mean(latency_list)), 2)
        ))

    return benchmark_results