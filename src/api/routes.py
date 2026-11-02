import numpy as np
from fastapi import APIRouter, HTTPException
from typing import List

from src.models import (
    SearchRequest, SearchResponse, NodeResult, NodeModel,
    BenchmarkResponse, MetricScore
)
from src.data_loader import data_loader
from src.algorithms.pagerank import (
    compute_classical_pagerank_lexical
)
from src.algorithms.rag import compute_semantic_rag_scores
from src.algorithms.hybrid import compute_linear_fusion, compute_rrf_fusion
from src.algorithms.benchmark import run_benchmark_suite
from src.algorithms.llm import generate_rag_response

router = APIRouter()


def execute_internal_ranking(
    dataset_id: str,
    query: str,
    algorithm_mode: str,
    custom_edges=None,
    weight_pr: float = 0.5,
    weight_rag: float = 0.5
) -> tuple[List[NodeResult], bool, float]:
    """
    Core orchestrator that runs all 5 retrieval algorithms and produces multi-metric rankings.
    Returns: (ranked_results, is_relevant_flag, max_raw_similarity)
    """
    dataset = data_loader.get_dataset(dataset_id)
    nodes = dataset.nodes
    # Use dynamically modified edges from the interactive sandbox if provided, else benchmark edges
    active_edges = custom_edges if custom_edges is not None else dataset.edges

    # 1. Classical PageRank with Lexical TF-IDF modulation
    norm_pr_lex, raw_pr_lex, raw_pr, raw_tfidf = compute_classical_pagerank_lexical(query, nodes, active_edges)

    # 2. Dense Semantic Search with raw threshold guardrail
    norm_rag, raw_rag, is_relevant, max_raw_sim = compute_semantic_rag_scores(
        query=query, nodes=nodes, dataset_id=dataset_id, similarity_threshold=0.30
    )

    # 3. Linear Score-Level Fusion
    linear_scores = compute_linear_fusion(norm_pr_lex, norm_rag, alpha=weight_pr)

    # 4. Reciprocal Rank Fusion
    norm_rrf, raw_rrf = compute_rrf_fusion(norm_pr_lex, norm_rag, k=60)

    # Build detailed result list with both normalized and raw metric breakdown
    results = []
    for i, node in enumerate(nodes):
        results.append(NodeResult(
            node=node,
            pagerank_score=round(float(norm_pr_lex[i]), 4),
            raw_pagerank_score=round(float(raw_pr_lex[i]), 6),
            rag_score=round(float(norm_rag[i]), 4),
            raw_rag_score=round(float(raw_rag[i]), 4),
            hybrid_linear_score=round(float(linear_scores[i]), 4),
            rrf_score=round(float(norm_rrf[i]), 4),
            raw_rrf_score=round(float(raw_rrf[i]), 6),
            rank_position=0
        ))
        
    # Determine sorting criterion based on selected algorithm mode
    sort_keys = {
        "pagerank_lexical": lambda x: x.pagerank_score,
        "rag_semantic": lambda x: x.rag_score,
        "hybrid_linear": lambda x: x.hybrid_linear_score,
        "hybrid_rrf": lambda x: x.rrf_score
    }
    key_func = sort_keys.get(algorithm_mode, lambda x: x.rrf_score)
    results.sort(key=key_func, reverse=True)

    # Assign final ordinal rank positions (1-indexed)
    for rank, item in enumerate(results, start=1):
        item.rank_position = rank

    return results, is_relevant, max_raw_sim


@router.get("/datasets")
def list_datasets():
    """Returns mapping of available domain datasets for frontend selection."""
    return data_loader.get_available_datasets()


@router.get("/dataset/{dataset_id}")
def get_dataset_details(dataset_id: str):
    """Returns full dataset graph structure (nodes, edges, description) for Cytoscape initialization."""
    try:
        dataset = data_loader.get_dataset(dataset_id)
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "description": dataset.description,
            "nodes": [n.model_dump() for n in dataset.nodes],
            "edges": [e.model_dump() for e in dataset.edges]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/search", response_model=SearchResponse)
def execute_search(request: SearchRequest):
    """
    Executes search across the specified domain dataset using the requested algorithm.
    Generates strict LLM synthesis if semantic relevance threshold is satisfied.
    """
    try:
        ranked_results, is_relevant, max_raw_sim = execute_internal_ranking(
            dataset_id=request.dataset_id,
            query=request.query,
            algorithm_mode=request.algorithm_mode,
            custom_edges=request.custom_edges,
            weight_pr=request.weight_pagerank,
            weight_rag=request.weight_rag
        )

        # Handle out-of-domain or irrelevant queries
        if not is_relevant:
            return SearchResponse(
                query=request.query,
                dataset_id=request.dataset_id,
                is_relevant=False,
                message=f"No relevant academic literature found in domain '{request.dataset_id}' (Max raw similarity: {max_raw_sim:.3f} < 0.30).",
                results=[],
                llm_response=None
            )

        # Generate grounded LLM response using the Top-1 ranked paper
        top_result = ranked_results[0]
        llm_answer = generate_rag_response(
            query=request.query,
            top_paper_title=top_result.node.title,
            top_paper_abstract=top_result.node.abstract
        )

        return SearchResponse(
            query=request.query,
            dataset_id=request.dataset_id,
            is_relevant=True,
            message="Search and ranking executed successfully.",
            results=ranked_results,
            llm_response=llm_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark(dataset_id: str = "nlp"):
    """
    Executes IR evaluation benchmark (P@1, MRR, NDCG@3, Latency) across ground-truth queries.
    """
    try:
        dataset = data_loader.get_dataset(dataset_id)

        # Search execution callback matching benchmark runner signature
        def search_adapter(d, q, mode):
            res, _, _ = execute_internal_ranking(d.dataset_id, q, mode)
            return [item.node for item in res]

        metrics = run_benchmark_suite(dataset, search_adapter)
        return BenchmarkResponse(
            dataset_id=dataset_id,
            total_queries=len(dataset.benchmark_queries),
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution error: {str(e)}")