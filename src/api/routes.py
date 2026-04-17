from fastapi import APIRouter
from src.models import SearchQuery, SearchResult, NodeScore
from src.mock_data import MOCK_GRAPH
from src.algorithms.pagerank import calculate_pagerank
from src.algorithms.rag import calculate_rag_scores
from src.algorithms.hybrid import calculate_hybrid_scores

router = APIRouter()

@router.post("/search", response_model=SearchResult)
def search_graph(query: SearchQuery):
    nodes = MOCK_GRAPH.nodes
    edges = MOCK_GRAPH.edges
    
    # 1. PageRank Score (based purely on connections)
    pr_scores = calculate_pagerank(nodes, edges)
    
    # 2. RAG Score (Semantic similarity query vs abstract)
    rag_scores = calculate_rag_scores(query.query, nodes)
    
    # 3. Hybrid Score
    hybrid_scores = calculate_hybrid_scores(
        pr_scores, 
        rag_scores, 
        query.weight_pagerank, 
        query.weight_rag
    )
    
    # Compile Results
    results = []
    for node in nodes:
        node_score = NodeScore(
            node=node,
            pagerank_score=pr_scores.get(node.id, 0.0),
            rag_score=rag_scores.get(node.id, 0.0),
            hybrid_score=hybrid_scores.get(node.id, 0.0)
        )
        results.append(node_score)
        
    # Optional: Sort results by hybrid score descending
    results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
    return SearchResult(query=query.query, results=results)
