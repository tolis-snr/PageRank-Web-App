def calculate_hybrid_scores(pagerank_scores, rag_scores, w_pagerank, w_rag):
    """
    Combines the normalized PageRank and RAG scores based on provided weights.
    Ensure scores are between 0 and 1.
    """
    hybrid_scores = {}
    for node_id in pagerank_scores:
        pr = pagerank_scores.get(node_id, 0.0)
        rag = rag_scores.get(node_id, 0.0)
        
        hybrid = (w_pagerank * pr) + (w_rag * rag)
        hybrid_scores[node_id] = hybrid
        
    return hybrid_scores
