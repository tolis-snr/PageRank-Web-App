import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained model statically so it only loads once
try:
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    # all-MiniLM-L6-v2 is small and fast for mock purposes
except Exception as e:
    print(f"Warning: Could not load sentence-transformers model. {e}")
    model = None

def calculate_rag_scores(query: str, nodes):
    """
    Calculates Cosine Similarity between query and node abstracts.
    Returns a dictionary of node_id -> normalized_score (0 to 1).
    """
    if not nodes or model is None:
        return {node.id: 0.0 for node in nodes}

    # Embed query
    query_embedding = model.encode(query)

    # Embed node abstracts
    abstracts = [node.abstract for node in nodes]
    node_embeddings = model.encode(abstracts)

    # Compute Cosine Similarity
    # np.dot(a, b) / (norm(a) * norm(b))
    query_norm = np.linalg.norm(query_embedding)
    scores = []
    
    if query_norm == 0:
        return {node.id: 0.0 for node in nodes}

    for emb in node_embeddings:
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            scores.append(0.0)
        else:
            sim = np.dot(query_embedding, emb) / (query_norm * emb_norm)
            scores.append(sim)
            
    scores = np.array(scores)

    # Min-Max Normalization
    min_val = np.min(scores)
    max_val = np.max(scores)
    
    if max_val - min_val > 0:
        normalized_scores = (scores - min_val) / (max_val - min_val)
    else:
        normalized_scores = np.zeros(len(scores))

    return {nodes[i].id: float(normalized_scores[i]) for i in range(len(nodes))}
