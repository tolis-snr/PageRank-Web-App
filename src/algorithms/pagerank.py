import numpy as np

def calculate_pagerank(nodes, edges, max_iterations=100, d=0.85, tol=1.0e-6):
    """
    Calculates PageRank for a given list of nodes and edges using Power Iteration.
    Returns a dictionary of node_id -> normalized_score (0 to 1).
    """
    if not nodes:
        return {}

    n = len(nodes)
    node_ids = [node.id for node in nodes]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    # Initialize adjacency matrix (M)
    M = np.zeros((n, n))
    
    # Track out-links
    out_degree = np.zeros(n)
    for edge in edges:
        # Avoid KeyError if edge has unknown id
        if edge.source in node_index and edge.target in node_index:
            src_idx = node_index[edge.source]
            tgt_idx = node_index[edge.target]
            M[tgt_idx, src_idx] += 1.0 # edge from source to target
            out_degree[src_idx] += 1
            
    # Normalize columns of M
    for i in range(n):
        if out_degree[i] > 0:
            M[:, i] = M[:, i] / out_degree[i]
        else:
            # Handle dangling nodes (pages with no outlinks) by adding links to all pages
            M[:, i] = 1.0 / n

    # Power Iteration
    # Initialize PageRank vector v evenly
    v = np.ones(n) / n
    damping_vector = np.ones(n) * ((1 - d) / n)

    for _ in range(max_iterations):
        prev_v = v
        v = d * np.dot(M, v) + damping_vector
        # Check for convergence
        if np.linalg.norm(v - prev_v, 1) < tol:
            break

    # Min-Max Normalization (0 to 1)
    min_val = np.min(v)
    max_val = np.max(v)
    
    if max_val - min_val > 0:
        normalized_v = (v - min_val) / (max_val - min_val)
    else:
        normalized_v = np.zeros(n) # If all are same, set to 0 (or 1)

    return {node_ids[i]: float(normalized_v[i]) for i in range(n)}
