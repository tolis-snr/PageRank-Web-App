# Graph RAG vs PageRank - Step 1 Implementation Plan

This plan covers the first build pass only: create the FastAPI backend skeleton, implement PageRank and semantic similarity, and verify the scoring pipeline on the provided 7-node research paper graph. The frontend, graph visualization, and LLM response box will come later.

## Starting Strategy

1. Build the smallest working FastAPI backend.
2. Implement the two scoring engines first: PageRank and semantic similarity.
3. Add the hybrid scorer on top of those two outputs.
4. Use the 7-node mock graph from `Scenario/scenario.md` as the initial test dataset.
5. Verify ranking behavior through the API before moving to Cytoscape.js or any UI work.

## Step 1 Scope

### Environment and Dependencies

Set up a local Python environment and install the core packages needed for the backend:

- `fastapi`
- `uvicorn`
- `numpy`
- `sentence-transformers`
- `pydantic`
- `pytest` for verification

Prefer a standard `venv` setup so the project stays easy to reproduce.

### Proposed Project Structure

Create a simple modular layout:

- `main.py` for FastAPI startup
- `models.py` for request and response schemas
- `mock_data.py` for the 7-node citation graph
- `algorithms/pagerank.py` for authority scoring
- `algorithms/rag.py` for semantic scoring
- `algorithms/hybrid.py` for score combination
- `api/routes.py` for the `/search` endpoint

### Core Algorithm Work

Implement the backend logic in this order:

1. Build the adjacency matrix from the citation edges.
2. Compute PageRank with power iteration until convergence.
3. Normalize PageRank scores to a `0-1` range.
4. Embed the query and paper abstracts.
5. Compute cosine similarity between the query and each paper.
6. Normalize semantic scores to a `0-1` range.
7. Combine both scores with the hybrid formula.

### Mock Data

Use the scenario graph as the first dataset:

- 7 research-paper nodes
- citation edges that make the graph intentionally asymmetric
- one irrelevant but semantically similar spam-like node
- one highly authoritative but semantically irrelevant node
- one clearly relevant and authoritative target paper

This dataset is useful because it demonstrates the thesis point clearly: PageRank alone can reward authority, RAG alone can be fooled by semantic overlap, and the hybrid model should balance both.

## Implementation Order

### Phase 1: Backend Skeleton

Set up the FastAPI app, wire the route module, and define the request and response models. At this stage the endpoint can return placeholder values while the scoring modules are being built.

### Phase 2: PageRank Module

Implement the citation graph conversion, the iterative rank calculation, and the normalization step. Validate the result with a tiny hand-checkable matrix before using the 7-node graph.

### Phase 3: Semantic Retrieval Module

Implement embedding generation and cosine similarity for the query against all abstracts. Keep the implementation isolated so the model can later be swapped if needed.

### Phase 4: Hybrid Scoring

Add a weighted merge of the normalized PageRank and semantic scores. Ensure the weights are explicit in the API request so the comparison stays transparent for the thesis demo.

### Phase 5: Search Endpoint

Expose a `POST /search` endpoint that accepts a query and weights, runs all three scorers, and returns ranked nodes with their individual scores and final hybrid score.

## Verification Plan

Test the scoring logic before adding any UI:

- Use a small synthetic matrix to confirm PageRank converges as expected.
- Run a query that should match the Transformer-related paper strongly and confirm the semantic scorer ranks it highly.
- Check that the hybrid output prefers the paper that is both relevant and authoritative.
- Verify the API response shape through FastAPI’s `/docs` page.

## Next Step After This Plan

Implement the backend structure and algorithms first. Once those are stable, move to the frontend with Cytoscape.js and score-driven node styling.
