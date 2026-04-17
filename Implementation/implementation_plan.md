# Graph RAG vs PageRank - Step 1 Implementation Plan

This plan details the approach for Step 1: creating the FastAPI backend structure, installing dependencies, and writing the Python algorithms for PageRank and Cosine Similarity. We will not be implementing the frontend or the LLM integration in this step.

## User Review Required

Please review the proposed backend structure and algorithm approach below. Let me know if you would like me to adjust the directory structure, the chosen dependency versions, or the testing methodology before we begin coding. 

> [!IMPORTANT]  
> We will need to set up a Python virtual environment to keep these dependencies isolated. Do you have a preference for the package manager (e.g., standard `venv` + `pip`, `poetry`, or `conda`)?

## Proposed Changes

### Dependency & Environment Setup
We will initialize the Python environment and define the required libraries.

#### [NEW] `requirements.txt`
This file will include the core dependencies:
- `fastapi` & `uvicorn` (for the server)
- `numpy` (for matrix operations & PageRank)
- `sentence-transformers` (for creating RAG embeddings & cosine similarity)
- `pydantic` (for data validation)

### Backend Architecture

We will set up a modular FastAPI structure to keep the routing, logic, and data models cleanly separated.

#### [NEW] `main.py`
The entry point of the FastAPI application. It will initialize the app and register API routes.

#### [NEW] `models.py`
Will contain Pydantic schemas representing:
- `Node` (Title, Abstract)
- `Edge` (Source, Target)
- `Graph` (List of Nodes and Edges)
- `SearchQuery` (Text query, weights for PageRank and RAG)
- `NodeScore` (Calculated metrics)

#### [NEW] `algorithms/pagerank.py`
Will contain the `numpy`-based Power Iteration algorithm.
- Build an Adjacency Matrix from the given edges.
- Perform matrix multiplication until convergence.
- Apply Min-Max normalization so the resulting scores map uniformly between 0 and 1.

#### [NEW] `algorithms/rag.py`
Will handle vector math.
- Use `sentence-transformers` to convert the `SearchQuery` and all node `Abstract`s into embeddings.
- Implement Cosine Similarity using `numpy`.
- Apply Min-Max normalization to the similarities so they sit on the same `0-1` scale as PageRank.

#### [NEW] `algorithms/hybrid.py`
A simple utility to merge scores:
- Function: `(Weight_A * Normalized_PageRank) + (Weight_B * Normalized_RAG)`

#### [NEW] `api/routes.py`
Will contain the REST endpoints.
- `POST /search`: Receives a query, calculates scores over a mock graph, and returns nodes augmented with their individual and combined scores.

#### [NEW] `mock_data.py`
- We need a small, hardcoded graph of "Research Papers" (nodes with titles/abstracts and citation edges) to test our algorithms on before building actual databases or frontends.

## Open Questions

1. **Virtual Environment**: As noted above, which environment manager should I use (e.g., `python -m venv`)? 
2. **Mock Data Detail**: How many mock nodes/papers would you like to start with? I suggest around 5-10 nodes to make it easy to visibly verify the math.

## Verification Plan

### Automated/Unit Tests
- Write a basic test script or run a Python interactive shell to feed a specific dummy 3x3 adjacency matrix into the `pagerank.py` module and verify the math converges properly.
- Send mock text to `rag.py` ensuring embeddings are generated and similar texts produce higher normalizations.

### Manual Verification
- Start the server using `uvicorn main:app --reload`.
- Hit the `POST /search` endpoint using FastAPI's built-in Swagger UI (`/docs`) with various weight balances to visually inspect that the JSON response returns correct combined scores.
