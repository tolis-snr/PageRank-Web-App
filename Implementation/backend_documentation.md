# Backend Documentation - Graph RAG vs PageRank

This document outlines the architecture, logic, and data structures of the backend system implemented for the Graph RAG vs PageRank Bachelor Thesis application. The backend is built entirely with Python and FastAPI.

## 1. File Structure

All application source code resides in the `src/` directory to separate it from the root-level architectural and scenario documentation:

```text
src/
├── main.py                    # Application entry point and server setup
├── models.py                  # Pydantic schemas for data validation
├── mock_data.py               # Sample 8-node mock graph for testing
├── api/
│   └── routes.py              # FastAPI REST endpoints
└── algorithms/
    ├── pagerank.py            # NumPy implementation of PageRank
    ├── rag.py                 # Vector Embeddings and Cosine Similarity math
    └── hybrid.py              # Math for aggregating PR and RAG scores
```

---

## 2. Data Models (`src/models.py`)

The application enforces strict data types natively using `Pydantic` models:

- **`NodeBase`**: Represents a research paper. Contains `id`, `title`, and `abstract`.
- **`EdgeBase`**: Represents a citation. Contains `source` (citing node ID) and `target` (cited node ID).
- **`GraphBase`**: The entire network graph consisting of lists of `NodeBase` and `EdgeBase`.
- **`SearchQuery`**: The input JSON parsed from the frontend. Contains a text `query`, `weight_pagerank`, and `weight_rag`.
- **`NodeScore`**: A compound object containing the node mapping alongside its calculated `pagerank_score`, `rag_score`, and `hybrid_score`.
- **`SearchResult`**: The final JSON response wrapper returning the query string alongside the sorted `List[NodeScore]`.

---

## 3. Core Algorithms

The calculation logic is split into separate files in the `algorithms/` folder to ensure modularity.

### 3.1. PageRank (`pagerank.py`)
Calculates the Authority Score of a node relying purely on the network connections (edges/citations). it bypasses natural language and processes structural significance.
- **Math**: Leverages `numpy` matrix calculations. Constructs an Adjacency Matrix `M` mapped to indices. 
- **Operations**: Applies a Power Iteration approach looping until convergence, controlled by a damping factor (`d=0.85`, representing the probability a user continues clicking links). It properly accounts for dangling nodes (nodes with 0 out-links) to prevent probability leakage.
- **Normalization**: Applies Min-Max Normalization dynamically scaling the converging vector between `0.0` and `1.0`.

### 3.2. Vector RAG Search (`rag.py`)
Calculates the Semantic Relevance score comparing the user's string query against a node's abstract content via Natural Language Processing (NLP).
- **Math**: Driven by the `sentence-transformers` library utilizing the lightweight model `all-MiniLM-L6-v2`. This model generates dense embeddings (vectors of float numbers) representing semantic meaning.
- **Operations**: It embeds the user text `query` and all graph `abstract` outputs, utilizing Cosine Similarity algebra (`numpy.dot() / norms`) to measure the geometric angle/distance between the text elements. 
- **Normalization**: Applies Min-Max Normalization mapping similarity scores strictly from `0.0` to `1.0`.

### 3.3. Hybrid Weighting (`hybrid.py`)
Aggregates the structural authority mapping with semantic text mapping.
- **Math**: `(Weight_PR * PR_Score) + (Weight_RAG * RAG_Score)`. 
- **Functionality**: Loops over the previously normalized dictionaries, marrying the isolated scores mathematically to determine the single dynamic sizing ratio necessary for the frontend application rendering.

---

## 4. API Endpoints

The backend currently exposes one primary route required by the application via `src/api/routes.py`.

- **Endpoint**: `POST /api/search`
- **Request Body**:
  ```json
  {
    "query": "transformer neural networks",
    "weight_pagerank": 0.5,
    "weight_rag": 0.5
  }
  ```
- **Execution Flow**: 
  1. Pulls the base nodes and edges from `MOCK_GRAPH`.
  2. Executes `calculate_pagerank(nodes, edges)`.
  3. Executes `calculate_rag_scores(query.query, nodes)`.
  4. Calls `calculate_hybrid_scores()` using the provided floating point weights.
  5. Maps the independent numerical metrics to a `NodeScore` object.
  6. Reorders all nodes in descending order utilizing the computed `hybrid_score`.

---

## 5. Sub-system Configurations

- **CORS**: As the frontend will operate using Vanilla JS strictly uncoupled from the server routing, `main.py` is configured utilizing FastAPI's `CORSMiddleware`, configured globally to accept connections allowing headers, models, or configurations sent from diverse development ports (like `127.0.0.1:5500`). 

## 6. How to Run

Install all standard dependencies generated within the `requirements.txt` inside your virtual environment and launch the FastAPI Uvicorn engine pointing to the src app bundle:

```bash
uvicorn src.main:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to test endpoints visually.
