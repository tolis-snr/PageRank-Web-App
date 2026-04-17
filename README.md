# Graph RAG vs PageRank — Bachelor Thesis Web App

An educational web application built for a Bachelor Thesis that compares three search/ranking strategies on a graph of research papers:

1. **PageRank** — authority scoring based purely on graph structure (citations)
2. **RAG (Semantic Search)** — relevance scoring based on NLP vector embeddings
3. **Hybrid Graph RAG** — a weighted combination of both

The repository currently contains the **Python/FastAPI backend**. The frontend (Vanilla HTML/CSS/JS + Cytoscape.js) is planned for a future step.

---

## Project Structure

```
.
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── models.py                  # Pydantic data models
│   ├── mock_data.py               # Sample 8-node research-paper graph
│   ├── api/
│   │   └── routes.py              # REST API endpoints
│   └── algorithms/
│       ├── pagerank.py            # PageRank via Power Iteration (NumPy)
│       ├── rag.py                 # Semantic similarity via sentence-transformers
│       └── hybrid.py             # Weighted combination of PR and RAG scores
├── Implementation/
│   ├── ARCHITECTURE.md            # High-level system architecture
│   ├── backend_documentation.md   # Detailed backend documentation
│   ├── implementation_plan.md     # Full project implementation plan
│   └── step1_implementation_plan.md
├── Scenario/
│   ├── scenario.md                # Walkthrough scenario (in Greek) demonstrating all three algorithms
│   ├── mockup.svg                 # UI mockup
│   └── svgviewer-png-output.png
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Server | [Uvicorn](https://www.uvicorn.org/) |
| Matrix math / PageRank | [NumPy](https://numpy.org/) |
| Text embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Data validation | [Pydantic](https://docs.pydantic.dev/) |
| Testing | [pytest](https://pytest.org/) |

---

## Algorithms

### 1. PageRank (`src/algorithms/pagerank.py`)
Calculates authority scores based on the citation graph structure, completely ignoring text content.
- Builds a column-stochastic adjacency matrix `M` from the edge list.
- Applies **Power Iteration** until convergence (damping factor `d = 0.85`, max 100 iterations, tolerance `1e-6`).
- Dangling nodes (no outbound links) are handled by distributing their probability uniformly across all nodes.
- Output scores are **Min-Max normalised** to `[0, 1]`.

### 2. Vector RAG Search (`src/algorithms/rag.py`)
Calculates semantic relevance between the user's query and each node's abstract.
- Encodes the query and all node abstracts using the `all-MiniLM-L6-v2` sentence-transformer model.
- Computes **Cosine Similarity** (`numpy.dot / norms`) between the query embedding and each abstract embedding.
- Output scores are **Min-Max normalised** to `[0, 1]`.

### 3. Hybrid Scoring (`src/algorithms/hybrid.py`)
Combines the two normalised scores into a single ranking signal:

```
hybrid_score = (weight_pagerank × pagerank_score) + (weight_rag × rag_score)
```

The caller supplies both weights; a 50/50 split is the default.

---

## API

### `POST /api/search`

Accepts a JSON body and returns all graph nodes ranked by their computed scores.

**Request body**
```json
{
  "query": "transformer neural networks",
  "weight_pagerank": 0.5,
  "weight_rag": 0.5
}
```

**Response** (`SearchResult`)
```json
{
  "query": "transformer neural networks",
  "results": [
    {
      "node": { "id": "1", "title": "Attention Is All You Need", "abstract": "..." },
      "pagerank_score": 0.82,
      "rag_score": 0.95,
      "hybrid_score": 0.885
    }
    ...
  ]
}
```

Results are sorted in **descending order** by `hybrid_score`.

### `GET /`

Health-check endpoint. Returns `{"message": "Welcome to Graph RAG vs PageRank Backend"}`.

Interactive API docs are available at `http://127.0.0.1:8000/docs` once the server is running.

---

## Mock Data (`src/mock_data.py`)

The backend ships with a built-in graph of **8 research papers** (nodes) and **7 citation links** (edges) for development and demonstration purposes:

| ID | Paper |
|---|---|
| 1 | Attention Is All You Need |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers |
| 3 | The PageRank Citation Ranking: Bringing Order to the Web |
| 4 | Graph Neural Networks: A Review of Methods and Applications |
| 5 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| 6 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks |
| 7 | Node2Vec: Scalable Feature Learning for Networks |
| 8 | Graph RAG: Graph based Retrieval Augmented Generation |

---

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/tolis-snr/PageRank-Web-App.git
cd PageRank-Web-App

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the server

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Interactive Swagger docs: `http://127.0.0.1:8000/docs`.

---

## License

See [LICENSE](LICENSE).
