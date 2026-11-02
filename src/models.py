from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class NodeModel(BaseModel):
    """Represents an individual academic paper node."""
    id: str
    title: str
    abstract: str
    year: int


class EdgeModel(BaseModel):
    """Represents a directed citation edge from source to target."""
    source: str
    target: str


class BenchmarkQueryModel(BaseModel):
    """Represents ground-truth test queries for IR evaluation."""
    query: str
    ground_truth_relevant_ids: List[str]


class DatasetModel(BaseModel):
    """Complete dataset representation containing graph topology and benchmarks."""
    dataset_id: str
    name: str
    description: str
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    benchmark_queries: List[BenchmarkQueryModel]


class SearchRequest(BaseModel):
    """Search request payload supporting dynamic topology and multiple ranking algorithms."""
    dataset_id: str = "nlp"
    query: str
    algorithm_mode: str = Field(
        default="hybrid_rrf",
        description="One of: 'pagerank_lexical', 'rag_semantic', 'hybrid_linear', 'hybrid_rrf'"
    )
    custom_edges: Optional[List[EdgeModel]] = None  # Topology sandbox support
    weight_pagerank: float = 0.5                     # Only applicable for hybrid_linear
    weight_rag: float = 0.5                          # Only applicable for hybrid_linear


class NodeResult(BaseModel):
    """Scored node metadata and multi-metric breakdown for UI explainability."""
    node: NodeModel
    pagerank_score: float          # Normalized [0.0, 1.0] for visual bar & graph sizing
    raw_pagerank_score: float      # Raw TF-IDF * PageRank product
    rag_score: float               # Normalized [0.0, 1.0]
    raw_rag_score: float           # Raw Cosine Similarity [-1.0, 1.0]
    hybrid_linear_score: float     # Normalized [0.0, 1.0]
    rrf_score: float               # Normalized [0.0, 1.0]
    raw_rrf_score: float           # Raw Reciprocal Rank sum (e.g. ~0.0325)
    rank_position: int


class SearchResponse(BaseModel):
    """Full search response including rankings, threshold status, and LLM output."""
    query: str
    dataset_id: str
    is_relevant: bool
    message: Optional[str] = None
    results: List[NodeResult]
    llm_response: Optional[str] = None


class MetricScore(BaseModel):
    """Evaluation metrics for a specific retrieval algorithm."""
    algorithm_name: str
    precision_at_1: float
    mrr: float
    ndcg_at_3: float
    latency_ms: float


class BenchmarkResponse(BaseModel):
    """Comparative evaluation benchmark results across all 5 retrieval methods."""
    dataset_id: str
    total_queries: int
    metrics: List[MetricScore]