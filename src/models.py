from pydantic import BaseModel
from typing import List

class NodeBase(BaseModel):
    id: str
    title: str
    abstract: str

class EdgeBase(BaseModel):
    source: str
    target: str

class GraphBase(BaseModel):
    nodes: List[NodeBase]
    edges: List[EdgeBase]

class SearchQuery(BaseModel):
    query: str
    weight_pagerank: float = 0.5
    weight_rag: float = 0.5

class NodeScore(BaseModel):
    node: NodeBase
    pagerank_score: float = 0.0
    rag_score: float = 0.0
    hybrid_score: float = 0.0

class SearchResult(BaseModel):
    query: str
    results: List[NodeScore]
