import os
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.models import NodeModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model_instance = None

# In-memory RAM cache for document embeddings: {dataset_id: np.ndarray}
_document_embeddings_cache: Dict[str, np.ndarray] = {}


def get_embedding_model() -> SentenceTransformer:
    """Retrieves or lazily loads the shared SentenceTransformer singleton."""
    global _model_instance
    if _model_instance is None:
        print(f"[RAG] Loading dense embedding model into memory: {EMBEDDING_MODEL_NAME}...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def get_cached_doc_embeddings(dataset_id: str, nodes: List[NodeModel]) -> np.ndarray:
    """
    Retrieves pre-computed document embeddings from memory.
    Encodes the dataset corpus strictly once per server lifetime.
    """
    if dataset_id not in _document_embeddings_cache:
        model = get_embedding_model()
        corpus = [f"{node.title}. {node.abstract}" for node in nodes]
        print(f"[RAG] Computing and caching embeddings for dataset: '{dataset_id}' ({len(nodes)} papers)...")
        embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        _document_embeddings_cache[dataset_id] = embeddings
    return _document_embeddings_cache[dataset_id]


def compute_semantic_rag_scores(
    query: str,
    nodes: List[NodeModel],
    dataset_id: str = "nlp",
    similarity_threshold: float = 0.30
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    """
    Computes dense semantic scores with in-memory caching.
    Returns: (normalized_scores, raw_similarities, is_relevant_flag, max_raw_similarity)
    """
    model = get_embedding_model()
    doc_embeddings = get_cached_doc_embeddings(dataset_id, nodes)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    raw_similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    max_raw_score = float(np.max(raw_similarities))

    if max_raw_score < similarity_threshold:
        return np.zeros(len(nodes)), raw_similarities, False, max_raw_score

    min_val, max_val = np.min(raw_similarities), np.max(raw_similarities)
    if max_val - min_val > 0:
        normalized_scores = (raw_similarities - min_val) / (max_val - min_val)
    else:
        normalized_scores = np.ones(len(nodes))

    return normalized_scores, raw_similarities, True, max_raw_score