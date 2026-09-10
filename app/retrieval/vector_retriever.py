from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.config.settings import settings
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.chroma_store import ChromaVectorStore


class BaseRetriever(ABC):
    """Abstract Base Class for vector retrieval services."""

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant context chunks for a given query string."""
        pass


class VectorRetriever(BaseRetriever):
    """
    Hybrid Semantic & Keyword Retriever utilizing BaseVectorStore abstraction.
    Blends dense vector similarity search with sparse keyword matching to maximize context relevance.
    """

    def __init__(self, vector_store: Optional[BaseVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else settings.TOP_K
        
        # 1. Fetch dense vector results from ChromaDB
        candidate_k = min(self.vector_store.get_count(), max(k * 4, 20))
        vector_results = self.vector_store.query(query_text=query, top_k=candidate_k)
        
        if not vector_results:
            return []

        # 2. Extract query keywords (excluding common stop words)
        stopwords = {
            "what", "is", "the", "in", "of", "to", "for", "a", "an", "and", "are", "this",
            "which", "document", "uploaded", "tell", "me", "does", "do", "how", "many"
        }
        words = [w.lower().strip("?,.!") for w in query.split() if w.lower().strip("?,.!") not in stopwords and len(w) > 2]

        if words:
            try:
                scored_items = []
                for v_rank, item in enumerate(vector_results):
                    doc = item.get("text", "")
                    meta = item.get("metadata", {})
                    doc_lower = doc.lower()
                    kw_hits = sum(1 for w in words if w in doc_lower)

                    v_score = 1.0 / (60 + v_rank)
                    combined_score = (0.6 * v_score) + (0.4 * (kw_hits / max(len(words), 1)))

                    item_copy = dict(item)
                    item_copy["hybrid_score"] = combined_score
                    scored_items.append(item_copy)

                if scored_items:
                    scored_items.sort(key=lambda x: x["hybrid_score"], reverse=True)
                    return scored_items[:k]
            except Exception:
                pass

        return vector_results[:k]


# Backwards compatibility alias
DenseRetriever = VectorRetriever
