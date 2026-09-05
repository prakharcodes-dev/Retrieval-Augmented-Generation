import hashlib
from typing import List, Optional, Any
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from app.config.settings import settings


class EmbeddingService(EmbeddingFunction):
    """
    Dedicated Embedding Service providing consistent vector embeddings for document chunks and queries.
    Configurable via settings or direct parameters (supports OpenAI, SentenceTransformers, and Mock providers).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY
        self._st_model = None

        valid_openai_key = (
            self.api_key
            and self.api_key.startswith("sk-")
            and not self.api_key.startswith("sk-your-actual")
        )

        if self.provider == "openai" and valid_openai_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Notice: Failed to initialize OpenAI client ({e}). Falling back to mock embeddings.")
                self.provider = "mock"
        elif self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name or "all-MiniLM-L6-v2")
            except ImportError:
                print("Notice: sentence-transformers not installed. Falling back to mock embeddings.")
                self.provider = "mock"
        else:
            self.provider = "mock"

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB EmbeddingFunction signature compatibility."""
        return self.embed_documents(list(input))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document chunk texts."""
        if not texts:
            return []

        if self.provider == "openai" and hasattr(self, "client"):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                print(f"Error generating OpenAI embeddings ({e}). Falling back to deterministic mock.")
                return [self._mock_embed(text) for text in texts]

        elif self.provider == "sentence-transformers" and self._st_model:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                print(f"Error generating SentenceTransformer embeddings ({e}). Falling back to mock.")
                return [self._mock_embed(text) for text in texts]

        # Fallback Mock Provider
        return [self._mock_embed(text) for text in texts]

    def embed_query(self, input: Any = None, text: Any = None) -> Any:
        """Generates an embedding vector for query text (compatible with ChromaDB and direct callers)."""
        target = input if input is not None else text
        if not target:
            return []

        if isinstance(target, str):
            res = self.embed_documents([target])
            return res[0] if res else []
        elif isinstance(target, list):
            return self.embed_documents(target)

        return self.embed_documents([str(target)])

    def name(self) -> str:
        return f"embedding_service_{self.provider}_{self.model_name}"

    def _mock_embed(self, text: str) -> List[float]:
        """Generates a deterministic normalized pseudo-embedding vector of dimension 384 based on text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(float(b) / 255.0) - 0.5 for b in h]
        full_vec = (vec * (384 // len(vec) + 1))[:384]
        norm = sum(x * x for x in full_vec) ** 0.5 or 1.0
        return [x / norm for x in full_vec]
