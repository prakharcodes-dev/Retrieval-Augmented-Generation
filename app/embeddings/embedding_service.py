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

        if self.provider == "auto":
            valid_openai_key = (
                self.api_key
                and self.api_key.startswith("sk-")
                and not self.api_key.startswith("sk-your-actual")
            )
            if valid_openai_key:
                self.provider = "openai"
            elif settings.GEMINI_API_KEY or settings.LLM_API_KEY:
                self.provider = "gemini"
                self.api_key = settings.GEMINI_API_KEY or settings.LLM_API_KEY
            else:
                raise ValueError("No embedding API key is configured. Set OPENAI_API_KEY or GEMINI_API_KEY in Streamlit secrets.")

        if self.provider == "gemini":
            self.api_key = self.api_key or settings.GEMINI_API_KEY or settings.LLM_API_KEY
            if not self.api_key:
                raise ValueError("Gemini embeddings require GEMINI_API_KEY in Streamlit secrets.")
            # The configured model name is OpenAI's default; use Gemini's actual model ID.
            self.model_name = "gemini-embedding-001"
            try:
                from google import genai
                from google.genai import types
                self._genai = genai.Client(api_key=self.api_key)
                self._genai_types = types
            except ImportError as exc:
                raise RuntimeError("Gemini embedding support requires google-genai.") from exc
            return

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
                raise RuntimeError("OpenAI embedding client could not be initialized.") from e
        elif self.provider == "openai":
            raise ValueError("OpenAI embeddings require a valid OPENAI_API_KEY.")
        elif self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name or "all-MiniLM-L6-v2")
            except Exception as exc:
                raise RuntimeError("SentenceTransformer embedding model could not be loaded.") from exc
        elif self.provider not in ("openai", "sentence-transformers", "mock"):
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
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
                raise RuntimeError(f"OpenAI embedding request failed: {e}") from e

        if self.provider == "gemini":
            vectors: List[List[float]] = []
            # Keep remote embedding calls bounded while embedding chunks in batches.
            for start in range(0, len(texts), 64):
                vectors.extend(self._gemini_embed_batch(texts[start:start + 64]))
            return vectors

        elif self.provider == "sentence-transformers" and self._st_model:
            try:
                embeddings = self._st_model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                raise RuntimeError(f"SentenceTransformer embedding request failed: {e}") from e

        # Fallback Mock Provider
        return [self._mock_embed(text) for text in texts]

    def embed_query(self, input: Any = None, text: Any = None) -> Any:
        """Generates an embedding vector for query text (compatible with ChromaDB and direct callers)."""
        target = input if input is not None else text
        if not target:
            return []

        if isinstance(target, str) and self.provider == "gemini":
            return self._gemini_embed(target, "RETRIEVAL_QUERY")
        if isinstance(target, str):
            res = self.embed_documents([target])
            return res[0] if res else []
        elif isinstance(target, list):
            return self.embed_documents(target)

        return self.embed_documents([str(target)])

    def _gemini_embed(self, text: str, task_type: str) -> List[float]:
        """Create semantic vectors through Gemini; never substitute random mock vectors on failure."""
        try:
            result = self._genai.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=self._genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            return result.embeddings[0].values
        except Exception as exc:
            raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

    def _gemini_embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            result = self._genai.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=self._genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            embeddings = [item.values for item in result.embeddings]
            if len(embeddings) != len(texts):
                raise ValueError("Gemini returned a different number of vectors than input texts.")
            return embeddings
        except Exception as exc:
            raise RuntimeError(f"Gemini batch embedding request failed: {exc}") from exc

    @staticmethod
    def name() -> str:
        return "rag_embedding_service"

    def get_config(self) -> dict:
        """Expose the embedding configuration to Chroma for persistence compatibility."""
        return {"provider": self.provider, "model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "EmbeddingService":
        return EmbeddingService(
            provider=config["provider"],
            model_name=config["model_name"],
        )

    def _mock_embed(self, text: str) -> List[float]:
        """Generates a deterministic normalized pseudo-embedding vector of dimension 384 based on text hash."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(float(b) / 255.0) - 0.5 for b in h]
        full_vec = (vec * (384 // len(vec) + 1))[:384]
        norm = sum(x * x for x in full_vec) ** 0.5 or 1.0
        return [x / norm for x in full_vec]
