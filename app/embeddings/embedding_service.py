import hashlib
from typing import List, Optional, Any
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from app.config.settings import settings
from app.core.exceptions import EmbeddingError


class EmbeddingService(EmbeddingFunction):
    """
    Dedicated Embedding Service providing consistent vector embeddings for document chunks and queries.
    Configurable via settings or direct parameters (supports Gemini, SentenceTransformers, and Mock providers).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.model_name = (model_name or settings.EMBEDDING_MODEL).strip()
        configured_gemini = (settings.GEMINI_API_KEY or "").strip()
        self.api_key = (api_key or configured_gemini).strip()
        self._st_model = None

        if self.provider == "auto":
            self.provider = "gemini"

        if self.provider == "gemini":
            if not self.api_key or self._is_placeholder_key(self.api_key):
                raise EmbeddingError(
                    "Gemini embeddings require GEMINI_API_KEY. Set it in local .env or Streamlit Secrets."
                )
            self.model_name = "gemini-embedding-001"
            self.embedding_dimension = 768
            try:
                from google import genai
                from google.genai import types
                self._genai = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=30_000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
                self._genai_types = types
            except ImportError as exc:
                raise RuntimeError("Gemini embedding support requires google-genai.") from exc
            except Exception as exc:
                raise RuntimeError("Gemini embedding client could not be initialized. Check SDK installation and configuration.") from exc
            return

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name or "all-MiniLM-L6-v2")
                self.embedding_dimension = self._st_model.get_sentence_embedding_dimension()
            except Exception as exc:
                raise RuntimeError("SentenceTransformer embedding model could not be loaded.") from exc
        elif self.provider not in ("sentence-transformers", "mock"):
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        else:
            self.provider = "mock"
            self.embedding_dimension = 384

    @staticmethod
    def _is_placeholder_key(value: str) -> bool:
        normalized = value.strip().lower()
        return any(marker in normalized for marker in ("your-", "your_", "paste-your", "placeholder"))

    def __call__(self, input: Documents) -> Embeddings:
        """ChromaDB EmbeddingFunction signature compatibility."""
        return self.embed_documents(list(input))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document chunk texts."""
        if not texts:
            return []

        if self.provider == "gemini":
            vectors: List[List[float]] = []
            # Keep remote embedding calls bounded while embedding chunks in batches.
            for start in range(0, len(texts), 64):
                try:
                    vectors.extend(self._gemini_embed_batch(texts[start:start + 64]))
                except EmbeddingError:
                    raise
                except Exception as exc:
                    raise self._embedding_error("Gemini", exc) from exc
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
            raise self._embedding_error("Gemini", exc) from exc

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
            raise self._embedding_error("Gemini", exc) from exc

    @staticmethod
    def _embedding_error(provider: str, exc: Exception) -> EmbeddingError:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (401, 403):
            message = f"{provider} rejected the configured API key. Check that the key is valid and enabled for embeddings."
        elif code == 429:
            message = f"{provider} embedding quota or rate limit was reached. Wait and retry."
        elif code == 503:
            message = f"{provider} embeddings are temporarily unavailable. Wait and retry."
        else:
            message = f"{provider} embedding request failed. Check provider configuration and network connectivity."
        return EmbeddingError(message)

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
