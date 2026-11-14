from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Store implementations."""

    @abstractmethod
    def add_chunks(self, chunks: List["Chunk"], batch_size: int = 100) -> int:
        """Stores document chunks into the vector database."""
        pass

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries vector database for top_k relevant chunk results."""
        pass

    @abstractmethod
    def iter_chunks(self, batch_size: int = 200) -> Iterable[Dict[str, Any]]:
        """Iterates stored chunks in bounded batches."""
        pass

    @abstractmethod
    def iter_chunk_ids(self, batch_size: int = 1000) -> Iterable[str]:
        """Iterates stable chunk IDs without reading document text or embeddings."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str, file_hash: Optional[str] = None) -> bool:
        """Deletes chunks for one document, never the entire collection."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Returns total number of chunks stored in the vector database."""
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_existing_hashes(self) -> set:
        pass

    @abstractmethod
    def has_file_hash(self, file_hash: str) -> bool:
        pass

