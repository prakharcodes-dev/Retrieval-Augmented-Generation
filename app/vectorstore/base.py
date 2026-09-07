from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingestion.chunker import Chunk


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Store implementations."""

    @abstractmethod
    def add_chunks(self, chunks: List["Chunk"]) -> int:
        """Stores document chunks into the vector database."""
        pass

    @abstractmethod
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Queries vector database for top_k relevant chunk results."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Returns total number of chunks stored in the vector database."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clears/resets the vector store collection."""
        pass
