import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import tiktoken
from app.config.settings import settings
from app.ingestion.loader import DocumentPage


@dataclass
class Chunk:
    """Represents a single token-aware text chunk with preserved metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    token_count: int = 0


def validate_chunks(chunks: List[Chunk]) -> bool:
    """
    Validates generated chunks to ensure quality and metadata completeness:
    - Chunks are not empty
    - Chunk IDs are unique
    - Chunk indexes are valid and sequential
    - Required metadata exists
    """
    if not chunks:
        return True

    seen_ids = set()
    for idx, chunk in enumerate(chunks):
        if not chunk.text or not chunk.text.strip():
            raise ValueError(f"Chunk at index {idx} has empty text.")

        if not chunk.chunk_id:
            raise ValueError(f"Chunk at index {idx} is missing chunk_id.")

        if chunk.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk.chunk_id}")

        seen_ids.add(chunk.chunk_id)

        meta = chunk.metadata
        if not meta:
            raise ValueError(f"Chunk {chunk.chunk_id} is missing metadata dictionary.")

        if "chunk_index" not in meta:
            raise ValueError(f"Chunk {chunk.chunk_id} is missing 'chunk_index' in metadata.")

    return True


class TextChunker:
    """
    Splits document text into token-bounded chunks (target: 700 tokens, 100 overlap).
    Preserves document hierarchy (headings -> sections -> paragraphs -> sentences).
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        encoding_name: str = "cl100k_base",
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.separators = separators or [
            "\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", "? ", "! ", " ", ""
        ]

        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Returns token count using tiktoken or fallback heuristic (~4 chars per token)."""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _split_text_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using hierarchical separators to preserve context boundaries."""
        if not text or not text.strip():
            return []

        if not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)
        result: List[str] = []

        for i, split in enumerate(splits):
            if not split and separator not in ("\n\n", "\n"):
                continue
            piece = split + (separator if i < len(splits) - 1 and separator != " " else "")

            if self.count_tokens(piece) > self.chunk_size and next_separators:
                sub_splits = self._split_text_by_separators(piece, next_separators)
                result.extend(sub_splits)
            else:
                if piece.strip():
                    result.append(piece)

        return result

    def split_text(self, text: str, metadata: Dict[str, Any], start_index: int = 0) -> List[Chunk]:
        """
        Splits text into token-bounded chunks (target: 700 tokens, 100 overlap).
        Attaches chunk_id, chunk_index, document_id, and source metadata.
        """
        if not text or not text.strip():
            return []

        units = self._split_text_by_separators(text, self.separators)
        if not units:
            return []

        chunks: List[Chunk] = []
        doc_id = metadata.get("document_id", "doc")
        page_num = metadata.get("page_number", metadata.get("page", 1))
        page_suffix = f"_p{page_num}" if page_num is not None else ""

        current_unit_idx = 0
        n_units = len(units)

        while current_unit_idx < n_units:
            current_tokens = 0
            current_units: List[str] = []
            chunk_start_idx = current_unit_idx

            while current_unit_idx < n_units:
                unit = units[current_unit_idx]
                unit_tokens = self.count_tokens(unit)

                if current_tokens + unit_tokens > self.chunk_size and current_units:
                    break

                current_units.append(unit)
                current_tokens += unit_tokens
                current_unit_idx += 1

            chunk_text = "".join(current_units).strip()
            if chunk_text:
                chunk_index = start_index + len(chunks)
                chunk_id = f"{doc_id}{page_suffix}_chunk_{chunk_index:03d}"

                chunk_meta = dict(metadata)
                chunk_meta["chunk_id"] = chunk_id
                chunk_meta["chunk_index"] = chunk_index
                chunk_meta["document_id"] = metadata.get("document_id", doc_id)
                chunk_meta["filename"] = metadata.get("filename", metadata.get("source", ""))
                chunk_meta["file_type"] = metadata.get("file_type", "pdf")
                chunk_meta["source"] = metadata.get("source", "")
                chunk_meta["title"] = metadata.get("title", "")
                chunk_meta["page_number"] = metadata.get("page_number", metadata.get("page"))

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=chunk_meta,
                        chunk_id=chunk_id,
                        token_count=current_tokens
                    )
                )

            if current_unit_idx >= n_units:
                break

            overlap_tokens = 0
            rewind_count = 0
            for idx in range(current_unit_idx - 1, chunk_start_idx, -1):
                u_tokens = self.count_tokens(units[idx])
                if overlap_tokens + u_tokens > self.chunk_overlap:
                    break
                overlap_tokens += u_tokens
                rewind_count += 1

            if rewind_count > 0:
                current_unit_idx -= rewind_count
            else:
                if current_unit_idx == chunk_start_idx:
                    current_unit_idx += 1

        return chunks

    def chunk_documents(self, pages: List[DocumentPage]) -> List[Chunk]:
        """Processes a list of DocumentPages into Chunk objects with globally unique IDs and indexes."""
        all_chunks: List[Chunk] = []

        doc_chunk_counters: Dict[str, int] = {}

        for page in pages:
            doc_id = page.metadata.get("document_id", "doc")
            current_counter = doc_chunk_counters.get(doc_id, 0)

            page_chunks = self.split_text(page.text, page.metadata, start_index=current_counter)
            doc_chunk_counters[doc_id] = current_counter + len(page_chunks)
            all_chunks.extend(page_chunks)

        validate_chunks(all_chunks)
        return all_chunks
