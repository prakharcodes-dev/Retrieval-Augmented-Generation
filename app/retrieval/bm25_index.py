"""Persistent, dependency-free BM25 index synchronized with the Chroma chunk IDs."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class BM25Index:
    """SQLite-backed sparse index. Chroma remains the source of truth for chunk text."""

    K1 = 1.5
    B = 0.75

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    length INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS terms (
                    term TEXT NOT NULL,
                    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, chunk_id)
                );
                CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS ingestion_status (
                    document_id TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_pages INTEGER NOT NULL DEFAULT 0,
                    processed_chunks INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database_path, timeout=10.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        return db

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [token.casefold() for token in _TOKEN_RE.findall(text or "")]

    def upsert_chunks(self, chunks: Iterable[Any]) -> int:
        rows = []
        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None) or chunk.get("chunk_id")
            text = getattr(chunk, "text", None) if not isinstance(chunk, dict) else chunk.get("text")
            metadata = getattr(chunk, "metadata", None) if not isinstance(chunk, dict) else chunk.get("metadata", {})
            if not chunk_id or not text:
                raise ValueError("Sparse index received a chunk without an ID or text.")
            metadata = metadata or {}
            rows.append((str(chunk_id), str(metadata.get("document_id", "")), str(text), metadata))

        if not rows:
            return 0
        with self._lock, self._connect() as db:
            for chunk_id, document_id, text, metadata in rows:
                db.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
                tokens = self._tokens(text)
                db.execute(
                    "INSERT INTO chunks(chunk_id, document_id, text, metadata, length) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, document_id, text, json.dumps(metadata, ensure_ascii=False), len(tokens)),
                )
                frequencies = Counter(tokens)
                db.executemany(
                    "INSERT INTO terms(term, chunk_id, frequency) VALUES (?, ?, ?)",
                    ((term, chunk_id, frequency) for term, frequency in frequencies.items()),
                )
        return len(rows)

    def search(self, query: str, top_k: int, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        terms = list(dict.fromkeys(self._tokens(query)))[:64]
        if not terms or top_k <= 0:
            return []
        placeholders = ",".join("?" for _ in terms)
        with self._lock, self._connect() as db:
            corpus = db.execute("SELECT COUNT(*) AS n, COALESCE(AVG(length), 0) AS avg_len FROM chunks").fetchone()
            total_docs, avg_len = int(corpus["n"]), float(corpus["avg_len"])
            if not total_docs:
                return []
            df_rows = db.execute(
                f"SELECT term, COUNT(*) AS df FROM terms WHERE term IN ({placeholders}) GROUP BY term", terms
            ).fetchall()
            document_frequency = {row["term"]: int(row["df"]) for row in df_rows}
            posting_rows = db.execute(
                f"SELECT t.term, t.frequency, c.chunk_id, c.document_id, c.text, c.metadata, c.length "
                f"FROM terms t JOIN chunks c ON c.chunk_id = t.chunk_id "
                f"WHERE t.term IN ({placeholders})",
                terms,
            ).fetchall()

        scores: Dict[str, float] = {}
        records: Dict[str, Dict[str, Any]] = {}
        for row in posting_rows:
            metadata = json.loads(row["metadata"])
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            term = row["term"]
            df = document_frequency.get(term, 0)
            if not df:
                continue
            tf = int(row["frequency"])
            doc_len = int(row["length"])
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            norm = tf + self.K1 * (1.0 - self.B + self.B * doc_len / max(avg_len, 1.0))
            scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0.0) + idf * (tf * (self.K1 + 1.0)) / norm
            records[row["chunk_id"]] = {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": metadata,
                "document_id": row["document_id"],
                "source": metadata.get("source", ""),
                "page_number": metadata.get("page_number", metadata.get("page")),
            }
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        for chunk_id in ranked:
            records[chunk_id]["sparse_score"] = scores[chunk_id]
        return [records[chunk_id] for chunk_id in ranked]

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def clear_chunks(self) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM chunks")

    def chunk_ids(self) -> set[str]:
        with self._connect() as db:
            return {row[0] for row in db.execute("SELECT chunk_id FROM chunks")}

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def set_status(
        self,
        document_id: str,
        file_hash: str,
        filename: str,
        status: str,
        processed_pages: int = 0,
        processed_chunks: int = 0,
        error: Optional[str] = None,
    ) -> None:
        if status not in {"pending", "processing", "completed", "failed"}:
            raise ValueError(f"Unknown ingestion status: {status}")
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO ingestion_status
                   (document_id, file_hash, filename, status, processed_pages, processed_chunks, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(document_id) DO UPDATE SET
                     file_hash=excluded.file_hash, filename=excluded.filename, status=excluded.status,
                     processed_pages=excluded.processed_pages, processed_chunks=excluded.processed_chunks,
                     error=excluded.error, updated_at=CURRENT_TIMESTAMP""",
                (document_id, file_hash, filename, status, processed_pages, processed_chunks, error),
            )

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT s.document_id, s.file_hash, s.filename, s.status, s.processed_pages,
                          s.processed_chunks, s.error, s.updated_at, COUNT(c.chunk_id) AS stored_chunks
                   FROM ingestion_status s LEFT JOIN chunks c ON c.document_id=s.document_id
                   GROUP BY s.document_id ORDER BY s.updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def remove_document_status(self, document_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM ingestion_status WHERE document_id = ?", (document_id,))

    def forget_document(self, document_id: str) -> None:
        """Remove document chunks and lifecycle metadata in one SQLite transaction."""
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            db.execute("DELETE FROM ingestion_status WHERE document_id = ?", (document_id,))

    def get_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM ingestion_status WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def reset(self) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM ingestion_status")

    def sync_from_store(self, store: Any) -> None:
        """Repair an interrupted cross-index write from Chroma, without loading all text at once."""
        vector_ids = set(store.iter_chunk_ids()) if hasattr(store, "iter_chunk_ids") else {
            item["chunk_id"] for item in store.iter_chunks()
        }
        if vector_ids == self.chunk_ids():
            return
        self.clear_chunks()
        for item in store.iter_chunks():
            self.upsert_chunks([item])
        known_docs = {item["document_id"] for item in self.list_documents()}
        for doc_id in {item.get("document_id", "") for item in store.iter_chunks()} - {""}:
            if doc_id not in known_docs:
                self.set_status(doc_id, "", doc_id, "completed")
