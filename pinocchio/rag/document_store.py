"""Document Store — ingest, chunk, embed, and retrieve documents for RAG.

Supports text-based documents (TXT, MD, PDF text extraction, CSV, JSON).
Documents are split into chunks, optionally embedded via the EmbeddingClient,
and stored in a SQLite database for persistent retrieval.

Usage::

    store = DocumentStore(data_dir="data")
    store.set_embedding_client(embedding_client)

    # Ingest a document
    doc_id = store.ingest("notes.md")

    # Search
    results = store.search("quantum entanglement", top_k=5)
    for chunk in results:
        print(chunk.text[:100], chunk.score)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    """A chunk of a document with optional embedding."""

    chunk_id: str = ""
    doc_id: str = ""
    source: str = ""       # file path or URL
    text: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    score: float = 0.0     # relevance score (filled by search)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the chunk to a JSON-friendly dictionary."""
        d = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata,
        }
        if self.embedding:
            d["embedding"] = self.embedding
        return d


# ---------------------------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 512   # tokens (approx)
_DEFAULT_OVERLAP = 64       # tokens overlap between chunks


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English, ~1.5 per char for CJK."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    words = len(text.split())
    return int(cjk * 1.5 + words * 1.3)


def _chunk_text(
    text: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks by paragraph/sentence boundaries."""
    if _estimate_tokens(text) <= chunk_size:
        return [text]

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current_tokens + para_tokens > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Keep overlap
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                pt = _estimate_tokens(p)
                if overlap_tokens + pt > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += pt
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Document Store
# ---------------------------------------------------------------------------

class DocumentStore:
    """Persistent document store with chunking and vector search."""

    def __init__(self, data_dir: str = "data") -> None:
        """Initialise the store, creating the SQLite DB on first use.

        Args:
            data_dir: Directory for the SQLite database file.
        """
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "rag_documents.db"
        self._embedding_client: Any | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Create the SQLite schema if it doesn't exist."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)
            """)
            conn.commit()

    def set_embedding_client(self, client: Any) -> None:
        """Set the embedding client for vector search."""
        self._embedding_client = client

    # -- Ingestion ------------------------------------------------------

    def ingest(
        self,
        path: str,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Ingest a document: read, chunk, embed, store. Returns doc_id."""
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = self._read_document(p)
        if not text.strip():
            raise ValueError(f"Document is empty: {path}")

        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Check for duplicate
        with sqlite3.connect(str(self._db_path)) as conn:
            existing = conn.execute(
                "SELECT doc_id FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if existing:
                return existing[0]  # already ingested

        doc_id = str(uuid.uuid4())[:8]
        chunks = _chunk_text(text, chunk_size, overlap)

        # Embed chunks if client available
        embeddings: list[list[float]] = []
        if self._embedding_client:
            try:
                for chunk in chunks:
                    emb = self._embedding_client.embed(chunk)
                    embeddings.append(emb)
            except Exception:
                embeddings = []

        # Store in SQLite
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, source, content_hash, chunk_count, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, str(p), content_hash, len(chunks),
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                emb_blob = (
                    json.dumps(embeddings[i]).encode()
                    if i < len(embeddings) else None
                )
                conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, i, chunk_text, emb_blob),
                )
            conn.commit()

        return doc_id

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> str:
        """Ingest raw text directly (no file needed). Returns doc_id."""
        if not text.strip():
            raise ValueError("Text is empty")

        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_id = str(uuid.uuid4())[:8]
        chunks = _chunk_text(text, chunk_size, overlap)

        embeddings: list[list[float]] = []
        if self._embedding_client:
            try:
                for chunk in chunks:
                    embeddings.append(self._embedding_client.embed(chunk))
            except Exception:
                embeddings = []

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, source, content_hash, chunk_count) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, source, content_hash, len(chunks)),
            )
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                emb_blob = (
                    json.dumps(embeddings[i]).encode()
                    if i < len(embeddings) else None
                )
                conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, i, chunk_text, emb_blob),
                )
            conn.commit()

        return doc_id

    # -- Search ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Search for relevant chunks using vector similarity + keyword fallback."""
        # Try vector search first
        if self._embedding_client:
            try:
                query_emb = self._embedding_client.embed(query)
                return self._vector_search(query_emb, top_k, doc_id)
            except Exception:
                pass

        # Fallback: keyword search
        return self._keyword_search(query, top_k, doc_id)

    def _vector_search(
        self,
        query_emb: list[float],
        top_k: int,
        doc_id: str | None,
    ) -> list[DocumentChunk]:
        """Vector cosine similarity search."""
        results: list[tuple[float, DocumentChunk]] = []

        with sqlite3.connect(str(self._db_path)) as conn:
            if doc_id:
                rows = conn.execute(
                    "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.embedding, "
                    "d.source FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                    "WHERE c.doc_id = ? AND c.embedding IS NOT NULL",
                    (doc_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, c.embedding, "
                    "d.source FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                    "WHERE c.embedding IS NOT NULL",
                ).fetchall()

        for row in rows:
            chunk_id, d_id, idx, text, emb_blob, source = row
            if emb_blob:
                emb = json.loads(emb_blob)
                score = _cosine_similarity(query_emb, emb)
                results.append((score, DocumentChunk(
                    chunk_id=chunk_id, doc_id=d_id, source=source,
                    text=text, chunk_index=idx, score=score,
                )))

        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        doc_id: str | None,
    ) -> list[DocumentChunk]:
        """Simple keyword overlap search as fallback."""
        query_words = set(query.lower().split())

        with sqlite3.connect(str(self._db_path)) as conn:
            if doc_id:
                rows = conn.execute(
                    "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, d.source "
                    "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id "
                    "WHERE c.doc_id = ?",
                    (doc_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text, d.source "
                    "FROM chunks c JOIN documents d ON c.doc_id = d.doc_id",
                ).fetchall()

        results: list[tuple[float, DocumentChunk]] = []
        for row in rows:
            chunk_id, d_id, idx, text, source = row
            chunk_words = set(text.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                results.append((score, DocumentChunk(
                    chunk_id=chunk_id, doc_id=d_id, source=source,
                    text=text, chunk_index=idx, score=score,
                )))

        results.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in results[:top_k]]

    # -- Management -----------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT doc_id, source, chunk_count, created_at, metadata FROM documents "
                "ORDER BY created_at DESC",
            ).fetchall()
        return [
            {
                "doc_id": r[0], "source": r[1], "chunk_count": r[2],
                "created_at": r[3], "metadata": json.loads(r[4] or "{}"),
            }
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_chunk_count(self) -> int:
        """Return total number of chunks across all documents."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return row[0] if row else 0

    def get_document_count(self) -> int:
        """Return total number of ingested documents."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0] if row else 0

    # -- File reading ---------------------------------------------------

    @staticmethod
    def _read_document(path: Path) -> str:
        """Read text from supported file formats."""
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml",
                       ".yml", ".toml", ".csv", ".xml", ".html", ".css",
                       ".rst", ".tex", ".sql", ".sh", ".log", ".cfg", ".ini",
                       ".r", ".rb", ".java", ".c", ".cpp", ".h", ".go",
                       ".rs", ".swift", ".kt"):
            return path.read_text(encoding="utf-8", errors="replace")
        # Unsupported formats
        raise ValueError(f"Unsupported document format: {suffix}")
