"""RAG (Retrieval-Augmented Generation) knowledge base.

Provides a lightweight, self-contained document store that supports:

* **Ingestion** — read plain-text and Markdown files, split into
  overlapping chunks, and (optionally) compute embedding vectors.
* **Hybrid search** — vector cosine-similarity search when embeddings
  are available, with automatic keyword-fallback otherwise.
* **SQLite persistence** — all chunks and metadata are stored in a
  local SQLite database so they survive across restarts.

Quick start::

    from pinocchio.rag import DocumentStore

    store = DocumentStore()
    store.ingest("docs/guide.md")
    results = store.search("how to configure memory", top_k=5)
    for chunk in results:
        print(chunk.text[:120])
"""

from pinocchio.rag.document_store import DocumentStore, DocumentChunk

__all__ = ["DocumentStore", "DocumentChunk"]
