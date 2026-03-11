"""Tests for Gap 3: RAG document store."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pinocchio.rag.document_store import (
    DocumentChunk,
    DocumentStore,
    _chunk_text,
    _cosine_similarity,
    _estimate_tokens,
)


class TestDocumentChunk:
    def test_defaults(self):
        c = DocumentChunk()
        assert c.chunk_id == ""
        assert c.text == ""
        assert c.embedding == []
        assert c.score == 0.0

    def test_to_dict(self):
        c = DocumentChunk(chunk_id="c1", doc_id="d1", text="hello", source="test.md")
        d = c.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["source"] == "test.md"
        assert "embedding" not in d  # empty embedding excluded

    def test_to_dict_with_embedding(self):
        c = DocumentChunk(embedding=[0.1, 0.2, 0.3])
        d = c.to_dict()
        assert "embedding" in d
        assert len(d["embedding"]) == 3


class TestChunkingUtilities:
    def test_estimate_tokens_english(self):
        text = "Hello world this is a test"
        tokens = _estimate_tokens(text)
        assert tokens > 0

    def test_estimate_tokens_cjk(self):
        text = "你好世界"
        tokens = _estimate_tokens(text)
        assert tokens > 0

    def test_chunk_text_short(self):
        text = "Short text"
        chunks = _chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long(self):
        # Create a long text with clear paragraph boundaries
        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity([], []) == 0.0

    def test_cosine_similarity_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestDocumentStore:
    @pytest.fixture
    def store(self, tmp_path):
        return DocumentStore(data_dir=str(tmp_path))

    def test_init_creates_db(self, tmp_path):
        store = DocumentStore(data_dir=str(tmp_path))
        assert (tmp_path / "rag_documents.db").exists()

    def test_ingest_text(self, store):
        doc_id = store.ingest_text("Hello world, this is a test document.")
        assert doc_id
        assert store.get_document_count() == 1
        assert store.get_chunk_count() >= 1

    def test_ingest_text_empty_raises(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.ingest_text("")

    def test_ingest_file(self, store, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("# Test\n\nThis is a test document with some content.", encoding="utf-8")
        doc_id = store.ingest(str(doc))
        assert doc_id
        assert store.get_document_count() == 1

    def test_ingest_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.ingest("/nonexistent/doc.txt")

    def test_ingest_duplicate(self, store):
        id1 = store.ingest_text("Same content", source="a")
        # Different source but same content hash - deduplication happens at file level
        # ingest_text doesn't deduplicate, so this creates a second doc
        id2 = store.ingest_text("Same content", source="b")
        # Both should succeed
        assert id1 and id2

    def test_ingest_file_duplicate_content(self, store, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("Exact same content", encoding="utf-8")
        id1 = store.ingest(str(doc))
        id2 = store.ingest(str(doc))
        assert id1 == id2  # Same file, same hash → same doc_id

    def test_keyword_search(self, store):
        store.ingest_text("Quantum computing uses qubits for parallel computation.")
        store.ingest_text("Classical computers use bits for sequential processing.")

        results = store.search("quantum qubits")
        assert len(results) > 0
        # First result should be the quantum doc
        assert "quantum" in results[0].text.lower() or "qubit" in results[0].text.lower()

    def test_search_top_k(self, store):
        for i in range(5):
            store.ingest_text(f"Document {i} about topic {i}")
        results = store.search("document topic", top_k=2)
        assert len(results) <= 2

    def test_search_by_doc_id(self, store):
        id1 = store.ingest_text("Alpha document about music")
        id2 = store.ingest_text("Beta document about science")

        results = store.search("document", doc_id=id1)
        for r in results:
            assert r.doc_id == id1

    def test_list_documents(self, store):
        store.ingest_text("Doc 1", source="source_1")
        store.ingest_text("Doc 2", source="source_2")
        docs = store.list_documents()
        assert len(docs) == 2
        assert all("doc_id" in d for d in docs)

    def test_delete_document(self, store):
        doc_id = store.ingest_text("To be deleted")
        assert store.get_document_count() == 1
        assert store.delete_document(doc_id) is True
        assert store.get_document_count() == 0

    def test_delete_nonexistent(self, store):
        assert store.delete_document("fake_id") is False

    def test_unsupported_format(self, store, tmp_path):
        doc = tmp_path / "test.pdf"
        doc.write_bytes(b"%PDF-1.4 binary content")
        with pytest.raises(ValueError, match="Unsupported"):
            store.ingest(str(doc))

    def test_vector_search_with_embedding_client(self, store):
        client = MagicMock()
        # Return different embeddings for different texts
        client.embed.side_effect = lambda t: [1.0, 0.0] if "quantum" in t.lower() else [0.0, 1.0]
        store.set_embedding_client(client)

        store.ingest_text("Quantum mechanics is fascinating")
        store.ingest_text("Cooking pasta is easy")

        results = store.search("quantum physics")
        assert len(results) > 0

    def test_get_counts_empty(self, store):
        assert store.get_document_count() == 0
        assert store.get_chunk_count() == 0

    def test_python_file_readable(self, store, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')", encoding="utf-8")
        doc_id = store.ingest(str(f))
        assert doc_id
