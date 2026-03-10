"""Tests for embedding / vector search: EmbeddingClient, memory embedding search, and schema persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pinocchio.models.schemas import EpisodicRecord, SemanticEntry


class TestEmbeddingClient:
    """Tests for EmbeddingClient."""

    def test_cosine_similarity_identical(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        vec = [1.0, 2.0, 3.0]
        sim = EmbeddingClient.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = EmbeddingClient.cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        a = [0.0, 0.0]
        b = [1.0, 2.0]
        sim = EmbeddingClient.cosine_similarity(a, b)
        assert sim == 0.0

    def test_embed_calls_api(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "test"

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = mock_response
        client._client = mock_openai

        result = client.embed("hello")
        assert result == [0.1, 0.2, 0.3]
        mock_openai.embeddings.create.assert_called_once()

    def test_embed_batch(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "test"

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2], index=1),
            MagicMock(embedding=[0.3, 0.4], index=0),
        ]
        mock_openai = MagicMock()
        mock_openai.embeddings.create.return_value = mock_response
        client._client = mock_openai

        results = client.embed_batch(["a", "b"])
        assert results == [[0.3, 0.4], [0.1, 0.2]]

    def test_embed_batch_empty(self):
        from pinocchio.utils.llm_client import EmbeddingClient

        client = EmbeddingClient.__new__(EmbeddingClient)
        client.model = "test"
        assert client.embed_batch([]) == []


class TestEpisodicMemoryEmbeddingSearch:
    """Tests for EpisodicMemory.search_by_embedding()."""

    def test_search_by_embedding(self, tmp_data_dir):
        from pinocchio.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        ep1 = EpisodicRecord(user_intent="coding task", embedding=[1.0, 0.0, 0.0])
        ep2 = EpisodicRecord(user_intent="math task", embedding=[0.0, 1.0, 0.0])
        ep3 = EpisodicRecord(user_intent="no embedding")
        mem.add(ep1)
        mem.add(ep2)
        mem.add(ep3)

        results = mem.search_by_embedding([1.0, 0.0, 0.0], limit=2)
        assert len(results) >= 1
        assert results[0].user_intent == "coding task"

    def test_search_by_embedding_empty(self, tmp_data_dir):
        from pinocchio.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        results = mem.search_by_embedding([1.0, 0.0], limit=5)
        assert results == []


class TestSemanticMemoryEmbeddingSearch:
    """Tests for SemanticMemory.search_by_embedding()."""

    def test_search_by_embedding(self, tmp_data_dir):
        from pinocchio.memory.semantic_memory import SemanticMemory

        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        e1 = SemanticEntry(knowledge="Python is great", domain="programming", embedding=[0.9, 0.1, 0.0])
        e2 = SemanticEntry(knowledge="Cats are cute", domain="animals", embedding=[0.0, 0.1, 0.9])
        mem.add(e1)
        mem.add(e2)

        results = mem.search_by_embedding([1.0, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0].knowledge == "Python is great"


class TestMemoryManagerEmbeddingIntegration:
    """Tests for MemoryManager with embedding client."""

    def test_store_episode_auto_embeds(self, tmp_data_dir):
        from pinocchio.memory.memory_manager import MemoryManager

        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.1, 0.2, 0.3]
        mm.set_embedding_client(mock_emb)

        ep = EpisodicRecord(user_intent="test", strategy_used="direct")
        mm.store_episode(ep)

        assert ep.embedding == [0.1, 0.2, 0.3]
        mock_emb.embed.assert_called_once()

    def test_store_knowledge_auto_embeds(self, tmp_data_dir):
        from pinocchio.memory.memory_manager import MemoryManager

        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_emb = MagicMock()
        mock_emb.embed.return_value = [0.4, 0.5, 0.6]
        mm.set_embedding_client(mock_emb)

        entry = SemanticEntry(knowledge="Python is typed", domain="lang")
        mm.store_knowledge(entry)

        assert entry.embedding == [0.4, 0.5, 0.6]

    def test_store_episode_embedding_failure_graceful(self, tmp_data_dir):
        from pinocchio.memory.memory_manager import MemoryManager

        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_emb = MagicMock()
        mock_emb.embed.side_effect = Exception("API down")
        mm.set_embedding_client(mock_emb)

        ep = EpisodicRecord(user_intent="test")
        mm.store_episode(ep)
        assert ep.embedding == []


class TestSchemaEmbeddingPersistence:
    """Test that embedding field round-trips through JSON serialization."""

    def test_episodic_record_embedding_roundtrip(self):
        ep = EpisodicRecord(user_intent="test", embedding=[0.1, 0.2, 0.3])
        d = ep.to_dict()
        assert d["embedding"] == [0.1, 0.2, 0.3]
        restored = EpisodicRecord.from_dict(d)
        assert restored.embedding == [0.1, 0.2, 0.3]

    def test_episodic_record_no_embedding_in_dict(self):
        ep = EpisodicRecord(user_intent="test")
        d = ep.to_dict()
        assert "embedding" not in d

    def test_semantic_entry_embedding_roundtrip(self):
        entry = SemanticEntry(knowledge="fact", embedding=[0.5, 0.6])
        d = entry.to_dict()
        assert d["embedding"] == [0.5, 0.6]
        restored = SemanticEntry.from_dict(d)
        assert restored.embedding == [0.5, 0.6]
