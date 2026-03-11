"""Tests for the three-part memory system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import Modality, TaskType
from pinocchio.models.schemas import EpisodicRecord, SemanticEntry, ProceduralEntry


# ──────────────────────────────────────────────────────────────────
# Episodic Memory
# ──────────────────────────────────────────────────────────────────


class TestEpisodicMemory:
    def test_add_and_count(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        assert mem.count == 0
        mem.add(EpisodicRecord(task_type=TaskType.ANALYSIS))
        assert mem.count == 1

    def test_persistence(self, tmp_data_dir):
        path = f"{tmp_data_dir}/ep.json"
        mem1 = EpisodicMemory(path)
        mem1.add(EpisodicRecord(task_type=TaskType.CODE_GENERATION, outcome_score=8))
        mem1.add(EpisodicRecord(task_type=TaskType.ANALYSIS, outcome_score=6))

        # Reload from disk
        mem2 = EpisodicMemory(path)
        assert mem2.count == 2
        assert mem2.all()[0].task_type == TaskType.CODE_GENERATION

    def test_get_by_id(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        ep = EpisodicRecord(task_type=TaskType.CONVERSATION)
        mem.add(ep)
        found = mem.get(ep.episode_id)
        assert found is not None
        assert found.task_type == TaskType.CONVERSATION

    def test_get_missing_returns_none(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        assert mem.get("nonexistent") is None

    def test_search_by_task_type(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        for _ in range(3):
            mem.add(EpisodicRecord(task_type=TaskType.CODE_GENERATION))
        mem.add(EpisodicRecord(task_type=TaskType.ANALYSIS))
        results = mem.search_by_task_type(TaskType.CODE_GENERATION)
        assert len(results) == 3

    def test_search_by_modality(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(modalities=[Modality.TEXT, Modality.IMAGE]))
        mem.add(EpisodicRecord(modalities=[Modality.TEXT]))
        mem.add(EpisodicRecord(modalities=[Modality.IMAGE, Modality.AUDIO]))
        results = mem.search_by_modality(Modality.IMAGE)
        assert len(results) == 2

    def test_search_by_keyword(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(user_intent="sort algorithm", lessons=["use quicksort"]))
        mem.add(EpisodicRecord(user_intent="image classification"))
        results = mem.search_by_keyword("quicksort")
        assert len(results) == 1

    def test_find_similar(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT],
        ))
        mem.add(EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT, Modality.IMAGE],
        ))
        mem.add(EpisodicRecord(
            task_type=TaskType.ANALYSIS,
            modalities=[Modality.TEXT],
        ))
        results = mem.find_similar(TaskType.CODE_GENERATION, [Modality.TEXT])
        assert len(results) >= 2
        # The first result should be the one with the best match
        assert results[0].task_type == TaskType.CODE_GENERATION

    def test_average_score(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(outcome_score=8))
        mem.add(EpisodicRecord(outcome_score=6))
        mem.add(EpisodicRecord(outcome_score=10))
        assert mem.average_score() == 8.0
        assert mem.average_score(last_n=2) == 8.0  # (6+10)/2

    def test_average_score_empty(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        assert mem.average_score() == 0.0

    def test_error_frequency(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(error_patterns=["timeout", "parse_error"]))
        mem.add(EpisodicRecord(error_patterns=["timeout"]))
        freq = mem.error_frequency()
        assert freq["timeout"] == 2
        assert freq["parse_error"] == 1

    def test_recent_lessons(self, tmp_data_dir):
        mem = EpisodicMemory(f"{tmp_data_dir}/ep.json")
        mem.add(EpisodicRecord(lessons=["L1", "L2"]))
        mem.add(EpisodicRecord(lessons=["L3"]))
        lessons = mem.recent_lessons(limit=2)
        # reversed episodes: ep2["L3"], ep1["L1","L2"] → extend → ["L3","L1","L2"][:2]
        assert lessons == ["L3", "L1"]


# ──────────────────────────────────────────────────────────────────
# Semantic Memory
# ──────────────────────────────────────────────────────────────────


class TestSemanticMemory:
    def test_add_and_count(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        assert mem.count == 0
        mem.add(SemanticEntry(domain="math", knowledge="2+2=4"))
        assert mem.count == 1

    def test_persistence(self, tmp_data_dir):
        path = f"{tmp_data_dir}/sem.json"
        mem1 = SemanticMemory(path)
        mem1.add(SemanticEntry(domain="coding", knowledge="DRY principle"))
        mem2 = SemanticMemory(path)
        assert mem2.count == 1
        assert mem2.all()[0].knowledge == "DRY principle"

    def test_search_by_domain(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        mem.add(SemanticEntry(domain="coding", knowledge="K1"))
        mem.add(SemanticEntry(domain="coding_patterns", knowledge="K2"))
        mem.add(SemanticEntry(domain="math", knowledge="K3"))
        results = mem.search_by_domain("coding")
        assert len(results) == 2  # "coding" and "coding_patterns"

    def test_search_by_keyword(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        mem.add(SemanticEntry(domain="design", knowledge="Use dependency injection for testability"))
        mem.add(SemanticEntry(domain="testing", knowledge="Mock external services"))
        results = mem.search_by_keyword("testability")
        assert len(results) == 1

    def test_update_confidence(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        entry = SemanticEntry(domain="x", knowledge="y", confidence=0.5)
        mem.add(entry)
        mem.update_confidence(entry.entry_id, 0.9)
        updated = mem.get(entry.entry_id)
        assert updated.confidence == 0.9
        assert updated.updated_at  # timestamp set

    def test_update_confidence_clamps(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        entry = SemanticEntry(domain="x", knowledge="y")
        mem.add(entry)
        mem.update_confidence(entry.entry_id, 1.5)
        assert mem.get(entry.entry_id).confidence == 1.0
        mem.update_confidence(entry.entry_id, -0.5)
        assert mem.get(entry.entry_id).confidence == 0.0

    def test_get_high_confidence(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        mem.add(SemanticEntry(domain="a", knowledge="low", confidence=0.3))
        mem.add(SemanticEntry(domain="b", knowledge="high", confidence=0.9))
        high = mem.get_high_confidence(threshold=0.7)
        assert len(high) == 1
        assert high[0].knowledge == "high"

    def test_needs_synthesis(self, tmp_data_dir):
        mem = SemanticMemory(f"{tmp_data_dir}/sem.json")
        assert mem.needs_synthesis("coding", 5) is False
        assert mem.needs_synthesis("coding", 10) is True


# ──────────────────────────────────────────────────────────────────
# Procedural Memory
# ──────────────────────────────────────────────────────────────────


class TestProceduralMemory:
    def test_add_and_count(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        assert mem.count == 0
        mem.add(ProceduralEntry(name="proc1", task_type=TaskType.ANALYSIS))
        assert mem.count == 1

    def test_persistence(self, tmp_data_dir):
        path = f"{tmp_data_dir}/proc.json"
        mem1 = ProceduralMemory(path)
        mem1.add(ProceduralEntry(name="proc1", task_type=TaskType.ANALYSIS, success_rate=0.8))
        mem2 = ProceduralMemory(path)
        assert mem2.count == 1
        assert mem2.all()[0].name == "proc1"

    def test_best_procedure(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        mem.add(ProceduralEntry(name="slow", task_type=TaskType.ANALYSIS, success_rate=0.5))
        mem.add(ProceduralEntry(name="fast", task_type=TaskType.ANALYSIS, success_rate=0.9))
        mem.add(ProceduralEntry(name="other", task_type=TaskType.CODE_GENERATION, success_rate=1.0))
        best = mem.best_procedure(TaskType.ANALYSIS)
        assert best is not None
        assert best.name == "fast"

    def test_best_procedure_no_match(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        assert mem.best_procedure(TaskType.CONVERSATION) is None

    def test_record_usage_updates_success_rate(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        entry = ProceduralEntry(name="p1", task_type=TaskType.ANALYSIS, success_rate=1.0, usage_count=1)
        mem.add(entry)

        mem.record_usage(entry.entry_id, success=False)
        updated = mem.get(entry.entry_id)
        assert updated.usage_count == 2
        assert updated.success_rate == 0.5  # (1*1.0 + 0) / 2
        assert updated.last_used  # timestamp set

    def test_record_usage_incremental(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        entry = ProceduralEntry(name="p1", task_type=TaskType.ANALYSIS, success_rate=0.0, usage_count=0)
        mem.add(entry)

        mem.record_usage(entry.entry_id, success=True)
        assert mem.get(entry.entry_id).success_rate == 1.0
        mem.record_usage(entry.entry_id, success=True)
        assert mem.get(entry.entry_id).success_rate == 1.0
        mem.record_usage(entry.entry_id, success=False)
        assert abs(mem.get(entry.entry_id).success_rate - 2 / 3) < 0.01

    def test_refine_steps(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        entry = ProceduralEntry(name="p1", steps=["old1", "old2"])
        mem.add(entry)
        mem.refine_steps(entry.entry_id, ["new1", "new2", "new3"])
        assert mem.get(entry.entry_id).steps == ["new1", "new2", "new3"]

    def test_top_procedures_requires_min_usage(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        mem.add(ProceduralEntry(name="p1", success_rate=1.0, usage_count=1))  # too few
        mem.add(ProceduralEntry(name="p2", success_rate=0.8, usage_count=5))
        top = mem.top_procedures(limit=5)
        assert len(top) == 1
        assert top[0].name == "p2"

    def test_search_by_name(self, tmp_data_dir):
        mem = ProceduralMemory(f"{tmp_data_dir}/proc.json")
        mem.add(ProceduralEntry(name="code_analysis_v2"))
        mem.add(ProceduralEntry(name="summarize_v1"))
        results = mem.search_by_name("analysis")
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────
# Memory Manager
# ──────────────────────────────────────────────────────────────────


class TestMemoryManager:
    def test_initialization(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        assert mm.episodic.count == 0
        assert mm.semantic.count == 0
        assert mm.procedural.count == 0

    def test_store_episode(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        ep = EpisodicRecord(task_type=TaskType.ANALYSIS, outcome_score=7)
        mm.store_episode(ep)
        assert mm.episodic.count == 1

    def test_recall_returns_all_components(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        # Seed some data
        mm.store_episode(EpisodicRecord(
            task_type=TaskType.CODE_GENERATION,
            modalities=[Modality.TEXT],
            lessons=["use types"],
        ))
        mm.store_knowledge(SemanticEntry(domain="code_generation", knowledge="typing helps"))
        mm.store_procedure(ProceduralEntry(
            name="code_gen_v1",
            task_type=TaskType.CODE_GENERATION,
            success_rate=0.9,
        ))

        recall = mm.recall(TaskType.CODE_GENERATION, [Modality.TEXT])
        assert len(recall["similar_episodes"]) >= 1
        assert len(recall["relevant_knowledge"]) >= 1
        assert recall["best_procedure"] is not None
        assert recall["best_procedure"].name == "code_gen_v1"
        assert len(recall["recent_lessons"]) >= 1

    def test_improvement_trend_insufficient_data(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        trend = mm.improvement_trend()
        assert trend["trend"] == "insufficient_data"

    def test_improvement_trend_with_data(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        for i in range(15):
            mm.store_episode(EpisodicRecord(outcome_score=5 + (i % 3)))
        trend = mm.improvement_trend(window=5)
        assert trend["trend"] in ("improving", "declining", "stable")
        assert "recent_avg" in trend

    def test_summary(self, tmp_data_dir):
        mm = MemoryManager(data_dir=tmp_data_dir)
        mm.store_episode(EpisodicRecord(outcome_score=7))
        summary = mm.summary()
        assert summary["episodic_count"] == 1
        assert summary["avg_score"] == 7.0


# =====================================================================
# Embedding failure logging
# =====================================================================


class TestMemoryManagerLogging:
    """store_episode and store_knowledge should log embedding failures."""

    def test_store_episode_logs_embedding_failure(self, tmp_data_dir):
        from unittest.mock import MagicMock, patch
        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_client = MagicMock()
        mock_client.embed = MagicMock(side_effect=RuntimeError("down"))
        mm.set_embedding_client(mock_client)
        ep = EpisodicRecord(
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
            user_intent="test",
            strategy_used="direct",
            outcome_score=8,
            lessons=["learned"],
        )
        with patch("pinocchio.memory.memory_manager._logger") as mock_log:
            mm.store_episode(ep)
            mock_log.warning.assert_called_once()

    def test_store_knowledge_logs_embedding_failure(self, tmp_data_dir):
        from unittest.mock import MagicMock, patch
        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_client = MagicMock()
        mock_client.embed = MagicMock(side_effect=RuntimeError("down"))
        mm.set_embedding_client(mock_client)
        entry = SemanticEntry(
            domain="test",
            knowledge="test knowledge",
        )
        with patch("pinocchio.memory.memory_manager._logger") as mock_log:
            mm.store_knowledge(entry)
            mock_log.warning.assert_called_once()

    def test_recall_logs_embedding_failure(self, tmp_data_dir):
        from unittest.mock import MagicMock, patch
        mm = MemoryManager(data_dir=tmp_data_dir)
        mock_client = MagicMock()
        mock_client.embed = MagicMock(side_effect=RuntimeError("down"))
        mm.set_embedding_client(mock_client)
        with patch("pinocchio.memory.memory_manager._logger") as mock_log:
            mm.recall(TaskType.QUESTION_ANSWERING, [Modality.TEXT], keyword="test")
            mock_log.warning.assert_called_once()
