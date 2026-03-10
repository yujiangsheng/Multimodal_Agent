"""Tests for the dual-axis memory system: temporal tiers + content stores.

Covers MemoryTier enum, tier-aware queries in all three content stores,
consolidation logic, MemoryManager dual-axis facade, and evaluation
agent completeness detection.
"""

import pytest
from unittest.mock import MagicMock

from pinocchio.models.enums import MemoryTier, Modality, TaskType
from pinocchio.models.schemas import (
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
    EvaluationResult,
    MultimodalInput,
    PerceptionResult,
    StrategyResult,
    AgentMessage,
)
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.memory.episodic_memory import EpisodicMemory
from pinocchio.memory.semantic_memory import SemanticMemory
from pinocchio.memory.procedural_memory import ProceduralMemory
from pinocchio.agents.unified_agent import PinocchioAgent
from pinocchio.utils.logger import PinocchioLogger


# =====================================================================
# MemoryTier enum
# =====================================================================

class TestMemoryTierEnum:
    def test_values(self):
        assert MemoryTier.WORKING == "working"
        assert MemoryTier.LONG_TERM == "long_term"
        assert MemoryTier.PERSISTENT == "persistent"

    def test_member_count(self):
        assert len(MemoryTier) == 3


# =====================================================================
# Schema defaults — memory_tier
# =====================================================================

class TestSchemaMemoryTier:
    def test_episodic_default_tier(self):
        rec = EpisodicRecord()
        assert rec.memory_tier == MemoryTier.LONG_TERM

    def test_episodic_custom_tier(self):
        rec = EpisodicRecord(memory_tier=MemoryTier.PERSISTENT)
        assert rec.memory_tier == MemoryTier.PERSISTENT

    def test_episodic_to_from_dict(self):
        rec = EpisodicRecord(memory_tier=MemoryTier.PERSISTENT)
        d = rec.to_dict()
        assert d["memory_tier"] == "persistent"
        restored = EpisodicRecord.from_dict(d)
        assert restored.memory_tier == MemoryTier.PERSISTENT

    def test_semantic_default_tier(self):
        entry = SemanticEntry()
        assert entry.memory_tier == MemoryTier.LONG_TERM

    def test_semantic_to_from_dict(self):
        entry = SemanticEntry(memory_tier=MemoryTier.PERSISTENT, domain="test")
        d = entry.to_dict()
        assert d["memory_tier"] == "persistent"
        restored = SemanticEntry.from_dict(d)
        assert restored.memory_tier == MemoryTier.PERSISTENT

    def test_procedural_default_tier(self):
        entry = ProceduralEntry()
        assert entry.memory_tier == MemoryTier.LONG_TERM

    def test_procedural_to_from_dict(self):
        entry = ProceduralEntry(memory_tier=MemoryTier.PERSISTENT, name="test")
        d = entry.to_dict()
        assert d["memory_tier"] == "persistent"
        restored = ProceduralEntry.from_dict(d)
        assert restored.memory_tier == MemoryTier.PERSISTENT


# =====================================================================
# EvaluationResult completeness fields
# =====================================================================

class TestEvaluationResultCompleteness:
    def test_defaults(self):
        result = EvaluationResult()
        assert result.is_complete is True
        assert result.incompleteness_details == ""

    def test_incomplete_result(self):
        result = EvaluationResult(
            is_complete=False,
            incompleteness_details="Response was truncated mid-sentence.",
        )
        assert result.is_complete is False
        assert "truncated" in result.incompleteness_details


# =====================================================================
# Episodic Memory tier-aware queries
# =====================================================================

class TestEpisodicMemoryTiers:
    def test_get_by_tier_empty(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        assert em.get_by_tier(MemoryTier.PERSISTENT) == []

    def test_get_by_tier(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(episode_id="e1", memory_tier=MemoryTier.LONG_TERM))
        em.add(EpisodicRecord(episode_id="e2", memory_tier=MemoryTier.PERSISTENT))
        em.add(EpisodicRecord(episode_id="e3", memory_tier=MemoryTier.LONG_TERM))
        assert len(em.get_by_tier(MemoryTier.LONG_TERM)) == 2
        assert len(em.get_by_tier(MemoryTier.PERSISTENT)) == 1

    def test_get_persistent(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(episode_id="e1", memory_tier=MemoryTier.PERSISTENT))
        em.add(EpisodicRecord(episode_id="e2"))
        assert len(em.get_persistent()) == 1
        assert em.get_persistent()[0].episode_id == "e1"

    def test_promote_to_persistent(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        em.add(EpisodicRecord(episode_id="e1"))
        assert em.promote_to_persistent("e1") is True
        assert em.get("e1").memory_tier == MemoryTier.PERSISTENT

    def test_promote_nonexistent_returns_false(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        assert em.promote_to_persistent("missing") is False

    def test_consolidate_high_value(self, tmp_path):
        em = EpisodicMemory(str(tmp_path / "ep.json"))
        # High value: high score + lessons + long_term
        em.add(EpisodicRecord(
            episode_id="good",
            outcome_score=9,
            lessons=["learned something"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Low value: high score but no lessons
        em.add(EpisodicRecord(
            episode_id="nolesson",
            outcome_score=9,
            lessons=[],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Already persistent
        em.add(EpisodicRecord(
            episode_id="already_persistent",
            outcome_score=10,
            lessons=["great"],
            memory_tier=MemoryTier.PERSISTENT,
        ))
        candidates = em.consolidate_high_value()
        assert len(candidates) == 1
        assert candidates[0].episode_id == "good"


# =====================================================================
# Semantic Memory tier-aware queries
# =====================================================================

class TestSemanticMemoryTiers:
    def test_get_by_tier(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.add(SemanticEntry(entry_id="s1", memory_tier=MemoryTier.LONG_TERM))
        sm.add(SemanticEntry(entry_id="s2", memory_tier=MemoryTier.PERSISTENT))
        assert len(sm.get_by_tier(MemoryTier.LONG_TERM)) == 1
        assert len(sm.get_persistent()) == 1

    def test_promote_to_persistent(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        sm.add(SemanticEntry(entry_id="s1"))
        assert sm.promote_to_persistent("s1") is True
        assert sm.get("s1").memory_tier == MemoryTier.PERSISTENT

    def test_promote_nonexistent_returns_false(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        assert sm.promote_to_persistent("missing") is False

    def test_consolidate_high_confidence(self, tmp_path):
        sm = SemanticMemory(str(tmp_path / "sem.json"))
        # Good candidate: high confidence + 2+ sources + long-term
        sm.add(SemanticEntry(
            entry_id="good",
            confidence=0.95,
            source_episodes=["ep1", "ep2"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Too low confidence
        sm.add(SemanticEntry(
            entry_id="low_conf",
            confidence=0.5,
            source_episodes=["ep1", "ep2"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Too few sources
        sm.add(SemanticEntry(
            entry_id="few_src",
            confidence=0.95,
            source_episodes=["ep1"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        candidates = sm.consolidate_high_confidence()
        assert len(candidates) == 1
        assert candidates[0].entry_id == "good"


# =====================================================================
# Procedural Memory tier-aware queries
# =====================================================================

class TestProceduralMemoryTiers:
    def test_get_by_tier(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(entry_id="p1", memory_tier=MemoryTier.LONG_TERM))
        pm.add(ProceduralEntry(entry_id="p2", memory_tier=MemoryTier.PERSISTENT))
        assert len(pm.get_by_tier(MemoryTier.LONG_TERM)) == 1
        assert len(pm.get_persistent()) == 1

    def test_promote_to_persistent(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        pm.add(ProceduralEntry(entry_id="p1"))
        assert pm.promote_to_persistent("p1") is True
        assert pm.get("p1").memory_tier == MemoryTier.PERSISTENT

    def test_promote_nonexistent_returns_false(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        assert pm.promote_to_persistent("missing") is False

    def test_consolidate_proven(self, tmp_path):
        pm = ProceduralMemory(str(tmp_path / "proc.json"))
        # Good candidate: high usage + high success rate + long-term
        pm.add(ProceduralEntry(
            entry_id="proven",
            usage_count=10,
            success_rate=0.9,
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Too few uses
        pm.add(ProceduralEntry(
            entry_id="new",
            usage_count=2,
            success_rate=0.95,
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Low success rate
        pm.add(ProceduralEntry(
            entry_id="unreliable",
            usage_count=10,
            success_rate=0.5,
            memory_tier=MemoryTier.LONG_TERM,
        ))
        candidates = pm.consolidate_proven()
        assert len(candidates) == 1
        assert candidates[0].entry_id == "proven"


# =====================================================================
# MemoryManager dual-axis
# =====================================================================

class TestMemoryManagerDualAxis:
    def test_working_memory_attribute(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        assert mm.working is not None
        assert mm.working.count == 0

    def test_recall_includes_working_context(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.working.add_conversation_turn("user", "hello")
        result = mm.recall(task_type=TaskType.QUESTION_ANSWERING, modalities=[Modality.TEXT])
        assert "working_context" in result
        assert "hello" in result["working_context"]

    def test_recall_includes_persistent_knowledge(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.semantic.add(SemanticEntry(
            entry_id="pk1",
            domain="test",
            memory_tier=MemoryTier.PERSISTENT,
        ))
        result = mm.recall(task_type=TaskType.QUESTION_ANSWERING, modalities=[Modality.TEXT])
        assert "persistent_knowledge" in result
        assert len(result["persistent_knowledge"]) == 1

    def test_reset_working_memory(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.working.add_conversation_turn("user", "hi")
        assert mm.working.count == 1
        mm.reset_working_memory()
        assert mm.working.count == 0

    def test_consolidate_empty(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        result = mm.consolidate()
        assert result == {"episodic": 0, "semantic": 0, "procedural": 0}

    def test_consolidate_promotes_to_persistent(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        # Add a high-value episodic record
        mm.episodic.add(EpisodicRecord(
            episode_id="e_high",
            outcome_score=9,
            lessons=["important lesson"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Add a high-confidence semantic entry
        mm.semantic.add(SemanticEntry(
            entry_id="s_high",
            confidence=0.95,
            source_episodes=["ep1", "ep2"],
            memory_tier=MemoryTier.LONG_TERM,
        ))
        # Add a proven procedure
        mm.procedural.add(ProceduralEntry(
            entry_id="p_high",
            usage_count=10,
            success_rate=0.9,
            memory_tier=MemoryTier.LONG_TERM,
        ))

        result = mm.consolidate()
        assert result["episodic"] == 1
        assert result["semantic"] == 1
        assert result["procedural"] == 1

        # Verify tiers were actually updated
        assert mm.episodic.get("e_high").memory_tier == MemoryTier.PERSISTENT
        assert mm.semantic.get("s_high").memory_tier == MemoryTier.PERSISTENT
        assert mm.procedural.get("p_high").memory_tier == MemoryTier.PERSISTENT

    def test_summary_includes_temporal_axis(self, tmp_path):
        mm = MemoryManager(data_dir=str(tmp_path))
        mm.working.add_conversation_turn("user", "test")
        s = mm.summary()
        assert "working_memory" in s
        assert "persistent_episodes" in s
        assert "persistent_knowledge" in s
        assert "persistent_procedures" in s
        assert s["working_memory"]["total_items"] == 1


# =====================================================================
# Evaluation completeness detection
# =====================================================================

class TestEvaluationCompletenessDetection:
    """Test the heuristic completeness checking in PinocchioAgent."""

    def test_heuristic_empty_response(self):
        ok, reason = PinocchioAgent._heuristic_completeness_check("")
        assert ok is False
        assert "empty" in reason.lower()

    def test_heuristic_whitespace_only(self):
        ok, reason = PinocchioAgent._heuristic_completeness_check("   ")
        assert ok is False
        assert "empty" in reason.lower()

    def test_heuristic_short_response(self):
        ok, reason = PinocchioAgent._heuristic_completeness_check("Hi")
        assert ok is False
        assert "short" in reason.lower()

    def test_heuristic_unbalanced_code_fences(self):
        text = "Here is the code:\n```python\ndef foo():\n    pass"
        ok, reason = PinocchioAgent._heuristic_completeness_check(text)
        assert ok is False
        assert "code" in reason.lower()

    def test_heuristic_balanced_code_fences_ok(self):
        text = "Here is the code:\n```python\ndef foo():\n    pass\n```\nDone."
        ok, reason = PinocchioAgent._heuristic_completeness_check(text)
        assert ok is True

    def test_heuristic_no_terminal_punctuation(self):
        # Long enough (> 200 chars) but ends with a special char not in terminal set
        text = "x" * 201 + "\x00"
        ok, reason = PinocchioAgent._heuristic_completeness_check(text)
        assert ok is False
        assert "punctuation" in reason.lower()

    def test_heuristic_proper_terminal_punctuation(self):
        text = "This is a complete response that answers the question properly."
        ok, reason = PinocchioAgent._heuristic_completeness_check(text)
        assert ok is True
        assert reason == ""

    def test_heuristic_short_but_valid_10_plus(self):
        ok, reason = PinocchioAgent._heuristic_completeness_check("Enough now.")
        assert ok is True

    def test_heuristic_ends_with_word(self):
        """Responses ending with a normal word should pass."""
        ok, reason = PinocchioAgent._heuristic_completeness_check(
            "x" * 201 + "complete"
        )
        assert ok is True

    def test_heuristic_chinese_punctuation(self):
        ok, reason = PinocchioAgent._heuristic_completeness_check(
            "\u8fd9\u662f\u4e00\u4e2a\u5b8c\u6574\u7684\u56de\u7b54\u3002"
        )
        assert ok is True

    def test_eval_incomplete_caps_quality(self, tmp_path):
        """When heuristic flags incomplete, quality should be capped at 5."""
        llm = MagicMock()
        llm.model = "test-model"
        llm.temperature = 0.7
        llm.max_tokens = 4096
        llm.ask_json.return_value = {
            "task_completion": "complete",
            "output_quality": 9,
            "strategy_effectiveness": 8,
            "is_complete": True,  # LLM says complete
        }
        memory = MemoryManager(data_dir=str(tmp_path))
        logger = PinocchioLogger()
        agent = PinocchioAgent(llm, memory, logger)

        # Very short response — heuristic will override LLM
        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(content="ok"),
        )
        assert result.is_complete is False
        assert result.output_quality <= 5
        assert result.task_completion == "partial"

    def test_eval_complete_preserves_quality(self, tmp_path):
        """Complete response should preserve LLM's quality score."""
        llm = MagicMock()
        llm.model = "test-model"
        llm.temperature = 0.7
        llm.max_tokens = 4096
        llm.ask_json.return_value = {
            "task_completion": "complete",
            "output_quality": 9,
            "strategy_effectiveness": 8,
            "is_complete": True,
        }
        memory = MemoryManager(data_dir=str(tmp_path))
        logger = PinocchioLogger()
        agent = PinocchioAgent(llm, memory, logger)

        result = agent.evaluate(
            user_input=MultimodalInput(text="test"),
            perception=PerceptionResult(),
            strategy=StrategyResult(),
            response=AgentMessage(
                content="This is a thorough and complete response to the user's question."
            ),
        )
        assert result.is_complete is True
        assert result.output_quality == 9
        assert result.task_completion == "complete"
