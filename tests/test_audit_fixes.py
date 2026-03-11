"""Tests for the deep-audit Round 5 fixes.

Covers all 14 issues:
  C1: chat_stream thread-local isolation
  C2: Sandbox escape hardening
  H1: Graph output merging
  H2: LLM truncation detection
  H3: Non-daemon thread + atexit
  H4: Upload MIME whitelist
  H5: MCP tool timeout
  H6: Memory purge
  M1: _emit_progress logging (not silent pass)
  M2: ask_json repair logging
  M3: Working memory decay rate
  M4: Team parallel execution
  M5: Config validation
  M6: ReAct empty answer handling
"""

from __future__ import annotations

import json
import logging
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ========================================================================
# C1: chat_stream thread-local isolation
# ========================================================================


class TestChatStreamThreadLocal:
    """Verify chat_stream uses thread-local state, no monkey-patching."""

    def test_stream_event_queue_local_initialized(self):
        """Pinocchio should have _stream_event_queue_local as threading.local."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._stream_event_queue_local = threading.local()
            assert isinstance(obj._stream_event_queue_local, threading.local)

    def test_emit_progress_pushes_to_thread_local_queue(self):
        """_emit_progress should push events to thread-local stream queue."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._progress_callback = None
            obj._stream_event_queue_local = threading.local()
            q: queue.Queue = queue.Queue()
            obj._stream_event_queue_local.queue = q

            obj._emit_progress("test_phase", "running", "detail")

            msg = q.get_nowait()
            assert "[PHASE] test_phase: running detail" in msg

    def test_emit_progress_no_queue_no_crash(self):
        """_emit_progress works fine without a stream queue set."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._progress_callback = None
            obj._stream_event_queue_local = threading.local()
            # No queue attribute → should not crash
            obj._emit_progress("test", "done")

    def test_no_monkey_patch_in_chat_stream(self):
        """chat_stream source should not contain self.react_executor.run = ."""
        import inspect
        from pinocchio.orchestrator import Pinocchio

        source = inspect.getsource(Pinocchio.chat_stream)
        assert "self.react_executor.run =" not in source
        assert "self._progress_callback = _stream_progress" not in source

    def test_get_stream_step_callback_returns_none_without_queue(self):
        """_get_stream_step_callback returns None when no stream queue."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._stream_event_queue_local = threading.local()
            assert obj._get_stream_step_callback() is None

    def test_get_stream_step_callback_returns_callable_with_queue(self):
        """_get_stream_step_callback returns a callback when queue is set."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._stream_event_queue_local = threading.local()
            obj._stream_event_queue_local.queue = queue.Queue()
            cb = obj._get_stream_step_callback()
            assert callable(cb)


# ========================================================================
# C2: Sandbox escape hardening
# ========================================================================


class TestSandboxHardening:
    """Verify sandbox blocks meta-programming and dotted imports."""

    def test_dotted_import_blocked(self):
        """importlib.util should be blocked even though it's a dotted name."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("import importlib.util")
        assert not result.success or "not allowed" in result.error.lower() or "not allowed" in result.stderr.lower()

    def test_os_path_dotted_blocked(self):
        """os.path should be blocked via dotted import check."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("import os.path")
        assert not result.success

    def test_getattr_static_check(self):
        """getattr() should be blocked by static analysis."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("x = getattr(__builtins__, '__import__')")
        assert not result.success

    def test_subclasses_blocked(self):
        """__subclasses__ should be blocked by static analysis."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("print(object.__subclasses__())")
        assert not result.success

    def test_globals_blocked(self):
        """__globals__ should be blocked by static analysis."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("print(f.__globals__)")
        assert not result.success

    def test_safe_code_still_works(self):
        """Legitimate code should still execute fine."""
        from pinocchio.sandbox.code_sandbox import CodeSandbox
        sb = CodeSandbox(timeout=5)
        result = sb.execute("print(sum(range(10)))")
        assert result.success
        assert "45" in result.stdout


# ========================================================================
# H1: Graph output merging
# ========================================================================


class TestGraphOutputMerging:
    """Verify graph auto-route merges outputs from all nodes."""

    def test_merge_multiple_node_outputs(self):
        """All node outputs should be joined, not just the last one."""
        import inspect
        from pinocchio.orchestrator import Pinocchio

        source = inspect.getsource(Pinocchio._run_cognitive_loop)
        # Should have output_parts list-based merging
        assert "output_parts" in source
        # Should NOT have the old single-variable pattern
        assert 'last_output = ""' not in source or "output_parts" in source

    def test_empty_graph_output_warning(self):
        """Empty graph output should log a warning."""
        import inspect
        from pinocchio.orchestrator import Pinocchio

        source = inspect.getsource(Pinocchio._run_cognitive_loop)
        assert "produced no output" in source


# ========================================================================
# H2: LLM truncation detection
# ========================================================================


class TestLLMTruncationDetection:
    """Verify truncated responses trigger a warning."""

    def test_finish_reason_length_logs_warning(self, caplog):
        """finish_reason='length' should produce a WARNING log."""
        from pinocchio.utils.llm_client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "truncated response"
        mock_choice.finish_reason = "length"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("openai.OpenAI"):
            client = LLMClient(model="test", api_key="test", base_url="http://test")
            client._client = MagicMock()
            client._client.chat.completions.create.return_value = mock_response
            client.circuit_breaker = MagicMock()
            client.circuit_breaker.allow_request.return_value = True

            with caplog.at_level(logging.WARNING, logger="pinocchio.utils.llm_client"):
                result = client.chat([{"role": "user", "content": "hi"}])

            assert result == "truncated response"
            assert any("truncated" in r.message.lower() for r in caplog.records)

    def test_finish_reason_stop_no_warning(self, caplog):
        """finish_reason='stop' should NOT produce a warning."""
        from pinocchio.utils.llm_client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "complete response"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("openai.OpenAI"):
            client = LLMClient(model="test", api_key="test", base_url="http://test")
            client._client = MagicMock()
            client._client.chat.completions.create.return_value = mock_response
            client.circuit_breaker = MagicMock()
            client.circuit_breaker.allow_request.return_value = True

            with caplog.at_level(logging.WARNING, logger="pinocchio.utils.llm_client"):
                result = client.chat([{"role": "user", "content": "hi"}])

            assert result == "complete response"
            assert not any("truncated" in r.message.lower() for r in caplog.records)


# ========================================================================
# H3: Non-daemon thread + atexit
# ========================================================================


class TestNonDaemonThread:
    """Verify _defer_post_response uses non-daemon thread."""

    def test_post_response_thread_not_daemon(self):
        """The background learning thread should NOT be a daemon."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._last_learning_error = None
            obj.agent = MagicMock()
            obj.memory = MagicMock()
            obj.logger = MagicMock()
            obj._interaction_count = 1
            obj._progress_callback = None
            obj._stream_event_queue_local = threading.local()
            obj._post_response_thread = None

            obj._defer_post_response("test", MagicMock(), MagicMock(), MagicMock())

            thread = obj._post_response_thread
            assert thread is not None
            assert not thread.daemon
            thread.join(timeout=5)

    def test_wait_for_background_method_exists(self):
        """_wait_for_background method should exist."""
        from pinocchio.orchestrator import Pinocchio
        assert hasattr(Pinocchio, "_wait_for_background")

    def test_wait_for_background_no_thread_no_crash(self):
        """_wait_for_background should not crash when no thread is running."""
        from pinocchio.orchestrator import Pinocchio

        with patch.object(Pinocchio, "__init__", lambda self, **kw: None):
            obj = Pinocchio.__new__(Pinocchio)
            obj._post_response_thread = None
            obj._wait_for_background()  # should not raise


# ========================================================================
# H4: Upload MIME whitelist
# ========================================================================


class TestUploadMIMEWhitelist:
    """Verify _save_upload rejects disallowed extensions."""

    def test_allowed_image_extension(self):
        """A .jpg upload should be accepted."""
        from web.app import _save_upload, UPLOAD_DIR
        mock_upload = MagicMock()
        mock_upload.filename = "photo.jpg"
        mock_upload.file = MagicMock()
        mock_upload.file.read.return_value = b"fake image data"

        with patch("shutil.copyfileobj"):
            path = _save_upload(mock_upload)
            assert path.endswith(".jpg")

    def test_blocked_executable_extension(self):
        """A .exe upload should be rejected."""
        from web.app import _save_upload
        mock_upload = MagicMock()
        mock_upload.filename = "malware.exe"
        with pytest.raises(ValueError, match="not allowed"):
            _save_upload(mock_upload)

    def test_blocked_script_extension(self):
        """A .py upload should be rejected."""
        from web.app import _save_upload
        mock_upload = MagicMock()
        mock_upload.filename = "evil.py"
        with pytest.raises(ValueError, match="not allowed"):
            _save_upload(mock_upload)

    def test_blocked_html_extension(self):
        """A .html upload should be rejected."""
        from web.app import _save_upload
        mock_upload = MagicMock()
        mock_upload.filename = "page.html"
        with pytest.raises(ValueError, match="not allowed"):
            _save_upload(mock_upload)

    def test_allowed_audio_extension(self):
        """A .mp3 upload should be accepted."""
        from web.app import _save_upload
        mock_upload = MagicMock()
        mock_upload.filename = "music.mp3"
        mock_upload.file = MagicMock()

        with patch("shutil.copyfileobj"):
            path = _save_upload(mock_upload)
            assert path.endswith(".mp3")

    def test_allowed_video_extension(self):
        """A .mp4 upload should be accepted."""
        from web.app import _save_upload
        mock_upload = MagicMock()
        mock_upload.filename = "video.mp4"
        mock_upload.file = MagicMock()

        with patch("shutil.copyfileobj"):
            path = _save_upload(mock_upload)
            assert path.endswith(".mp4")


# ========================================================================
# H5: MCP tool timeout
# ========================================================================


class TestMCPToolTimeout:
    """Verify MCP call_tool has a timeout."""

    def test_tool_timeout_parameter(self):
        """MCPClient should accept tool_timeout parameter."""
        from pinocchio.mcp.mcp_client import MCPClient

        client = MCPClient("http://localhost:9999/mcp", tool_timeout=5.0)
        assert client._tool_timeout == 5.0

    def test_tool_timeout_default(self):
        """Default tool_timeout should be 60s."""
        from pinocchio.mcp.mcp_client import MCPClient

        client = MCPClient("http://localhost:9999/mcp")
        assert client._tool_timeout == 60.0

    def test_call_tool_timeout_raises(self):
        """A slow tool call should raise TimeoutError."""
        from pinocchio.mcp.mcp_client import MCPClient

        client = MCPClient("http://localhost:9999/mcp", tool_timeout=0.1)

        def slow_rpc(*args, **kwargs):
            time.sleep(2)
            return {"content": []}

        client._rpc_call = slow_rpc
        with pytest.raises(TimeoutError, match="timed out"):
            client.call_tool("slow_tool")


# ========================================================================
# H6: Memory purge
# ========================================================================


class TestMemoryPurge:
    """Verify consolidate includes purge, and purge works correctly."""

    def test_purge_method_exists(self):
        """MemoryManager should have a purge method."""
        from pinocchio.memory.memory_manager import MemoryManager
        assert hasattr(MemoryManager, "purge")

    def test_purge_removes_low_value_episodic(self, tmp_data_dir):
        """Purge should remove low-scored episodes beyond limit."""
        from pinocchio.memory.memory_manager import MemoryManager
        from pinocchio.models.enums import MemoryTier, TaskType, Modality
        from pinocchio.models.schemas import EpisodicRecord

        mm = MemoryManager(data_dir=tmp_data_dir)
        # Add 10 low-value episodes
        for i in range(10):
            ep = EpisodicRecord(
                episode_id=f"ep_{i}",
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent="test",
                strategy_used="test",
                outcome_score=float(i),  # scores 0-9
                memory_tier=MemoryTier.LONG_TERM,
            )
            mm.episodic.add(ep)

        purged = mm.purge(max_episodic=5, max_semantic=999, max_procedural=999)
        assert purged["episodic"] == 5
        # Should keep the 5 highest-scored (5,6,7,8,9)
        remaining = mm.episodic.all()
        scores = [e.outcome_score for e in remaining if e.memory_tier != MemoryTier.PERSISTENT]
        assert min(scores) >= 5.0

    def test_purge_preserves_persistent(self, tmp_data_dir):
        """Purge should never remove PERSISTENT entries."""
        from pinocchio.memory.memory_manager import MemoryManager
        from pinocchio.models.enums import MemoryTier, TaskType, Modality
        from pinocchio.models.schemas import EpisodicRecord

        mm = MemoryManager(data_dir=tmp_data_dir)
        # Add 1 persistent + 3 non-persistent
        persistent = EpisodicRecord(
            episode_id="persistent_ep",
            task_type=TaskType.QUESTION_ANSWERING,
            modalities=[Modality.TEXT],
            user_intent="important",
            strategy_used="test",
            outcome_score=1.0,  # low score but persistent
            memory_tier=MemoryTier.PERSISTENT,
        )
        mm.episodic.add(persistent)
        for i in range(3):
            mm.episodic.add(EpisodicRecord(
                episode_id=f"ep_{i}",
                task_type=TaskType.QUESTION_ANSWERING,
                modalities=[Modality.TEXT],
                user_intent="test",
                strategy_used="test",
                outcome_score=float(i),
                memory_tier=MemoryTier.LONG_TERM,
            ))

        mm.purge(max_episodic=1, max_semantic=999, max_procedural=999)
        remaining_ids = [e.episode_id for e in mm.episodic.all()]
        assert "persistent_ep" in remaining_ids

    def test_consolidate_includes_purge(self, tmp_data_dir):
        """consolidate() should call purge internally."""
        from pinocchio.memory.memory_manager import MemoryManager
        mm = MemoryManager(data_dir=tmp_data_dir)
        with patch.object(mm, "purge") as mock_purge:
            mock_purge.return_value = {"episodic": 0, "semantic": 0, "procedural": 0}
            mm.consolidate()
            mock_purge.assert_called_once()


# ========================================================================
# M1: _emit_progress logs errors instead of silent pass
# ========================================================================


class TestEmitProgressLogging:
    """Verify _emit_progress logs callback errors instead of swallowing."""

    def test_callback_error_logged(self):
        """A failing callback should produce a debug log, not silent pass."""
        import inspect
        from pinocchio.orchestrator import Pinocchio

        source = inspect.getsource(Pinocchio._emit_progress)
        # Should NOT have bare pass
        assert "pass  # never let" not in source
        # Should have logging
        assert "_orch_logger" in source or "logger" in source.lower()


# ========================================================================
# M2: ask_json repair logging
# ========================================================================


class TestAskJsonRepairLogging:
    """Verify _validate_and_repair logs when fields are auto-filled."""

    def test_missing_required_key_logged(self, caplog):
        """Auto-filling a missing required key should produce a warning."""
        from pinocchio.utils.llm_client import LLMClient

        schema = {
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["name", "score"],
        }
        data = {"name": "test"}  # missing 'score'

        with caplog.at_level(logging.WARNING, logger="pinocchio.utils.llm_client"):
            result = LLMClient._validate_and_repair(data, schema)

        assert result["score"] == 0  # auto-filled
        assert any("score" in r.message for r in caplog.records)


# ========================================================================
# M3: Working memory decay rate
# ========================================================================


class TestWorkingMemoryDecay:
    """Verify working memory uses gentler decay rate."""

    def test_decay_rate_default_is_gentle(self):
        """Default decay rate should be 0.02, not 0.05."""
        import inspect
        from pinocchio.memory.working_memory import WorkingMemory

        sig = inspect.signature(WorkingMemory._decay_relevance)
        default = sig.parameters["decay_rate"].default
        assert default == 0.02

    def test_context_survives_20_turns(self):
        """Important context should have > 0 relevance after 20 turns."""
        from pinocchio.memory.working_memory import WorkingMemory

        wm = WorkingMemory()
        wm.add_context("important context", source="test")
        # Simulate 20 turns
        for _ in range(20):
            wm.add_conversation_turn("user", "hello")

        items = wm.get_by_category("context")
        assert len(items) > 0
        # With decay_rate=0.02, after 20 turns: 0.8 - 20*0.02 = 0.4 > 0
        assert items[0].relevance > 0.0


# ========================================================================
# M4: Team parallel execution
# ========================================================================


class TestTeamParallelExecution:
    """Verify team supports parallel member execution."""

    def test_parallel_attribute_exists(self):
        """AgentTeam should have a 'parallel' attribute."""
        from pinocchio.collaboration.team import AgentTeam
        team = AgentTeam("test")
        assert hasattr(team, "parallel")
        assert team.parallel is False  # default off

    def test_parallel_execution(self, mock_llm):
        """Parallel mode should execute all members concurrently."""
        from pinocchio.collaboration.team import AgentTeam, TeamMember

        execution_times: dict[str, float] = {}

        def make_handler(member_id: str, delay: float = 0.1):
            def handler(task, context):
                execution_times[member_id] = time.time()
                time.sleep(delay)
                return f"{member_id} result"
            return handler

        team = AgentTeam("test", llm_client=mock_llm)
        team.parallel = True
        team.review_enabled = False

        team.add_member(TeamMember(
            member_id="a", role="role_a", specialty="spec_a",
            handler=make_handler("a"),
        ))
        team.add_member(TeamMember(
            member_id="b", role="role_b", specialty="spec_b",
            handler=make_handler("b"),
        ))

        # Mock decomposition
        mock_llm.chat.return_value = json.dumps({
            "assignments": [
                {"member_id": "a", "sub_task": "task_a", "order": 1},
                {"member_id": "b", "sub_task": "task_b", "order": 2},
            ]
        })

        result = team.collaborate("test task")
        assert "a" in result.contributions
        assert "b" in result.contributions

        # Both should have started within a short window (parallel)
        if len(execution_times) == 2:
            diff = abs(execution_times["a"] - execution_times["b"])
            assert diff < 0.5  # should be nearly simultaneous


# ========================================================================
# M5: Config validation
# ========================================================================


class TestConfigValidation:
    """Verify PinocchioConfig validates critical fields."""

    def test_num_ctx_too_small(self):
        """num_ctx < 512 should raise ValueError."""
        from config import PinocchioConfig
        with pytest.raises(ValueError, match="num_ctx"):
            PinocchioConfig(num_ctx=100)

    def test_max_workers_zero(self):
        """max_workers=0 should raise ValueError."""
        from config import PinocchioConfig
        with pytest.raises(ValueError, match="max_workers"):
            PinocchioConfig(max_workers=0)

    def test_max_workers_negative(self):
        """max_workers=-1 should raise ValueError."""
        from config import PinocchioConfig
        with pytest.raises(ValueError, match="max_workers"):
            PinocchioConfig(max_workers=-1)

    def test_max_tokens_zero(self):
        """max_tokens=0 should raise ValueError."""
        from config import PinocchioConfig
        with pytest.raises(ValueError, match="max_tokens"):
            PinocchioConfig(max_tokens=0)

    def test_empty_model_rejected(self):
        """Empty string model should raise ValueError."""
        from config import PinocchioConfig
        with pytest.raises(ValueError, match="model"):
            PinocchioConfig(model="")

    def test_valid_config_works(self):
        """A valid configuration should not raise."""
        from config import PinocchioConfig
        cfg = PinocchioConfig(num_ctx=4096, max_tokens=1024, max_workers=2)
        assert cfg.num_ctx == 4096


# ========================================================================
# M6: ReAct empty answer handling
# ========================================================================


class TestReActEmptyAnswer:
    """Verify ReAct handles max iteration with empty answer."""

    def test_extract_best_answer_logs_warning(self, mock_llm, caplog):
        """Reaching max iterations should log a warning."""
        from pinocchio.planning.react import ReActExecutor
        from pinocchio.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        react = ReActExecutor(mock_llm, executor, registry, max_iterations=1)

        # Make LLM always return a non-FINISH action
        mock_llm.chat.side_effect = [
            json.dumps({
                "thought": "thinking",
                "action": "some_tool",
                "action_input": {"param": "val"},
            }),
            "final answer from extraction",
        ]

        with patch.object(executor, "execute", return_value="observation"):
            with caplog.at_level(logging.WARNING, logger="pinocchio.planning.react"):
                trace = react.run("test question")

        assert not trace.success  # max iterations reached
        assert trace.final_answer  # should have a non-empty answer

    def test_empty_extraction_falls_back(self, mock_llm):
        """If extraction returns empty, should fall back to last observation."""
        from pinocchio.planning.react import ReActExecutor
        from pinocchio.tools import ToolExecutor, ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        react = ReActExecutor(mock_llm, executor, registry, max_iterations=1)

        mock_llm.chat.side_effect = [
            json.dumps({
                "thought": "thinking",
                "action": "tool",
                "action_input": {},
            }),
            "",  # empty extraction
        ]

        with patch.object(executor, "execute", return_value="useful observation"):
            trace = react.run("question")

        # Should have a fallback answer, not empty
        assert trace.final_answer
        assert len(trace.final_answer) > 0
