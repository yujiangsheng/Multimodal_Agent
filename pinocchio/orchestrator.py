"""Pinocchio Orchestrator — the conductor of the cognitive loop.

This module contains the :class:`Pinocchio` class, the **sole public
entry point** for users of the library.  It coordinates the unified
:class:`~pinocchio.agents.unified_agent.PinocchioAgent` through the
complete self-evolving cognitive cycle::

    PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → (META-REFLECT)

Responsibilities
----------------
1. **Cognitive-loop coordination** — drive all 6 phases for every
   interaction, passing context between skill methods.
2. **Fast-path optimisation** — short text-only messages skip the
   heavy phases and go directly to a single LLM call.
3. **Multimodal routing** — detect which modality processors are
   needed, run them in parallel (or sequentially), and inject the
   resulting text descriptions into the execution context.
4. **Response completeness** — an outer retry loop re-invokes
   ``continue_response()`` if the evaluator flags the output as
   incomplete (up to ``MAX_COMPLETION_RETRIES``).
5. **Background learning** — LEARN + META-REFLECT + memory
   consolidation run in a daemon thread so the user gets their
   response immediately after EVALUATE.
6. **Session management** — conversation history, interaction
   counter, adaptive ``UserModel``, and ``reset()`` / ``status()``.
7. **Error recovery** — catch exceptions in any phase and return
   a safe fallback response.

Example
-------
>>> from pinocchio import Pinocchio
>>> agent = Pinocchio()
>>> print(agent.chat("Explain quantum entanglement simply."))
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pinocchio.agents.unified_agent import PinocchioAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.models.schemas import (
    AgentMessage,
    MultimodalInput,
    UserModel,
)
from pinocchio.multimodal.text_processor import TextProcessor
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.multimodal.audio_processor import AudioProcessor
from pinocchio.multimodal.video_processor import VideoProcessor
from pinocchio.tools import Tool, ToolExecutor, ToolRegistry, tool as tool_decorator
from pinocchio.utils.context_manager import ContextManager
from pinocchio.utils.conversation_store import ConversationStore
from pinocchio.utils.input_guard import InputGuard
from pinocchio.utils.llm_client import EmbeddingClient, LLMClient
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.utils.resource_monitor import ResourceMonitor
from pinocchio.utils.response_cache import ResponseCache

# ── New subsystems (Gaps 1–7) ──
from pinocchio.planning.planner import TaskPlanner
from pinocchio.planning.react import ReActExecutor
from pinocchio.sandbox.code_sandbox import CodeSandbox
from pinocchio.rag.document_store import DocumentStore
from pinocchio.mcp.mcp_client import MCPToolBridge
from pinocchio.graph.agent_graph import AgentGraph, GraphExecutor
from pinocchio.collaboration.team import AgentTeam, TeamMember
from pinocchio.tracing.tracer import Tracer


class Pinocchio:
    """Top-level orchestrator for the self-evolving multimodal agent.

    This is the **only class** most users need to interact with.  It
    provides a simple ``chat()`` / ``async_chat()`` interface that
    internally drives the full 6-phase cognitive loop.

    Parameters
    ----------
    model : str
        LLM model name for the Ollama backend (default ``qwen3-vl:4b``).
    api_key : str | None
        API key for the OpenAI-compatible endpoint.
    base_url : str | None
        Base URL for the LLM API (default ``http://localhost:11434/v1``).
    data_dir : str
        Directory where persistent memory JSON files are stored.
    verbose : bool
        If ``True``, log hardware info and phase transitions to stdout.
    max_workers : int | None
        Max parallel threads for modality preprocessing.  ``None`` =
        auto-detect from hardware.
    parallel_modalities : bool
        Whether to preprocess modalities (vision/audio/video) in
        parallel threads.
    meta_reflect_interval : int
        Number of interactions between meta-reflection triggers.
    num_ctx : int
        Ollama context-window size (lower = less KV-cache memory).

    Usage
    -----
    >>> agent = Pinocchio()
    >>> response = agent.chat("Explain quantum entanglement simply.")
    >>> print(response)

    Multimodal::

        >>> response = agent.chat(
        ...     "What's in this image?",
        ...     image_paths=["photo.jpg"],
        ... )

    Async::

        >>> response = await agent.async_chat("Hello!")
    """

    GREETING = (
        "你好，我是 Pinocchio — 一个持续进化的多模态智能体。\n"
        "每一次对话都让我变得更好。我会记住有效的策略、从错误中学习、"
        "并不断精进我的推理能力。让我们开始吧。"
    )

    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        api_key: str | None = None,
        base_url: str | None = None,
        data_dir: str = "data",
        verbose: bool = True,
        max_workers: int | None = None,
        parallel_modalities: bool = True,
        meta_reflect_interval: int = 5,
        num_ctx: int = 8192,
    ) -> None:
        # Shared infrastructure
        self.llm = LLMClient(model=model, api_key=api_key, base_url=base_url, num_ctx=num_ctx)
        self.memory = MemoryManager(data_dir=data_dir)
        self.logger = PinocchioLogger()

        # Unified cognitive agent (all 6 skills in one)
        self.agent = PinocchioAgent(
            self.llm, self.memory, self.logger,
            meta_reflect_interval=meta_reflect_interval,
        )

        # Tool framework
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_defaults()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.agent.set_tools(self.tool_registry, self.tool_executor)

        # Embedding client for vector-based memory search
        try:
            self._embedding_client = EmbeddingClient(
                api_key=api_key, base_url=base_url,
            )
            self.memory.set_embedding_client(self._embedding_client)
        except Exception:
            self._embedding_client = None

        # Multimodal processor pool
        self.text_proc = TextProcessor(self.llm, self.memory, self.logger)
        self.vision_proc = VisionProcessor(self.llm, self.memory, self.logger)
        self.audio_proc = AudioProcessor(self.llm, self.memory, self.logger)
        self.video_proc = VideoProcessor(self.llm, self.memory, self.logger)

        # Resource detection & parallelism
        self._resource_monitor = ResourceMonitor()
        self._resources = self._resource_monitor.snapshot()
        self._max_workers = max_workers or self._resources.recommended_workers
        self._parallel_modalities = parallel_modalities

        # Input guard (prompt injection defense)
        self._input_guard = InputGuard()

        # Context window manager (summarise old turns)
        self._context_manager = ContextManager(
            self.llm,
            max_context_tokens=int(num_ctx * 0.6),  # leave 40 % for response
        )

        # Response cache (avoid redundant LLM calls)
        self._response_cache = ResponseCache(capacity=256, ttl_seconds=600)

        # Session state
        self.user_model = UserModel()
        self.conversation_history: list[dict[str, str]] = []
        self._interaction_count = 0
        self._verbose = verbose
        self._lock = threading.Lock()  # protects session state mutations
        self._post_response_thread: threading.Thread | None = None

        # Conversation persistence (SQLite)
        self._conversation_store = ConversationStore(
            Path(data_dir) / "conversations.db"
        )
        session = self._conversation_store.create_session()
        self._current_session_id: str = session.id
        self._last_user_msg_id: int | None = None
        self._last_asst_msg_id: int | None = None

        # Remove leftover empty sessions from earlier restarts / tests
        self._conversation_store.purge_empty_sessions(keep_id=session.id)

        # ── New subsystems (Gaps 1–7) ──
        # Gap 1: Multi-step task planner (ReAct)
        self.planner = TaskPlanner(self.llm)
        self.react_executor = ReActExecutor(
            self.llm, self.tool_executor, self.tool_registry,
        )

        # Gap 2: Code sandbox
        self.sandbox = CodeSandbox()

        # Gap 3: RAG document store
        self.document_store = DocumentStore(data_dir=data_dir)
        if self._embedding_client:
            self.document_store.set_embedding_client(self._embedding_client)

        # Gap 4: MCP protocol bridge
        self.mcp_bridge = MCPToolBridge(self.tool_registry)

        # Gap 5: Agent graph (workflow engine)
        self.graph_executor = GraphExecutor(max_workers=self._max_workers)

        # Gap 6: Multi-agent collaboration
        self.team = AgentTeam("default_team", llm_client=self.llm)

        # Gap 7: Structured tracing
        self.tracer = Tracer()

        # Log detected hardware
        if verbose:
            r = self._resources
            gpu_label = (
                f"{r.gpus[0].name} ({r.total_vram_mb:,} MB VRAM)"
                if r.has_gpu else "none"
            )
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Hardware: {r.cpu_count_physical} cores, "
                f"{r.ram_total_mb:,} MB RAM, GPU: {gpu_label} | "
                f"Workers: {self._max_workers}, "
                f"Parallel modalities: {self._parallel_modalities}",
            )

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------

    def chat(
        self,
        text: str | None = None,
        *,
        image_paths: list[str] | None = None,
        audio_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        """Process a user message through the full cognitive loop.

        Parameters
        ----------
        text : User's text message (may be None for pure media input).
        image_paths : Optional list of image file paths or URLs.
        audio_paths : Optional list of audio file paths.
        video_paths : Optional list of video file paths.

        Returns
        -------
        str : The agent's response text.
        """
        with self._lock:
            self._interaction_count += 1
            interaction_num = self._interaction_count
            self.user_model.interaction_count = interaction_num
        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{interaction_num}",
        )

        # ── Input validation & sanitisation ──
        if text:
            guard_result = self._input_guard.validate(text)
            if guard_result.threats:
                self.logger.warn(
                    AgentRole.ORCHESTRATOR,
                    f"Input guard: {guard_result.threat_summary}",
                )
            text = guard_result.sanitised_text

        user_input = MultimodalInput(
            text=text,
            image_paths=image_paths or [],
            audio_paths=audio_paths or [],
            video_paths=video_paths or [],
        )

        try:
            response = self._run_cognitive_loop(user_input)
        except Exception as exc:
            self.logger.error(AgentRole.ORCHESTRATOR, f"Cognitive loop error: {exc}")
            traceback.print_exc()
            response = AgentMessage(
                content="抱歉，处理过程中出现了意外错误。请重新尝试您的请求。",
                confidence=0.0,
            )

        # Store in conversation history
        with self._lock:
            user_msg_id = None
            if text:
                user_msg_id = self._conversation_store.add_message(
                    self._current_session_id, "user", text,
                )
                self.conversation_history.append(
                    {"role": "user", "content": text, "id": user_msg_id}
                )
            asst_msg_id = self._conversation_store.add_message(
                self._current_session_id, "assistant", response.content,
            )
            self.conversation_history.append(
                {"role": "assistant", "content": response.content, "id": asst_msg_id}
            )
            self._last_user_msg_id = user_msg_id
            self._last_asst_msg_id = asst_msg_id

            # Auto-title from first user message
            if text and interaction_num == 1:
                title = text[:40].replace("\n", " ")
                if len(text) > 40:
                    title += "…"
                self._conversation_store.update_session_title(
                    self._current_session_id, title,
                )

        return response.content

    async def async_chat(
        self,
        text: str | None = None,
        *,
        image_paths: list[str] | None = None,
        audio_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> str:
        """Async version of :meth:`chat` — runs the cognitive loop off the event loop.

        This wraps the synchronous cognitive loop in ``asyncio.to_thread``
        so it doesn't block other coroutines.  For truly async modality
        preprocessing, use :class:`AsyncLLMClient`.

        Usage
        -----
        >>> response = await agent.async_chat("Explain quantum entanglement.")
        """
        return await asyncio.to_thread(
            self.chat,
            text,
            image_paths=image_paths,
            audio_paths=audio_paths,
            video_paths=video_paths,
        )

    def greet(self) -> str:
        """Return the initialization greeting."""
        return self.GREETING

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        text: str | None = None,
        *,
        image_paths: list[str] | None = None,
        audio_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """Stream the agent's response token-by-token.

        Uses the fast path (simple text-only, ≤ FAST_PATH_MAX_LENGTH).
        For multimodal or complex input, falls back to yielding the full
        ``chat()`` response as a single chunk.
        """
        with self._lock:
            self._interaction_count += 1
            interaction_num = self._interaction_count
            self.user_model.interaction_count = interaction_num

        # ── Input validation ──
        if text:
            guard_result = self._input_guard.validate(text)
            if guard_result.threats:
                self.logger.warn(
                    AgentRole.ORCHESTRATOR,
                    f"Input guard: {guard_result.threat_summary}",
                )
            text = guard_result.sanitised_text

        user_input = MultimodalInput(
            text=text,
            image_paths=image_paths or [],
            audio_paths=audio_paths or [],
            video_paths=video_paths or [],
        )

        # Only stream the fast-path (simple text-only messages)
        if self._is_simple_input(user_input):
            if user_input.text:
                self.memory.working.add_conversation_turn("user", user_input.text)

            _FAST_SYSTEM = (
                "You are Pinocchio, a helpful multimodal AI assistant that learns "
                "and evolves. Respond directly to the user. Be thorough, accurate, "
                "and helpful. Write in the same language the user uses.\n\n"
                f"[USER PROFILE] {self._user_model_context()}"
            )
            messages: list[dict[str, Any]] = self._context_manager.build_messages(
                _FAST_SYSTEM, self.conversation_history,
            )
            messages.append({"role": "user", "content": user_input.text or ""})

            collected: list[str] = []
            for chunk in self.llm.chat_stream(messages):
                collected.append(chunk)
                yield chunk

            full_text = "".join(collected)
            self.memory.working.add_conversation_turn("assistant", full_text[:500])
            with self._lock:
                user_msg_id = None
                if text:
                    user_msg_id = self._conversation_store.add_message(
                        self._current_session_id, "user", text,
                    )
                    self.conversation_history.append(
                        {"role": "user", "content": text, "id": user_msg_id}
                    )
                asst_msg_id = self._conversation_store.add_message(
                    self._current_session_id, "assistant", full_text,
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": full_text, "id": asst_msg_id}
                )
                self._last_user_msg_id = user_msg_id
                self._last_asst_msg_id = asst_msg_id
        else:
            # Fallback: run full cognitive loop and yield result as one chunk
            response = self.chat(text, image_paths=image_paths,
                                 audio_paths=audio_paths, video_paths=video_paths)
            yield response

    def reset(self) -> None:
        """Reset session state (keeps persistent memory)."""
        self.conversation_history.clear()
        self._interaction_count = 0
        self.user_model = UserModel()
        self.memory.reset_working_memory()
        self._context_manager.reset()
        self._response_cache.clear()
        # Start a fresh session
        session = self._conversation_store.create_session()
        self._current_session_id = session.id
        self._last_user_msg_id = None
        self._last_asst_msg_id = None

    def status(self) -> dict[str, Any]:
        """Return a summary of the agent's current state."""
        self._resources = self._resource_monitor.snapshot(refresh=True)
        return {
            "interaction_count": self._interaction_count,
            "session_id": self._current_session_id,
            "memory_summary": self.memory.summary(),
            "improvement_trend": self.memory.improvement_trend(),
            "user_model": {
                "expertise": self.user_model.expertise.value,
                "style": self.user_model.style.value,
                "interests": self.user_model.domains_of_interest,
            },
            "resources": self._resources.to_dict(),
            "working_memory": self.memory.working.summary(),
            "context_manager": self._context_manager.stats(),
            "response_cache": self._response_cache.stats(),
            "rag": {
                "documents": self.document_store.get_document_count(),
                "chunks": self.document_store.get_chunk_count(),
            },
            "mcp": {
                "connected_servers": self.mcp_bridge.connected_servers,
            },
            "tracing": self.tracer.stats(),
            "team_members": list(self.team.members.keys()),
        }

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @property
    def current_session_id(self) -> str:
        """Return the active session's ID."""
        return self._current_session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all stored sessions (newest first)."""
        return [s.to_dict() for s in self._conversation_store.list_sessions()]

    def new_session(self, title: str = "新对话") -> dict[str, Any]:
        """Create a new session and switch to it."""
        self.conversation_history.clear()
        self._interaction_count = 0
        self.memory.reset_working_memory()
        self._context_manager.reset()
        self._response_cache.clear()
        session = self._conversation_store.create_session(title)
        self._current_session_id = session.id
        self._last_user_msg_id = None
        self._last_asst_msg_id = None
        return session.to_dict()

    def switch_session(self, session_id: str) -> dict[str, Any] | None:
        """Switch to an existing session, loading its messages."""
        session = self._conversation_store.get_session(session_id)
        if session is None:
            return None

        # Load messages into conversation_history
        stored = self._conversation_store.get_messages(session_id)
        self.conversation_history = [
            {"role": m.role, "content": m.content, "id": m.id}
            for m in stored
        ]
        self._current_session_id = session_id
        self._interaction_count = sum(
            1 for m in stored if m.role == "user"
        )
        self.user_model.interaction_count = self._interaction_count
        self.memory.reset_working_memory()
        self._context_manager.reset()
        self._response_cache.clear()

        # Re-seed working memory with recent turns
        for msg in self.conversation_history[-12:]:
            self.memory.working.add_conversation_turn(
                msg["role"], msg["content"][:500],
            )

        return session.to_dict()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.  Cannot delete the active session."""
        if session_id == self._current_session_id:
            return False
        return self._conversation_store.delete_session(session_id)

    def rename_session(self, session_id: str, title: str) -> bool:
        """Rename a session."""
        return self._conversation_store.update_session_title(session_id, title)

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return all messages for a session."""
        return [
            m.to_dict()
            for m in self._conversation_store.get_messages(session_id)
        ]

    # ------------------------------------------------------------------
    # Regenerate / Edit
    # ------------------------------------------------------------------

    def regenerate(self) -> str | None:
        """Remove the last assistant response and regenerate it.

        Returns the new response text, or ``None`` if there is nothing
        to regenerate.
        """
        # Find the last assistant message
        last_asst_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i]["role"] == "assistant":
                last_asst_idx = i
                break
        if last_asst_idx is None:
            return None

        # Find the preceding user message
        user_text = None
        for i in range(last_asst_idx - 1, -1, -1):
            if self.conversation_history[i]["role"] == "user":
                user_text = self.conversation_history[i]["content"]
                break
        if user_text is None:
            return None

        # Remove assistant message from history and store
        removed = self.conversation_history.pop(last_asst_idx)
        if "id" in removed:
            self._conversation_store.delete_messages_after(
                self._current_session_id,
                removed["id"] - 1,
            )

        # Re-run the cognitive loop (user message is already in history)
        user_input = MultimodalInput(text=user_text)
        try:
            response = self._run_cognitive_loop(user_input)
        except Exception as exc:
            response = AgentMessage(
                content="重新生成失败，请重试。", confidence=0.0,
            )

        # Save new assistant message
        asst_msg_id = self._conversation_store.add_message(
            self._current_session_id, "assistant", response.content,
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response.content, "id": asst_msg_id}
        )
        self._last_asst_msg_id = asst_msg_id
        return response.content

    def edit_and_regenerate(self, message_id: int, new_text: str) -> str | None:
        """Edit a user message and regenerate everything after it.

        Parameters
        ----------
        message_id : int
            Database ID of the message to edit.
        new_text : str
            New content for the message.

        Returns
        -------
        str | None
            The new assistant response, or ``None`` on failure.
        """
        # Find the message in conversation_history
        target_idx = None
        for i, msg in enumerate(self.conversation_history):
            if msg.get("id") == message_id and msg["role"] == "user":
                target_idx = i
                break
        if target_idx is None:
            return None

        # Update in store
        self._conversation_store.update_message(message_id, new_text)
        # Delete all messages after it in the store
        self._conversation_store.delete_messages_after(
            self._current_session_id, message_id,
        )

        # Truncate in-memory history
        self.conversation_history = self.conversation_history[: target_idx + 1]
        self.conversation_history[target_idx]["content"] = new_text

        # Regenerate
        user_input = MultimodalInput(text=new_text)
        try:
            response = self._run_cognitive_loop(user_input)
        except Exception:
            response = AgentMessage(
                content="重新生成失败，请重试。", confidence=0.0,
            )

        asst_msg_id = self._conversation_store.add_message(
            self._current_session_id, "assistant", response.content,
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response.content, "id": asst_msg_id}
        )
        self._last_asst_msg_id = asst_msg_id
        return response.content

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def register_tool(
        self,
        func: Callable[..., str],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        """Register a custom tool function.

        Example::

            def weather(city: str) -> str:
                return f"Sunny in {city}"

            agent.register_tool(weather, description="Get weather info")
        """
        return self.tool_registry.register_function(
            func, name=name, description=description, parameters=parameters,
        )

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        return self.tool_registry.unregister(name)

    def enable_tool(self, name: str) -> bool:
        """Re-enable a disabled tool."""
        return self.tool_registry.enable(name)

    def disable_tool(self, name: str) -> bool:
        """Temporarily disable a tool without removing it."""
        return self.tool_registry.disable(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return info about all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "enabled": t.enabled,
                "parameters": t.parameters,
            }
            for t in self.tool_registry._tools.values()
        ]

    def tool_stats(self) -> dict[str, Any]:
        """Return usage metrics for tool invocations."""
        return self.tool_executor.stats()

    # ------------------------------------------------------------------
    # User model context
    # ------------------------------------------------------------------

    def _user_model_context(self) -> str:
        """Build a concise user-profile string for prompt injection."""
        parts = [f"用户水平: {self.user_model.expertise.value}"]
        parts.append(f"沟通风格: {self.user_model.style.value}")
        if self.user_model.domains_of_interest:
            parts.append(
                f"兴趣领域: {', '.join(self.user_model.domains_of_interest)}"
            )
        parts.append(f"交互次数: {self.user_model.interaction_count}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Internal: Cognitive Loop
    # ------------------------------------------------------------------

    MAX_COMPLETION_RETRIES = 1  # reduced from 3 for faster response

    # Simple-input threshold: text-only, short messages skip heavy phases
    FAST_PATH_MAX_LENGTH = 500

    @staticmethod
    def _is_simple_input(user_input: MultimodalInput) -> bool:
        """Heuristic: text-only short messages don't need full cognitive loop."""
        if user_input.image_paths or user_input.audio_paths or user_input.video_paths:
            return False
        text = (user_input.text or "").strip()
        if not text or len(text) > Pinocchio.FAST_PATH_MAX_LENGTH:
            return False
        return True

    def _run_fast_path(self, user_input: MultimodalInput) -> AgentMessage:
        """Fast path: single LLM call for simple text-only inputs.

        Skips PERCEIVE, STRATEGIZE, EVALUATE — goes directly to the LLM
        with conversational context from working memory.  Roughly as fast
        as a raw Ollama call.
        """
        self.logger.log(AgentRole.ORCHESTRATOR, "Fast path — direct execution")

        user_text = user_input.text or ""

        # Build context-managed message list
        _FAST_SYSTEM = (
            "You are Pinocchio, a helpful multimodal AI assistant that learns "
            "and evolves. Respond directly to the user. Be thorough, accurate, "
            "and helpful. Write in the same language the user uses.\n\n"
            f"[USER PROFILE] {self._user_model_context()}"
        )
        messages = self._context_manager.build_messages(
            _FAST_SYSTEM, self.conversation_history,
        )
        messages.append({"role": "user", "content": user_text})

        # ── Cache lookup ──
        cache_key = ResponseCache.make_key(messages)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            self.logger.log(AgentRole.ORCHESTRATOR, "Fast path — cache hit")
            response_text = cached
        else:
            response_text = self.llm.chat(messages)
            self._response_cache.put(cache_key, response_text)

        # Store in working memory
        self.memory.working.add_conversation_turn("assistant", response_text[:500])

        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Fast path complete — {len(response_text)} chars",
        )

        return AgentMessage(
            role="assistant",
            content=response_text,
            confidence=0.8,
            metadata={"fast_path": True, "cache_hit": cached is not None},
        )

    def _run_cognitive_loop(self, user_input: MultimodalInput) -> AgentMessage:
        """Execute the full PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN pipeline.

        This is the heart of Pinocchio's self-evolving intelligence.  Every
        user interaction passes through all 6 phases, each handled by a
        dedicated sub-agent.  Phase 4.5 (retry loop) ensures response
        completeness, and Phase 6 meta-reflect fires periodically for
        higher-order self-improvement.
        """

        # ── Phase 0: Buffer the raw user input into working memory ──
        if user_input.text:
            self.memory.working.add_conversation_turn("user", user_input.text)

        # ── Fast path: simple text-only input → single LLM call ──
        if self._is_simple_input(user_input):
            return self._run_fast_path(user_input)

        # ── Start a trace for this interaction ──
        trace = self.tracer.create_trace(f"interaction_{self._interaction_count}")

        # ── Phase 0.5: PREPROCESS MODALITIES ──
        with trace.span("preprocess_modalities") as sp:
            modality_context = self._preprocess_modalities(user_input)
            modality_context["user_profile"] = self._user_model_context()
            sp.set_attribute("modalities", list(modality_context.keys()))

        # ── Phase 1: PERCEIVE ──
        with trace.span("perceive") as sp:
            perception = self.agent.perceive(
                user_input=user_input,
                modality_context=modality_context,
            )
            sp.set_attribute("task_type", str(getattr(perception, 'task_type', '')))
            sp.set_attribute("complexity", int(getattr(perception, 'complexity', 0)))

        # ── Phase 1.5: PLAN (if complex) ──
        plan = None
        complexity_val = int(getattr(perception, 'complexity', 0))
        task_type_val = str(getattr(perception, 'task_type', ''))
        if self.planner.should_plan(complexity_val, task_type_val):
            with trace.span("plan") as sp:
                plan = self.planner.decompose(user_input.text or "")
                sp.set_attribute("steps", plan.total_steps if plan else 0)

        # ── Phase 2: STRATEGIZE ──
        with trace.span("strategize") as sp:
            strategy = self.agent.strategize(perception=perception)

        # ── Phase 3: EXECUTE ──
        with trace.span("execute") as sp:
            response = self.agent.execute(
                user_input=user_input,
                perception=perception,
                strategy=strategy,
                modality_context=modality_context,
            )
            sp.set_attribute("response_length", len(response.content))

        # ── Phase 4: EVALUATE ──
        with trace.span("evaluate") as sp:
            evaluation = self.agent.evaluate(
                user_input=user_input,
                perception=perception,
                strategy=strategy,
                response=response,
            )
            sp.set_attribute("quality", getattr(evaluation, 'output_quality', 0))

        # ── Phase 4.5: COMPLETENESS RETRY LOOP ──
        # If the evaluator says the response is incomplete, ask the
        # execution agent to continue.  This is the *outer* retry loop;
        # each continue_response() call also has its own *inner* auto-
        # continuation loop for token-limit hits.
        retry_count = 0
        while not evaluation.is_complete and retry_count < self.MAX_COMPLETION_RETRIES:
            retry_count += 1
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Response incomplete (attempt {retry_count}/{self.MAX_COMPLETION_RETRIES}) "
                f"— requesting continuation: {evaluation.incompleteness_details}",
            )

            try:
                response = self.agent.continue_response(
                    user_input=user_input,
                    partial_response=response.content,
                    incompleteness_details=evaluation.incompleteness_details,
                )

                # Re-evaluate the continued response
                evaluation = self.agent.evaluate(
                    user_input=user_input,
                    perception=perception,
                    strategy=strategy,
                    response=response,
                )
            except Exception as retry_exc:
                self.logger.error(
                    AgentRole.ORCHESTRATOR,
                    f"Continuation attempt {retry_count} failed: {retry_exc}",
                )
                break  # Keep the best response we have so far

        if retry_count > 0:
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Completion retries used: {retry_count} — "
                f"final status: {evaluation.task_completion}, "
                f"complete: {evaluation.is_complete}",
            )

        # ── Phase 5+6: LEARN & META-REFLECT (deferred to background) ──
        # These phases update memory but don't affect the current response.
        # Running them in a background thread lets us return the response
        # immediately after evaluation, saving 2-3 LLM call latencies.
        user_text = user_input.text or "(non-text input)"
        self._defer_post_response(user_text, perception, strategy, evaluation)

        # Store assistant response in working memory
        self.memory.working.add_conversation_turn(
            "assistant", response.content[:500]
        )

        self.logger.separator()
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Interaction #{self._interaction_count} complete — "
            f"quality: {evaluation.output_quality}/10",
        )

        return response

    # ------------------------------------------------------------------
    # Internal: Deferred post-response work (background thread)
    # ------------------------------------------------------------------

    def _defer_post_response(
        self,
        user_text: str,
        perception: Any,
        strategy: Any,
        evaluation: Any,
    ) -> None:
        """Run LEARN + META-REFLECT + consolidation in a background thread."""

        def _background() -> None:
            try:
                # Phase 5: LEARN
                self.agent.learn(
                    user_input_text=user_text,
                    perception=perception,
                    strategy=strategy,
                    evaluation=evaluation,
                )

                # Phase 6: META-REFLECT (periodic)
                if self.agent.should_meta_reflect():
                    self.logger.log(
                        AgentRole.ORCHESTRATOR,
                        "Meta-reflection triggered — running higher-order analysis",
                    )
                    meta = self.agent.meta_reflect()
                    if meta.priority_improvements:
                        self.logger.log(
                            AgentRole.ORCHESTRATOR,
                            f"Top improvement priorities: {meta.priority_improvements[:3]}",
                        )

                # Periodic consolidation
                if self._interaction_count % 10 == 0 and self._interaction_count > 0:
                    promoted = self.memory.consolidate()
                    if any(v > 0 for v in promoted.values()):
                        self.logger.log(
                            AgentRole.ORCHESTRATOR,
                            f"Memory consolidation: {promoted}",
                        )
            except Exception as exc:
                self.logger.error(
                    AgentRole.ORCHESTRATOR,
                    f"Background post-response error: {exc}",
                )

        thread = threading.Thread(
            target=_background, name="pinocchio-post-response", daemon=True
        )
        self._post_response_thread = thread
        thread.start()

    # ------------------------------------------------------------------
    # Internal: Parallel Modality Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_modalities(self, user_input: MultimodalInput) -> dict[str, str]:
        """Pre-process non-text modalities in parallel.

        When the user sends images + audio + video simultaneously, each
        modality processor runs in its own thread.  The resulting text
        descriptions are returned as a dict keyed by modality name so
        downstream agents can incorporate them.
        """
        tasks: dict[str, tuple[Any, dict[str, Any]]] = {}

        if user_input.image_paths:
            tasks["vision"] = (
                self.vision_proc,
                {"task": "Describe the image(s) in detail", "image_paths": user_input.image_paths},
            )
        if user_input.audio_paths:
            tasks["audio"] = (
                self.audio_proc,
                {"task": "Transcribe and analyse the audio", "audio_paths": user_input.audio_paths},
            )
        if user_input.video_paths:
            tasks["video"] = (
                self.video_proc,
                {
                    "task": "Analyse the video content",
                    "video_paths": user_input.video_paths,
                    "vision_processor": self.vision_proc,
                    "audio_processor": self.audio_proc,
                },
            )

        if not tasks:
            return {}

        # Sequential or parallel depending on config & worker count
        n_tasks = len(tasks)
        use_parallel = self._parallel_modalities and self._max_workers > 1 and n_tasks > 1

        if use_parallel:
            self.logger.log(
                AgentRole.ORCHESTRATOR,
                f"Parallel modality preprocessing: {list(tasks.keys())} "
                f"({self._max_workers} workers)",
            )
            results: dict[str, str] = {}
            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, n_tasks),
                thread_name_prefix="modality",
            ) as pool:
                futures = {
                    pool.submit(proc.run, **kwargs): name
                    for name, (proc, kwargs) in tasks.items()
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        results[name] = fut.result()
                    except Exception as exc:
                        self.logger.error(
                            AgentRole.ORCHESTRATOR,
                            f"Modality '{name}' failed: {exc}",
                        )
                        results[name] = f"(error processing {name})"
            return results

        # Sequential fallback
        self.logger.log(
            AgentRole.ORCHESTRATOR,
            f"Sequential modality preprocessing: {list(tasks.keys())}",
        )
        results = {}
        for name, (proc, kwargs) in tasks.items():
            try:
                results[name] = proc.run(**kwargs)
            except Exception as exc:
                self.logger.error(
                    AgentRole.ORCHESTRATOR,
                    f"Modality '{name}' failed: {exc}",
                )
                results[name] = f"(error processing {name})"
        return results
