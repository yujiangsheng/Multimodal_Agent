"""BaseAgent — abstract foundation for all Pinocchio components.

Every cognitive-loop agent and multimodal processor inherits from this
class.  It provides shared access to:

- **LLM client** — ``self.llm`` (:class:`~pinocchio.utils.llm_client.LLMClient`)
- **Memory manager** — ``self.memory`` (:class:`~pinocchio.memory.memory_manager.MemoryManager`)
- **Logger** — ``self.logger`` (:class:`~pinocchio.utils.logger.PinocchioLogger`)

Subclasses must implement :meth:`run`, which is the main entry point
for multimodal processors.  The :class:`PinocchioAgent` overrides
``run()`` to raise ``NotImplementedError`` (use named skill methods
like ``perceive()``, ``strategize()``, etc. instead).

Example::

    class MyProcessor(BaseAgent):
        role = AgentRole.TEXT_PROCESSOR

        def run(self, **kwargs) -> str:
            return self.llm.ask(system="You are helpful.", user=kwargs["text"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger


class BaseAgent(ABC):
    """Abstract base class shared by all Pinocchio components.

    Provides three shared capabilities:

    - **LLM access** — ``self.llm`` for language model inference
    - **Memory access** — ``self.memory`` for all three content stores
      (episodic, semantic, procedural) plus working memory
    - **Structured logging** — ``self._log()`` / ``self._warn()`` /
      ``self._error()`` with role-tagged colour output

    Parameters
    ----------
    llm : LLMClient
        Synchronous OpenAI-compatible LLM client.
    memory : MemoryManager
        Unified façade over the dual-axis memory system.
    logger : PinocchioLogger
        Colour-coded structured logger.
    """

    role: AgentRole = AgentRole.ORCHESTRATOR  # override in subclass

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        logger: PinocchioLogger,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.logger = logger

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute this agent's primary function.

        Subclasses must override this method.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.logger.info(self.role, msg)

    def _warn(self, msg: str) -> None:
        self.logger.warn(self.role, msg)

    def _error(self, msg: str) -> None:
        self.logger.error(self.role, msg)
