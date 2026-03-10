"""Pinocchio utility modules.

Shared infrastructure used across the agent system:

* :class:`LLMClient` / :class:`AsyncLLMClient` — OpenAI-compatible LLM API wrapper
* :class:`PinocchioLogger` — colour-coded, role-tagged structured logging
* :class:`ResourceMonitor` — dynamic CPU / GPU / RAM detection
* :class:`ParallelExecutor` — resource-aware concurrent task runner
* :class:`InputGuard` — prompt injection detection and input validation
* :class:`ContextManager` — intelligent context window management
* :class:`ResponseCache` — LRU response cache with TTL
"""

from pinocchio.utils.llm_client import LLMClient, AsyncLLMClient, EmbeddingClient
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.utils.resource_monitor import ResourceMonitor, ResourceSnapshot, GPUInfo
from pinocchio.utils.parallel_executor import ParallelExecutor
from pinocchio.utils.input_guard import InputGuard, ValidationResult
from pinocchio.utils.context_manager import ContextManager, estimate_tokens
from pinocchio.utils.response_cache import ResponseCache

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "EmbeddingClient",
    "PinocchioLogger",
    "ResourceMonitor",
    "ResourceSnapshot",
    "GPUInfo",
    "ParallelExecutor",
    "InputGuard",
    "ValidationResult",
    "ContextManager",
    "estimate_tokens",
    "ResponseCache",
]
