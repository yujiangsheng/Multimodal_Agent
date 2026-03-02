"""Pinocchio utility modules.

Shared infrastructure used across the agent system:

* :class:`LLMClient` / :class:`AsyncLLMClient` — OpenAI-compatible LLM API wrapper
* :class:`PinocchioLogger` — colour-coded, role-tagged structured logging
* :class:`ResourceMonitor` — dynamic CPU / GPU / RAM detection
* :class:`ParallelExecutor` — resource-aware concurrent task runner
"""

from pinocchio.utils.llm_client import LLMClient, AsyncLLMClient
from pinocchio.utils.logger import PinocchioLogger
from pinocchio.utils.resource_monitor import ResourceMonitor, ResourceSnapshot, GPUInfo
from pinocchio.utils.parallel_executor import ParallelExecutor

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "PinocchioLogger",
    "ResourceMonitor",
    "ResourceSnapshot",
    "GPUInfo",
    "ParallelExecutor",
]
