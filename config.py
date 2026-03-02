"""Pinocchio agent configuration.

Centralises all tuneable parameters into a single ``PinocchioConfig``
dataclass.  Every field can be overridden via an environment variable
(see table below), making it easy to adjust behaviour in CI, Docker,
or production without touching code.

Environment Variables
---------------------
========================= ========================================
Variable                  Description
========================= ========================================
``PINOCCHIO_MODEL``       LLM model name (default: ``qwen2.5-omni``)
``OLLAMA_API_KEY``        API key (default: ``ollama``)
``OPENAI_BASE_URL``       Base URL for the OpenAI-compatible API
``PINOCCHIO_DATA_DIR``    Directory for persistent memory files
``PINOCCHIO_MAX_WORKERS`` Max parallel worker threads (auto if unset)
``PINOCCHIO_PARALLEL``    Enable parallel modality preprocessing
========================= ========================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PinocchioConfig:
    """Configuration for the Pinocchio agent.

    All values can be overridden via environment variables prefixed with
    ``PINOCCHIO_``.
    """

    # LLM settings (Qwen2.5-Omni via Ollama local server)
    model: str = field(default_factory=lambda: os.getenv("PINOCCHIO_MODEL", "qwen2.5-omni"))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", "ollama"))
    base_url: str | None = field(default_factory=lambda: os.getenv(
        "OPENAI_BASE_URL", "http://localhost:11434/v1",
    ))
    temperature: float = 0.7
    max_tokens: int = 4096

    # Memory settings
    data_dir: str = field(default_factory=lambda: os.getenv("PINOCCHIO_DATA_DIR", "data"))

    # Behaviour settings
    meta_reflect_interval: int = 5
    verbose: bool = True

    # Resource / parallelism settings
    max_workers: int | None = field(
        default_factory=lambda: (
            int(os.getenv("PINOCCHIO_MAX_WORKERS"))
            if os.getenv("PINOCCHIO_MAX_WORKERS")
            else None  # None = auto-detect from hardware
        )
    )
    parallel_modalities: bool = field(
        default_factory=lambda: os.getenv("PINOCCHIO_PARALLEL", "true").lower() in ("1", "true", "yes")
    )
