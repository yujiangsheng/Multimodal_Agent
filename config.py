"""Pinocchio agent configuration.

Centralises all tuneable parameters into a single ``PinocchioConfig``
dataclass.  Every field can be overridden via an environment variable,
making it easy to adjust behaviour in CI, Docker, or production without
touching code.

Environment Variables
---------------------
==============================  ============================================
Variable                        Description
==============================  ============================================
``PINOCCHIO_MODEL``             LLM model name (default: ``qwen3-vl:4b``)
``OLLAMA_API_KEY``              API key (default: ``ollama``)
``OPENAI_BASE_URL``             Base URL for the OpenAI-compatible API
``PINOCCHIO_DATA_DIR``          Directory for persistent memory JSON files
``PINOCCHIO_NUM_CTX``           Ollama context window size (default: 8192)
``PINOCCHIO_MAX_WORKERS``       Max parallel worker threads (auto if unset)
``PINOCCHIO_PARALLEL``          Enable parallel modality preprocessing
==============================  ============================================

Example
-------
>>> from config import PinocchioConfig
>>> cfg = PinocchioConfig()
>>> print(cfg.model)   # "qwen3-vl:4b" (or PINOCCHIO_MODEL env value)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PinocchioConfig:
    """Configuration for the Pinocchio agent.

    All fields have sensible defaults for local development with Ollama.
    Every field can be overridden via an environment variable (see the
    module-level docstring for the full mapping).

    Usage
    -----
    >>> cfg = PinocchioConfig()
    >>> agent = Pinocchio(
    ...     model=cfg.model,
    ...     api_key=cfg.api_key,
    ...     base_url=cfg.base_url,
    ...     data_dir=cfg.data_dir,
    ... )
    """

    # LLM settings (Qwen3-VL via Ollama local server)
    model: str = field(default_factory=lambda: os.getenv("PINOCCHIO_MODEL", "qwen3-vl:4b"))
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", "ollama"))
    base_url: str | None = field(default_factory=lambda: os.getenv(
        "OPENAI_BASE_URL", "http://localhost:11434/v1",
    ))
    temperature: float = 0.7
    max_tokens: int = 16384

    # Memory settings
    data_dir: str = field(default_factory=lambda: os.getenv("PINOCCHIO_DATA_DIR", "data"))

    # Behaviour settings
    meta_reflect_interval: int = 5
    verbose: bool = True

    # Embedding model for vector-based memory search
    embedding_model: str = field(
        default_factory=lambda: os.getenv("PINOCCHIO_EMBEDDING_MODEL", "nomic-embed-text"),
    )

    # Ollama context window — lower = less KV cache memory, faster inference
    num_ctx: int = field(
        default_factory=lambda: int(os.getenv("PINOCCHIO_NUM_CTX", "8192"))
    )

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
