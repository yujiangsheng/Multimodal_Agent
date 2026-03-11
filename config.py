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
``PINOCCHIO_PROVIDER``          LLM provider preset name (see PROVIDER_PRESETS)
==============================  ============================================

Provider Presets
----------------
Use ``PINOCCHIO_PROVIDER`` to quickly switch between providers::

    PINOCCHIO_PROVIDER=openai         # OpenAI GPT-4o
    PINOCCHIO_PROVIDER=deepseek       # DeepSeek Chat
    PINOCCHIO_PROVIDER=ollama         # Local Ollama (default)
    PINOCCHIO_PROVIDER=dashscope      # Alibaba DashScope (Qwen)
    PINOCCHIO_PROVIDER=groq           # Groq cloud
    PINOCCHIO_PROVIDER=together       # Together AI
    PINOCCHIO_PROVIDER=anthropic_compat  # Anthropic via OpenAI compat proxy

Each preset sets model, base_url, and api_key_env.  You can still override
individual fields via their specific environment variables.

Example
-------
>>> from config import PinocchioConfig
>>> cfg = PinocchioConfig()
>>> print(cfg.model)   # "qwen3-vl:4b" (or PINOCCHIO_MODEL env value)
>>>
>>> cfg = PinocchioConfig.from_provider("openai")
>>> print(cfg.model)   # "gpt-4o"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# =====================================================================
# Provider presets
# =====================================================================

@dataclass(frozen=True)
class ProviderPreset:
    """Pre-configured LLM provider settings."""
    name: str
    model: str
    base_url: str
    api_key_env: str  # environment variable that holds the API key
    num_ctx: int = 8192


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        name="ollama",
        model="qwen3-vl:4b",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        num_ctx=8192,
    ),
    "openai": ProviderPreset(
        name="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        num_ctx=128000,
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        num_ctx=64000,
    ),
    "dashscope": ProviderPreset(
        name="dashscope",
        model="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        num_ctx=32000,
    ),
    "groq": ProviderPreset(
        name="groq",
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        num_ctx=32000,
    ),
    "together": ProviderPreset(
        name="together",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        num_ctx=32000,
    ),
    "anthropic_compat": ProviderPreset(
        name="anthropic_compat",
        model="claude-sonnet-4-20250514",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        num_ctx=200000,
    ),
    "siliconflow": ProviderPreset(
        name="siliconflow",
        model="Qwen/Qwen3-8B",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        num_ctx=32000,
    ),
}


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
    >>>
    >>> # Quick provider switch
    >>> cfg = PinocchioConfig.from_provider("openai")
    >>> cfg = PinocchioConfig.from_provider("deepseek")
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

    def __post_init__(self) -> None:
        """Apply provider preset from PINOCCHIO_PROVIDER env var if set."""
        provider_name = os.getenv("PINOCCHIO_PROVIDER", "").lower()
        if provider_name and provider_name in PROVIDER_PRESETS:
            preset = PROVIDER_PRESETS[provider_name]
            # Only override fields that still have their default values
            if not os.getenv("PINOCCHIO_MODEL"):
                self.model = preset.model
            if not os.getenv("OPENAI_BASE_URL"):
                self.base_url = preset.base_url
            if not os.getenv("PINOCCHIO_NUM_CTX"):
                self.num_ctx = preset.num_ctx
            # Always resolve API key from the preset's env var
            if not os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY") == "ollama":
                key = os.getenv(preset.api_key_env, "")
                if key:
                    self.api_key = key

    @classmethod
    def from_provider(
        cls,
        provider: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> PinocchioConfig:
        """Create a config from a provider preset with optional overrides.

        Parameters
        ----------
        provider : Provider name (see ``PROVIDER_PRESETS``).
        model : Override the preset's default model.
        api_key : Explicit API key (otherwise read from the preset's env var).
        **kwargs : Any other ``PinocchioConfig`` fields to override.

        Raises
        ------
        ValueError : If the provider name is not recognised.

        Example
        -------
        >>> cfg = PinocchioConfig.from_provider("openai", model="gpt-4o-mini")
        >>> cfg = PinocchioConfig.from_provider("deepseek")
        """
        if provider.lower() not in PROVIDER_PRESETS:
            available = ", ".join(sorted(PROVIDER_PRESETS.keys()))
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {available}"
            )
        preset = PROVIDER_PRESETS[provider.lower()]
        resolved_key = api_key or os.getenv(preset.api_key_env, "")
        if not resolved_key:
            resolved_key = "ollama"  # fallback to avoid empty string
        return cls(
            model=model or preset.model,
            api_key=resolved_key,
            base_url=preset.base_url,
            num_ctx=preset.num_ctx,
            **kwargs,
        )

    @staticmethod
    def available_providers() -> list[str]:
        """Return a list of supported provider preset names."""
        return sorted(PROVIDER_PRESETS.keys())
