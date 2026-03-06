"""Shared pytest configuration and fixtures for the Pinocchio test suite.

Fixtures provided:
    * ``tmp_data_dir``  — temporary directory (str) for memory persistence tests
    * ``mock_llm``      — MagicMock mimicking :class:`LLMClient`
    * ``mock_logger``   — MagicMock mimicking :class:`PinocchioLogger`
    * ``memory_manager`` — real :class:`MemoryManager` backed by ``tmp_data_dir``

See ``pyproject.toml`` [tool.pytest.ini_options] for pytest settings.
"""

from __future__ import annotations

import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so ``import config`` etc. work
# even when tests are invoked without ``pip install -e .``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory for memory persistence tests."""
    return str(tmp_path)


@pytest.fixture
def mock_llm():
    """Return a MagicMock that mimics :class:`LLMClient`.

    Pre-configures ``ask_json`` to return an empty dict by default.
    Individual tests can override with ``mock_llm.ask_json.return_value = ...``.
    """
    llm = MagicMock()
    llm.model = "test-model"
    llm.temperature = 0.7
    llm.max_tokens = 4096
    llm.ask_json.return_value = {}
    llm.ask.return_value = "mock response"
    llm.chat.return_value = "mock response"
    return llm


@pytest.fixture
def mock_logger():
    """Return a MagicMock that mimics :class:`PinocchioLogger`."""
    logger = MagicMock()
    return logger


@pytest.fixture
def memory_manager(tmp_data_dir):
    """Return a real :class:`MemoryManager` backed by a temp directory."""
    from pinocchio.memory.memory_manager import MemoryManager
    return MemoryManager(data_dir=tmp_data_dir)


@pytest.fixture(autouse=True)
def _disable_fast_path(monkeypatch):
    """Disable fast path for all tests so the full cognitive loop is exercised.

    Individual tests that need to verify fast-path behaviour can re-enable it
    by setting ``agent.FAST_PATH_MAX_LENGTH = 500`` in the test body.
    """
    from pinocchio.orchestrator import Pinocchio
    monkeypatch.setattr(Pinocchio, "FAST_PATH_MAX_LENGTH", 0)
