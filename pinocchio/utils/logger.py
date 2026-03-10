"""Pinocchio Logger — structured logging for agent activity.

Provides a consistent, colour-coded logging façade that tags every message
with the originating agent role for easy tracing.

Features
--------
- **Role-tagged output** — each log line is prefixed with the agent role
  (ORCHESTRATOR, PERCEPTION, STRATEGY, etc.)
- **ANSI colour coding** — distinct colours per role for readability in
  terminals; automatically disabled for non-TTY output (pipes, files)
- **Phase markers** — visual separators for cognitive-loop transitions
- **Structured data** — optional dict payload rendered as indented JSON

Usage
-----
>>> from pinocchio.utils.logger import PinocchioLogger
>>> logger = PinocchioLogger()
>>> logger.info(AgentRole.ORCHESTRATOR, "Starting cognitive loop")
>>> logger.phase("Phase 1: PERCEIVE")
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from pinocchio.models.enums import AgentRole

# Colour codes (ANSI)
_COLOURS = {
    AgentRole.ORCHESTRATOR: "\033[1;36m",      # Bold Cyan
    AgentRole.PERCEPTION: "\033[0;33m",         # Yellow
    AgentRole.STRATEGY: "\033[0;35m",           # Magenta
    AgentRole.EXECUTION: "\033[0;32m",          # Green
    AgentRole.EVALUATION: "\033[0;34m",         # Blue
    AgentRole.LEARNING: "\033[1;33m",           # Bold Yellow
    AgentRole.META_REFLECTION: "\033[1;35m",    # Bold Magenta
    AgentRole.TEXT_PROCESSOR: "\033[0;37m",      # White
    AgentRole.VISION_PROCESSOR: "\033[0;31m",   # Red
    AgentRole.AUDIO_PROCESSOR: "\033[0;36m",    # Cyan
    AgentRole.VIDEO_PROCESSOR: "\033[1;31m",    # Bold Red
}
_RESET = "\033[0m"


class PinocchioLogger:
    """Structured logger for the Pinocchio agent system.

    Parameters
    ----------
    level : int
        Standard :mod:`logging` level (default ``logging.INFO``).
    use_colour : bool | None
        ``True`` = force ANSI colours, ``False`` = plain text,
        ``None`` (default) = auto-detect based on ``sys.stdout.isatty()``.

    Examples
    --------
    >>> logger = PinocchioLogger()                     # auto colour
    >>> logger = PinocchioLogger(use_colour=False)     # no colour (for log files)
    """

    def __init__(self, level: int = logging.INFO, *, use_colour: bool | None = None) -> None:
        self._logger = logging.getLogger("pinocchio")
        self._logger.setLevel(level)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        # Resolve colour mode: explicit > auto-detect
        if use_colour is None:
            self._colour = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        else:
            self._colour = use_colour

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, role: AgentRole, message: str, data: dict[str, Any] | None = None) -> None:
        """Log a message tagged with the originating agent role."""
        if self._colour:
            colour = _COLOURS.get(role, "")
            prefix = f"{colour}[{role.value.upper()}]{_RESET}"
        else:
            prefix = f"[{role.value.upper()}]"
        line = f"{prefix} {message}"
        if data:
            line += f"\n  {json.dumps(data, ensure_ascii=False, indent=2)}"
        self._logger.info(line)

    def phase(self, phase_name: str) -> None:
        """Log a cognitive-loop phase transition with a visual separator."""
        bar = "─" * 50
        if self._colour:
            self._logger.info(f"\033[1;37m{bar}\n  ◆ {phase_name}\n{bar}{_RESET}")
        else:
            self._logger.info(f"{bar}\n  ◆ {phase_name}\n{bar}")

    def separator(self) -> None:
        """Print a visual separator line."""
        if self._colour:
            self._logger.info("\033[0;90m" + "═" * 60 + _RESET)
        else:
            self._logger.info("═" * 60)

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def info(self, role: AgentRole, msg: str) -> None:
        self.log(role, msg)

    def warn(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"⚠  {msg}")

    def error(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"✖  {msg}")

    def success(self, role: AgentRole, msg: str) -> None:
        self.log(role, f"✔  {msg}")
