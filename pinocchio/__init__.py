"""Pinocchio — a multimodal self-evolving agent.

Top-level package.  The primary public API is the :class:`Pinocchio`
orchestrator, which drives the 6-phase cognitive loop::

    from pinocchio import Pinocchio

    agent = Pinocchio()
    print(agent.chat("Hello, Pinocchio!"))

For multimodal input, pass file paths via keyword arguments::

    agent.chat("Describe this image", image_paths=["photo.jpg"])
    agent.chat("Summarise this recording", audio_paths=["meeting.wav"])

All internal sub-modules (agents, memory, models, multimodal, utils)
are implementation details and should not normally be imported directly.
"""

from pinocchio.orchestrator import Pinocchio
from pinocchio.tools import Tool, ToolRegistry, ToolExecutor, tool

__all__ = ["Pinocchio", "Tool", "ToolRegistry", "ToolExecutor", "tool"]
