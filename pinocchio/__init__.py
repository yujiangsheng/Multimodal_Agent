"""Pinocchio — a multimodal self-evolving agent.

Top-level package.  Import the orchestrator directly::

    from pinocchio import Pinocchio
    agent = Pinocchio()
    response = agent.chat("Hello, Pinocchio!")
"""

from pinocchio.orchestrator import Pinocchio

__all__ = ["Pinocchio"]
