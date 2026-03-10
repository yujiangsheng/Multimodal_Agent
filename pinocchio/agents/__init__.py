"""Pinocchio cognitive-loop agents.

:class:`PinocchioAgent` is the unified cognitive agent that consolidates
all six phases of the self-evolving cognitive loop into a single class
with distinct skill methods:

1. :meth:`~PinocchioAgent.perceive`      — PERCEIVE: analyse & classify input
2. :meth:`~PinocchioAgent.strategize`    — STRATEGIZE: select approach & plan
3. :meth:`~PinocchioAgent.execute`       — EXECUTE: generate the response
4. :meth:`~PinocchioAgent.evaluate`      — EVALUATE: score quality & effectiveness
5. :meth:`~PinocchioAgent.learn`         — LEARN: extract lessons, update memory
6. :meth:`~PinocchioAgent.meta_reflect`  — META-REFLECT: periodic higher-order analysis

Inheritance::

    BaseAgent (ABC)
    ├── PinocchioAgent     — cognitive loop (6 skills)
    ├── TextProcessor      — multimodal: text
    ├── VisionProcessor    — multimodal: image
    ├── AudioProcessor     — multimodal: audio
    └── VideoProcessor     — multimodal: video
"""

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.agents.unified_agent import PinocchioAgent

__all__ = [
    "BaseAgent",
    "PinocchioAgent",
]
