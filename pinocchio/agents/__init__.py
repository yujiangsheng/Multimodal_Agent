"""Pinocchio cognitive-loop sub-agents.

Six agents form the self-evolving cognitive loop:

1. :class:`PerceptionAgent`     — PERCEIVE: analyse & classify input
2. :class:`StrategyAgent`       — STRATEGIZE: select approach & plan
3. :class:`ExecutionAgent`      — EXECUTE: generate the response
4. :class:`EvaluationAgent`     — EVALUATE: score quality & effectiveness
5. :class:`LearningAgent`       — LEARN: extract lessons, update memory
6. :class:`MetaReflectionAgent` — META-REFLECT: periodic higher-order analysis

All inherit from :class:`BaseAgent` which provides shared LLM, memory,
and logging access.
"""

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.agents.perception_agent import PerceptionAgent
from pinocchio.agents.strategy_agent import StrategyAgent
from pinocchio.agents.execution_agent import ExecutionAgent
from pinocchio.agents.evaluation_agent import EvaluationAgent
from pinocchio.agents.learning_agent import LearningAgent
from pinocchio.agents.meta_reflection_agent import MetaReflectionAgent

__all__ = [
    "BaseAgent",
    "PerceptionAgent",
    "StrategyAgent",
    "ExecutionAgent",
    "EvaluationAgent",
    "LearningAgent",
    "MetaReflectionAgent",
]
