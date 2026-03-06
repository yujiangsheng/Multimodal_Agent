"""MetaReflectionAgent — Phase 6 of the cognitive loop (META-REFLECT).

Triggered periodically (every N interactions) to perform higher-order
reflection over recent performance trends, detect cognitive biases, assess
learning efficiency, and produce an evolution plan.

Skills / Capabilities
─────────────────────
1. **Pattern Analysis**
   Analyse accumulated episodes to identify recurring error types, domains
   of strength, domains of weakness, and the trajectory of strategy evolution.

2. **Cognitive Bias Detection**
   Check if the agent is over-relying on certain strategies, avoiding certain
   task types, or miscalibrating its confidence.

3. **Learning Efficiency Assessment**
   Evaluate whether enough value is being extracted from each interaction,
   whether lessons are appropriately specific vs. general, and whether
   procedural memory is actually improving performance.

4. **Evolution Plan Generation**
   Produce a ranked list of priority improvement areas, experimental
   strategies to try, and knowledge gaps to actively fill.

5. **Improvement Trend Tracking**
   Compare recent performance metrics against historical averages to detect
   improvement or regression.

6. **Strategy Portfolio Rebalancing**
   Identify over-used and under-used strategies and recommend diversification.

7. **Cross-Domain Knowledge Transfer**
   Spot opportunities to apply lessons from one domain to another, promoting
   cross-pollination of insights.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import AgentRole
from pinocchio.models.schemas import MetaReflectionResult
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger

_SYSTEM_PROMPT = """\
You are the Meta-Reflection sub-agent of Pinocchio, a self-evolving multimodal AI.
You perform periodic higher-order reflection on the agent's overall performance.

You are given a summary of recent episodes, error frequencies, improvement trends,
and memory statistics.  Analyse these and produce a meta-reflection report.

Output valid JSON:
{
  "recurring_errors": ["list of recurring error patterns"],
  "strength_domains": ["domains where agent performs well"],
  "weakness_domains": ["domains needing improvement"],
  "strategy_trajectory": "how strategies have evolved recently",
  "bias_check": "assessment of cognitive biases detected",
  "learning_efficiency": "assessment of learning quality",
  "priority_improvements": ["ranked list of areas to improve"],
  "experimental_strategies": ["novel strategies to try"],
  "knowledge_gaps": ["information the agent should seek"],
  "analysis": "overall meta-reflection narrative (3-5 sentences)"
}
"""

_DEFAULT_META_REFLECT_INTERVAL = 5  # trigger every N interactions


class MetaReflectionAgent(BaseAgent):
    """Performs periodic meta-cognitive reflection over accumulated experience."""

    role = AgentRole.META_REFLECTION

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        logger: PinocchioLogger,
        *,
        meta_reflect_interval: int | None = None,
    ) -> None:
        super().__init__(llm, memory, logger)
        self._interval = meta_reflect_interval or _DEFAULT_META_REFLECT_INTERVAL

    def should_trigger(self) -> bool:
        """Check if it's time for a meta-reflection cycle."""
        total_episodes = self.memory.episodic.count
        return total_episodes > 0 and total_episodes % self._interval == 0

    def run(self, **kwargs: Any) -> MetaReflectionResult:  # type: ignore[override]
        self.logger.phase("Phase 6: META-REFLECT 元反思")
        self._log("Performing higher-order self-reflection…")

        # Gather analytics
        summary = self.memory.summary()
        trend = self.memory.improvement_trend(window=self._interval)
        recent_lessons = self.memory.episodic.recent_lessons(limit=15)
        error_freq = self.memory.episodic.error_frequency()

        context = (
            f"=== MEMORY SUMMARY ===\n"
            f"Total episodes: {summary['episodic_count']}\n"
            f"Semantic entries: {summary['semantic_count']}\n"
            f"Procedures: {summary['procedural_count']}\n"
            f"Average score: {summary['avg_score']}\n"
            f"Top procedures: {summary['top_procedures']}\n\n"
            f"=== IMPROVEMENT TREND (window={self._interval}) ===\n{trend}\n\n"
            f"=== ERROR FREQUENCY ===\n{error_freq}\n\n"
            f"=== RECENT LESSONS ===\n"
            + "\n".join(f"  - {l}" for l in recent_lessons)
        )

        llm_result = self.llm.ask_json(system=_SYSTEM_PROMPT, user=context, max_tokens=2048)

        result = MetaReflectionResult(
            recurring_errors=llm_result.get("recurring_errors", []),
            strength_domains=llm_result.get("strength_domains", []),
            weakness_domains=llm_result.get("weakness_domains", []),
            strategy_trajectory=llm_result.get("strategy_trajectory", ""),
            bias_check=llm_result.get("bias_check", ""),
            learning_efficiency=llm_result.get("learning_efficiency", ""),
            priority_improvements=llm_result.get("priority_improvements", []),
            experimental_strategies=llm_result.get("experimental_strategies", []),
            knowledge_gaps=llm_result.get("knowledge_gaps", []),
            raw_analysis=llm_result.get("analysis", ""),
        )

        self._log(f"Strengths: {result.strength_domains}")
        self._warn(f"Weaknesses: {result.weakness_domains}")
        self._log(f"Priority improvements: {result.priority_improvements}")
        self.logger.separator()

        return result
