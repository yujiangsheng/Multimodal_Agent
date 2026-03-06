"""StrategyAgent — Phase 2 of the cognitive loop (STRATEGIZE).

Given the perception analysis from Phase 1, this agent selects or constructs
the best approach for fulfilling the user's request, including fallback plans
and risk assessment.

Skills / Capabilities
─────────────────────
1. **Strategy Selection**
   Search procedural memory for proven strategies that match the current task
   type.  If a high-success-rate procedure exists, adapt and reuse it.

2. **Novel Strategy Construction**
   When no existing procedure matches, reason from first principles and design
   a new strategy.  Flag these as experimental so the Learning agent can track
   their effectiveness.

3. **Risk Assessment**
   Identify what could go wrong with the chosen strategy and estimate the
   likelihood of failure.

4. **Fallback Planning**
   Define an alternative approach to use if the primary strategy fails during
   execution.

5. **Modality Pipeline Design**
   For multimodal tasks, specify the order and method of cross-modal processing
   (e.g., "image → caption → reasoning → text").

6. **Fusion Strategy Selection**
   Choose between early, late, or hybrid multimodal fusion based on the nature
   of the task and modalities involved.

7. **Lesson Integration**
   Incorporate relevant lessons from past episodes to avoid known failure modes.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole, FusionStrategy
from pinocchio.models.schemas import PerceptionResult, StrategyResult

_SYSTEM_PROMPT = """\
You are the Strategy sub-agent of Pinocchio, a self-evolving multimodal AI.
Given a perception analysis, you must produce a strategy for completing the task.

Context about past performance will be provided.  Use it to avoid repeating
mistakes and to prefer proven approaches.

Output valid JSON with exactly these keys:
{
  "selected_strategy": "descriptive name of the strategy",
  "basis": "why this strategy was chosen (reference past episodes if applicable)",
  "risk_assessment": "what could go wrong",
  "fallback_plan": "alternative approach if Primary fails",
  "modality_pipeline": "ordered processing pipeline, e.g. image→caption→reasoning→text",
  "fusion_strategy": one of [early_fusion, late_fusion, hybrid_fusion],
  "is_novel": true/false,
  "analysis": "free-text strategic reasoning (2-4 sentences)"
}
"""


class StrategyAgent(BaseAgent):
    """Produces a StrategyResult given insight from the PerceptionAgent."""

    role = AgentRole.STRATEGY

    def run(  # type: ignore[override]
        self,
        perception: PerceptionResult,
        **kwargs: Any,
    ) -> StrategyResult:
        self.logger.phase("Phase 2: STRATEGIZE 策略")
        self._log("Selecting strategy…")

        # Check procedural memory for a proven approach
        best_proc = self.memory.procedural.best_procedure(perception.task_type)
        proc_context = ""
        if best_proc:
            proc_context = (
                f"\nProven procedure found: \"{best_proc.name}\" "
                f"(success rate {best_proc.success_rate:.0%}, used {best_proc.usage_count}x).\n"
                f"Steps: {best_proc.steps}"
            )
            self._log(f"Found proven procedure: {best_proc.name} ({best_proc.success_rate:.0%} success)")
        else:
            self._log("No proven procedure found — will design from first principles")

        lessons_context = ""
        if perception.relevant_lessons:
            lessons_context = "\nLessons from past episodes:\n" + "\n".join(
                f"  - {l}" for l in perception.relevant_lessons
            )

        user_prompt = (
            f"Perception Summary:\n"
            f"  Task type: {perception.task_type.value}\n"
            f"  Complexity: {perception.complexity.value}/5\n"
            f"  Modalities: {[m.value for m in perception.modalities]}\n"
            f"  Confidence: {perception.confidence.value}\n"
            f"  Ambiguities: {perception.ambiguities}\n"
            f"  Analysis: {perception.raw_analysis}\n"
            f"{proc_context}"
            f"{lessons_context}"
        )

        llm_result = self.llm.ask_json(system=_SYSTEM_PROMPT, user=user_prompt, max_tokens=2048)

        try:
            fusion = FusionStrategy(llm_result.get("fusion_strategy", "late_fusion"))
        except ValueError:
            self._warn(f"Invalid fusion_strategy from LLM: {llm_result.get('fusion_strategy')}; defaulting to LATE_FUSION")
            fusion = FusionStrategy.LATE_FUSION

        result = StrategyResult(
            selected_strategy=llm_result.get("selected_strategy", "default"),
            basis=llm_result.get("basis", ""),
            risk_assessment=llm_result.get("risk_assessment", ""),
            fallback_plan=llm_result.get("fallback_plan", ""),
            modality_pipeline=llm_result.get("modality_pipeline", "text→reasoning→text"),
            fusion_strategy=fusion,
            is_novel=llm_result.get("is_novel", True),
            raw_analysis=llm_result.get("analysis", ""),
        )

        self._log(f"Strategy: {result.selected_strategy} | Novel: {result.is_novel}")
        return result
