"""LearningAgent — Phase 5 of the cognitive loop (LEARN).

Extracts and consolidates learnings from the completed interaction, updating
all three memory stores (episodic, semantic, procedural) with new insights.

Skills / Capabilities
─────────────────────
1. **Lesson Extraction**
   Distil concrete, reusable lessons from the evaluation results — not vague
   platitudes but specific, actionable insights.

2. **Episodic Memory Update**
   Create a structured ``EpisodicRecord`` capturing the full interaction trace
   and store it in episodic memory.

3. **Semantic Memory Update**
   When a lesson is generalizable across tasks, create or reinforce a
   ``SemanticEntry`` in semantic memory.

4. **Procedural Memory Update**
   If a strategy proved effective (score ≥ 7), codify it as a
   ``ProceduralEntry`` or refine an existing one.  If it failed, downgrade
   the procedure's success rate.

5. **Strategy Refinement Proposal**
   Suggest specific modifications to the strategy for next time (what to
   keep, change, or drop).

6. **Skill Gap Identification**
   Identify areas where the agent's capability was insufficient, providing
   direction for deliberate practice.

7. **Knowledge Synthesis Triggering**
   When a domain reaches the episode threshold (10+), trigger a synthesis
   pass to distil high-level heuristics from accumulated episodes.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole, TaskType
from pinocchio.models.schemas import (
    PerceptionResult,
    StrategyResult,
    EvaluationResult,
    LearningResult,
    EpisodicRecord,
    SemanticEntry,
    ProceduralEntry,
)

_SYSTEM_PROMPT = """\
You are the Learning sub-agent of Pinocchio, a self-evolving multimodal AI.
Based on an interaction's perception, strategy, and evaluation, extract
actionable lessons and suggest memory updates.

Output valid JSON with these keys:
{
  "new_lessons": ["lesson 1", "lesson 2", ...],
  "strategy_refinements": "specific changes for next time",
  "skill_gap": "area needing improvement or empty string",
  "self_improvement_action": "concrete next step to get better",
  "semantic_knowledge": "a generalizable insight to store (or empty string)",
  "should_save_procedure": true/false,
  "procedure_name": "short name for the procedure (if saving)",
  "procedure_steps": ["step1", "step2", ...] (if saving)
}

Be specific. "Try harder" is NOT a useful lesson. Prefer "When handling X-type
tasks, start by Y instead of Z because…".
"""


class LearningAgent(BaseAgent):
    """Consolidates learnings and updates the three memory stores."""

    role = AgentRole.LEARNING

    def run(  # type: ignore[override]
        self,
        user_input_text: str,
        perception: PerceptionResult,
        strategy: StrategyResult,
        evaluation: EvaluationResult,
        **kwargs: Any,
    ) -> LearningResult:
        self.logger.phase("Phase 5: LEARN 学习")
        self._log("Extracting lessons and updating memory…")

        learn_prompt = (
            f"=== PERCEPTION ===\n"
            f"Task type: {perception.task_type.value}\n"
            f"Complexity: {perception.complexity.value}\n\n"
            f"=== STRATEGY ===\n"
            f"Strategy: {strategy.selected_strategy}\n"
            f"Novel: {strategy.is_novel}\n"
            f"Risk: {strategy.risk_assessment}\n\n"
            f"=== EVALUATION ===\n"
            f"Quality: {evaluation.output_quality}/10\n"
            f"Strategy effectiveness: {evaluation.strategy_effectiveness}/10\n"
            f"Went well: {evaluation.went_well}\n"
            f"Went wrong: {evaluation.went_wrong}\n"
            f"Surprises: {evaluation.surprises}\n"
        )

        llm_result = self.llm.ask_json(system=_SYSTEM_PROMPT, user=learn_prompt, max_tokens=2048)

        # --- 1. Build episodic record ---
        episode = EpisodicRecord(
            task_type=perception.task_type,
            modalities=perception.modalities,
            user_intent=user_input_text[:200],
            strategy_used=strategy.selected_strategy,
            outcome_score=evaluation.output_quality,
            lessons=llm_result.get("new_lessons", []),
            error_patterns=[e for e in evaluation.went_wrong],
            improvement_notes=llm_result.get("strategy_refinements", ""),
        )
        self.memory.store_episode(episode)
        self._log(f"Stored episode {episode.episode_id} (score: {episode.outcome_score}/10)")

        # --- 2. Semantic memory update ---
        semantic_updates: list[str] = []
        semantic_knowledge = llm_result.get("semantic_knowledge", "")
        if semantic_knowledge:
            entry = SemanticEntry(
                domain=perception.task_type.value,
                knowledge=semantic_knowledge,
                source_episodes=[episode.episode_id],
                confidence=evaluation.output_quality / 10.0,
            )
            self.memory.store_knowledge(entry)
            semantic_updates.append(semantic_knowledge)
            self._log(f"Added semantic knowledge entry: {entry.entry_id}")

        # --- 3. Procedural memory update ---
        procedural_updates: list[str] = []
        if llm_result.get("should_save_procedure") and evaluation.output_quality >= 7:
            proc = ProceduralEntry(
                task_type=perception.task_type,
                name=llm_result.get("procedure_name", strategy.selected_strategy),
                description=f"Strategy for {perception.task_type.value} tasks",
                steps=llm_result.get("procedure_steps", []),
                success_rate=evaluation.output_quality / 10.0,
                usage_count=1,
            )
            self.memory.store_procedure(proc)
            procedural_updates.append(proc.name)
            self._log(f"Saved new procedure: {proc.name}")
        elif not llm_result.get("should_save_procedure"):
            # Update existing procedure success rate if one was used
            best = self.memory.procedural.best_procedure(perception.task_type)
            if best:
                success = evaluation.output_quality >= 6
                self.memory.record_procedure_usage(best.entry_id, success)

        result = LearningResult(
            new_lessons=llm_result.get("new_lessons", []),
            episodic_update=f"Episode {episode.episode_id} stored",
            semantic_updates=semantic_updates,
            procedural_updates=procedural_updates,
            strategy_refinements=llm_result.get("strategy_refinements", ""),
            skill_gap=llm_result.get("skill_gap", ""),
            self_improvement_action=llm_result.get("self_improvement_action", ""),
            raw_analysis=str(llm_result),
        )

        if result.skill_gap:
            self._warn(f"Skill gap identified: {result.skill_gap}")

        return result
