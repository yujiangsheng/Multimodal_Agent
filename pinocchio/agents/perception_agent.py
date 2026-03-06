"""PerceptionAgent — Phase 1 of the cognitive loop (PERCEIVE).

Responsible for holistic analysis of every incoming user input *before*
any action is taken.  This agent determines what modalities are present,
classifies the task, assesses complexity, retrieves relevant past episodes,
and flags ambiguities.

Skills / Capabilities
─────────────────────
1. **Modality Detection**
   Identify which modalities (text, image, audio, video) are present in the
   input and estimate how central each modality is to the task.

2. **Task Classification**
   Map the user request to one of the predefined ``TaskType`` categories
   (QA, code generation, analysis, creative writing, multimodal reasoning, …).

3. **Complexity Assessment**
   Rate task complexity on a 1–5 scale based on the number of reasoning steps,
   domain expertise required, and modality interactions involved.

4. **Memory Retrieval**
   Query episodic memory for similar past interactions and extract the most
   relevant lessons learned.

5. **Ambiguity Detection**
   Flag under-specified or potentially mis-interpretable aspects of the input
   so that downstream agents can request clarification or apply fallback
   strategies.

6. **Confidence Estimation**
   Produce an overall confidence level (low / medium / high) for the agent's
   ability to handle this request given current knowledge.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import (
    AgentRole,
    Modality,
    TaskType,
    Complexity,
    ConfidenceLevel,
)
from pinocchio.models.schemas import MultimodalInput, PerceptionResult

_SYSTEM_PROMPT = """\
You are the Perception sub-agent of Pinocchio, a self-evolving multimodal AI.
Your job is to ANALYSE the user's input and produce a structured perception report.

You must output valid JSON with exactly these keys:
{
  "task_type": one of [question_answering, content_generation, analysis, translation,
                       summarization, code_generation, creative_writing,
                       multimodal_reasoning, conversation, tool_use, unknown],
  "complexity": integer 1-5,
  "confidence": one of [low, medium, high],
  "ambiguities": [list of strings or empty list],
  "analysis": "free-text analysis of the input (2-4 sentences)"
}

Be precise. If the input contains images or other media references, note them.
"""


class PerceptionAgent(BaseAgent):
    """Analyses incoming input to produce a structured PerceptionResult."""

    role = AgentRole.PERCEPTION

    def run(  # type: ignore[override]
        self,
        user_input: MultimodalInput,
        **kwargs: Any,
    ) -> PerceptionResult:
        self.logger.phase("Phase 1: PERCEIVE 感知")
        self._log("Analysing incoming input…")

        modalities = user_input.modalities
        self._log(f"Detected modalities: {[m.value for m in modalities]}")

        # --- Retrieve similar past episodes ---
        # We need a preliminary task type guess for memory lookup.
        # Ask the LLM for classification and ambiguity analysis.
        user_text = user_input.text or "(non-text input)"
        llm_result = self.llm.ask_json(
            system=_SYSTEM_PROMPT,
            user=f"User input:\n{user_text}",
            max_tokens=2048,
        )

        try:
            task_type = TaskType(llm_result.get("task_type", "unknown"))
        except ValueError:
            self._warn(f"Invalid task_type from LLM: {llm_result.get('task_type')}; defaulting to UNKNOWN")
            task_type = TaskType.UNKNOWN

        try:
            complexity = Complexity(int(llm_result.get("complexity", 3)))
        except (ValueError, TypeError):
            self._warn(f"Invalid complexity from LLM: {llm_result.get('complexity')}; defaulting to MODERATE")
            complexity = Complexity.MODERATE

        try:
            confidence = ConfidenceLevel(llm_result.get("confidence", "medium"))
        except ValueError:
            self._warn(f"Invalid confidence from LLM: {llm_result.get('confidence')}; defaulting to MEDIUM")
            confidence = ConfidenceLevel.MEDIUM

        ambiguities: list[str] = llm_result.get("ambiguities", [])
        if not isinstance(ambiguities, list):
            ambiguities = []
        raw_analysis: str = llm_result.get("analysis", "")

        # --- Memory lookup ---
        similar = self.memory.episodic.find_similar(task_type, modalities, limit=3)
        similar_ids = [ep.episode_id for ep in similar]
        lessons: list[str] = []
        for ep in similar:
            lessons.extend(ep.lessons)

        self._log(f"Task type: {task_type.value} | Complexity: {complexity.value} | Confidence: {confidence.value}")
        if similar_ids:
            self._log(f"Found {len(similar_ids)} similar past episode(s)")
        if ambiguities:
            self._warn(f"Ambiguities detected: {ambiguities}")

        return PerceptionResult(
            modalities=modalities,
            task_type=task_type,
            complexity=complexity,
            similar_episodes=similar_ids,
            relevant_lessons=lessons[:5],
            confidence=confidence,
            ambiguities=ambiguities,
            raw_analysis=raw_analysis,
        )
