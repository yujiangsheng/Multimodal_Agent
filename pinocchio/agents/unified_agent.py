"""PinocchioAgent — unified cognitive agent with six skill methods.

Consolidates all six phases of the self-evolving cognitive loop into
one class.  Each phase is implemented as a distinct public method:

    PERCEIVE → STRATEGIZE → EXECUTE → EVALUATE → LEARN → META-REFLECT

Skills
------
1. :meth:`perceive`      — analyse & classify user input
2. :meth:`strategize`    — select approach, retrieve proven procedures
3. :meth:`execute`       — generate the user-facing response
4. :meth:`evaluate`      — score quality, detect truncation
5. :meth:`learn`         — extract lessons, update all memory stores
6. :meth:`meta_reflect`  — periodic higher-order self-analysis

Helper methods:

- :meth:`continue_response`  — complete a truncated response
- :meth:`should_meta_reflect` — check if meta-reflection is due
- :meth:`_heuristic_completeness_check` — fast truncation detection

The agent inherits from :class:`BaseAgent` and uses the shared LLM,
memory, and logging infrastructure.

Example::

    agent = PinocchioAgent(llm, memory, logger)
    perception = agent.perceive(user_input=input_obj)
    strategy   = agent.strategize(perception=perception)
    response   = agent.execute(user_input=input_obj, perception=perception, strategy=strategy)
    evaluation = agent.evaluate(user_input=input_obj, perception=perception, strategy=strategy, response=response)
    learning   = agent.learn(user_input_text="...", perception=perception, strategy=strategy, evaluation=evaluation)
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.memory.memory_manager import MemoryManager
from pinocchio.models.enums import (
    AgentRole,
    Complexity,
    ConfidenceLevel,
    FusionStrategy,
    Modality,
    TaskType,
)
from pinocchio.models.schemas import (
    AgentMessage,
    EpisodicRecord,
    EvaluationResult,
    LearningResult,
    MetaReflectionResult,
    MultimodalInput,
    PerceptionResult,
    ProceduralEntry,
    SemanticEntry,
    StrategyResult,
)
from pinocchio.multimodal.vision_processor import VisionProcessor
from pinocchio.utils.llm_client import LLMClient
from pinocchio.utils.logger import PinocchioLogger

if TYPE_CHECKING:
    from pinocchio.tools import ToolExecutor, ToolRegistry


# =====================================================================
# System Prompts (one per skill)
# =====================================================================

_PERCEIVE_PROMPT = """\
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

_STRATEGIZE_PROMPT = """\
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

_EXECUTE_PROMPT = """\
You are the Execution sub-agent of Pinocchio, a self-evolving multimodal AI.
Your task is to produce the BEST possible response to the user's request.

You are given:
1. The original user input
2. A perception analysis
3. A strategy plan

Follow the strategy's modality pipeline.  If an intermediate step seems to
fail, note the issue and switch to the fallback plan.

Your output must be the final, polished response to the user.
Write in the same language the user uses.
Be thorough, accurate, and helpful.

CRITICAL: Your response MUST be COMPLETE.  Never stop mid-sentence,
mid-list, or mid-thought.  Always end with a proper conclusion.
If you are writing code, close all brackets and functions.
If you are writing a numbered list, finish ALL items.
If explaining multiple aspects, cover each one before stopping.
"""

_CONTINUATION_PROMPT = """\
You are the Execution sub-agent of Pinocchio, a self-evolving multimodal AI.
The previous response was INCOMPLETE — it was cut off before finishing.

You are given:
1. The original user input
2. The partial response that was already generated

Your job is to CONTINUE and COMPLETE the response from where it left off.
Do NOT repeat what was already said.  Start exactly where the previous
response ended and finish the thought/list/code/explanation.

Your output should seamlessly continue the partial response and bring it
to a proper, natural conclusion.
Write in the same language the user uses.
"""

_EVALUATE_PROMPT = """\
You are the Evaluation sub-agent of Pinocchio, a self-evolving multimodal AI.
Your job is to rigorously evaluate the quality of a response that was just
generated.

You are given:
1. The user's original input
2. Perception analysis
3. Strategy used
4. The actual response produced

You MUST pay special attention to **response completeness**:
- Is the response complete and well-formed?
- Does the response end naturally with a proper conclusion?
- Does it address ALL parts of the user's question?
- Is it cut off mid-sentence, mid-list, or mid-thought?
- Are there numbered lists that start but don't finish?
- Are there promises of explanation that are never delivered?
- For code: is the code complete with all functions closed?

Output valid JSON with exactly these keys:
{
  "task_completion": "complete" or "partial" or "failed",
  "output_quality": integer 1-10,
  "strategy_effectiveness": integer 1-10,
  "went_well": ["list of positives"],
  "went_wrong": ["list of negatives or empty"],
  "surprises": ["unexpected elements encountered or empty"],
  "cross_modal_coherence": integer 1-10 (use 5 if single modality),
  "is_complete": true/false (is the response fully formed without truncation?),
  "incompleteness_details": "why the response is incomplete, or empty string",
  "analysis": "free-text evaluation summary (2-4 sentences)"
}

IMPORTANT: A truncated or incomplete response MUST get:
  - is_complete: false
  - task_completion: "partial" or "failed"
  - output_quality: no higher than 5

Be honest and critical.  Over-generous ratings reduce learning quality.
"""

_LEARN_PROMPT = """\
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

_META_REFLECT_PROMPT = """\
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


# =====================================================================
# Constants
# =====================================================================

_DEFAULT_META_REFLECT_INTERVAL = 5
_MAX_AUTO_CONTINUATIONS = 2

# Characters that signal a natural sentence/paragraph ending.
# Used by both _looks_complete() and _heuristic_completeness_check().
_TERMINAL_CHARS = set(
    ".!?。！？…\"'`）)]}】》❯→~—–-：:；;、，,"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "\n"
)


def _looks_complete(text: str) -> bool:
    """Quick heuristic: does the text end at a sentence boundary?"""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) < 10:
        return False
    if stripped.count("```") % 2 != 0:
        return False
    if stripped[-1] in _TERMINAL_CHARS:
        return True
    if len(stripped) < 200:
        return True
    return False


# =====================================================================
# Unified Agent
# =====================================================================


class PinocchioAgent(BaseAgent):
    """Unified cognitive agent with six skill methods.

    Each skill method corresponds to one phase of the cognitive loop.
    The orchestrator calls them in sequence::

        perception  = agent.perceive(user_input=…)
        strategy    = agent.strategize(perception=perception)
        response    = agent.execute(user_input=…, perception=…, strategy=…)
        evaluation  = agent.evaluate(user_input=…, perception=…, strategy=…, response=…)
        learning    = agent.learn(user_input_text=…, perception=…, strategy=…, evaluation=…)
        meta_result = agent.meta_reflect()   # periodic

    The ``run()`` method is intentionally disabled — always use the
    named skill methods above.

    Parameters
    ----------
    llm : LLMClient
        Shared LLM client for all cognitive phases.
    memory : MemoryManager
        Unified dual-axis memory manager.
    logger : PinocchioLogger
        Structured colour-coded logger.
    meta_reflect_interval : int | None
        Number of interactions between meta-reflection triggers
        (default: 5).
    """

    role = AgentRole.ORCHESTRATOR

    # Keep a class-level reference for image encoding
    _encode_image = staticmethod(VisionProcessor._encode_image)

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        logger: PinocchioLogger,
        *,
        meta_reflect_interval: int | None = None,
    ) -> None:
        super().__init__(llm, memory, logger)
        self._meta_reflect_interval = meta_reflect_interval or _DEFAULT_META_REFLECT_INTERVAL
        self._tool_registry: ToolRegistry | None = None
        self._tool_executor: ToolExecutor | None = None

    def set_tools(self, registry: ToolRegistry, executor: ToolExecutor) -> None:
        """Attach a tool registry and executor for the EXECUTE phase."""
        self._tool_registry = registry
        self._tool_executor = executor

    def run(self, **kwargs: Any) -> Any:
        """Not used directly — use specific skill methods."""
        raise NotImplementedError(
            "Use specific skill methods: perceive, strategize, execute, "
            "evaluate, learn, meta_reflect"
        )

    # =================================================================
    # Skill 1: PERCEIVE
    # =================================================================

    def perceive(
        self,
        user_input: MultimodalInput,
        **kwargs: Any,
    ) -> PerceptionResult:
        """Analyse incoming input to produce a structured PerceptionResult."""
        self.logger.phase("Phase 1: PERCEIVE 感知")
        self.logger.info(AgentRole.PERCEPTION, "Analysing incoming input…")

        modalities = user_input.modalities
        self.logger.info(AgentRole.PERCEPTION, f"Detected modalities: {[m.value for m in modalities]}")

        user_text = user_input.text or "(non-text input)"
        llm_result = self.llm.ask_json(
            system=_PERCEIVE_PROMPT,
            user=f"User input:\n{user_text}",
            max_tokens=2048,
        )

        try:
            task_type = TaskType(llm_result.get("task_type", "unknown"))
        except ValueError:
            self.logger.warn(AgentRole.PERCEPTION, f"Invalid task_type from LLM: {llm_result.get('task_type')}; defaulting to UNKNOWN")
            task_type = TaskType.UNKNOWN

        try:
            complexity = Complexity(int(llm_result.get("complexity", 3)))
        except (ValueError, TypeError):
            self.logger.warn(AgentRole.PERCEPTION, f"Invalid complexity from LLM: {llm_result.get('complexity')}; defaulting to MODERATE")
            complexity = Complexity.MODERATE

        try:
            confidence = ConfidenceLevel(llm_result.get("confidence", "medium"))
        except ValueError:
            self.logger.warn(AgentRole.PERCEPTION, f"Invalid confidence from LLM: {llm_result.get('confidence')}; defaulting to MEDIUM")
            confidence = ConfidenceLevel.MEDIUM

        ambiguities: list[str] = llm_result.get("ambiguities", [])
        if not isinstance(ambiguities, list):
            ambiguities = []
        raw_analysis: str = llm_result.get("analysis", "")

        similar = self.memory.episodic.find_similar(task_type, modalities, limit=3)
        similar_ids = [ep.episode_id for ep in similar]
        lessons: list[str] = []
        for ep in similar:
            lessons.extend(ep.lessons)

        self.logger.info(AgentRole.PERCEPTION, f"Task type: {task_type.value} | Complexity: {complexity.value} | Confidence: {confidence.value}")
        if similar_ids:
            self.logger.info(AgentRole.PERCEPTION, f"Found {len(similar_ids)} similar past episode(s)")
        if ambiguities:
            self.logger.warn(AgentRole.PERCEPTION, f"Ambiguities detected: {ambiguities}")

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

    # =================================================================
    # Skill 2: STRATEGIZE
    # =================================================================

    def strategize(
        self,
        perception: PerceptionResult,
        **kwargs: Any,
    ) -> StrategyResult:
        """Produce a StrategyResult given insight from the perceive phase."""
        self.logger.phase("Phase 2: STRATEGIZE 策略")
        self.logger.info(AgentRole.STRATEGY, "Selecting strategy…")

        best_proc = self.memory.procedural.best_procedure(perception.task_type)
        proc_context = ""
        if best_proc:
            proc_context = (
                f"\nProven procedure found: \"{best_proc.name}\" "
                f"(success rate {best_proc.success_rate:.0%}, used {best_proc.usage_count}x).\n"
                f"Steps: {best_proc.steps}"
            )
            self.logger.info(AgentRole.STRATEGY, f"Found proven procedure: {best_proc.name} ({best_proc.success_rate:.0%} success)")
        else:
            self.logger.info(AgentRole.STRATEGY, "No proven procedure found — will design from first principles")

        lessons_context = ""
        if perception.relevant_lessons:
            lessons_context = "\nLessons from past episodes:\n" + "\n".join(
                f"  - {lesson}" for lesson in perception.relevant_lessons
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

        llm_result = self.llm.ask_json(system=_STRATEGIZE_PROMPT, user=user_prompt, max_tokens=2048)

        try:
            fusion = FusionStrategy(llm_result.get("fusion_strategy", "late_fusion"))
        except ValueError:
            self.logger.warn(AgentRole.STRATEGY, f"Invalid fusion_strategy from LLM: {llm_result.get('fusion_strategy')}; defaulting to LATE_FUSION")
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

        self.logger.info(AgentRole.STRATEGY, f"Strategy: {result.selected_strategy} | Novel: {result.is_novel}")
        return result

    # =================================================================
    # Skill 3: EXECUTE
    # =================================================================

    def _resolve_image_urls(self, paths: list[str]) -> list[str]:
        """Convert a list of file paths / URLs to URLs the LLM can consume."""
        return [
            p if p.startswith(("http://", "https://", "data:")) else self._encode_image(p)
            for p in paths
        ]

    def execute(
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        modality_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        """Generate the user-facing response following the strategy plan."""
        self.logger.phase("Phase 3: EXECUTE 执行")
        self.logger.info(AgentRole.EXECUTION, f"Executing strategy: {strategy.selected_strategy}")

        user_text = user_input.text or "(non-text input)"

        modality_section = ""
        if modality_context:
            parts = []
            for mod_name, mod_desc in modality_context.items():
                parts.append(f"[{mod_name.upper()}] {mod_desc}")
            modality_section = (
                "\n\n=== MULTIMODAL CONTEXT ===\n"
                + "\n".join(parts)
            )

        # Inject tool descriptions into the execution prompt
        tool_section = ""
        exec_prompt = _EXECUTE_PROMPT
        if self._tool_registry and self._tool_registry.count > 0:
            tool_desc = self._tool_registry.to_prompt_description()
            tool_section = (
                "\n\n=== AVAILABLE TOOLS ===\n"
                f"{tool_desc}\n\n"
                "To use a tool, include a tool-call block in your response:\n"
                "```tool_call\n"
                '{"tool": "tool_name", "arguments": {"param": "value"}}\n'
                "```\n"
                "You may include multiple tool-call blocks. The results will "
                "be injected automatically and you should incorporate them "
                "into your final answer."
            )

        context_prompt = (
            f"=== PERCEPTION ===\n"
            f"Task type: {perception.task_type.value}\n"
            f"Complexity: {perception.complexity.value}/5\n"
            f"Modalities: {[m.value for m in perception.modalities]}\n"
            f"Ambiguities: {perception.ambiguities}\n\n"
            f"=== STRATEGY ===\n"
            f"Strategy: {strategy.selected_strategy}\n"
            f"Pipeline: {strategy.modality_pipeline}\n"
            f"Fusion: {strategy.fusion_strategy.value}\n"
            f"Risk: {strategy.risk_assessment}\n"
            f"Fallback: {strategy.fallback_plan}\n\n"
            f"=== USER INPUT ===\n{user_text}"
            f"{modality_section}"
            f"{tool_section}"
        )

        if user_input.image_paths:
            self.logger.info(AgentRole.EXECUTION, "Building multimodal (vision) request…")
            image_urls = self._resolve_image_urls(user_input.image_paths)
            vision_msg = self.llm.build_vision_message(context_prompt, image_urls)
            messages = [
                {"role": "system", "content": exec_prompt},
                vision_msg,
            ]
            response_text = self.llm.chat(messages)
        else:
            response_text = self.llm.ask(system=exec_prompt, user=context_prompt)

        # ---- Tool-call detection and execution ----
        response_text = self._process_tool_calls(response_text, exec_prompt, user_text)

        # Auto-continuation when token limit is hit
        finish_reason = self.llm.last_finish_reason
        auto_rounds = 0
        while finish_reason == "length" and auto_rounds < _MAX_AUTO_CONTINUATIONS:
            auto_rounds += 1
            self.logger.info(
                AgentRole.EXECUTION,
                f"Token limit reached (finish_reason=length) — "
                f"auto-continuing ({auto_rounds}/{_MAX_AUTO_CONTINUATIONS})…",
            )
            cont_prompt = (
                f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
                f"=== PARTIAL RESPONSE (already generated) ===\n"
                f"{response_text}\n\n"
                f"Continue the response from where it left off. "
                f"Do NOT repeat what was already said. "
                f"Bring the response to a proper, natural conclusion."
            )
            chunk = self.llm.ask(system=_CONTINUATION_PROMPT, user=cont_prompt)
            if not chunk.strip():
                self.logger.info(AgentRole.EXECUTION, "Continuation returned empty — stopping auto-continue")
                break
            response_text = response_text.rstrip() + "\n" + chunk.lstrip()
            finish_reason = self.llm.last_finish_reason

        if finish_reason != "length" and not _looks_complete(response_text):
            self.logger.info(AgentRole.EXECUTION, "Response ended naturally but looks incomplete — one more continuation")
            cont_prompt = (
                f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
                f"=== PARTIAL RESPONSE ===\n{response_text}\n\n"
                f"The response above seems to end abruptly. "
                f"Please finish it with a proper conclusion. "
                f"Do NOT repeat what was already said."
            )
            extra = self.llm.ask(system=_CONTINUATION_PROMPT, user=cont_prompt)
            if extra.strip():
                response_text = response_text.rstrip() + "\n" + extra.lstrip()

        was_truncated = finish_reason == "length"

        self.logger.info(AgentRole.EXECUTION, f"Execution complete — response length: {len(response_text)} chars")
        if auto_rounds > 0:
            self.logger.info(AgentRole.EXECUTION, f"Auto-continuation rounds: {auto_rounds}")
        if was_truncated:
            self.logger.warn(AgentRole.EXECUTION, "Response still truncated after all auto-continuation attempts")

        return AgentMessage(
            role="assistant",
            content=response_text,
            confidence=0.8 if strategy.is_novel else 0.9,
            metadata={
                "strategy": strategy.selected_strategy,
                "is_novel_strategy": strategy.is_novel,
                "finish_reason": finish_reason,
                "was_truncated": was_truncated,
            },
        )

    # -- Tool-call helpers -----------------------------------------------

    _TOOL_CALL_RE = re.compile(
        r"```tool_call\s*\n(.*?)\n\s*```",
        re.DOTALL,
    )

    _MAX_TOOL_ROUNDS = 5

    def _process_tool_calls(
        self,
        response_text: str,
        system_prompt: str,
        user_text: str,
    ) -> str:
        """Detect ``tool_call`` blocks, run them, and let the LLM revise."""
        if not self._tool_executor:
            return response_text

        rounds = 0
        while rounds < self._MAX_TOOL_ROUNDS:
            matches = self._TOOL_CALL_RE.findall(response_text)
            if not matches:
                break
            rounds += 1
            results: list[str] = []
            for raw_json in matches:
                self.logger.info(AgentRole.EXECUTION, f"Tool call detected: {raw_json[:120]}")
                result = self._tool_executor.parse_and_execute(raw_json)
                results.append(result)
                self.logger.info(AgentRole.EXECUTION, f"Tool result: {result[:200]}")

            # Strip tool_call blocks from the response
            cleaned = self._TOOL_CALL_RE.sub("", response_text).strip()

            tool_results_section = "\n".join(
                f"[Tool result {i+1}] {r}" for i, r in enumerate(results)
            )
            followup = (
                f"=== ORIGINAL REQUEST ===\n{user_text}\n\n"
                f"=== YOUR PREVIOUS RESPONSE ===\n{cleaned}\n\n"
                f"=== TOOL RESULTS ===\n{tool_results_section}\n\n"
                "Now incorporate the tool results above into a final, "
                "complete answer. Do NOT include tool_call blocks."
            )
            response_text = self.llm.ask(system=system_prompt, user=followup)

        return response_text

    # ------------------------------------------------------------------

    def continue_response(
        self,
        user_input: MultimodalInput,
        partial_response: str,
        incompleteness_details: str = "",
        **kwargs: Any,
    ) -> AgentMessage:
        """Continue an incomplete response from where it left off."""
        self.logger.info(AgentRole.EXECUTION, "Continuing incomplete response…")

        user_text = user_input.text or "(non-text input)"

        continuation_prompt = (
            f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
            f"=== PARTIAL RESPONSE (already generated) ===\n"
            f"{partial_response}\n\n"
            f"=== INCOMPLETENESS DIAGNOSIS ===\n"
            f"{incompleteness_details or 'Response was cut off before completion.'}\n\n"
            f"Continue the response from where it left off.  "
            f"Do NOT repeat what was already said."
        )

        if user_input.image_paths:
            image_urls = self._resolve_image_urls(user_input.image_paths)
            vision_msg = self.llm.build_vision_message(
                continuation_prompt, image_urls
            )
            messages = [
                {"role": "system", "content": _CONTINUATION_PROMPT},
                vision_msg,
            ]
            continuation_text = self.llm.chat(messages)
        else:
            continuation_text = self.llm.ask(
                system=_CONTINUATION_PROMPT, user=continuation_prompt
            )

        merged = partial_response.rstrip() + "\n" + continuation_text.lstrip()

        finish_reason = self.llm.last_finish_reason
        auto_rounds = 0
        while finish_reason == "length" and auto_rounds < _MAX_AUTO_CONTINUATIONS:
            auto_rounds += 1
            self.logger.info(
                AgentRole.EXECUTION,
                f"Continuation hit token limit — "
                f"auto-continuing ({auto_rounds}/{_MAX_AUTO_CONTINUATIONS})…",
            )
            cont_prompt = (
                f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
                f"=== PARTIAL RESPONSE ===\n{merged}\n\n"
                f"Continue from where this left off. "
                f"Do NOT repeat what was already said."
            )
            chunk = self.llm.ask(system=_CONTINUATION_PROMPT, user=cont_prompt)
            if not chunk.strip():
                break
            merged = merged.rstrip() + "\n" + chunk.lstrip()
            finish_reason = self.llm.last_finish_reason

        if finish_reason != "length" and not _looks_complete(merged):
            self.logger.info(AgentRole.EXECUTION, "Continued response looks incomplete — one more pass")
            cont_prompt = (
                f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
                f"=== PARTIAL RESPONSE ===\n{merged}\n\n"
                f"The response above seems to end abruptly. "
                f"Please finish it with a proper conclusion."
            )
            extra = self.llm.ask(system=_CONTINUATION_PROMPT, user=cont_prompt)
            if extra.strip():
                merged = merged.rstrip() + "\n" + extra.lstrip()

        self.logger.info(
            AgentRole.EXECUTION,
            f"Continuation complete — added {len(merged) - len(partial_response)} chars "
            f"(total: {len(merged)})",
        )

        return AgentMessage(
            role="assistant",
            content=merged,
            confidence=0.7,
            metadata={
                "continued": True,
                "finish_reason": self.llm.last_finish_reason,
                "was_truncated": self.llm.last_finish_reason == "length",
            },
        )

    # =================================================================
    # Skill 4: EVALUATE
    # =================================================================

    @staticmethod
    def _heuristic_completeness_check(text: str) -> tuple[bool, str]:
        """Quick heuristic check for obvious truncation signals.

        Returns (is_likely_complete, reason_if_not).
        """
        if not text or not text.strip():
            return False, "Response is empty"

        stripped = text.strip()

        if stripped.count("```") % 2 != 0:
            return False, "Unbalanced code fences — response appears truncated mid-code-block"

        if stripped[-1] not in _TERMINAL_CHARS and len(stripped) > 200:
            return False, "Response does not end with terminal punctuation — may be truncated"

        if len(stripped) < 10:
            return False, "Response is suspiciously short"

        return True, ""

    def evaluate(
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        response: AgentMessage,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the execution output and produce an EvaluationResult."""
        self.logger.phase("Phase 4: EVALUATE 评估")
        self.logger.info(AgentRole.EVALUATION, "Evaluating output quality…")

        user_text = user_input.text or "(non-text input)"

        heuristic_ok, heuristic_reason = self._heuristic_completeness_check(
            response.content
        )

        if response.metadata.get("was_truncated"):
            heuristic_ok = False
            heuristic_reason = (
                heuristic_reason or
                "Response was truncated by token limit (finish_reason=length)"
            )

        eval_prompt = (
            f"=== USER INPUT ===\n{user_text}\n\n"
            f"=== PERCEPTION ===\n"
            f"Task type: {perception.task_type.value}\n"
            f"Complexity: {perception.complexity.value}/5\n\n"
            f"=== STRATEGY ===\n"
            f"Strategy: {strategy.selected_strategy}\n"
            f"Pipeline: {strategy.modality_pipeline}\n\n"
            f"=== RESPONSE ===\n{response.content[:4000]}"
        )

        if not heuristic_ok:
            eval_prompt += (
                f"\n\n=== COMPLETENESS WARNING ===\n"
                f"Heuristic check flagged: {heuristic_reason}\n"
                f"Pay extra attention to whether the response is complete."
            )

        llm_result = self.llm.ask_json(system=_EVALUATE_PROMPT, user=eval_prompt, max_tokens=2048)

        def _safe_int(val: Any, default: int, lo: int = 1, hi: int = 10) -> int:
            try:
                return max(lo, min(hi, int(val)))
            except (ValueError, TypeError):
                return default

        def _safe_bool(val: Any, default: bool) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() not in ("false", "0", "no", "")
            return default

        llm_is_complete = _safe_bool(llm_result.get("is_complete", True), True)
        final_is_complete = llm_is_complete and heuristic_ok

        incompleteness_details = llm_result.get("incompleteness_details", "")
        if not heuristic_ok and not incompleteness_details:
            incompleteness_details = heuristic_reason

        task_completion = llm_result.get("task_completion", "complete")
        output_quality = _safe_int(llm_result.get("output_quality", 5), 5)

        if not final_is_complete:
            if task_completion == "complete":
                task_completion = "partial"
            output_quality = min(output_quality, 5)

        result = EvaluationResult(
            task_completion=task_completion,
            output_quality=output_quality,
            strategy_effectiveness=_safe_int(llm_result.get("strategy_effectiveness", 5), 5),
            went_well=llm_result.get("went_well", []),
            went_wrong=llm_result.get("went_wrong", []),
            surprises=llm_result.get("surprises", []),
            cross_modal_coherence=_safe_int(llm_result.get("cross_modal_coherence", 5), 5),
            is_complete=final_is_complete,
            incompleteness_details=incompleteness_details,
            user_satisfaction="awaiting",
            raw_analysis=llm_result.get("analysis", ""),
        )

        self.logger.info(
            AgentRole.EVALUATION,
            f"Quality: {result.output_quality}/10 | "
            f"Strategy: {result.strategy_effectiveness}/10 | "
            f"Status: {result.task_completion} | "
            f"Complete: {result.is_complete}",
        )
        if not result.is_complete:
            self.logger.warn(AgentRole.EVALUATION, f"Response incomplete: {result.incompleteness_details}")
        if result.went_wrong:
            self.logger.warn(AgentRole.EVALUATION, f"Issues found: {result.went_wrong}")

        return result

    # =================================================================
    # Skill 5: LEARN
    # =================================================================

    def learn(
        self,
        user_input_text: str,
        perception: PerceptionResult,
        strategy: StrategyResult,
        evaluation: EvaluationResult,
        **kwargs: Any,
    ) -> LearningResult:
        """Consolidate learnings and update the three memory stores."""
        self.logger.phase("Phase 5: LEARN 学习")
        self.logger.info(AgentRole.LEARNING, "Extracting lessons and updating memory…")

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

        llm_result = self.llm.ask_json(system=_LEARN_PROMPT, user=learn_prompt, max_tokens=2048)

        # 1. Build episodic record
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
        self.logger.info(AgentRole.LEARNING, f"Stored episode {episode.episode_id} (score: {episode.outcome_score}/10)")

        # 2. Semantic memory update
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
            self.logger.info(AgentRole.LEARNING, f"Added semantic knowledge entry: {entry.entry_id}")

        # 3. Procedural memory update
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
            self.logger.info(AgentRole.LEARNING, f"Saved new procedure: {proc.name}")
        elif not llm_result.get("should_save_procedure"):
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
            self.logger.warn(AgentRole.LEARNING, f"Skill gap identified: {result.skill_gap}")

        return result

    # =================================================================
    # Skill 6: META-REFLECT
    # =================================================================

    def should_meta_reflect(self) -> bool:
        """Check if it's time for a meta-reflection cycle."""
        total_episodes = self.memory.episodic.count
        return total_episodes > 0 and total_episodes % self._meta_reflect_interval == 0

    def meta_reflect(self, **kwargs: Any) -> MetaReflectionResult:
        """Perform periodic higher-order self-reflection."""
        self.logger.phase("Phase 6: META-REFLECT 元反思")
        self.logger.info(AgentRole.META_REFLECTION, "Performing higher-order self-reflection…")

        summary = self.memory.summary()
        trend = self.memory.improvement_trend(window=self._meta_reflect_interval)
        recent_lessons = self.memory.episodic.recent_lessons(limit=15)
        error_freq = self.memory.episodic.error_frequency()

        context = (
            f"=== MEMORY SUMMARY ===\n"
            f"Total episodes: {summary['episodic_count']}\n"
            f"Semantic entries: {summary['semantic_count']}\n"
            f"Procedures: {summary['procedural_count']}\n"
            f"Average score: {summary['avg_score']}\n"
            f"Top procedures: {summary['top_procedures']}\n\n"
            f"=== IMPROVEMENT TREND (window={self._meta_reflect_interval}) ===\n{trend}\n\n"
            f"=== ERROR FREQUENCY ===\n{error_freq}\n\n"
            f"=== RECENT LESSONS ===\n"
            + "\n".join(f"  - {lesson}" for lesson in recent_lessons)
        )

        llm_result = self.llm.ask_json(system=_META_REFLECT_PROMPT, user=context, max_tokens=2048)

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

        self.logger.info(AgentRole.META_REFLECTION, f"Strengths: {result.strength_domains}")
        self.logger.warn(AgentRole.META_REFLECTION, f"Weaknesses: {result.weakness_domains}")
        self.logger.info(AgentRole.META_REFLECTION, f"Priority improvements: {result.priority_improvements}")
        self.logger.separator()

        return result
