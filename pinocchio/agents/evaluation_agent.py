"""EvaluationAgent — Phase 4 of the cognitive loop (EVALUATE).

Performs rigorous self-assessment after the Execution phase completes.
It scores the output quality, strategy effectiveness, cross-modal coherence,
and identifies what went well, what went wrong, and any surprise factors.

A key responsibility is **response completeness checking**: detecting
truncated, abruptly-ended, or incomplete responses so the orchestrator
can trigger re-execution.

Skills / Capabilities
─────────────────────
1. **Output Quality Scoring**
   Rate the response quality on a 1–10 scale considering accuracy,
   completeness, clarity, and helpfulness.

2. **Response Completeness Verification**
   Check whether the response is fully formed — does it end naturally,
   cover all parts of the user's question, and reach a proper conclusion?
   Detect truncation, mid-sentence breaks, and unfinished lists.

3. **Strategy Effectiveness Scoring**
   Evaluate how well the chosen strategy served the task: was it efficient?
   Did it lead to a good result on the first attempt?

4. **Cross-Modal Coherence Assessment**
   For multimodal outputs, check if information across modalities is consistent
   and mutually reinforcing.

5. **Success / Failure Analysis**
   Enumerate specific things that went well and things that went wrong
   during execution.

6. **Surprise Factor Identification**
   Flag unexpected elements that were encountered during processing — these
   are high-value learning signals.

7. **User Satisfaction Inference**
   Based on the user's subsequent feedback (if available), infer satisfaction
   level; otherwise mark as "awaiting".

8. **Completion Status Classification**
   Determine whether the task was fully completed, partially addressed, or
   failed entirely.
"""

from __future__ import annotations

from typing import Any

from pinocchio.agents.base_agent import BaseAgent
from pinocchio.models.enums import AgentRole
from pinocchio.models.schemas import (
    MultimodalInput,
    PerceptionResult,
    StrategyResult,
    AgentMessage,
    EvaluationResult,
)

_SYSTEM_PROMPT = """\
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


class EvaluationAgent(BaseAgent):
    """Evaluates the execution output and produces an EvaluationResult."""

    role = AgentRole.EVALUATION

    @staticmethod
    def _heuristic_completeness_check(text: str) -> tuple[bool, str]:
        """Quick heuristic check for obvious truncation signals.

        Returns (is_likely_complete, reason_if_not).
        """
        if not text or not text.strip():
            return False, "Response is empty"

        stripped = text.strip()

        # Check for unbalanced code fences
        if stripped.count("```") % 2 != 0:
            return False, "Unbalanced code fences — response appears truncated mid-code-block"

        # Check if ends mid-sentence (no terminal punctuation)
        # Include Chinese punctuation, markdown endings, and common closings
        terminal = set(".!?。！？…\n\"'`）)]}】》❯→~—–-：:；;、，,"
                       "0123456789"  # ends with a number (lists, dates, etc.)
                       "abcdefghijklmnopqrstuvwxyz"  # ends with a word
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if stripped[-1] not in terminal and len(stripped) > 200:
            return False, "Response does not end with terminal punctuation — may be truncated"

        # Very short response — only flag truly empty/trivial ones
        if len(stripped) < 10:
            return False, "Response is suspiciously short"

        return True, ""

    def run(  # type: ignore[override]
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        response: AgentMessage,
        **kwargs: Any,
    ) -> EvaluationResult:
        self.logger.phase("Phase 4: EVALUATE 评估")
        self._log("Evaluating output quality…")

        user_text = user_input.text or "(non-text input)"

        # --- Step 1: Heuristic pre-check for truncation ---
        heuristic_ok, heuristic_reason = self._heuristic_completeness_check(
            response.content
        )

        # Also check if the LLM itself signalled truncation via finish_reason
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

        llm_result = self.llm.ask_json(system=_SYSTEM_PROMPT, user=eval_prompt, max_tokens=2048)

        def _safe_int(val: Any, default: int, lo: int = 1, hi: int = 10) -> int:
            """Parse an int from LLM output, clamping to [lo, hi]."""
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

        # Combine LLM assessment with heuristic check
        llm_is_complete = _safe_bool(llm_result.get("is_complete", True), True)
        final_is_complete = llm_is_complete and heuristic_ok

        incompleteness_details = llm_result.get("incompleteness_details", "")
        if not heuristic_ok and not incompleteness_details:
            incompleteness_details = heuristic_reason

        # If incomplete, cap quality and force partial/failed completion
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

        self._log(
            f"Quality: {result.output_quality}/10 | "
            f"Strategy: {result.strategy_effectiveness}/10 | "
            f"Status: {result.task_completion} | "
            f"Complete: {result.is_complete}"
        )
        if not result.is_complete:
            self._warn(f"Response incomplete: {result.incompleteness_details}")
        if result.went_wrong:
            self._warn(f"Issues found: {result.went_wrong}")

        return result
