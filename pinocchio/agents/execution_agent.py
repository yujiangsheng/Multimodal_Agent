"""ExecutionAgent — Phase 3 of the cognitive loop (EXECUTE).

Carries out the task according to the strategy produced in Phase 2.
This is the agent that generates the actual user-facing response.  It follows
the strategic plan step-by-step, monitors intermediate results, and triggers
adaptive re-planning when something goes wrong.

Skills / Capabilities
─────────────────────
1. **Plan Execution**
   Follow the strategy plan step by step, generating intermediate outputs
   and assembling them into a coherent final result.

2. **Adaptive Re-Planning**
   If an intermediate step produces unexpected or low-quality output, pause
   and switch to the fallback plan rather than blindly continuing.

3. **Multimodal Output Assembly**
   Construct responses that may combine text, images, and other modalities,
   ensuring cross-modal consistency.

4. **Quality Monitoring**
   Continuously assess intermediate results against expected standards
   and adjust approach in real-time.

5. **Tool Invocation**
   When the strategy calls for external tool use (web search, code execution,
   image generation, etc.), dispatch the appropriate tool call and incorporate
   the result.

6. **Context Window Management**
   Efficiently manage the LLM context by summarising prior steps and retaining
   only the most relevant information.

7. **Cross-Modal Consistency Enforcement**
   When producing multimodal output, verify that information across modalities
   is coherent and non-contradictory.
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
)
from pinocchio.multimodal.vision_processor import VisionProcessor

_SYSTEM_PROMPT = """\
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

# Maximum number of auto-continuation rounds when finish_reason="length"
_MAX_AUTO_CONTINUATIONS = 2

# Terminal punctuation set for sentence-completeness check
_TERMINAL_CHARS = set(
    ".!?。！？…\"'`）)]}】》~—–-"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


def _looks_complete(text: str) -> bool:
    """Quick heuristic: does the text end at a sentence boundary?"""
    stripped = text.strip()
    if not stripped:
        return False
    # Very short — cannot judge, assume incomplete
    if len(stripped) < 10:
        return False
    # Unbalanced code fences
    if stripped.count("```") % 2 != 0:
        return False
    # Ends with a recognisable terminal character
    if stripped[-1] in _TERMINAL_CHARS:
        return True
    # For shorter texts (< 200 chars) be lenient
    if len(stripped) < 200:
        return True
    return False


class ExecutionAgent(BaseAgent):
    """Generates the user-facing response following the strategy plan."""

    role = AgentRole.EXECUTION

    # Reuse VisionProcessor's image encoding to avoid duplication
    _encode_image = staticmethod(VisionProcessor._encode_image)

    def _resolve_image_urls(self, paths: list[str]) -> list[str]:
        """Convert a list of file paths / URLs to URLs the LLM can consume."""
        return [
            p if p.startswith(("http://", "https://", "data:")) else self._encode_image(p)
            for p in paths
        ]

    def run(  # type: ignore[override]
        self,
        user_input: MultimodalInput,
        perception: PerceptionResult,
        strategy: StrategyResult,
        modality_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AgentMessage:
        self.logger.phase("Phase 3: EXECUTE 执行")
        self._log(f"Executing strategy: {strategy.selected_strategy}")

        user_text = user_input.text or "(non-text input)"

        # Build modality context section if multimodal preprocessing was done
        modality_section = ""
        if modality_context:
            parts = []
            for mod_name, mod_desc in modality_context.items():
                parts.append(f"[{mod_name.upper()}] {mod_desc}")
            modality_section = (
                "\n\n=== MULTIMODAL CONTEXT ===\n"
                + "\n".join(parts)
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
        )

        # Vision path: if images are present, build a multimodal message
        if user_input.image_paths:
            self._log("Building multimodal (vision) request…")
            image_urls = self._resolve_image_urls(user_input.image_paths)
            vision_msg = self.llm.build_vision_message(context_prompt, image_urls)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                vision_msg,
            ]
            response_text = self.llm.chat(messages)
        else:
            response_text = self.llm.ask(system=_SYSTEM_PROMPT, user=context_prompt)

        # ── Auto-continuation: keep generating when token limit is hit ──
        # When the LLM returns finish_reason="length", it means the response
        # was truncated by the max_tokens limit rather than ending naturally.
        # We loop here, feeding the partial output back as context and asking
        # the LLM to continue.  Max rounds: _MAX_AUTO_CONTINUATIONS (5).
        finish_reason = self.llm.last_finish_reason
        auto_rounds = 0
        while finish_reason == "length" and auto_rounds < _MAX_AUTO_CONTINUATIONS:
            auto_rounds += 1
            self._log(
                f"Token limit reached (finish_reason=length) — "
                f"auto-continuing ({auto_rounds}/{_MAX_AUTO_CONTINUATIONS})…"
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
                self._log("Continuation returned empty — stopping auto-continue")
                break
            response_text = response_text.rstrip() + "\n" + chunk.lstrip()
            finish_reason = self.llm.last_finish_reason

        # ── Final sentence-completeness heuristic ──
        # After auto-continuation (or if no continuation was needed), check
        # if the text looks like it ends at a natural sentence boundary.
        # If not, issue one more LLM call to wrap up the response.
        # This catches cases where the LLM stopped naturally but mid-thought.
        if finish_reason != "length" and not _looks_complete(response_text):
            self._log("Response ended naturally but looks incomplete — one more continuation")
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

        was_truncated = finish_reason == "length"  # still truncated after all retries

        self._log(f"Execution complete — response length: {len(response_text)} chars")
        if auto_rounds > 0:
            self._log(f"Auto-continuation rounds: {auto_rounds}")
        if was_truncated:
            self._warn("Response still truncated after all auto-continuation attempts")

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

    def continue_response(
        self,
        user_input: MultimodalInput,
        partial_response: str,
        incompleteness_details: str = "",
        **kwargs: Any,
    ) -> AgentMessage:
        """Continue an incomplete response from where it left off.

        Called by the orchestrator when the evaluation agent detects
        that the initial response was truncated or incomplete.
        Includes its own auto-continuation loop for token-limit hits.
        """
        self._log("Continuing incomplete response…")

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

        # Vision path: if images are present, build a multimodal message
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

        # Merge partial + continuation
        merged = partial_response.rstrip() + "\n" + continuation_text.lstrip()

        # ── Auto-continuation within continue_response ──
        finish_reason = self.llm.last_finish_reason
        auto_rounds = 0
        while finish_reason == "length" and auto_rounds < _MAX_AUTO_CONTINUATIONS:
            auto_rounds += 1
            self._log(
                f"Continuation hit token limit — "
                f"auto-continuing ({auto_rounds}/{_MAX_AUTO_CONTINUATIONS})…"
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

        # Final sentence-completeness check
        if finish_reason != "length" and not _looks_complete(merged):
            self._log("Continued response looks incomplete — one more pass")
            cont_prompt = (
                f"=== ORIGINAL USER REQUEST ===\n{user_text}\n\n"
                f"=== PARTIAL RESPONSE ===\n{merged}\n\n"
                f"The response above seems to end abruptly. "
                f"Please finish it with a proper conclusion."
            )
            extra = self.llm.ask(system=_CONTINUATION_PROMPT, user=cont_prompt)
            if extra.strip():
                merged = merged.rstrip() + "\n" + extra.lstrip()

        self._log(
            f"Continuation complete — added {len(merged) - len(partial_response)} chars "
            f"(total: {len(merged)})"
        )

        return AgentMessage(
            role="assistant",
            content=merged,
            confidence=0.7,  # slightly lower confidence for continued responses
            metadata={
                "continued": True,
                "finish_reason": self.llm.last_finish_reason,
                "was_truncated": self.llm.last_finish_reason == "length",
            },
        )
