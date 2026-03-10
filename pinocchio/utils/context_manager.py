"""Context Manager — intelligent context window management.

Prevents context-window overflows by:

1. **Token estimation** — fast word-based approximation (no external
   tokenizer dependency).
2. **Conversation summarisation** — when the conversation exceeds a
   configurable token budget, older turns are compressed into a concise
   summary via the LLM.
3. **Selective injection** — builds message lists that fit within the
   token budget by keeping the system prompt, the summary of old turns,
   and the most recent turns.

Usage
-----
>>> from pinocchio.utils.context_manager import ContextManager
>>> cm = ContextManager(llm_client, max_context_tokens=6000)
>>> messages = cm.build_messages(system_prompt, conversation_history)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pinocchio.utils.llm_client import LLMClient


# =====================================================================
# Token estimation
# =====================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.

    Uses a heuristic of ~1.3 tokens per word for English and ~2 tokens
    per character for CJK text.  Good enough for budget management
    without requiring a model-specific tokenizer.
    """
    if not text:
        return 0
    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3400' <= c <= '\u4dbf'
                    or '\uf900' <= c <= '\ufaff')
    non_cjk = len(text) - cjk_chars
    # CJK: ~1.5 tokens per character; Latin: ~1.3 tokens per word
    words = non_cjk / 4.5 if non_cjk > 0 else 0  # avg word len ~4.5 chars
    return int(words * 1.3 + cjk_chars * 1.5)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # role + overhead: ~4 tokens per message
        total += 4
    return total


# =====================================================================
# Summarisation prompt
# =====================================================================

_SUMMARISE_PROMPT = (
    "You are a conversation summariser. Condense the following "
    "conversation into a brief, information-dense summary (max 200 words). "
    "Preserve key facts, decisions, user preferences, and any unresolved "
    "questions. Write in the same language the user uses."
)


# =====================================================================
# Context Manager
# =====================================================================

class ContextManager:
    """Manage conversation context to fit within the LLM's token window.

    Parameters
    ----------
    llm : LLMClient
        Used for summarising old conversation turns.
    max_context_tokens : int
        Token budget for the full message list sent to the LLM.
        Should be less than the model's actual context window to leave
        room for the response.
    summary_trigger_ratio : float
        When conversation tokens exceed this fraction of the budget,
        summarisation is triggered.  Default 0.7 (70 %).
    keep_recent_turns : int
        Minimum number of recent turns to keep verbatim (not summarised).
    """

    def __init__(
        self,
        llm: LLMClient,
        max_context_tokens: int = 5000,
        summary_trigger_ratio: float = 0.7,
        keep_recent_turns: int = 6,
    ) -> None:
        self._llm = llm
        self.max_context_tokens = max_context_tokens
        self._trigger_threshold = int(max_context_tokens * summary_trigger_ratio)
        self.keep_recent_turns = keep_recent_turns
        self._cached_summary: str = ""
        self._summarised_turn_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_messages(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build a message list that fits within the token budget.

        Steps:
        1. Start with the system prompt.
        2. If conversation is long, summarise old turns and prepend
           the summary as a system note.
        3. Append recent turns verbatim.
        4. If still over budget, drop oldest recent turns one by one.

        Returns the final message list ready for ``llm.chat()``.
        """
        system_tokens = estimate_tokens(system_prompt) + 4
        remaining_budget = self.max_context_tokens - system_tokens

        # Check if summarisation is needed
        conv_tokens = sum(estimate_tokens(m.get("content", "")) + 4 for m in conversation)
        if conv_tokens > self._trigger_threshold and len(conversation) > self.keep_recent_turns:
            self._maybe_summarise(conversation)

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # If we have a summary of old turns, inject it
        if self._cached_summary and self._summarised_turn_count > 0:
            summary_note = f"[Summary of earlier conversation ({self._summarised_turn_count} turns)]\n{self._cached_summary}"
            messages.append({"role": "system", "content": summary_note})
            remaining_budget -= estimate_tokens(summary_note) + 4
            # Only include turns after the summarised ones
            recent = conversation[self._summarised_turn_count:]
        else:
            recent = list(conversation)

        # Drop oldest recent turns if still over budget
        while recent and sum(estimate_tokens(m.get("content", "")) + 4 for m in recent) > remaining_budget:
            if len(recent) <= 2:  # keep at least the last exchange
                break
            recent.pop(0)

        messages.extend(recent)
        return messages

    @property
    def has_summary(self) -> bool:
        return bool(self._cached_summary)

    @property
    def summary(self) -> str:
        return self._cached_summary

    @property
    def summarised_turn_count(self) -> int:
        return self._summarised_turn_count

    def reset(self) -> None:
        """Clear cached summary (call on session reset)."""
        self._cached_summary = ""
        self._summarised_turn_count = 0

    def stats(self) -> dict[str, Any]:
        """Return context management statistics."""
        return {
            "max_context_tokens": self.max_context_tokens,
            "summarised_turns": self._summarised_turn_count,
            "has_summary": self.has_summary,
            "summary_length": len(self._cached_summary),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_summarise(self, conversation: list[dict[str, str]]) -> None:
        """Summarise old turns if not already summarised."""
        # How many turns to summarise: everything except the recent N
        cut = len(conversation) - self.keep_recent_turns
        if cut <= self._summarised_turn_count:
            return  # already summarised up to this point

        old_turns = conversation[: cut]
        text_to_summarise = "\n".join(
            f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
            for m in old_turns
        )

        try:
            self._cached_summary = self._llm.ask(
                system=_SUMMARISE_PROMPT,
                user=text_to_summarise,
                max_tokens=400,
            )
            self._summarised_turn_count = cut
        except Exception:
            pass  # keep the old summary if LLM call fails
