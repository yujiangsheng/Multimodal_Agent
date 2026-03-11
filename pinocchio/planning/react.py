"""ReAct Executor — Thought → Action → Observation iterative loop.

Implements the ReAct (Reason + Act) pattern for tool-augmented reasoning.
Each iteration:
1. **Thought** — the LLM reasons about what to do next
2. **Action** — the LLM chooses a tool to call (or decides to finish)
3. **Observation** — the tool result is fed back for the next iteration

The loop terminates when the LLM produces a ``FINISH`` action or the
max iteration limit is reached.

Usage::

    executor = ReActExecutor(llm, tool_executor, tool_registry)
    trace = executor.run("What is the SHA-256 hash of 'hello world'?")
    print(trace.final_answer)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pinocchio.tools import ToolExecutor, ToolRegistry
from pinocchio.utils.llm_client import LLMClient


@dataclass
class ReActStep:
    """A single Thought-Action-Observation iteration."""

    iteration: int = 0
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class ReActTrace:
    """Complete execution trace of a ReAct loop."""

    question: str = ""
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = True
    total_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "total_iterations": self.total_iterations,
        }

    def summary(self) -> str:
        lines = [f"ReAct: {self.question}"]
        for s in self.steps:
            lines.append(f"  [{s.iteration}] Think: {s.thought[:80]}...")
            if s.action == "FINISH":
                lines.append(f"       → FINISH")
            else:
                lines.append(f"       → {s.action}({json.dumps(s.action_input, ensure_ascii=False)[:60]})")
                lines.append(f"       ← {s.observation[:80]}...")
        return "\n".join(lines)


_MAX_ITERATIONS = 8

_REACT_SYSTEM = """\
You are an AI assistant that uses tools to answer questions step by step.

Available tools:
{tools}

For each step, respond in this EXACT JSON format:
{{
  "thought": "your reasoning about what to do next",
  "action": "tool_name",
  "action_input": {{"param": "value"}}
}}

When you have enough information to give the final answer, use:
{{
  "thought": "I now have all the information needed",
  "action": "FINISH",
  "action_input": {{"answer": "your final answer"}}
}}

Rules:
- Always think before acting.
- Use tools when you need external information or computation.
- Do NOT guess — use tools to verify.
- You MUST eventually call FINISH to provide the answer.
- Write your thoughts and answers in the same language the user uses."""


class ReActExecutor:
    """Execute a question through a ReAct reasoning loop with tool use."""

    def __init__(
        self,
        llm: LLMClient,
        tool_executor: ToolExecutor,
        tool_registry: ToolRegistry,
        *,
        max_iterations: int = _MAX_ITERATIONS,
    ) -> None:
        self._llm = llm
        self._tool_executor = tool_executor
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations

    def run(self, question: str, context: str = "") -> ReActTrace:
        """Run the ReAct loop until FINISH or max iterations."""
        trace = ReActTrace(question=question)

        tools_desc = self._tool_registry.to_prompt_description()
        system_msg = _REACT_SYSTEM.format(tools=tools_desc)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
        ]
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        messages.append({"role": "user", "content": user_content})

        for i in range(1, self._max_iterations + 1):
            # Get LLM response
            raw = self._llm.chat(messages, json_mode=True)
            parsed = self._parse_response(raw)

            step = ReActStep(
                iteration=i,
                thought=parsed.get("thought", ""),
                action=parsed.get("action", "FINISH"),
                action_input=parsed.get("action_input", {}),
            )

            if step.action == "FINISH":
                step.observation = "(done)"
                trace.steps.append(step)
                trace.final_answer = step.action_input.get("answer", raw)
                break

            # Execute tool
            observation = self._tool_executor.execute(
                step.action, step.action_input
            )
            step.observation = observation
            trace.steps.append(step)

            # Feed observation back
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue reasoning.",
            })
        else:
            # Max iterations reached — extract best answer
            trace.final_answer = self._extract_best_answer(messages)
            trace.success = False

        trace.total_iterations = len(trace.steps)
        return trace

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse the LLM's JSON response, with fallback."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            # Best-effort: treat as a FINISH
            return {
                "thought": "Could not parse structured response",
                "action": "FINISH",
                "action_input": {"answer": raw},
            }

    def _extract_best_answer(self, messages: list[dict[str, Any]]) -> str:
        """Extract a final answer when max iterations are reached."""
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of iterations. "
                "Based on all the information gathered so far, provide your "
                "best final answer now."
            ),
        })
        return self._llm.chat(messages)
