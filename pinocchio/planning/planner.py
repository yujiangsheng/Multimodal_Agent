"""Task Planner — decomposes complex tasks into executable sub-steps.

Implements Plan-and-Solve style planning: the LLM generates a structured
plan before execution, enabling multi-step reasoning with intermediate
tool use and self-correction.

Usage::

    planner = TaskPlanner(llm)
    plan = planner.decompose("Research quantum computing and write a summary")
    for step in plan.steps:
        print(f"{step.order}. {step.description} [tool: {step.tool_hint}]")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pinocchio.utils.llm_client import LLMClient


@dataclass
class TaskStep:
    """A single step in a task plan."""

    order: int = 0
    description: str = ""
    tool_hint: str | None = None  # suggested tool to use (optional)
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"  # pending / running / completed / failed
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise this step to a JSON-friendly dictionary."""
        return {
            "order": self.order,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskStep:
        """Reconstruct a :class:`TaskStep` from a previously serialised dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskPlan:
    """A structured plan with ordered steps."""

    goal: str = ""
    steps: list[TaskStep] = field(default_factory=list)
    reasoning: str = ""
    is_complex: bool = False

    @property
    def total_steps(self) -> int:
        """Total number of steps in the plan."""
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        """Number of steps that have finished successfully."""
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def is_done(self) -> bool:
        """``True`` when every step is completed or failed."""
        return all(s.status in ("completed", "failed") for s in self.steps)

    @property
    def current_step(self) -> TaskStep | None:
        """Return the first step still in *pending* status, or ``None``."""
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full plan to a JSON-friendly dictionary."""
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "is_complex": self.is_complex,
        }

    def summary(self) -> str:
        """Human-readable progress summary with status markers."""
        lines = [f"Plan: {self.goal} ({self.completed_steps}/{self.total_steps} done)"]
        for s in self.steps:
            marker = {"pending": "○", "running": "●", "completed": "✓", "failed": "✗"}
            lines.append(f"  {marker.get(s.status, '?')} {s.order}. {s.description}")
        return "\n".join(lines)


_PLAN_PROMPT = """\
You are a task planning expert. Decompose the user's request into a clear, \
ordered list of actionable sub-steps.

Rules:
1. Each step should be a single, concrete action.
2. Include tool hints when a step needs a specific tool (calculator, web_fetch, \
file_reader, shell_command, etc.).
3. Mark dependencies between steps (which steps must complete before others).
4. If the task is simple (1-2 steps), still create a plan but set is_complex=false.

Respond in JSON:
{
  "goal": "brief summary of the overall task",
  "reasoning": "why you chose this decomposition",
  "is_complex": true/false,
  "steps": [
    {
      "order": 1,
      "description": "what to do in this step",
      "tool_hint": "tool_name or null",
      "depends_on": []
    }
  ]
}"""

_COMPLEXITY_THRESHOLD = 3  # Complexity >= this triggers planning


class TaskPlanner:
    """Decomposes complex user requests into structured multi-step plans."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialise the planner with an LLM client for plan generation."""
        self._llm = llm

    def should_plan(self, complexity: int, task_type: str = "") -> bool:
        """Decide whether this task needs multi-step planning."""
        planning_types = {"analysis", "code_generation", "multimodal_reasoning"}
        if complexity >= _COMPLEXITY_THRESHOLD:
            return True
        if task_type in planning_types and complexity >= 2:
            return True
        return False

    def decompose(self, user_text: str, context: str = "") -> TaskPlan:
        """Decompose a user request into a TaskPlan with ordered steps."""
        user_prompt = user_text
        if context:
            user_prompt = f"Context:\n{context}\n\nUser request:\n{user_text}"

        raw = self._llm.ask_json(system=_PLAN_PROMPT, user=user_prompt)

        steps = []
        for s in raw.get("steps", []):
            steps.append(TaskStep(
                order=s.get("order", len(steps) + 1),
                description=s.get("description", ""),
                tool_hint=s.get("tool_hint"),
                depends_on=s.get("depends_on", []),
            ))

        return TaskPlan(
            goal=raw.get("goal", user_text[:100]),
            steps=steps,
            reasoning=raw.get("reasoning", ""),
            is_complex=raw.get("is_complex", len(steps) > 2),
        )

    def replan(
        self, original_plan: TaskPlan, failed_step: TaskStep, error: str
    ) -> TaskPlan:
        """Re-plan after a step failure, adjusting remaining steps."""
        context = (
            f"Original plan: {json.dumps(original_plan.to_dict(), ensure_ascii=False)}\n"
            f"Failed step #{failed_step.order}: {failed_step.description}\n"
            f"Error: {error}\n\n"
            f"Create a revised plan that works around this failure."
        )
        return self.decompose(original_plan.goal, context=context)
