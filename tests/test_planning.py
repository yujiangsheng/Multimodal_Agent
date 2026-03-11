"""Tests for Gap 1: Multi-step task planner (TaskPlanner + ReActExecutor)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pinocchio.planning.planner import TaskPlanner, TaskPlan, TaskStep
from pinocchio.planning.react import ReActExecutor, ReActTrace, ReActStep


# ---------------------------------------------------------------------------
# TaskStep / TaskPlan dataclasses
# ---------------------------------------------------------------------------

class TestTaskStep:
    def test_defaults(self):
        step = TaskStep(description="do something")
        assert step.description == "do something"
        assert step.status == "pending"
        assert step.tool_hint is None
        assert step.order == 0

    def test_to_dict(self):
        step = TaskStep(description="x", tool_hint="calc", status="done", order=1)
        d = step.to_dict()
        assert d["description"] == "x"
        assert d["tool_hint"] == "calc"
        assert d["status"] == "done"

    def test_from_dict(self):
        d = {"order": 2, "description": "step 2", "tool_hint": "web_fetch"}
        step = TaskStep.from_dict(d)
        assert step.order == 2
        assert step.description == "step 2"


class TestTaskPlan:
    def test_from_steps(self):
        steps = [TaskStep(description=f"Step {i}") for i in range(3)]
        plan = TaskPlan(goal="test goal", steps=steps)
        assert plan.goal == "test goal"
        assert plan.total_steps == 3

    def test_summary(self):
        plan = TaskPlan(
            goal="my goal",
            steps=[TaskStep(description="A"), TaskStep(description="B")],
        )
        s = plan.summary()
        assert "my goal" in s
        assert "A" in s

    def test_to_dict(self):
        plan = TaskPlan(goal="g", steps=[TaskStep(description="s1")])
        d = plan.to_dict()
        assert d["goal"] == "g"
        assert len(d["steps"]) == 1

    def test_completed_steps(self):
        plan = TaskPlan(
            goal="g",
            steps=[
                TaskStep(description="a", status="completed"),
                TaskStep(description="b", status="pending"),
            ],
        )
        assert plan.completed_steps == 1
        assert plan.is_done is False

    def test_current_step(self):
        plan = TaskPlan(
            goal="g",
            steps=[
                TaskStep(description="a", status="completed"),
                TaskStep(description="b", status="pending"),
            ],
        )
        assert plan.current_step.description == "b"

    def test_is_done(self):
        plan = TaskPlan(
            goal="g",
            steps=[
                TaskStep(description="a", status="completed"),
                TaskStep(description="b", status="failed"),
            ],
        )
        assert plan.is_done is True


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------

class TestTaskPlanner:
    def test_should_plan_simple(self):
        llm = MagicMock()
        planner = TaskPlanner(llm)
        assert planner.should_plan(2, "question_answering") is False

    def test_should_plan_complex(self):
        llm = MagicMock()
        planner = TaskPlanner(llm)
        assert planner.should_plan(4, "analysis") is True

    def test_should_plan_by_task_type(self):
        llm = MagicMock()
        planner = TaskPlanner(llm)
        assert planner.should_plan(2, "analysis") is True

    def test_decompose_success(self):
        llm = MagicMock()
        llm.ask_json.return_value = {
            "goal": "research topic",
            "reasoning": "need multi-step approach",
            "is_complex": True,
            "steps": [
                {"order": 1, "description": "search for info", "tool_hint": "web_fetch"},
                {"order": 2, "description": "summarize findings"},
            ]
        }

        planner = TaskPlanner(llm)
        plan = planner.decompose("Research quantum computing")
        assert plan.goal == "research topic"
        assert plan.total_steps == 2
        assert plan.steps[0].tool_hint == "web_fetch"

    def test_decompose_empty_response(self):
        llm = MagicMock()
        llm.ask_json.return_value = {}

        planner = TaskPlanner(llm)
        plan = planner.decompose("test")
        assert plan is not None
        assert plan.total_steps == 0

    def test_replan(self):
        llm = MagicMock()
        llm.ask_json.return_value = {
            "goal": "revised plan",
            "steps": [{"order": 1, "description": "new approach"}],
        }

        planner = TaskPlanner(llm)
        old_plan = TaskPlan(
            goal="old",
            steps=[TaskStep(order=1, description="old step", status="failed")],
        )
        failed_step = old_plan.steps[0]
        new_plan = planner.replan(old_plan, failed_step, "step failed due to timeout")
        assert new_plan is not None
        assert new_plan.goal == "revised plan"


# ---------------------------------------------------------------------------
# ReActExecutor
# ---------------------------------------------------------------------------

class TestReActExecutor:
    def test_init(self):
        llm = MagicMock()
        executor_mock = MagicMock()
        registry = MagicMock()
        react = ReActExecutor(llm, executor_mock, registry)
        assert react._max_iterations == 8

    def test_run_simple_thought(self):
        response = json.dumps({
            "thought": "I know the answer",
            "action": "FINISH",
            "action_input": {"answer": "42"},
        })
        llm = MagicMock()
        llm.chat.return_value = response

        registry = MagicMock()
        registry.to_prompt_description.return_value = "- calculator(expression)"

        executor_mock = MagicMock()
        react = ReActExecutor(llm, executor_mock, registry)
        trace = react.run("What is 6 * 7?")

        assert trace.final_answer == "42"
        assert trace.total_iterations >= 1

    def test_run_with_tool_call(self):
        tool_action = json.dumps({
            "thought": "I need to calculate",
            "action": "calculator",
            "action_input": {"expression": "6 * 7"},
        })
        final = json.dumps({
            "thought": "Got the result",
            "action": "FINISH",
            "action_input": {"answer": "42"},
        })

        llm = MagicMock()
        llm.chat.side_effect = [tool_action, final]

        registry = MagicMock()
        registry.to_prompt_description.return_value = "- calculator(expression)"

        executor_mock = MagicMock()
        executor_mock.execute.return_value = "42"

        react = ReActExecutor(llm, executor_mock, registry)
        trace = react.run("What is 6 * 7?")

        assert trace.final_answer == "42"
        assert executor_mock.execute.called

    def test_max_iterations_exceeded(self):
        loop_response = json.dumps({
            "thought": "thinking...",
            "action": "calculator",
            "action_input": {"expression": "1+1"},
        })

        llm = MagicMock()
        llm.chat.return_value = loop_response

        registry = MagicMock()
        registry.to_prompt_description.return_value = "- calculator(expression)"

        executor_mock = MagicMock()
        executor_mock.execute.return_value = "2"

        react = ReActExecutor(llm, executor_mock, registry, max_iterations=3)
        trace = react.run("loop forever")
        assert trace.total_iterations == 3

    def test_react_trace(self):
        trace = ReActTrace()
        assert trace.total_iterations == 0
        assert trace.final_answer == ""
        assert trace.steps == []

    def test_react_step(self):
        step = ReActStep(iteration=1, thought="thinking", action="calc")
        d = step.to_dict()
        assert d["iteration"] == 1
        assert d["thought"] == "thinking"

    def test_react_trace_summary(self):
        trace = ReActTrace(
            question="What is 2+2?",
            steps=[
                ReActStep(iteration=1, thought="need to calc", action="calc",
                          action_input={"expr": "2+2"}, observation="4"),
                ReActStep(iteration=2, thought="done", action="FINISH",
                          action_input={"answer": "4"}),
            ],
            final_answer="4",
        )
        s = trace.summary()
        assert "2+2" in s
