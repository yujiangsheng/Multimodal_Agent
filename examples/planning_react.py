"""任务规划与 ReAct 推理循环示例。

演示 Plan-and-Solve 分解和 Thought → Action → Observation 循环。
# 离线可运行（使用 mock LLM）
"""

from __future__ import annotations

from pinocchio.planning import TaskPlanner, TaskPlan, TaskStep


# ---------------------------------------------------------------------------
# 1. 直接构造 TaskPlan（无需 LLM）
# ---------------------------------------------------------------------------

def manual_plan_demo():
    """手动创建计划并追踪进度。"""
    plan = TaskPlan(
        goal="Research quantum computing and write a summary",
        reasoning="Multi-step task requiring search and synthesis",
        is_complex=True,
        steps=[
            TaskStep(order=1, description="Search for recent papers", tool_hint="web_fetch"),
            TaskStep(order=2, description="Extract key findings", depends_on=[1]),
            TaskStep(order=3, description="Write executive summary", depends_on=[2]),
        ],
    )

    print("=== 手动创建计划 ===")
    print(plan.summary())
    print(f"Total: {plan.total_steps}, Done: {plan.completed_steps}, Is done: {plan.is_done}")

    # 模拟步骤执行
    for step in plan.steps:
        step.status = "completed"
        step.result = f"Result for step {step.order}"

    print("\n完成后:")
    print(plan.summary())
    print(f"Is done: {plan.is_done}")
    print()


# ---------------------------------------------------------------------------
# 2. 使用 LLM 自动分解（需要 Ollama）
# ---------------------------------------------------------------------------

def llm_plan_demo():
    """使用 LLM 自动分解任务。"""
    from pinocchio.utils.llm_client import LLMClient

    llm = LLMClient()
    planner = TaskPlanner(llm)

    # 判断是否需要规划
    needs_plan = planner.should_plan(complexity=4, task_type="analysis")
    print(f"需要规划: {needs_plan}")  # True

    # 自动分解
    plan = planner.decompose("分析 Python 3.13 的新特性并写一份技术总结")
    print(plan.summary())

    # 序列化
    import json
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 3. TaskStep 序列化往返
# ---------------------------------------------------------------------------

def serialization_demo():
    """步骤的序列化与反序列化。"""
    step = TaskStep(
        order=1,
        description="Fetch data from API",
        tool_hint="web_fetch",
        depends_on=[],
        status="completed",
        result="200 OK - 42 records",
    )

    d = step.to_dict()
    reconstructed = TaskStep.from_dict(d)

    print("=== 序列化往返 ===")
    print(f"Original:     {step.description} [{step.status}]")
    print(f"Reconstructed: {reconstructed.description} [{reconstructed.status}]")
    print()


if __name__ == "__main__":
    manual_plan_demo()
    serialization_demo()
    # llm_plan_demo()  # 取消注释以使用 LLM（需要 ollama serve）
