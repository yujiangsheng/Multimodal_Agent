"""多智能体协作示例。

演示 AgentTeam 的成员管理和协作执行流程。
需要运行 Ollama (ollama serve) 才能使用 LLM 协调。
"""

from __future__ import annotations

from pinocchio.collaboration import AgentTeam, TeamMember


def team_with_custom_handlers():
    """使用自定义 handler — 无需 LLM。
    # 离线可运行
    """
    team = AgentTeam("demo_team")

    # 每个 handler 接收 (sub_task: str, context: str) -> str
    team.add_member(TeamMember(
        member_id="analyst",
        role="data_analysis",
        specialty="Statistical analysis and data interpretation",
        handler=lambda task, ctx: f"[Analyst] Analyzed: {task[:50]}. Found 3 key trends.",
    ))
    team.add_member(TeamMember(
        member_id="writer",
        role="technical_writing",
        specialty="Clear, concise technical documentation",
        handler=lambda task, ctx: f"[Writer] Wrote report based on: {ctx[:50]}",
    ))

    # 由于没有 LLM，需要手动设置一个 mock LLM
    # 或直接传入 handler 使每个成员独立工作
    class _MockLLM:
        def chat(self, messages, **kw):
            import json
            return json.dumps({
                "assignments": [
                    {"member_id": "analyst", "sub_task": "Analyze sales data", "order": 1},
                    {"member_id": "writer", "sub_task": "Write summary report", "order": 2},
                ]
            })

    team.set_llm_client(_MockLLM())
    result = team.collaborate("Analyze Q3 sales and write a report")

    print("=== 自定义 Handler 协作 ===")
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Contributors: {list(result.contributions.keys())}")
    for mid, output in result.contributions.items():
        print(f"  [{mid}]: {output}")
    print(f"Final: {result.final_output[:100]}")
    print(f"Messages: {len(result.messages)}")
    print(f"Elapsed: {result.elapsed_ms:.1f}ms")
    print()


def member_management():
    """成员管理操作。
    # 离线可运行
    """
    team = AgentTeam("project_team")

    team.add_member(TeamMember(member_id="dev", role="developer", specialty="Python"))
    team.add_member(TeamMember(member_id="qa", role="testing", specialty="pytest"))
    team.add_member(TeamMember(member_id="pm", role="management", specialty="Agile"))

    print("=== 成员管理 ===")
    print(f"Members: {list(team.members.keys())}")

    team.remove_member("pm")
    print(f"After removing PM: {list(team.members.keys())}")
    print()


def llm_collaboration():
    """使用真实 LLM 进行协作（需要 ollama serve）。"""
    from pinocchio.utils.llm_client import LLMClient

    llm = LLMClient()
    team = AgentTeam("research_team", llm_client=llm)

    team.add_member(TeamMember(
        member_id="researcher",
        role="research",
        specialty="Finding and analyzing academic papers",
    ))
    team.add_member(TeamMember(
        member_id="synthesizer",
        role="synthesis",
        specialty="Combining multiple sources into coherent summaries",
    ))

    result = team.collaborate("总结量子计算在密码学领域的最新进展")
    print("=== LLM 协作 ===")
    print(result.final_output)


if __name__ == "__main__":
    team_with_custom_handlers()
    member_management()
    # llm_collaboration()  # 取消注释（需要 ollama serve）
