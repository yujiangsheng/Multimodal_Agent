"""Multi-agent collaboration framework.

Supports teams of specialized agents that communicate via message passing,
with coordinator-driven task delegation and result aggregation.

The workflow follows a coordinator pattern:

1. **Decompose** — the coordinator LLM splits the user task into sub-tasks.
2. **Assign** — each sub-task is routed to the best-matching team member.
3. **Execute** — members process their assignments sequentially, each
   receiving context from prior members.
4. **Synthesize** — the coordinator merges all contributions into a
   final coherent response.

Quick start::

    from pinocchio.collaboration import AgentTeam, TeamMember

    team = AgentTeam("review_team", llm_client=llm)
    team.add_member(TeamMember(member_id="analyst", role="analysis",
                               specialty="Data analysis"))
    team.add_member(TeamMember(member_id="writer", role="writing",
                               specialty="Technical writing"))
    result = team.collaborate("Analyse Q3 sales and draft a summary")
    print(result.final_output)
"""

from pinocchio.collaboration.team import (
    AgentTeam,
    TeamMember,
    TeamMessage,
    TeamResult,
)

__all__ = ["AgentTeam", "TeamMember", "TeamMessage", "TeamResult"]
