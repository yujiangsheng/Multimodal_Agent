"""Multi-agent collaboration framework.

Supports teams of specialized agents that communicate via message passing,
with coordinator-driven task delegation and result aggregation.
"""

from pinocchio.collaboration.team import (
    AgentTeam,
    TeamMember,
    TeamMessage,
    TeamResult,
)

__all__ = ["AgentTeam", "TeamMember", "TeamMessage", "TeamResult"]
