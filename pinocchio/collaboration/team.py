"""Multi-agent collaboration — AgentTeam for cooperative task execution.

Provides a team abstraction where each member has a role, specialty, and
an LLM-backed processing function. A coordinator member delegates
sub-tasks, collects results, and synthesizes the final output.

Usage::

    team = AgentTeam("research_team", llm_client=llm)

    team.add_member(TeamMember(
        member_id="researcher",
        role="research",
        specialty="Finding and analyzing information",
    ))
    team.add_member(TeamMember(
        member_id="writer",
        role="writing",
        specialty="Creating clear, polished prose",
    ))

    result = team.collaborate("Write a report on quantum computing trends")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_COORDINATOR_PROMPT = """\
You are a team coordinator. Given a task, break it into sub-tasks and
assign each to the most appropriate team member.

Team members:
{members}

Task: {task}

Respond in JSON:
{{
  "assignments": [
    {{"member_id": "...", "sub_task": "...", "order": 1}},
    ...
  ]
}}
"""

_MEMBER_PROMPT = """\
You are a specialist with role: {role}.
Specialty: {specialty}

Previous context from other team members:
{context}

Your assignment:
{sub_task}

Provide your best work for this assignment.
"""

_SYNTHESIS_PROMPT = """\
You are synthesizing outputs from a team of specialists into a final answer.

Original task: {task}

Individual contributions:
{contributions}

Produce a coherent, well-structured final response that integrates all contributions.
"""


@dataclass
class TeamMessage:
    """A message exchanged between team members."""

    sender: str = ""
    recipient: str = ""     # empty = broadcast
    content: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content[:200],
            "timestamp": self.timestamp,
        }


@dataclass
class TeamMember:
    """A member of the agent team.

    Parameters
    ----------
    member_id : str
        Unique identifier.
    role : str
        Functional role (e.g., "researcher", "coder", "reviewer").
    specialty : str
        Description of domain expertise.
    handler : callable or None
        Custom ``(sub_task: str, context: str) -> str`` handler.
        If None, the team's shared LLM client is used.
    """

    member_id: str = ""
    role: str = ""
    specialty: str = ""
    handler: Any = None  # Optional Callable[[str, str], str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamResult:
    """Result from a team collaboration."""

    task: str = ""
    final_output: str = ""
    contributions: dict[str, str] = field(default_factory=dict)
    messages: list[TeamMessage] = field(default_factory=list)
    elapsed_ms: float = 0.0
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task[:100],
            "final_output": self.final_output[:500],
            "contributors": list(self.contributions.keys()),
            "message_count": len(self.messages),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "success": self.success,
        }


class AgentTeam:
    """A collaborative team of specialized agents.

    The team uses a coordinator pattern: given a task, the coordinator
    (powered by the LLM) breaks it into sub-tasks, assigns them to
    members, and synthesizes the final output.
    """

    def __init__(self, name: str = "default_team", *, llm_client: Any = None) -> None:
        self.name = name
        self._llm = llm_client
        self._members: dict[str, TeamMember] = {}
        self._messages: list[TeamMessage] = []

    def add_member(self, member: TeamMember) -> None:
        if not member.member_id:
            raise ValueError("TeamMember must have a non-empty member_id")
        self._members[member.member_id] = member

    def remove_member(self, member_id: str) -> bool:
        return self._members.pop(member_id, None) is not None

    @property
    def members(self) -> dict[str, TeamMember]:
        return dict(self._members)

    @property
    def message_log(self) -> list[TeamMessage]:
        return list(self._messages)

    def set_llm_client(self, llm_client: Any) -> None:
        self._llm = llm_client

    def collaborate(self, task: str) -> TeamResult:
        """Execute a collaborative task.

        1. Coordinator decomposes task into assignments.
        2. Each member processes its assignment (sequentially, respecting order).
        3. Coordinator synthesizes all outputs.

        Returns a TeamResult with the final answer and all contributions.
        """
        start = time.time()
        self._messages = []

        if not self._members:
            return TeamResult(task=task, success=False, final_output="No team members")

        if not self._llm:
            return TeamResult(task=task, success=False, final_output="No LLM client")

        # Step 1: Decompose
        assignments = self._decompose(task)
        if not assignments:
            # Fallback: let first member handle everything
            first = next(iter(self._members.values()))
            assignments = [{"member_id": first.member_id, "sub_task": task, "order": 1}]

        # Step 2: Execute assignments
        contributions: dict[str, str] = {}
        context_parts: list[str] = []

        # Sort by order
        assignments.sort(key=lambda a: a.get("order", 0))

        for assignment in assignments:
            mid = assignment.get("member_id", "")
            sub_task = assignment.get("sub_task", "")
            member = self._members.get(mid)

            if not member:
                logger.warning("Unknown member '%s', skipping", mid)
                continue

            context = "\n---\n".join(context_parts) if context_parts else "(none yet)"
            output = self._execute_member(member, sub_task, context)

            contributions[mid] = output
            context_parts.append(f"[{member.role}]: {output}")

            self._messages.append(TeamMessage(
                sender=mid, content=output, timestamp=time.time(),
            ))

        # Step 3: Synthesize
        final_output = self._synthesize(task, contributions)

        elapsed = (time.time() - start) * 1000
        return TeamResult(
            task=task,
            final_output=final_output,
            contributions=contributions,
            messages=self._messages,
            elapsed_ms=elapsed,
        )

    def _decompose(self, task: str) -> list[dict[str, Any]]:
        """Use the LLM to decompose a task into member assignments."""
        members_desc = "\n".join(
            f"- {m.member_id} (role: {m.role}, specialty: {m.specialty})"
            for m in self._members.values()
        )
        prompt = _COORDINATOR_PROMPT.format(members=members_desc, task=task)

        try:
            response = self._llm.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.3,
            )
            data = json.loads(response)
            return data.get("assignments", [])
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Decomposition failed: %s", exc)
            return []

    def _execute_member(
        self, member: TeamMember, sub_task: str, context: str,
    ) -> str:
        """Execute a sub-task using a member's handler or the shared LLM."""
        if member.handler:
            try:
                return member.handler(sub_task, context)
            except Exception as exc:
                return f"Error: {exc}"

        # Use LLM
        prompt = _MEMBER_PROMPT.format(
            role=member.role,
            specialty=member.specialty,
            context=context,
            sub_task=sub_task,
        )
        try:
            return self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
            )
        except Exception as exc:
            return f"Error: {exc}"

    def _synthesize(self, task: str, contributions: dict[str, str]) -> str:
        """Synthesize individual contributions into a final response."""
        if len(contributions) == 1:
            return next(iter(contributions.values()))

        contribs_text = "\n\n".join(
            f"[{mid}]:\n{text}" for mid, text in contributions.items()
        )
        prompt = _SYNTHESIS_PROMPT.format(task=task, contributions=contribs_text)

        try:
            return self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
            )
        except Exception as exc:
            logger.warning("Synthesis failed: %s", exc)
            return contribs_text
