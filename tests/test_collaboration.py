"""Tests for Gap 6: Multi-agent collaboration (AgentTeam)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from pinocchio.collaboration.team import (
    AgentTeam,
    TeamMember,
    TeamMessage,
    TeamResult,
)


class TestTeamMessage:
    def test_defaults(self):
        m = TeamMessage()
        assert m.sender == ""
        assert m.content == ""

    def test_to_dict(self):
        m = TeamMessage(sender="agent_a", content="Hello team", timestamp=1000.0)
        d = m.to_dict()
        assert d["sender"] == "agent_a"
        assert d["timestamp"] == 1000.0


class TestTeamMember:
    def test_defaults(self):
        m = TeamMember(member_id="r1", role="researcher")
        assert m.member_id == "r1"
        assert m.role == "researcher"
        assert m.handler is None


class TestTeamResult:
    def test_defaults(self):
        r = TeamResult()
        assert r.success is True
        assert r.final_output == ""

    def test_to_dict(self):
        r = TeamResult(
            task="test task",
            final_output="result here",
            contributions={"a": "work_a"},
            elapsed_ms=150.0,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["contributors"] == ["a"]
        assert d["elapsed_ms"] == 150.0


class TestAgentTeam:
    def test_add_member(self):
        team = AgentTeam("test_team")
        team.add_member(TeamMember(member_id="r1", role="researcher"))
        assert "r1" in team.members

    def test_add_member_empty_id_raises(self):
        team = AgentTeam()
        with pytest.raises(ValueError, match="non-empty"):
            team.add_member(TeamMember(member_id=""))

    def test_remove_member(self):
        team = AgentTeam()
        team.add_member(TeamMember(member_id="r1", role="x"))
        assert team.remove_member("r1") is True
        assert team.remove_member("r1") is False

    def test_no_members_returns_failure(self):
        llm = MagicMock()
        team = AgentTeam(llm_client=llm)
        result = team.collaborate("Do something")
        assert result.success is False
        assert "No team members" in result.final_output

    def test_no_llm_returns_failure(self):
        team = AgentTeam()
        team.add_member(TeamMember(member_id="r1", role="x"))
        result = team.collaborate("Do something")
        assert result.success is False
        assert "No LLM" in result.final_output

    def test_collaborate_with_custom_handlers(self):
        team = AgentTeam("custom_team", llm_client=MagicMock())

        # Override decompose to skip LLM
        def researcher(sub_task, context):
            return f"Research findings about: {sub_task}"

        def writer(sub_task, context):
            return f"Written report using: {context}"

        team.add_member(TeamMember(
            member_id="researcher",
            role="research",
            specialty="Finding info",
            handler=researcher,
        ))
        team.add_member(TeamMember(
            member_id="writer",
            role="writing",
            specialty="Writing reports",
            handler=writer,
        ))

        # Mock the decompose to return simple assignments
        llm = MagicMock()
        llm.chat.side_effect = [
            # decompose
            json.dumps({"assignments": [
                {"member_id": "researcher", "sub_task": "find data", "order": 1},
                {"member_id": "writer", "sub_task": "write report", "order": 2},
            ]}),
        ]
        team.set_llm_client(llm)

        result = team.collaborate("Write a research report")
        assert result.success is True
        assert len(result.contributions) == 2
        assert "researcher" in result.contributions
        assert "writer" in result.contributions

    def test_collaborate_single_member(self):
        """With one member, output should be that member's work (no synthesis)."""
        def worker(sub_task, context):
            return "Done!"

        llm = MagicMock()
        llm.chat.return_value = json.dumps({"assignments": [
            {"member_id": "solo", "sub_task": "do everything", "order": 1},
        ]})

        team = AgentTeam(llm_client=llm)
        team.add_member(TeamMember(
            member_id="solo", role="generalist",
            specialty="All", handler=worker,
        ))

        result = team.collaborate("Just do it")
        assert result.success is True
        assert result.final_output == "Done!"

    def test_collaborate_decompose_fails(self):
        """If LLM decompose returns bad JSON, fallback to first member."""
        def backup(sub_task, context):
            return "Fallback answer"

        llm = MagicMock()
        llm.chat.return_value = "not valid json"

        team = AgentTeam(llm_client=llm)
        team.add_member(TeamMember(
            member_id="backup", role="backup",
            specialty="everything", handler=backup,
        ))

        result = team.collaborate("Handle this")
        assert result.success is True
        assert result.final_output == "Fallback answer"

    def test_collaborate_llm_member(self):
        """Member without handler uses LLM."""
        llm = MagicMock()
        # Call 1: decompose
        llm.chat.side_effect = [
            json.dumps({"assignments": [
                {"member_id": "agent", "sub_task": "analyze data", "order": 1},
            ]}),
            # Call 2: member execution
            "Analysis complete: data looks good",
        ]

        team = AgentTeam(llm_client=llm)
        team.add_member(TeamMember(
            member_id="agent", role="analyst",
            specialty="Data analysis",
        ))

        result = team.collaborate("Analyze the dataset")
        assert result.success is True
        assert llm.chat.call_count == 2

    def test_message_log(self):
        def worker(sub_task, context):
            return "output"

        llm = MagicMock()
        llm.chat.return_value = json.dumps({"assignments": [
            {"member_id": "w", "sub_task": "work", "order": 1},
        ]})

        team = AgentTeam(llm_client=llm)
        team.add_member(TeamMember(member_id="w", role="worker", handler=worker))
        team.collaborate("test")

        assert len(team.message_log) == 1
        assert team.message_log[0].sender == "w"

    def test_unknown_member_skipped(self):
        """Assignments referencing unknown members are skipped."""
        def known_handler(sub_task, context):
            return "known output"

        llm = MagicMock()
        llm.chat.return_value = json.dumps({"assignments": [
            {"member_id": "ghost", "sub_task": "vanish", "order": 1},
            {"member_id": "real", "sub_task": "work", "order": 2},
        ]})

        team = AgentTeam(llm_client=llm)
        team.add_member(TeamMember(member_id="real", role="worker", handler=known_handler))
        result = team.collaborate("test")
        assert result.success is True
        assert "real" in result.contributions
        assert "ghost" not in result.contributions


# ========================================================================
# Round 6 — I9: Team parallel execution thread-safety
# ========================================================================


class TestTeamParallelThreadSafety:
    """Parallel team execution must use a lock to protect shared state."""

    def test_parallel_contributions_consistent(self):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            json.dumps({"assignments": [
                {"member_id": f"m{i}", "sub_task": f"task {i}", "order": i}
                for i in range(8)
            ]}),
            *[f"output_{i}" for i in range(8)],
            json.dumps({"reviews": [
                {"member_id": f"m{i}", "quality_score": 8, "feedback": "ok"}
                for i in range(8)
            ]}),
            "final answer",
        ]

        team = AgentTeam("test_team", llm_client=mock_llm)
        team.parallel = True
        team.review_enabled = False
        for i in range(8):
            team.add_member(TeamMember(
                member_id=f"m{i}", role=f"role_{i}", specialty=f"spec_{i}",
            ))

        result = team.collaborate("do stuff")
        assert len(result.contributions) == 8
        for i in range(8):
            assert f"m{i}" in result.contributions


# ========================================================================
# Round 6 — I14: Team parallel mode runs review round
# ========================================================================


class TestParallelReviewRound:
    """When parallel=True and review_enabled=True, the review phase must
    still be invoked after parallel execution completes."""

    def test_parallel_review_called(self):
        from unittest.mock import patch as _patch

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            json.dumps({"assignments": [
                {"member_id": "m1", "sub_task": "t1", "order": 1},
                {"member_id": "m2", "sub_task": "t2", "order": 2},
            ]}),
            "output_1",
            "output_2",
            json.dumps({"reviews": [
                {"member_id": "m1", "quality_score": 8, "feedback": "good"},
                {"member_id": "m2", "quality_score": 8, "feedback": "good"},
            ]}),
            "synthesized",
        ]

        team = AgentTeam("review_team", llm_client=mock_llm)
        team.parallel = True
        team.review_enabled = True
        team.add_member(TeamMember(member_id="m1", role="r1", specialty="s1"))
        team.add_member(TeamMember(member_id="m2", role="r2", specialty="s2"))

        with _patch.object(team, "_review_and_refine", wraps=team._review_and_refine) as spy:
            team.collaborate("test task")
            spy.assert_called_once()
