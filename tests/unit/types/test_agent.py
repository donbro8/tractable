"""Unit tests for tractable/types/agent.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tractable.types.agent import (
    AgentCheckpoint,
    AgentContext,
    AuditEntry,
    ChangeVelocity,
    TemporalAgentContext,
)
from tractable.types.enums import TaskPhase
from tractable.types.temporal import ChangeNotification
from tractable.types.enums import ChangeRelevance

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


# ── AgentContext ───────────────────────────────────────────────────────


def test_agent_context_defaults() -> None:
    ctx = AgentContext(
        agent_id="agent-1",
        base_template="api_maintainer",
        system_prompt="You are...",
        repo_architectural_summary="REST API with FastAPI.",
    )
    assert ctx.known_patterns == []
    assert ctx.pinned_instructions == []
    assert ctx.user_overrides == {}
    assert ctx.last_refreshed is None


def test_agent_context_with_values() -> None:
    ctx = AgentContext(
        agent_id="agent-1",
        base_template="api_maintainer",
        system_prompt="prompt",
        repo_architectural_summary="summary",
        known_patterns=["prefer async", "use pydantic"],
        last_refreshed=NOW,
    )
    assert len(ctx.known_patterns) == 2
    assert ctx.last_refreshed == NOW


def test_agent_context_defaults_independent() -> None:
    c1 = AgentContext(agent_id="a1", base_template="t", system_prompt="p", repo_architectural_summary="s")
    c2 = AgentContext(agent_id="a2", base_template="t", system_prompt="p", repo_architectural_summary="s")
    c1.known_patterns.append("pattern")
    assert c2.known_patterns == []


def test_agent_context_model_dump() -> None:
    ctx = AgentContext(
        agent_id="a",
        base_template="t",
        system_prompt="p",
        repo_architectural_summary="s",
    )
    data = ctx.model_dump()
    assert "agent_id" in data
    assert "last_refreshed" in data


# ── AgentCheckpoint ────────────────────────────────────────────────────


def test_agent_checkpoint_instantiation() -> None:
    cp = AgentCheckpoint(
        task_id="task-1",
        phase=TaskPhase.EXECUTING,
        progress_summary="Working on PR creation",
        files_modified=["src/api.py"],
        pending_actions=["run tests"],
        conversation_summary="...",
        token_usage=1500,
        created_at=NOW,
    )
    assert cp.phase is TaskPhase.EXECUTING
    assert cp.token_usage == 1500


def test_agent_checkpoint_all_phases_valid() -> None:
    for phase in TaskPhase:
        cp = AgentCheckpoint(
            task_id="t",
            phase=phase,
            progress_summary="",
            files_modified=[],
            pending_actions=[],
            conversation_summary="",
            token_usage=0,
            created_at=NOW,
        )
        assert cp.phase is phase


# ── AuditEntry — outcome validation ───────────────────────────────────


def test_audit_entry_valid_outcomes() -> None:
    for outcome in ("success", "failure", "escalated"):
        entry = AuditEntry(
            timestamp=NOW,
            agent_id="a1",
            action="file_write",
            outcome=outcome,  # type: ignore[arg-type]
        )
        assert entry.outcome == outcome


def test_audit_entry_invalid_outcome_raises() -> None:
    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp=NOW,
            agent_id="a1",
            action="test",
            outcome="invalid",  # type: ignore[arg-type]
        )


def test_audit_entry_optional_task_id() -> None:
    entry = AuditEntry(timestamp=NOW, agent_id="a1", action="ping", outcome="success")
    assert entry.task_id is None


def test_audit_entry_detail_default_empty() -> None:
    entry = AuditEntry(timestamp=NOW, agent_id="a1", action="ping", outcome="success")
    assert entry.detail == {}


# ── ChangeVelocity ─────────────────────────────────────────────────────


def test_change_velocity_defaults() -> None:
    v = ChangeVelocity(
        commits_last_24h=3,
        commits_last_7d=15,
        entities_changed_last_24h=12,
        entities_changed_last_7d=50,
        cross_repo_changes_last_7d=2,
    )
    assert v.hotspot_files == []
    assert v.hotspot_entities == []


# ── TemporalAgentContext ───────────────────────────────────────────────


def test_temporal_agent_context_defaults() -> None:
    ctx = TemporalAgentContext(
        agent_id="a1",
        base_template="api",
        system_prompt="p",
        repo_architectural_summary="s",
    )
    assert ctx.last_active is None
    assert ctx.last_known_head_sha is None
    assert ctx.recent_changes_digest == ""
    assert ctx.pending_notifications == []
    assert ctx.change_velocity is None


def test_temporal_agent_context_pending_notifications() -> None:
    from tractable.types.enums import ChangeRelevance
    notif = ChangeNotification(
        target_agent_id="a1",
        repo_name="repo",
        relevance=ChangeRelevance.DIRECT,
        change_summary="added fn",
        commit_sha="abc",
        requires_action=True,
    )
    ctx = TemporalAgentContext(
        agent_id="a1",
        base_template="api",
        system_prompt="p",
        repo_architectural_summary="s",
        pending_notifications=[notif],
    )
    assert len(ctx.pending_notifications) == 1


def test_temporal_agent_context_with_velocity() -> None:
    velocity = ChangeVelocity(
        commits_last_24h=5,
        commits_last_7d=20,
        entities_changed_last_24h=10,
        entities_changed_last_7d=40,
        cross_repo_changes_last_7d=1,
    )
    ctx = TemporalAgentContext(
        agent_id="a1",
        base_template="api",
        system_prompt="p",
        repo_architectural_summary="s",
        change_velocity=velocity,
    )
    assert ctx.change_velocity is not None
    assert ctx.change_velocity.commits_last_24h == 5


def test_temporal_agent_context_defaults_independent() -> None:
    c1 = TemporalAgentContext(agent_id="a1", base_template="t", system_prompt="p", repo_architectural_summary="s")
    c2 = TemporalAgentContext(agent_id="a2", base_template="t", system_prompt="p", repo_architectural_summary="s")
    c1.known_patterns.append("x")
    assert c2.known_patterns == []
