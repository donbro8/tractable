"""Unit tests for tractable/registry/main.py — _WakeOnWebhookPipeline.

Covers the fix for the duplicate `event` kwarg bug: verify that
process_change() emits event="agent_woke" via structlog and does NOT raise
when a matching agent is found.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import structlog

from tractable.protocols.reactivity import RepositoryChangeEvent, WebhookCommit
from tractable.registry.main import _WakeOnWebhookPipeline  # pyright: ignore[reportPrivateUsage]
from tractable.types.agent import AgentContext


def _make_event(repo_name: str = "test-repo") -> RepositoryChangeEvent:
    return RepositoryChangeEvent(
        event_id="test-evt-1",
        repo_name=repo_name,
        provider="github",
        event_type="push",
        ref="refs/heads/main",
        before_sha="0" * 40,
        after_sha="a" * 40,
        commits=[
            WebhookCommit(
                sha="a" * 40,
                message="Fix bug",
                author="dev",
                timestamp=datetime.now(tz=UTC),
                added_files=[],
                modified_files=["src/calc.py"],
                removed_files=[],
            )
        ],
        author="dev",
        timestamp=datetime.now(tz=UTC),
    )


def _make_ctx(agent_id: str, repo: str) -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        repo=repo,
        base_template="api_maintainer",
        system_prompt="Test agent.",
        repo_architectural_summary="",
        user_overrides={"status": "DORMANT"},
    )


@pytest.mark.asyncio
async def test_process_change_emits_agent_woke_log() -> None:
    """process_change() must emit event='agent_woke' via structlog for each matching agent."""
    agent_id = "test-agent-1"
    repo = "test-repo"
    ctx = _make_ctx(agent_id, repo)

    state_store = AsyncMock()
    state_store.list_agents = AsyncMock(return_value=[ctx])
    state_store.save_agent_context = AsyncMock()

    pipeline = _WakeOnWebhookPipeline(state_store)

    with structlog.testing.capture_logs() as logs:
        await pipeline.process_change(_make_event(repo_name=repo))

    wake_events = [r for r in logs if r.get("event") == "agent_woke"]
    assert wake_events, f"Expected event='agent_woke' in logs; got {logs}"
    assert wake_events[0]["agent_id"] == agent_id


@pytest.mark.asyncio
async def test_process_change_sets_last_active() -> None:
    """process_change() must set last_active and status='working' on matching agent."""
    agent_id = "test-agent-2"
    repo = "test-repo"
    ctx = _make_ctx(agent_id, repo)

    state_store = AsyncMock()
    state_store.list_agents = AsyncMock(return_value=[ctx])
    state_store.save_agent_context = AsyncMock()

    pipeline = _WakeOnWebhookPipeline(state_store)
    await pipeline.process_change(_make_event(repo_name=repo))

    state_store.save_agent_context.assert_awaited_once()
    saved_ctx: AgentContext = state_store.save_agent_context.call_args[0][1]
    assert saved_ctx.user_overrides.get("status") == "working"
    assert saved_ctx.user_overrides.get("last_active") is not None


@pytest.mark.asyncio
async def test_process_change_no_match_no_wake() -> None:
    """process_change() must not save or log when no agent matches the repo."""
    ctx = _make_ctx("other-agent", "other-repo")

    state_store = AsyncMock()
    state_store.list_agents = AsyncMock(return_value=[ctx])
    state_store.save_agent_context = AsyncMock()

    pipeline = _WakeOnWebhookPipeline(state_store)

    with structlog.testing.capture_logs() as logs:
        await pipeline.process_change(_make_event(repo_name="test-repo"))

    state_store.save_agent_context.assert_not_awaited()
    wake_events = [r for r in logs if r.get("event") == "agent_woke"]
    assert not wake_events
