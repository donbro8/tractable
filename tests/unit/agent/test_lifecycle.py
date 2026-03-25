"""Unit tests for tractable/agent/lifecycle.py — TASK-2.3.3.

Covers:
- AC-1: Debounce timer cancelled and reset on second notify_agent call
- AC-2: wake_agent on DORMANT → save_agent_context with status=WORKING
- AC-3: wake_agent on WORKING → warning event="agent_already_working", no save
- AC-4: sync_agent_repo clean → SyncResult(success=True, strategy_used="rebase")
- AC-5: sync_agent_repo conflict → RecoverableError
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.agent.lifecycle import LifecycleManager
from tractable.errors import RecoverableError
from tractable.types.agent import AgentCheckpoint, AgentContext, AgentStatus, AuditEntry
from tractable.types.enums import ChangeRelevance
from tractable.types.temporal import AgentReactivityConfig, ChangeNotification

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_notification(
    agent_id: str = "agent-1",
    relevance: ChangeRelevance = ChangeRelevance.DIRECT,
    requires_action: bool = True,
) -> ChangeNotification:
    return ChangeNotification(
        target_agent_id=agent_id,
        repo_name="my-repo",
        relevance=relevance,
        change_summary="A file changed",
        commit_sha="abc123",
        requires_action=requires_action,
    )


def _make_context(
    agent_id: str = "agent-1",
    status: AgentStatus | None = None,
) -> AgentContext:
    overrides: dict[str, Any] = {}
    if status is not None:
        overrides["status"] = status
    return AgentContext(
        agent_id=agent_id,
        base_template="test",
        system_prompt="",
        repo_architectural_summary="",
        user_overrides=overrides,
    )


class _MockStateStore:
    """In-memory AgentStateStore for testing."""

    def __init__(self, initial_context: AgentContext | None = None) -> None:
        self._ctx = initial_context or _make_context()
        self.save_agent_context = AsyncMock()
        self.saved_contexts: list[AgentContext] = []

        async def _save(agent_id: str, ctx: AgentContext) -> None:
            self.saved_contexts.append(ctx)

        self.save_agent_context.side_effect = _save

    async def get_agent_context(self, agent_id: str) -> AgentContext:
        return self._ctx

    async def get_checkpoint(self, agent_id: str, task_id: str) -> AgentCheckpoint | None:
        return None

    async def save_checkpoint(
        self, agent_id: str, task_id: str, checkpoint: AgentCheckpoint
    ) -> None:
        pass

    async def append_audit_entry(self, entry: AuditEntry) -> None:
        pass

    async def get_audit_log(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        return []


def _make_manager(
    state_store: _MockStateStore | None = None,
    status: AgentStatus | None = None,
    reactivity_configs: dict[str, AgentReactivityConfig] | None = None,
    working_dirs: dict[str, Path] | None = None,
) -> LifecycleManager:
    from unittest.mock import MagicMock

    store = state_store or _MockStateStore(initial_context=_make_context(status=status))
    graph = MagicMock()
    return LifecycleManager(
        state_store=store,
        graph=graph,
        registrations={},
        working_dirs=working_dirs or {},
        reactivity_configs=reactivity_configs,
    )


# ---------------------------------------------------------------------------
# AC-1: Debounce timer cancelled and reset on second notify_agent call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_schedules_debounce_timer() -> None:
    """First notify_agent call schedules a debounce timer."""
    mock_handle = MagicMock()
    mock_loop = MagicMock()
    mock_loop.call_later.return_value = mock_handle

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager()
        notification = _make_notification()
        await manager.notify_agent("agent-1", notification)

    assert mock_loop.call_later.call_count == 1
    # First positional arg is the delay (debounce_seconds default = 30)
    delay_arg = mock_loop.call_later.call_args[0][0]
    assert delay_arg == 30


@pytest.mark.asyncio
async def test_notify_cancels_existing_timer_on_second_call() -> None:
    """AC-1: Second notify_agent call cancels the first timer and schedules new one."""
    mock_handle_1 = MagicMock()
    mock_handle_2 = MagicMock()
    mock_loop = MagicMock()
    mock_loop.call_later.side_effect = [mock_handle_1, mock_handle_2]

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager()
        notification = _make_notification()

        await manager.notify_agent("agent-1", notification)
        await manager.notify_agent("agent-1", notification)

    # First timer must be cancelled before the second is scheduled.
    mock_handle_1.cancel.assert_called_once()
    assert mock_loop.call_later.call_count == 2


@pytest.mark.asyncio
async def test_notify_skips_timer_when_requires_action_false() -> None:
    """notify_agent does nothing when requires_action=False."""
    mock_loop = MagicMock()

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager()
        notification = _make_notification(requires_action=False)
        await manager.notify_agent("agent-1", notification)

    mock_loop.call_later.assert_not_called()


@pytest.mark.asyncio
async def test_notify_skips_timer_when_relevance_not_configured() -> None:
    """notify_agent does nothing when agent config disables that relevance level."""
    mock_loop = MagicMock()
    config = AgentReactivityConfig(wake_on_direct_change=False)

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager(reactivity_configs={"agent-1": config})
        notification = _make_notification(relevance=ChangeRelevance.DIRECT)
        await manager.notify_agent("agent-1", notification)

    mock_loop.call_later.assert_not_called()


@pytest.mark.asyncio
async def test_notify_uses_custom_debounce_seconds() -> None:
    """notify_agent uses debounce_seconds from AgentReactivityConfig."""
    mock_loop = MagicMock()
    config = AgentReactivityConfig(debounce_seconds=60)

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager(reactivity_configs={"agent-1": config})
        await manager.notify_agent("agent-1", _make_notification())

    delay_arg = mock_loop.call_later.call_args[0][0]
    assert delay_arg == 60


# ---------------------------------------------------------------------------
# AC-2: wake_agent on DORMANT → save_agent_context called with status=WORKING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_agent_dormant_transitions_to_working() -> None:
    """AC-2: wake_agent sets status=WORKING and calls save_agent_context."""
    store = _MockStateStore(initial_context=_make_context(status=AgentStatus.DORMANT))
    manager = _make_manager(state_store=store)

    await manager.wake_agent("agent-1", "direct_change")

    store.save_agent_context.assert_called_once()
    saved_ctx: AgentContext = store.saved_contexts[0]
    assert saved_ctx.user_overrides.get("status") == AgentStatus.WORKING


@pytest.mark.asyncio
async def test_wake_agent_idle_transitions_to_working() -> None:
    """wake_agent also works when agent is IDLE (not just DORMANT)."""
    store = _MockStateStore(initial_context=_make_context(status=AgentStatus.IDLE))
    manager = _make_manager(state_store=store)

    await manager.wake_agent("agent-1", "dependency_change")

    store.save_agent_context.assert_called_once()
    saved_ctx: AgentContext = store.saved_contexts[0]
    assert saved_ctx.user_overrides.get("status") == AgentStatus.WORKING


@pytest.mark.asyncio
async def test_wake_agent_sets_last_active_timestamp() -> None:
    """wake_agent writes last_active (ISO-8601) into user_overrides."""
    store = _MockStateStore(initial_context=_make_context())
    manager = _make_manager(state_store=store)

    await manager.wake_agent("agent-1", "direct_change")

    saved_ctx: AgentContext = store.saved_contexts[0]
    raw_ts = saved_ctx.user_overrides.get("last_active")
    assert raw_ts is not None
    # Must be parseable as a datetime.
    parsed = datetime.fromisoformat(str(raw_ts))
    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# AC-3: wake_agent on WORKING → warning, no save_agent_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_agent_already_working_no_save(caplog: object) -> None:
    """AC-3: wake_agent on WORKING state logs warning, does NOT call save."""
    store = _MockStateStore(initial_context=_make_context(status=AgentStatus.WORKING))
    manager = _make_manager(state_store=store)

    await manager.wake_agent("agent-1", "direct_change")

    store.save_agent_context.assert_not_called()


@pytest.mark.asyncio
async def test_wake_agent_already_working_logs_event() -> None:
    """AC-3: wake_agent on WORKING logs event='agent_already_working'."""
    import structlog.testing

    store = _MockStateStore(initial_context=_make_context(status=AgentStatus.WORKING))
    manager = _make_manager(state_store=store)

    with structlog.testing.capture_logs() as logs:
        await manager.wake_agent("agent-1", "direct_change")

    warning_events = [e for e in logs if e.get("event") == "agent_already_working"]
    assert len(warning_events) == 1
    assert warning_events[0]["agent_id"] == "agent-1"


# ---------------------------------------------------------------------------
# AC-4: sync_agent_repo clean → SyncResult(success=True, strategy_used="rebase")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_agent_repo_success() -> None:
    """AC-4: Clean git pull returns SyncResult(success=True, strategy_used='rebase')."""
    tmp_dir = Path("/tmp/fake-repo")
    store = _MockStateStore()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Fast-forwarded main to abc1234.\n",
            stderr="",
        )
        manager = _make_manager(
            state_store=store,
            working_dirs={"agent-1": tmp_dir},
        )
        result = await manager.sync_agent_repo("agent-1", "main")

    assert result.success is True
    assert result.strategy_used == "rebase"

    # Verify subprocess was called with correct args.
    call_args = mock_run.call_args[0][0]
    assert call_args == ["git", "pull", "--rebase", "origin", "main"]
    assert mock_run.call_args[1]["cwd"] == tmp_dir


# ---------------------------------------------------------------------------
# AC-5: sync_agent_repo conflict → RecoverableError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_agent_repo_conflict_raises_recoverable_error() -> None:
    """AC-5: git pull with CONFLICT in output raises RecoverableError."""
    tmp_dir = Path("/tmp/fake-repo")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="CONFLICT (content): Merge conflict in foo.py\n",
            stderr="error: Failed to merge in the changes.",
        )
        manager = _make_manager(working_dirs={"agent-1": tmp_dir})

        with pytest.raises(RecoverableError):
            await manager.sync_agent_repo("agent-1", "main")


@pytest.mark.asyncio
async def test_sync_agent_repo_nonzero_no_conflict_raises_recoverable_error() -> None:
    """Non-zero exit without CONFLICT keyword also raises RecoverableError."""
    tmp_dir = Path("/tmp/fake-repo")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )
        manager = _make_manager(working_dirs={"agent-1": tmp_dir})

        with pytest.raises(RecoverableError):
            await manager.sync_agent_repo("agent-1", "main")


@pytest.mark.asyncio
async def test_sync_agent_repo_no_working_dir_raises_recoverable_error() -> None:
    """sync_agent_repo raises RecoverableError when agent has no working dir."""
    manager = _make_manager(working_dirs={})

    with pytest.raises(RecoverableError):
        await manager.sync_agent_repo("agent-1", "main")


# ---------------------------------------------------------------------------
# get_agent_last_active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_agent_last_active_returns_none_when_never_active() -> None:
    """get_agent_last_active returns None for a fresh agent."""
    store = _MockStateStore(initial_context=_make_context())
    manager = _make_manager(state_store=store)

    result = await manager.get_agent_last_active("agent-1")
    assert result is None


@pytest.mark.asyncio
async def test_get_agent_last_active_returns_datetime_after_wake() -> None:
    """get_agent_last_active returns a datetime after wake_agent runs."""
    store = _MockStateStore(initial_context=_make_context())
    manager = _make_manager(state_store=store)

    await manager.wake_agent("agent-1", "direct_change")

    # Now update the store's in-memory context to reflect the saved state.
    store._ctx = store.saved_contexts[-1]  # type: ignore[attr-defined]

    result = await manager.get_agent_last_active("agent-1")
    assert result is not None
    assert isinstance(result, datetime)


# ---------------------------------------------------------------------------
# Dependency relevance routing in notify_agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_dependency_relevance_schedules_when_enabled() -> None:
    """DEPENDENCY relevance wakes agent when wake_on_dependency_change=True."""
    mock_loop = MagicMock()
    config = AgentReactivityConfig(wake_on_dependency_change=True)

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager(reactivity_configs={"agent-1": config})
        notification = _make_notification(relevance=ChangeRelevance.DEPENDENCY)
        await manager.notify_agent("agent-1", notification)

    mock_loop.call_later.assert_called_once()


@pytest.mark.asyncio
async def test_notify_consumer_relevance_skipped_by_default() -> None:
    """CONSUMER relevance does not wake agent by default."""
    mock_loop = MagicMock()

    with patch("asyncio.get_event_loop", return_value=mock_loop):
        manager = _make_manager()
        notification = _make_notification(relevance=ChangeRelevance.CONSUMER)
        await manager.notify_agent("agent-1", notification)

    mock_loop.call_later.assert_not_called()


# ---------------------------------------------------------------------------
# Working directory cleanup (TASK-3.3.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_working_dir_cleaned_up_on_completion(tmp_path: Path) -> None:
    """AC-3: shutil.rmtree is called with the working directory on task completion."""
    from tractable.agent.workflow import resume_task
    from tractable.errors import FatalError as _FatalError  # noqa: F401
    from tractable.types.enums import TaskPhase

    working_dir = tmp_path / "agent-work"
    working_dir.mkdir()

    mock_store = MagicMock()
    mock_store.get_checkpoint = AsyncMock(return_value=None)

    with patch("tractable.agent.workflow.build_workflow") as mock_build:
        mock_wf = MagicMock()
        mock_wf.ainvoke = AsyncMock(return_value={
            "agent_id": "agent-1",
            "task_id": "task-1",
            "phase": str(TaskPhase.COORDINATING),
            "plan": [],
            "files_changed": [],
            "test_results": {},
            "pr_url": None,
            "error": None,
            "token_count": 0,
            "current_model": "claude-sonnet-4-6",
            "messages": [],
        })
        mock_build.return_value = mock_wf

        with patch("shutil.rmtree") as mock_rmtree:
            await resume_task(
                agent_id="agent-1",
                task_id="task-1",
                task_description="test task",
                state_store=mock_store,
                tools={},
                graph=MagicMock(),
                working_dir=working_dir,
            )

    mock_rmtree.assert_any_call(str(working_dir), ignore_errors=True)


@pytest.mark.asyncio
async def test_working_dir_cleaned_up_on_fatal_error(tmp_path: Path) -> None:
    """AC-4: shutil.rmtree is called even when the workflow raises FatalError."""
    from tractable.agent.workflow import resume_task
    from tractable.errors import FatalError

    working_dir = tmp_path / "agent-work"
    working_dir.mkdir()

    mock_store = MagicMock()
    mock_store.get_checkpoint = AsyncMock(return_value=None)

    with patch("tractable.agent.workflow.build_workflow") as mock_build:
        mock_wf = MagicMock()
        mock_wf.ainvoke = AsyncMock(
            side_effect=FatalError("Token budget exhausted on Opus model")
        )
        mock_build.return_value = mock_wf

        with patch("shutil.rmtree") as mock_rmtree, pytest.raises(FatalError):
            await resume_task(
                agent_id="agent-1",
                task_id="task-1",
                task_description="test task",
                state_store=mock_store,
                tools={},
                graph=MagicMock(),
                working_dir=working_dir,
            )

    mock_rmtree.assert_any_call(str(working_dir), ignore_errors=True)
