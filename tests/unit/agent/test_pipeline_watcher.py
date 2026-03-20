"""Unit tests for pipeline_watcher.py and triage.py (TASK-2.4.5).

Acceptance criteria covered:
1. get_check_status with all checks passing → output["all_passed"] == True.
   Verified with mocked GitProvider.
2. get_check_status with one failed check → output["any_failed"] == True and
   output["failure_logs"] contains first 4000 chars of the mock log.
   Verified with mocked GitProvider.
3. PIPELINE_TRIAGE classifying "agent_caused" → state["phase"] == EXECUTING
   and new message added to state["messages"].
   Verified with mocked LLM (classify_fn).
4. PIPELINE_TRIAGE classifying "environment" → raises GovernanceError.
   Verified with mocked LLM.
5. pyright strict-mode clean (verified via CLI, not here).
6. ruff clean (verified via CLI, not here).

Additional coverage:
- unknown operation → ToolResult(success=False)
- all_passed=False when no check runs present
- PIPELINE_TRIAGE "flaky" path calls git_provider.rerun_failed_checks
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tractable.agent.nodes.triage import TriageState, build_triage_graph
from tractable.agent.tools.pipeline_watcher import PipelineWatcherTool
from tractable.errors import GovernanceError
from tractable.types.enums import TaskPhase
from tractable.types.git import CheckRunInfo

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tool(
    *,
    provider: AsyncMock | None = None,
    repo_id: str = "org/repo",
) -> PipelineWatcherTool:
    if provider is None:
        provider = AsyncMock()
    return PipelineWatcherTool(
        git_provider=provider,
        repo_id=repo_id,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


def _passing_run(name: str = "ci") -> CheckRunInfo:
    return CheckRunInfo(name=name, status="completed", conclusion="success", log_url=None)


def _failed_run(
    name: str = "ci", log_url: str | None = "https://logs.example.com/1"
) -> CheckRunInfo:
    return CheckRunInfo(name=name, status="completed", conclusion="failure", log_url=log_url)


def _base_triage_state(**overrides: object) -> TriageState:
    state: TriageState = {
        "agent_id": "agent-test",
        "task_id": "task-test",
        "repo_id": "org/repo",
        "pr_number": 42,
        "failure_logs": "ERROR: test_foo failed with AssertionError",
        "classification": None,
        "messages": [],
        "phase": TaskPhase.REVIEWING,
    }
    for k, v in overrides.items():
        state[k] = v  # type: ignore[literal-required]
    return state


# ── AC-1: all checks passing → all_passed=True ───────────────────────────────


@pytest.mark.asyncio
async def test_get_check_status_all_passed() -> None:
    provider = AsyncMock()
    provider.get_check_runs.return_value = [
        _passing_run("lint"),
        _passing_run("test"),
    ]
    tool = _make_tool(provider=provider)

    result = await tool.invoke({"operation": "get_check_status", "pr_number": 42})

    assert result.success
    assert isinstance(result.output, dict)
    assert result.output["all_passed"] is True
    assert result.output["any_failed"] is False
    assert "failure_logs" not in result.output
    provider.get_check_runs.assert_called_once_with("org/repo", 42)


# ── AC-2: one failed check → any_failed=True + failure_logs ──────────────────


@pytest.mark.asyncio
async def test_get_check_status_one_failed_includes_log() -> None:
    log_text = "X" * 5000  # longer than 4000 chars
    provider = AsyncMock()
    provider.get_check_runs.return_value = [
        _passing_run("lint"),
        _failed_run("test", log_url="https://logs.example.com/42"),
    ]
    provider.get_check_run_log.return_value = log_text

    tool = _make_tool(provider=provider)
    result = await tool.invoke({"operation": "get_check_status", "pr_number": 42})

    assert result.success
    assert isinstance(result.output, dict)
    assert result.output["any_failed"] is True
    assert result.output["all_passed"] is False
    assert "failure_logs" in result.output
    assert len(result.output["failure_logs"]) == 4000
    assert result.output["failure_logs"] == log_text[:4000]
    provider.get_check_run_log.assert_called_once_with("https://logs.example.com/42")


# ── AC-3: triage "agent_caused" → phase=EXECUTING + message ──────────────────


@pytest.mark.asyncio
async def test_triage_agent_caused_returns_executing_phase() -> None:
    async def mock_classify(_: str) -> str:
        return "agent_caused"

    git_provider = MagicMock()
    graph = build_triage_graph(classify_fn=mock_classify, git_provider=git_provider)

    result = await graph.ainvoke(_base_triage_state())

    assert result["phase"] == TaskPhase.EXECUTING
    assert len(result["messages"]) == 1
    assert "CI failure reason" in result["messages"][0]["content"]


# ── AC-4: triage "environment" → GovernanceError ─────────────────────────────


@pytest.mark.asyncio
async def test_triage_environment_raises_governance_error() -> None:
    async def mock_classify(_: str) -> str:
        return "environment"

    git_provider = MagicMock()
    graph = build_triage_graph(classify_fn=mock_classify, git_provider=git_provider)

    with pytest.raises(GovernanceError):
        await graph.ainvoke(_base_triage_state())


# ── Additional: triage "flaky" → rerun_failed_checks called ──────────────────


@pytest.mark.asyncio
async def test_triage_flaky_calls_rerun() -> None:
    async def mock_classify(_: str) -> str:
        return "flaky"

    git_provider = AsyncMock()
    graph = build_triage_graph(classify_fn=mock_classify, git_provider=git_provider)

    result = await graph.ainvoke(_base_triage_state())

    git_provider.rerun_failed_checks.assert_called_once_with("org/repo", 42)
    assert result["classification"] == "flaky"
    assert result["phase"] == TaskPhase.REVIEWING


# ── Additional: unknown operation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_operation_returns_error() -> None:
    tool = _make_tool()
    result = await tool.invoke({"operation": "does_not_exist"})
    assert not result.success
    assert result.error is not None
    assert "does_not_exist" in result.error


# ── Additional: empty check_runs → all_passed=False ──────────────────────────


@pytest.mark.asyncio
async def test_empty_check_runs_all_passed_false() -> None:
    provider = AsyncMock()
    provider.get_check_runs.return_value = []

    tool = _make_tool(provider=provider)
    result = await tool.invoke({"operation": "get_check_status", "pr_number": 1})

    assert result.success
    assert isinstance(result.output, dict)
    assert result.output["all_passed"] is False
    assert result.output["any_failed"] is False


# ── Additional: failed run with no log_url → failure_logs is empty string ────


@pytest.mark.asyncio
async def test_failed_run_no_log_url_gives_empty_failure_logs() -> None:
    provider = AsyncMock()
    provider.get_check_runs.return_value = [_failed_run(log_url=None)]

    tool = _make_tool(provider=provider)
    result = await tool.invoke({"operation": "get_check_status", "pr_number": 10})

    assert result.success
    assert isinstance(result.output, dict)
    assert result.output["any_failed"] is True
    assert result.output["failure_logs"] == ""
    provider.get_check_run_log.assert_not_called()
