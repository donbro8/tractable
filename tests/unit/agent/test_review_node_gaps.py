# pyright: reportPrivateUsage=false
"""Gap-fill tests for tractable/agent/nodes/review.py (TASK-3.3.4).

Covers paths not exercised by test_governance.py:
- _git_diff_stat_lines: git diff --stat output parsing (lines 65-76)
- reviewing_node: sensitive_path_blocked error propagation (lines 129-139)
- reviewing_node: linter gate failure (lines 151-153)
- reviewing_router: DONE_EDGE and RETRY_EDGE paths (line 200)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.agent.nodes.review import (
    DONE_EDGE,
    RETRY_EDGE,
    _git_diff_stat_lines,
    make_reviewing_node,
    reviewing_router,
)
from tractable.agent.state import AgentWorkflowState
from tractable.protocols.tool import ToolResult
from tractable.types.enums import TaskPhase


def _make_state(
    files_changed: list[str] | None = None,
    error: str | None = None,
    pr_url: str | None = None,
) -> AgentWorkflowState:
    state = AgentWorkflowState(
        agent_id="agent-test",
        task_id="task-test",
        task_description="test",
        phase=TaskPhase.REVIEWING,
        plan=[],
        files_changed=files_changed or [],
        test_results={},
        pr_url=pr_url,
        error=error,
        token_count=0,
        current_model="claude-sonnet-4-6",
        messages=[],
        resume_from=None,
    )
    state["replan_count"] = 0  # type: ignore[typeddict-unknown-key]
    return state


def _stub_state_store() -> MagicMock:
    store = MagicMock()
    store.save_checkpoint = AsyncMock()
    return store


# ── _git_diff_stat_lines ──────────────────────────────────────────────────────


def test_git_diff_stat_lines_parses_insertions_and_deletions() -> None:
    """Parsing a realistic git diff --stat summary returns insertions + deletions."""
    mock_result = MagicMock()
    mock_result.stdout = "3 files changed, 42 insertions(+), 7 deletions(-)\n"
    with patch("tractable.agent.nodes.review.subprocess.run", return_value=mock_result):
        count = _git_diff_stat_lines(["file1.py"])
    assert count == 49  # 42 + 7


def test_git_diff_stat_lines_empty_output_returns_zero() -> None:
    """Empty git diff --stat output returns 0."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    with patch("tractable.agent.nodes.review.subprocess.run", return_value=mock_result):
        count = _git_diff_stat_lines([])
    assert count == 0


def test_git_diff_stat_lines_only_insertions() -> None:
    """Output with only insertions (no deletions line) is parsed correctly."""
    mock_result = MagicMock()
    mock_result.stdout = "2 files changed, 10 insertions(+)\n"
    with patch("tractable.agent.nodes.review.subprocess.run", return_value=mock_result):
        count = _git_diff_stat_lines(["a.py"])
    assert count == 10


# ── reviewing_node: sensitive_path_blocked ────────────────────────────────────


@pytest.mark.asyncio
async def test_reviewing_node_sensitive_path_blocked_no_git_ops() -> None:
    """reviewing_node returns the sensitive-path error when no git_ops tool is present."""
    node = make_reviewing_node({}, _stub_state_store())
    state = _make_state(
        error="Sensitive path blocked: src/auth/tokens.py matches rule 'src/auth/**'"
    )

    result = await node(state)

    expected = "Sensitive path blocked: src/auth/tokens.py matches rule 'src/auth/**'"
    assert result["error"] == expected


@pytest.mark.asyncio
async def test_reviewing_node_sensitive_path_blocked_posts_pr_comment() -> None:
    """reviewing_node invokes git_ops with pr_comment when a sensitive-path error is present."""
    git_ops = MagicMock()
    git_ops.invoke = AsyncMock(return_value=ToolResult(success=True))

    node = make_reviewing_node({"git_ops": git_ops}, _stub_state_store())
    state = _make_state(
        error="Sensitive path blocked: db/migrations/001.sql matches rule '**/migrations/**'",
        pr_url="https://github.com/owner/repo/pull/5",
    )

    result = await node(state)

    git_ops.invoke.assert_called_once()
    call_params: dict[str, str] = git_ops.invoke.call_args[0][0]
    assert call_params.get("operation") == "pr_comment"
    assert "Sensitive path blocked" in call_params.get("body", "")
    assert result["error"] is not None


# ── reviewing_node: linter gate failure ───────────────────────────────────────


@pytest.mark.asyncio
async def test_reviewing_node_linter_gate_failure_returns_error() -> None:
    """reviewing_node returns the linter error when test_runner passes but linter fails."""
    test_runner = MagicMock()
    test_runner.invoke = AsyncMock(return_value=ToolResult(success=True))
    linter = MagicMock()
    linter.invoke = AsyncMock(return_value=ToolResult(success=False, error="E501 line too long"))

    node = make_reviewing_node(
        {"test_runner": test_runner, "linter": linter},
        _stub_state_store(),
    )
    state = _make_state()

    result = await node(state)

    assert result["error"] == "E501 line too long"


# ── reviewing_router ──────────────────────────────────────────────────────────


def test_reviewing_router_no_error_returns_done_edge() -> None:
    """reviewing_router returns DONE_EDGE when state['error'] is None."""
    state = _make_state(error=None)
    assert reviewing_router(state) == DONE_EDGE


def test_reviewing_router_with_error_returns_retry_edge() -> None:
    """reviewing_router returns RETRY_EDGE when state['error'] is set."""
    state = _make_state(error="test_runner gate failed")
    assert reviewing_router(state) == RETRY_EDGE
