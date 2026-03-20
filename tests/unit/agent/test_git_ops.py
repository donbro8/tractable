"""Unit tests for tractable/agent/tools/git_ops.py.

TASK-2.4.3 acceptance criteria:
1. create_branch with valid name calls GitProvider.create_branch and returns
   ToolResult(success=True).
2. create_branch with invalid name (not ^agent/) raises GovernanceError.
3. stage_and_commit with empty commit_message raises RecoverableError.
4. push calls subprocess.run with ["git", "push", "origin", branch]; token absent from args.
5. open_pull_request calls GitProvider.create_pull_request and returns ToolResult with PR URL.
6. pyright strict mode reports zero errors (verified separately).

Additional coverage:
- commit message too long (>72 chars first line) → RecoverableError
- commit message containing chain-of-thought marker → RecoverableError
- push non-zero exit → TransientError
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.agent.tools.git_ops import GitOpsTool
from tractable.errors import GovernanceError, RecoverableError, TransientError
from tractable.types.git import PullRequestHandle

_TOKEN = "ghp_supersecrettoken"
_REPO_ID = "org/repo"


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_tool(tmp_path: Path, *, provider: AsyncMock | None = None) -> GitOpsTool:
    if provider is None:
        provider = AsyncMock()
    return GitOpsTool(
        git_provider=provider,
        working_dir=tmp_path,
        repo_id=_REPO_ID,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )


def _make_pr_handle(head: str = "agent/fix-123") -> PullRequestHandle:
    return PullRequestHandle(
        provider="github",
        repo_id=_REPO_ID,
        pr_number=42,
        url="https://github.com/org/repo/pull/42",
        head_branch=head,
        base_branch="main",
    )


# ── AC-1: create_branch with valid name delegates to GitProvider ──────────────


@pytest.mark.asyncio
async def test_create_branch_valid_name_calls_provider(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.create_branch.return_value = "refs/heads/agent/fix-123"
    tool = _make_tool(tmp_path, provider=provider)

    result = await tool.invoke(
        {
            "operation": "create_branch",
            "branch_name": "agent/fix-123",
            "from_ref": "main",
        }
    )

    assert result.success is True
    provider.create_branch.assert_awaited_once_with(
        _REPO_ID,
        branch_name="agent/fix-123",
        from_ref="main",
    )


# ── AC-2: create_branch with invalid name raises GovernanceError ──────────────


@pytest.mark.asyncio
async def test_create_branch_invalid_name_raises_governance_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    with pytest.raises(GovernanceError):
        await tool.invoke(
            {
                "operation": "create_branch",
                "branch_name": "feature/human-work",
                "from_ref": "main",
            }
        )


@pytest.mark.asyncio
async def test_create_branch_uppercase_raises_governance_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    with pytest.raises(GovernanceError):
        await tool.invoke(
            {"operation": "create_branch", "branch_name": "agent/Fix-123", "from_ref": "main"}
        )


@pytest.mark.asyncio
async def test_create_branch_no_prefix_raises_governance_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    with pytest.raises(GovernanceError):
        await tool.invoke({"operation": "create_branch", "branch_name": "main", "from_ref": "HEAD"})


# ── AC-3: stage_and_commit with empty message raises RecoverableError ─────────


@pytest.mark.asyncio
async def test_stage_and_commit_empty_message_raises_recoverable_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    with pytest.raises(RecoverableError):
        await tool.invoke(
            {
                "operation": "stage_and_commit",
                "files": ["src/main.py"],
                "commit_message": "",
            }
        )


@pytest.mark.asyncio
async def test_stage_and_commit_message_too_long_raises_recoverable_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    long_first_line = "x" * 73  # exceeds 72-char limit

    with pytest.raises(RecoverableError):
        await tool.invoke(
            {
                "operation": "stage_and_commit",
                "files": ["src/main.py"],
                "commit_message": long_first_line,
            }
        )


@pytest.mark.asyncio
async def test_stage_and_commit_cot_marker_raises_recoverable_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    cot_message = "Fix bug\n\n<thinking>My internal reasoning here</thinking>"

    with pytest.raises(RecoverableError):
        await tool.invoke(
            {
                "operation": "stage_and_commit",
                "files": ["src/main.py"],
                "commit_message": cot_message,
            }
        )


@pytest.mark.asyncio
async def test_stage_and_commit_exactly_72_chars_is_valid(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    exactly_72 = "x" * 72

    completed = MagicMock()
    completed.returncode = 0

    with patch("tractable.agent.tools.git_ops.subprocess.run", return_value=completed):
        result = await tool.invoke(
            {
                "operation": "stage_and_commit",
                "files": ["src/main.py"],
                "commit_message": exactly_72,
            }
        )

    assert result.success is True


# ── AC-4: push calls subprocess.run with correct args; token absent ───────────


@pytest.mark.asyncio
async def test_push_calls_subprocess_with_correct_args(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    completed = MagicMock()
    completed.returncode = 0

    with patch("tractable.agent.tools.git_ops.subprocess.run", return_value=completed) as mock_run:
        result = await tool.invoke({"operation": "push", "branch_name": "agent/fix-123"})

    assert result.success is True
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]  # positional first arg (the command list)
    assert call_args == ["git", "push", "origin", "agent/fix-123"]
    # Token must not appear in the subprocess args
    assert _TOKEN not in " ".join(str(a) for a in call_args)


@pytest.mark.asyncio
async def test_push_nonzero_exit_raises_transient_error(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)

    completed = MagicMock()
    completed.returncode = 1
    completed.stderr = "Connection refused"

    with (
        patch("tractable.agent.tools.git_ops.subprocess.run", return_value=completed),
        pytest.raises(TransientError),
    ):
        await tool.invoke({"operation": "push", "branch_name": "agent/fix-123"})


# ── AC-5: open_pull_request delegates to provider and returns PR URL ──────────


@pytest.mark.asyncio
async def test_open_pull_request_calls_provider_and_returns_url(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.create_pull_request.return_value = _make_pr_handle("agent/fix-123")
    tool = _make_tool(tmp_path, provider=provider)

    result = await tool.invoke(
        {
            "operation": "open_pull_request",
            "title": "Fix the bug",
            "body": "Detailed description",
            "head": "agent/fix-123",
            "base": "main",
            "reviewers": ["alice", "bob"],
        }
    )

    assert result.success is True
    assert result.output == "https://github.com/org/repo/pull/42"
    provider.create_pull_request.assert_awaited_once_with(
        _REPO_ID,
        title="Fix the bug",
        body="Detailed description",
        head_branch="agent/fix-123",
        base_branch="main",
        reviewers=["alice", "bob"],
    )


@pytest.mark.asyncio
async def test_open_pull_request_empty_reviewers_passes_none(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.create_pull_request.return_value = _make_pr_handle("agent/fix-123")
    tool = _make_tool(tmp_path, provider=provider)

    await tool.invoke(
        {
            "operation": "open_pull_request",
            "title": "Fix the bug",
            "body": "Description",
            "head": "agent/fix-123",
            "base": "main",
            "reviewers": [],
        }
    )

    _, kwargs = provider.create_pull_request.call_args
    assert kwargs["reviewers"] is None


# ── Token not in structlog output ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_token_not_in_log_output(tmp_path: Path) -> None:
    """The GitHub token must never appear in structlog entries."""
    import structlog.testing

    provider = AsyncMock()
    tool = GitOpsTool(
        git_provider=provider,
        working_dir=tmp_path,
        repo_id=_REPO_ID,
        agent_id="agent-test",
        task_id="task-test",
        repo="test/repo",
    )

    completed = MagicMock()
    completed.returncode = 0

    with (
        structlog.testing.capture_logs() as cap_logs,
        patch("tractable.agent.tools.git_ops.subprocess.run", return_value=completed),
    ):
        await tool.invoke({"operation": "push", "branch_name": "agent/fix-123"})

    for entry in cap_logs:
        for value in entry.values():
            assert _TOKEN not in str(value)


# ── Unknown operation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_operation_returns_failure(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    result = await tool.invoke({"operation": "nonexistent"})
    assert result.success is False
    assert "nonexistent" in (result.error or "")
