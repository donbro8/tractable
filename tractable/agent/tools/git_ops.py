"""git_ops tool — branch, commit, push, and PR operations for the agent.

TASK-2.4.3: Implements the ``git_ops`` tool satisfying the ``Tool`` Protocol
(tech-spec.py §2.6).  Operations: create_branch, stage_and_commit, push,
open_pull_request.

Branch governance: agent-created branches must match ``^agent/[a-z0-9\\-]+$``;
any other name raises ``GovernanceError`` to prevent naming conflicts with human
conventions.

Commit message governance: the first line must be non-empty and ≤ 72 characters;
chain-of-thought markers (lines starting with ``<thinking>``) raise
``RecoverableError`` so the agent can retry with a corrected message.

Push errors: non-zero subprocess exit raises ``TransientError`` (network
failures are transient; the agent will retry with backoff).

Sources:
- tech-spec.py §2.1 — GitProvider Protocol
- tech-spec.py §2.6 — Tool Protocol
- CLAUDE.md        — Governance enforcement points, error taxonomy
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import structlog

from tractable.errors import GovernanceError, RecoverableError, TransientError
from tractable.protocols.git_provider import GitProvider
from tractable.protocols.tool import ToolResult
from tractable.types.git import PullRequestHandle

_log = structlog.get_logger()

_BRANCH_PATTERN = re.compile(r"^agent/[a-z0-9\-]+$")
_COT_MARKER_RE = re.compile(r"^<thinking>", re.MULTILINE)
_MAX_FIRST_LINE = 72


class GitOpsTool:
    """Git operations tool: branch creation, commit/push, and PR opening.

    Parameters
    ----------
    git_provider:
        Provider implementation satisfying the GitProvider Protocol.
    working_dir:
        Absolute path to the cloned repository root (subprocess cwd).
    repo_id:
        Repository identifier passed to the GitProvider (e.g. ``"org/repo"``).
    agent_id:
        Identifier of the running agent (included in every log entry).
    task_id:
        Identifier of the current task (included in every log entry).
    repo:
        Human-readable repository name (included in every log entry).
    """

    def __init__(
        self,
        git_provider: GitProvider,
        working_dir: Path,
        repo_id: str,
        agent_id: str,
        task_id: str,
        repo: str,
    ) -> None:
        self._git_provider = git_provider
        self._working_dir = working_dir.resolve()
        self._repo_id = repo_id
        self._agent_id = agent_id
        self._task_id = task_id
        self._repo = repo

    # ── Tool Protocol ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "git_ops"

    @property
    def description(self) -> str:
        return (
            "Create branches, stage and commit files, push to origin, and open "
            "pull requests. All branch names must follow the agent/ convention."
        )

    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        """Dispatch to the appropriate git operation.

        Expected ``params`` keys:

        - ``operation``: one of ``"create_branch"``, ``"stage_and_commit"``,
          ``"push"``, ``"open_pull_request"``
        - Operation-specific sub-parameters (see individual handlers).
        """
        operation: str = params.get("operation", "")

        if operation == "create_branch":
            return await self._create_branch(params)
        if operation == "stage_and_commit":
            return await self._stage_and_commit(params)
        if operation == "push":
            return await self._push(params)
        if operation == "open_pull_request":
            return await self._open_pull_request(params)
        if operation == "pr_comment":
            return self._pr_comment(params)

        return ToolResult(success=False, error=f"Unknown operation: {operation!r}")

    # ── Operations ───────────────────────────────────────────────────────────

    async def _create_branch(self, params: dict[str, Any]) -> ToolResult:
        branch_name: str = params.get("branch_name", "")
        from_ref: str = params.get("from_ref", "main")

        if not _BRANCH_PATTERN.match(branch_name):
            raise GovernanceError(
                f"Branch name {branch_name!r} does not match required pattern "
                r"'^agent/[a-z0-9\-]+$'. Agent branches must be prefixed with 'agent/'."
            )

        branch_ref = await self._git_provider.create_branch(
            self._repo_id,
            branch_name=branch_name,
            from_ref=from_ref,
        )

        _log.info(
            "branch_created",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            branch_name=branch_name,
            from_ref=from_ref,
        )

        return ToolResult(success=True, output=branch_ref)

    async def _stage_and_commit(self, params: dict[str, Any]) -> ToolResult:
        files: list[str] = params.get("files", [])
        commit_message: str = params.get("commit_message", "")

        self._validate_commit_message(commit_message)

        add_result = subprocess.run(
            ["git", "add", *files],
            capture_output=True,
            text=True,
            cwd=self._working_dir,
        )
        if add_result.returncode != 0:
            raise TransientError(
                f"git add failed (exit {add_result.returncode}): {add_result.stderr}"
            )

        commit_result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            cwd=self._working_dir,
        )
        if commit_result.returncode != 0:
            raise TransientError(
                f"git commit failed (exit {commit_result.returncode}): "
                f"{commit_result.stderr}"
            )

        _log.info(
            "files_committed",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            files=files,
        )

        return ToolResult(success=True)

    async def _push(self, params: dict[str, Any]) -> ToolResult:
        branch_name: str = params.get("branch_name", "")

        push_result = subprocess.run(
            ["git", "push", "origin", branch_name],
            capture_output=True,
            text=True,
            cwd=self._working_dir,
        )
        if push_result.returncode != 0:
            raise TransientError(
                f"git push failed (exit {push_result.returncode}): {push_result.stderr}"
            )

        _log.info(
            "branch_pushed",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            branch_name=branch_name,
        )

        return ToolResult(success=True)

    async def _open_pull_request(self, params: dict[str, Any]) -> ToolResult:
        title: str = params.get("title", "")
        body: str = params.get("body", "")
        head: str = params.get("head", "")
        base: str = params.get("base", "main")
        reviewers: list[str] = params.get("reviewers", [])

        pr_handle: PullRequestHandle = await self._git_provider.create_pull_request(
            self._repo_id,
            title=title,
            body=body,
            head_branch=head,
            base_branch=base,
            reviewers=reviewers if reviewers else None,
        )

        _log.info(
            "pull_request_opened",
            level="info",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            pr_number=pr_handle.pr_number,
            head_branch=pr_handle.head_branch,
        )

        return ToolResult(success=True, output=pr_handle.url)

    def _pr_comment(self, params: dict[str, Any]) -> ToolResult:
        """Log a PR comment notification for governance blocks.

        The GitProvider Protocol does not yet include a ``create_pr_comment``
        method, so comments are emitted as structured log entries and recorded
        in the audit trail.  A future milestone will add the provider method
        and replace this with a live API call.
        """
        pr_url: str = params.get("pr_url", "")
        body: str = params.get("body", "")
        _log.warning(
            "sensitive_path_pr_comment",
            agent_id=self._agent_id,
            task_id=self._task_id,
            repo=self._repo,
            pr_url=pr_url,
            body=body,
        )
        return ToolResult(success=True)

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_commit_message(self, message: str) -> None:
        """Raise RecoverableError if the commit message fails validation."""
        if not message:
            raise RecoverableError("Commit message must not be empty.")

        first_line = message.split("\n")[0]
        if len(first_line) > _MAX_FIRST_LINE:
            raise RecoverableError(
                f"Commit message first line is {len(first_line)} characters, "
                f"exceeding the {_MAX_FIRST_LINE}-character limit."
            )

        if _COT_MARKER_RE.search(message):
            raise RecoverableError(
                "Commit message contains chain-of-thought markers (e.g., lines "
                "starting with '<thinking>'). Remove internal reasoning before committing."
            )
