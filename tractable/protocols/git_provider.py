"""GitProvider Protocol — uniform interface for any git hosting platform.

Source: tech-spec.py §2.1
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from tractable.types.git import (
    BranchProtectionRules,
    CommitEntry,
    FileEntry,
    MergeResult,
    PullRequestHandle,
)


@runtime_checkable
class GitProvider(Protocol):
    """
    Uniform interface for interacting with any git hosting platform.
    Agents never touch git directly — they call these methods.
    """

    async def clone(
        self,
        repo_id: str,
        target_path: str,
        branch: str = "main",
        sparse_paths: Sequence[str] | None = None,
    ) -> str:
        """Clone or sparse-checkout a repository. Returns local path."""
        ...

    async def create_branch(
        self,
        repo_id: str,
        branch_name: str,
        from_ref: str = "main",
    ) -> str:
        """Create a new branch. Returns the branch ref."""
        ...

    async def create_pull_request(
        self,
        repo_id: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        reviewers: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> PullRequestHandle:
        """Create a PR/MR. Returns a handle for further operations."""
        ...

    async def merge_pull_request(
        self,
        repo_id: str,
        pr_handle: PullRequestHandle,
        strategy: Literal["merge", "squash", "rebase"] = "squash",
    ) -> MergeResult:
        """Merge a PR if checks pass."""
        ...

    async def get_file_content(
        self,
        repo_id: str,
        file_path: str,
        ref: str = "main",
    ) -> bytes:
        """Read a single file at a given ref."""
        ...

    async def list_files(
        self,
        repo_id: str,
        path: str = "",
        ref: str = "main",
    ) -> Sequence[FileEntry]:
        """List files in a directory."""
        ...

    async def get_diff(
        self,
        repo_id: str,
        base_ref: str,
        head_ref: str,
    ) -> str:
        """Get unified diff between two refs."""
        ...

    async def get_commit_history(
        self,
        repo_id: str,
        path: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> Sequence[CommitEntry]:
        """Get commit log, optionally filtered by path."""
        ...

    async def set_branch_protection(
        self,
        repo_id: str,
        branch: str,
        rules: BranchProtectionRules,
    ) -> None:
        """Configure branch protection rules."""
        ...
