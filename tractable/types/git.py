"""Git-related value types for the Tractable framework.

All models are pure data containers — no implementation logic.

Source: tech-spec.py §2.1 — Git Provider Protocol value types.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PullRequestHandle(BaseModel):
    """A reference to a pull/merge request on any git provider."""

    provider: str  # "github" | "gitlab" | "codecommit"
    repo_id: str
    pr_number: int | str
    url: str
    head_branch: str
    base_branch: str


class MergeResult(BaseModel):
    """Result of a merge operation."""

    success: bool
    merge_commit_sha: str | None = None
    error: str | None = None


class FileEntry(BaseModel):
    """A single file or directory entry from a repository listing."""

    path: str
    is_directory: bool
    size_bytes: int | None = None
    sha: str | None = None


class CommitEntry(BaseModel):
    """Metadata for a single git commit."""

    sha: str
    message: str
    author: str
    timestamp: datetime
    files_changed: list[str] = Field(default_factory=list)


class BranchProtectionRules(BaseModel):
    """Branch protection settings for a repository."""

    require_pr: bool = True
    required_approvals: int = 1
    require_status_checks: list[str] = Field(default_factory=list)
    restrict_push: bool = True


class CheckRunInfo(BaseModel):
    """A single CI check run from the git provider.

    Source: tech-spec.py §2.1 — GitProvider Protocol (CI check run value type).
    """

    name: str
    status: str  # "queued" | "in_progress" | "completed"
    conclusion: str | None = None  # "success" | "failure" | "skipped" | "cancelled" | ...
    log_url: str | None = None  # URL to fetch log text; None when not yet available
