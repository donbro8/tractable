"""Unit tests for tractable/types/git.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tractable.types.git import (
    BranchProtectionRules,
    CommitEntry,
    FileEntry,
    MergeResult,
    PullRequestHandle,
)

NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)


# ── PullRequestHandle ──────────────────────────────────────────────────


def test_pull_request_handle_instantiation() -> None:
    pr = PullRequestHandle(
        provider="github",
        repo_id="org/repo",
        pr_number=42,
        url="https://github.com/org/repo/pull/42",
        head_branch="feat/foo",
        base_branch="main",
    )
    assert pr.provider == "github"
    assert pr.pr_number == 42


def test_pull_request_handle_pr_number_can_be_str() -> None:
    pr = PullRequestHandle(
        provider="gitlab",
        repo_id="org/repo",
        pr_number="!7",
        url="https://gitlab.com/org/repo/-/merge_requests/7",
        head_branch="feat/bar",
        base_branch="main",
    )
    assert pr.pr_number == "!7"


def test_pull_request_handle_model_dump() -> None:
    pr = PullRequestHandle(
        provider="github",
        repo_id="org/repo",
        pr_number=1,
        url="https://example.com",
        head_branch="feat",
        base_branch="main",
    )
    data = pr.model_dump()
    assert set(data.keys()) == {"provider", "repo_id", "pr_number", "url", "head_branch", "base_branch"}


# ── MergeResult ────────────────────────────────────────────────────────


def test_merge_result_success() -> None:
    r = MergeResult(success=True, merge_commit_sha="abc123")
    assert r.success is True
    assert r.error is None


def test_merge_result_failure() -> None:
    r = MergeResult(success=False, error="conflict")
    assert r.success is False
    assert r.merge_commit_sha is None


# ── FileEntry ──────────────────────────────────────────────────────────


def test_file_entry_defaults() -> None:
    f = FileEntry(path="src/foo.py", is_directory=False)
    assert f.size_bytes is None
    assert f.sha is None


def test_file_entry_with_values() -> None:
    f = FileEntry(path="src/", is_directory=True, size_bytes=4096, sha="deadbeef")
    assert f.is_directory is True
    assert f.size_bytes == 4096


def test_file_entry_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        FileEntry(path="src/foo.py")  # type: ignore[call-arg]


# ── CommitEntry ────────────────────────────────────────────────────────


def test_commit_entry_default_files_changed() -> None:
    c = CommitEntry(sha="abc", message="init", author="Alice", timestamp=NOW)
    assert c.files_changed == []


def test_commit_entry_with_files() -> None:
    c = CommitEntry(
        sha="abc",
        message="fix",
        author="Bob",
        timestamp=NOW,
        files_changed=["src/a.py", "src/b.py"],
    )
    assert len(c.files_changed) == 2


def test_commit_entry_defaults_independent() -> None:
    """Mutable default lists must be independent per instance."""
    c1 = CommitEntry(sha="a", message="m", author="x", timestamp=NOW)
    c2 = CommitEntry(sha="b", message="m", author="y", timestamp=NOW)
    c1.files_changed.append("changed.py")
    assert c2.files_changed == []


# ── BranchProtectionRules ──────────────────────────────────────────────


def test_branch_protection_defaults() -> None:
    rules = BranchProtectionRules()
    assert rules.require_pr is True
    assert rules.required_approvals == 1
    assert rules.require_status_checks == []
    assert rules.restrict_push is True


def test_branch_protection_custom() -> None:
    rules = BranchProtectionRules(
        require_pr=False,
        required_approvals=2,
        require_status_checks=["ci/build", "ci/test"],
        restrict_push=False,
    )
    assert rules.required_approvals == 2
    assert len(rules.require_status_checks) == 2


def test_branch_protection_model_dump() -> None:
    rules = BranchProtectionRules()
    data = rules.model_dump()
    assert "require_pr" in data
    assert "required_approvals" in data
