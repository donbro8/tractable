"""Gap-fill tests for tractable/reactivity/webhook_receiver.py (TASK-3.3.4).

Covers paths not exercised by test_webhook_receiver.py:
- normalize_github_event: non-push event returns None (line 67)
- normalize_github_event: malformed JSON returns None (lines 71-72)
- normalize_github_event: commit loop body with valid timestamp (lines 80-87)
- normalize_github_event: invalid commit timestamp uses UTC fallback (line 87)
- normalize_github_event: non-integer pushed_at uses datetime.now fallback (line 106)
"""

from __future__ import annotations

import json
from typing import Any

from tractable.reactivity.webhook_receiver import normalize_github_event  # noqa: E402

# ── non-push event ────────────────────────────────────────────────────────────


def test_non_push_event_returns_none() -> None:
    """normalize_github_event returns None for non-push events (e.g. ping)."""
    body = json.dumps({"zen": "Keep it simple.", "hook_id": 1}).encode()
    headers = {"X-GitHub-Event": "ping", "X-GitHub-Delivery": "d-1"}
    assert normalize_github_event(headers, body) is None


def test_unsupported_event_type_returns_none() -> None:
    """normalize_github_event returns None for any unsupported event type."""
    body = json.dumps({"action": "opened"}).encode()
    headers = {"X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "d-2"}
    assert normalize_github_event(headers, body) is None


# ── malformed JSON ────────────────────────────────────────────────────────────


def test_malformed_json_returns_none() -> None:
    """normalize_github_event returns None when the body is not valid JSON."""
    headers = {"X-GitHub-Event": "push", "X-GitHub-Delivery": "d-3"}
    assert normalize_github_event(headers, b"not-valid-json{{{") is None


# ── commit loop: valid timestamp ──────────────────────────────────────────────


def test_push_with_valid_commit_timestamp_parsed_correctly() -> None:
    """normalize_github_event parses a valid ISO-8601 commit timestamp."""
    payload: dict[str, Any] = {
        "ref": "refs/heads/main",
        "before": "000",
        "after": "abc",
        "repository": {"full_name": "owner/repo", "pushed_at": 1742472000},
        "pusher": {"name": "dev"},
        "commits": [
            {
                "id": "sha-ok",
                "message": "feat: add feature",
                "author": {"name": "alice"},
                "timestamp": "2026-03-20T12:00:00+00:00",
                "added": ["src/new.py"],
                "modified": [],
                "removed": [],
            }
        ],
    }
    headers = {"X-GitHub-Event": "push", "X-GitHub-Delivery": "d-4"}
    event = normalize_github_event(headers, json.dumps(payload).encode())
    assert event is not None
    assert len(event.commits) == 1
    assert event.commits[0].sha == "sha-ok"
    assert event.commits[0].author == "alice"
    assert event.commits[0].timestamp.year == 2026
    assert event.commits[0].added_files == ["src/new.py"]


# ── commit loop: invalid timestamp ────────────────────────────────────────────


def test_commit_with_invalid_timestamp_uses_utc_fallback() -> None:
    """A commit with an unparseable timestamp defaults to datetime.now(UTC)."""
    payload: dict[str, Any] = {
        "ref": "refs/heads/main",
        "before": "000",
        "after": "abc",
        "repository": {"full_name": "owner/repo", "pushed_at": 1742472000},
        "pusher": {"name": "dev"},
        "commits": [
            {
                "id": "sha-bad",
                "message": "fix",
                "author": {"name": "bob"},
                "timestamp": "not-a-real-timestamp",
                "added": [],
                "modified": ["src/app.py"],
                "removed": [],
            }
        ],
    }
    headers = {"X-GitHub-Event": "push", "X-GitHub-Delivery": "d-5"}
    event = normalize_github_event(headers, json.dumps(payload).encode())
    assert event is not None
    assert len(event.commits) == 1
    # The commit timestamp should still be a valid timezone-aware datetime (fallback).
    assert event.commits[0].timestamp.tzinfo is not None


# ── pushed_at fallback ────────────────────────────────────────────────────────


def test_push_event_with_non_integer_pushed_at_uses_fallback() -> None:
    """normalize_github_event falls back to datetime.now() when pushed_at is None."""
    payload: dict[str, Any] = {
        "ref": "refs/heads/main",
        "before": "000",
        "after": "abc",
        "repository": {"full_name": "owner/repo", "pushed_at": None},
        "pusher": {"name": "dev"},
        "commits": [],
    }
    headers = {"X-GitHub-Event": "push", "X-GitHub-Delivery": "d-6"}
    event = normalize_github_event(headers, json.dumps(payload).encode())
    assert event is not None
    assert event.repo_name == "owner/repo"
    assert event.timestamp.tzinfo is not None
