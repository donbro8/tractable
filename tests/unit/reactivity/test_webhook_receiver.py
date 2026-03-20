"""Unit tests for webhook_receiver.py and ingestion_pipeline.py (TASK-2.6.1).

Covers:
- AC-1: Valid HMAC-SHA256 signature → HTTP 202.
- AC-2: Invalid signature → HTTP 401 + structlog event="webhook_rejected".
- AC-3: 3-file push calls get_file_content() x3 and apply_mutations() x1.
- AC-4: Duplicate event_id returns files_modified=0 without calling apply_mutations().
- AC-5: One file parse failure → warnings list populated, remaining files processed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tractable.protocols.graph_construction import ParseResult
from tractable.protocols.reactivity import (
    ChangeIngestionResult,
    RepositoryChangeEvent,
    WebhookCommit,
)
from tractable.reactivity.ingestion_pipeline import GitChangeIngestionPipeline
from tractable.reactivity.webhook_receiver import (
    verify_signature,
)
from tractable.registry.api import create_app
from tractable.types.temporal import TemporalMutationResult

# ── Helpers ────────────────────────────────────────────────────────────────

_SECRET = "test-webhook-secret"


def _make_signature(body: bytes, secret: str = _SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _push_payload(
    repo: str = "owner/repo",
    after: str = "abc123",
    delivery_id: str = "delivery-001",
    files_modified: list[str] | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Return (body_bytes, headers) for a minimal GitHub push event."""
    commits: list[dict[str, Any]] = []
    if files_modified:
        commits.append(
            {
                "id": "commit-sha-1",
                "message": "fix: update files",
                "author": {"name": "dev"},
                "timestamp": "2026-03-20T12:00:00+00:00",
                "added": [],
                "modified": files_modified,
                "removed": [],
            }
        )
    payload: dict[str, Any] = {
        "ref": "refs/heads/main",
        "before": "000000",
        "after": after,
        "repository": {"full_name": repo, "pushed_at": 1742472000},
        "pusher": {"name": "dev"},
        "commits": commits,
    }
    body = json.dumps(payload).encode()
    headers = {
        "X-GitHub-Event": "push",
        "X-GitHub-Delivery": delivery_id,
        "X-Hub-Signature-256": _make_signature(body),
        "content-type": "application/json",
    }
    return body, headers


def _noop_mutation_result() -> TemporalMutationResult:
    return TemporalMutationResult(
        entities_created=0,
        entities_updated=0,
        entities_deleted=0,
        edges_created=0,
        edges_deleted=0,
        timestamp=datetime.now(tz=UTC),
    )


# ── Stub pipeline for HTTP-layer tests ────────────────────────────────────


class _StubPipeline:
    async def process_change(self, event: RepositoryChangeEvent) -> ChangeIngestionResult:
        return ChangeIngestionResult(
            event_id=event.event_id,
            repo_name=event.repo_name,
            commit_sha=event.after_sha,
            files_added=0,
            files_modified=0,
            files_removed=0,
            parse_duration_ms=0,
            graph_mutations=_noop_mutation_result(),
        )


def _make_test_client() -> TestClient:
    app = create_app(pipeline=_StubPipeline(), webhook_secret=_SECRET)
    return TestClient(app, raise_server_exceptions=True)


# ── AC-1: valid signature returns 202 ─────────────────────────────────────


def test_valid_signature_returns_202() -> None:
    """AC-1: POST /webhooks/github with valid HMAC-SHA256 → HTTP 202."""
    client = _make_test_client()
    body, headers = _push_payload()
    response = client.post("/webhooks/github", content=body, headers=headers)
    assert response.status_code == 202


# ── AC-2: invalid signature returns 401 + structlog event ─────────────────


def test_invalid_signature_returns_401() -> None:
    """AC-2: Invalid signature → HTTP 401."""
    client = _make_test_client()
    body, headers = _push_payload()
    # Tamper with the signature.
    headers["X-Hub-Signature-256"] = "sha256=deadbeef"
    response = client.post("/webhooks/github", content=body, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "invalid_signature"}


def test_invalid_signature_logs_webhook_rejected() -> None:
    """AC-2: Invalid signature logs event='webhook_rejected' at warning level."""
    client = _make_test_client()
    body, headers = _push_payload()
    headers["X-Hub-Signature-256"] = "sha256=bad"

    with patch("tractable.reactivity.webhook_receiver._log") as mock_log:
        client.post("/webhooks/github", content=body, headers=headers)

    mock_log.warning.assert_called_once()
    call_kwargs = mock_log.warning.call_args
    # First positional arg is the event name.
    assert call_kwargs.args[0] == "webhook_rejected"


# ── AC-3: 3 modified files → get_file_content x3 and apply_mutations x1 ──


@pytest.mark.asyncio
async def test_three_file_push_calls_correct_counts() -> None:
    """AC-3: push with 3 modified .py files calls get_file_content x3 and apply_mutations x1."""
    files = ["src/a.py", "src/b.py", "src/c.py"]

    git_provider = AsyncMock()
    git_provider.get_file_content = AsyncMock(return_value=b"def foo(): pass\n")

    graph = AsyncMock()
    graph.apply_mutations = AsyncMock(return_value=_noop_mutation_result())

    # Parser returns an empty result (no entities extracted).
    parser = MagicMock()
    type(parser).supported_extensions = property(lambda _: frozenset({".py"}))
    parser.parse_file = AsyncMock(return_value=ParseResult(file_path="x.py", language="python"))

    # Redis: no duplicate
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()

    pipeline = GitChangeIngestionPipeline(
        git_provider=git_provider,
        graph=graph,
        parsers=[parser],
        redis_client=redis,
    )

    event = RepositoryChangeEvent(
        event_id="ev-ac3",
        repo_name="owner/repo",
        provider="github",
        event_type="push",
        ref="refs/heads/main",
        after_sha="abc123",
        commits=[
            WebhookCommit(
                sha="sha1",
                message="fix",
                author="dev",
                timestamp=datetime.now(tz=UTC),
                modified_files=files,
            )
        ],
        author="dev",
        timestamp=datetime.now(tz=UTC),
    )

    await pipeline.process_change(event)

    assert git_provider.get_file_content.call_count == 3
    graph.apply_mutations.assert_called_once()


# ── AC-4: duplicate event_id returns files_modified=0 ──────────────────────


@pytest.mark.asyncio
async def test_duplicate_event_returns_noop() -> None:
    """AC-4: Second call with same event_id returns files_modified=0 without apply_mutations."""
    git_provider = AsyncMock()
    graph = AsyncMock()
    graph.apply_mutations = AsyncMock(return_value=_noop_mutation_result())

    redis = AsyncMock()
    # Simulate already-seen event.
    redis.get = AsyncMock(return_value=b"1")
    redis.set = AsyncMock()

    pipeline = GitChangeIngestionPipeline(
        git_provider=git_provider,
        graph=graph,
        parsers=[],
        redis_client=redis,
    )

    event = RepositoryChangeEvent(
        event_id="ev-duplicate",
        repo_name="owner/repo",
        provider="github",
        event_type="push",
        ref="refs/heads/main",
        after_sha="abc123",
        commits=[],
        author="dev",
        timestamp=datetime.now(tz=UTC),
    )

    result = await pipeline.process_change(event)

    assert result.files_modified == 0
    graph.apply_mutations.assert_not_called()


# ── AC-5: parse failure → warning + continue ─────────────────────────────


@pytest.mark.asyncio
async def test_parse_failure_continues_remaining_files() -> None:
    """AC-5: Parse exception on one file → warning in result, remaining files processed."""
    files = ["src/ok.py", "src/bad.py"]

    git_provider = AsyncMock()
    git_provider.get_file_content = AsyncMock(return_value=b"x = 1")

    graph = AsyncMock()
    graph.apply_mutations = AsyncMock(return_value=_noop_mutation_result())

    call_count = 0

    async def _parse_file(file_path: str, content: bytes) -> ParseResult:
        nonlocal call_count
        call_count += 1
        if "bad" in file_path:
            raise ValueError("syntax error")
        return ParseResult(file_path=file_path, language="python")

    parser = MagicMock()
    type(parser).supported_extensions = property(lambda _: frozenset({".py"}))
    parser.parse_file = _parse_file

    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()

    pipeline = GitChangeIngestionPipeline(
        git_provider=git_provider,
        graph=graph,
        parsers=[parser],
        redis_client=redis,
    )

    event = RepositoryChangeEvent(
        event_id="ev-ac5",
        repo_name="owner/repo",
        provider="github",
        event_type="push",
        ref="refs/heads/main",
        after_sha="abc123",
        commits=[
            WebhookCommit(
                sha="sha1",
                message="fix",
                author="dev",
                timestamp=datetime.now(tz=UTC),
                modified_files=files,
            )
        ],
        author="dev",
        timestamp=datetime.now(tz=UTC),
    )

    result = await pipeline.process_change(event)

    # Warning recorded for the bad file.
    assert len(result.warnings) == 1
    assert "bad.py" in result.warnings[0]
    # apply_mutations still called (for the ok file).
    graph.apply_mutations.assert_called_once()


# ── verify_signature unit tests ────────────────────────────────────────────


def test_verify_signature_valid() -> None:
    """verify_signature returns True for matching signature."""
    body = b'{"foo": "bar"}'
    sig = _make_signature(body)
    assert verify_signature({"X-Hub-Signature-256": sig}, body, _SECRET) is True


def test_verify_signature_invalid() -> None:
    """verify_signature returns False for wrong signature."""
    body = b'{"foo": "bar"}'
    assert verify_signature({"X-Hub-Signature-256": "sha256=bad"}, body, _SECRET) is False


def test_verify_signature_missing_header() -> None:
    """verify_signature returns False when header is absent."""
    assert verify_signature({}, b"body", _SECRET) is False
