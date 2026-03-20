# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""FastAPI webhook receiver for GitHub push events.

TASK-2.6.1 — Implement FastAPI webhook receiver and ChangeIngestionPipeline.

Provides:
- ``verify_signature(headers, body, secret)`` — HMAC-SHA256 validation
- ``normalize_github_event(headers, body)`` — parse GitHub payload → RepositoryChangeEvent
- ``create_webhook_router(pipeline, webhook_secret)`` — FastAPI APIRouter with POST /webhooks/github
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Request, Response

from tractable.protocols.reactivity import (
    ChangeIngestionPipeline,
    RepositoryChangeEvent,
    WebhookCommit,
)

_log = structlog.get_logger()


def verify_signature(
    headers: dict[str, str],
    body: bytes,
    secret: str,
) -> bool:
    """Return True if the X-Hub-Signature-256 header matches the HMAC-SHA256 of body.

    GitHub computes the signature as:
        "sha256=" + HMAC-SHA256(secret, body).hexdigest()
    """
    signature: str | None = headers.get("X-Hub-Signature-256") or headers.get(
        "x-hub-signature-256"
    )
    if not signature:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def normalize_github_event(
    headers: dict[str, str],
    body: bytes,
) -> RepositoryChangeEvent | None:
    """Parse a raw GitHub webhook body into a ``RepositoryChangeEvent``.

    Returns ``None`` for unsupported event types (e.g. ping).
    """
    event_type_raw: str = headers.get("X-GitHub-Event") or headers.get(
        "x-github-event", ""
    )
    delivery_id: str = headers.get("X-GitHub-Delivery") or headers.get(
        "x-github-delivery", ""
    )

    if event_type_raw not in ("push",):
        return None

    try:
        payload: dict[str, Any] = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None

    repo_info: dict[str, Any] = payload.get("repository", {})
    repo_name: str = repo_info.get("full_name", "")

    commits_raw: list[dict[str, Any]] = payload.get("commits", [])
    commits: list[WebhookCommit] = []
    for c in commits_raw:
        author_info: dict[str, Any] = c.get("author") or {}
        author_name: str = str(author_info.get("name") or "")
        timestamp_str: str = str(c.get("timestamp") or "")
        try:
            ts: datetime = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            ts = datetime.now(tz=UTC)
        commits.append(
            WebhookCommit(
                sha=str(c.get("id") or ""),
                message=str(c.get("message") or ""),
                author=author_name,
                timestamp=ts,
                added_files=list(c.get("added") or []),
                modified_files=list(c.get("modified") or []),
                removed_files=list(c.get("removed") or []),
            )
        )

    pusher_info: dict[str, Any] = payload.get("pusher") or {}
    pusher_name: str = str(pusher_info.get("name") or "")

    pushed_at_raw: Any = payload.get("repository", {}).get("pushed_at")
    if isinstance(pushed_at_raw, (int, float)):
        event_timestamp: datetime = datetime.fromtimestamp(pushed_at_raw, tz=UTC)
    else:
        event_timestamp = datetime.now(tz=UTC)

    return RepositoryChangeEvent(
        event_id=delivery_id or f"github-push-{payload.get('after', '')}",
        repo_name=repo_name,
        provider="github",
        event_type="push",
        ref=payload.get("ref", ""),
        before_sha=payload.get("before"),
        after_sha=payload.get("after", ""),
        commits=commits,
        author=pusher_name,
        timestamp=event_timestamp,
    )


def create_webhook_router(
    pipeline: ChangeIngestionPipeline,
    webhook_secret: str,
) -> APIRouter:
    """Return a FastAPI ``APIRouter`` with ``POST /webhooks/github``.

    Signature verification is performed synchronously before accepting the
    request.  Ingestion is dispatched as a background task so the endpoint
    returns HTTP 202 immediately.
    """
    router = APIRouter()

    @router.post("/webhooks/github", status_code=202)
    async def handle_github_webhook(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> Response:
        body: bytes = await request.body()
        # Normalise header keys to canonical case for lookup.
        raw_headers: dict[str, str] = dict(request.headers)

        if not verify_signature(raw_headers, body, webhook_secret):
            _log.warning(
                "webhook_rejected",
                reason="invalid_signature",
            )
            return Response(
                content='{"error": "invalid_signature"}',
                status_code=401,
                media_type="application/json",
            )

        event: RepositoryChangeEvent | None = normalize_github_event(raw_headers, body)
        if event is not None:
            background_tasks.add_task(pipeline.process_change, event)

        return Response(status_code=202)

    return router
